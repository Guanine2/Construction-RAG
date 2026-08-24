import logging
import os
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
import pymupdf as fitz
import re 
from langchain_google_community import VertexAIRank
from langchain_classic.retrievers import ContextualCompressionRetriever

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.messages import HumanMessage

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

DOCS_DIR = PROJECT_ROOT / "docs"

if os.getenv("K_SERVICE"):
    BASE_DATA_DIR = Path("/mnt/rag_data")
else:
    BASE_DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()


DOCS_DIR = BASE_DATA_DIR / "docs"
CHROMA_DIR = BASE_DATA_DIR / "chroma_db"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "internal_documents")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "construction-rag-505118")
GCP_LOCATION = os.getenv("GCP_LOCATION", "global")
HTML_OUTPUT_DIR = PROJECT_ROOT / "extracted_html"
HTML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FETCH_K = 50
FINAL_K = 10

key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if key_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(PROJECT_ROOT / key_path)

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    project=GCP_PROJECT_ID,
    location=GCP_LOCATION,
    temperature=0.0
)

reranker = VertexAIRank(
    project_id=GCP_PROJECT_ID,
    location_id=GCP_LOCATION,
    ranking_config="default_ranking_config",
    model="semantic-ranker-fast@latest",
    top_n=FINAL_K,
)

system_prompt = """
You are an expert internal document intelligence assistant for engineering and legal files.

CRITICAL INSTRUCTIONS:
1. Grounding: Answer ONLY using the provided Context. If the context does not contain sufficient facts, explicitly state: "I cannot find this information in the provided documents."
2. Citation: For EVERY factual claim or number, explicitly cite the source number using brackets (e.g., [Source 1] or [Source 2]). Do NOT cite document filenames or page numbers.
3. Reasoning: Step-by-step, verify that your claims match the context before producing your final answer.

Context:
{context}
""".strip()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

document_prompt = PromptTemplate.from_template(
    "[Source {source_index} | File: {source} | Page: {page_number}]\n"
    "Document Content:\n{page_content}\n"
)

html_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML,
    chunk_size=1000,
    chunk_overlap=150
)

vector_store = None
retriever = None
rag_chain = None

logger = logging.getLogger(__name__)

def _make_cad_doc(text: str, bbox: List[float], page_num: int, file_name: str, method: str) -> Document:
    """Helper to instantiate standardized LangChain Documents with spatial provenance metadata."""
    prov = [{"page_no": page_num, "bbox": bbox, "type": method}]
    return Document(
        page_content=text,
        metadata={
            "source": file_name,
            "file_type": "pdf",
            "page_number": page_num,
            "page_numbers": str(page_num),
            "extraction_method": method,
            "bbox": json.dumps(bbox),
            "dl_prov": json.dumps(prov),
        }
    )
def _box_distance(b1: List[float], b2: List[float]) -> float:
    """Calculates the minimum edge-to-edge distance between two bounding boxes [x0, y0, x1, y1]."""
    dx = max(0.0, b1[0] - b2[2], b2[0] - b1[2])
    dy = max(0.0, b1[1] - b2[3], b2[1] - b1[3])
    return (dx ** 2 + dy ** 2) ** 0.5

def _make_cad_doc_multi_bbox(
    text: str,
    boxes: List[List[float]],
    page_num: int,
    file_name: str,
    method: str
) -> Document:
    """Creates a Document keeping all individual annotation bounding boxes in dl_prov."""
    prov = [{"page_no": page_num, "bbox": b, "type": method} for b in boxes]
    
    # Macro bounding box enclosing all items for top-level metadata
    enclosing_bbox = [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes)
    ]
    
    return Document(
        page_content=text,
        metadata={
            "source": file_name,
            "file_type": "pdf",
            "page_number": page_num,
            "page_numbers": str(page_num),
            "extraction_method": method,
            "bbox": json.dumps(enclosing_bbox),
            "dl_prov": json.dumps(prov),
        }
    )


def _chunk_shx_annotations(
    page: fitz.Page,
    page_num: int,
    file_name: str,
    min_word_cutoff: int = 10
) -> List[Document]:
    annots = page.annots()
    if not annots:
        return []

    extracted = []
    seen = set()

    for annot in annots:
        shx_text = _extract_shx_text_thorough(annot)
        if shx_text:
            clean_text = shx_text.replace("AutoCAD SHX Text:", "").strip()
            if clean_text:
                raw_rect = annot.rect
                bbox = [float(raw_rect.x0), float(raw_rect.y0), float(raw_rect.x1), float(raw_rect.y1)]
                
                dedup_key = (clean_text, round(raw_rect.x0, 1), round(raw_rect.y0, 1))
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    extracted.append({
                        "text": clean_text,
                        "bbox": bbox
                    })

    if not extracted:
        return []

    # Initialize each extracted annotation as its own spatial cluster
    clusters = [{"texts": [item["text"]], "boxes": [item["bbox"]]} for item in extracted]

    def cluster_envelope(cluster: dict) -> List[float]:
        boxes = cluster["boxes"]
        return [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes)
        ]

    def cluster_word_count(cluster: dict) -> int:
        return len(" ".join(cluster["texts"]).split())

    # Iterative spatial merging loop for short clusters
    while len(clusters) > 1:
        # Find the first cluster containing <= min_word_cutoff words
        short_idx = None
        for idx, cl in enumerate(clusters):
            if cluster_word_count(cl) <= min_word_cutoff:
                short_idx = idx
                break

        # Stop merging if all remaining clusters exceed min_word_cutoff words
        if short_idx is None:
            break

        short_cluster = clusters[short_idx]
        short_env = cluster_envelope(short_cluster)

        # Find the geometrically closest neighbor cluster on the 2D canvas
        best_neighbor_idx = None
        min_dist = float("inf")

        for idx, other_cl in enumerate(clusters):
            if idx == short_idx:
                continue
            dist = _box_distance(short_env, cluster_envelope(other_cl))
            if dist < min_dist:
                min_dist = dist
                best_neighbor_idx = idx

        if best_neighbor_idx is not None:
            # Merge short_cluster into its nearest spatial neighbor
            target = clusters[best_neighbor_idx]
            target["texts"].extend(short_cluster["texts"])
            target["boxes"].extend(short_cluster["boxes"])
            clusters.pop(short_idx)
        else:
            break

    # Convert finalized spatial clusters into LangChain Documents
    docs: List[Document] = []
    for cl in clusters:
        combined_text = " ".join(cl["texts"])
        docs.append(
            _make_cad_doc_multi_bbox(combined_text, cl["boxes"], page_num, file_name, "shx_annotation")
        )

    return docs


def _chunk_native_text_spatially(
    page: fitz.Page,
    page_num: int,
    file_name: str,
    max_chars: int = 1200,
    max_gap: float = 25.0
) -> List[Document]:
    """Extracts native text blocks and chunks them using spatial distance thresholds."""
    blocks = sorted(
        [{"text": b[4].strip(), "bbox": [float(x) for x in b[:4]]} for b in page.get_text("blocks") if b[4].strip()],
        key=lambda b: (b["bbox"][1], b["bbox"][0])
    )
    if not blocks:
        return []

    docs, cur_txt, cur_box = [], [], []

    def flush():
        if cur_txt:
            env = [min(b[0] for b in cur_box), min(b[1] for b in cur_box), max(b[2] for b in cur_box), max(b[3] for b in cur_box)]
            docs.append(_make_cad_doc("\n\n".join(cur_txt), env, page_num, file_name, "pymupdf_native_spatial"))
            cur_txt.clear()
            cur_box.clear()

    for b in blocks:
        if cur_box:
            prev_ymax, curr_ymin = cur_box[-1][3], b["bbox"][1]
            if (curr_ymin - prev_ymax > max_gap) or (curr_ymin < prev_ymax - 15.0) or (sum(map(len, cur_txt)) >= max_chars):
                flush()
        cur_txt.append(b["text"])
        cur_box.append(b["bbox"])

    flush()
    return docs

def _extract_page_vlm_html(page: fitz.Page, page_num: int) -> str:
    """Renders a PDF page to image and extracts structured HTML with bounding boxes using Gemini."""
    # Render at 200 DPI for high CAD text clarity
    pix = page.get_pixmap(dpi=300)
    base64_image = base64.b64encode(pix.tobytes("png")).decode("utf-8")

    prompt = f"""
    You are a high-precision document intelligence engine performing layout-aware text extraction.

    TASK:
    Extract ALL text from the provided document page while maintaining visual hierarchy.

    OUTPUT FORMAT:
    Return clean, semantic HTML directly as a single-line string. Do NOT output markdown code blocks (e.g., ```html). Do NOT include literal newline characters (\\n).

    EXTRACTION RULES:
    1. Top-Level Page Wrapper: Wrap the entire page content in a root <section> tag containing `data-page="{page_num}"`.
    2. Logical Grouping: Wrap visually or semantically related groups of elements (e.g., site plans, zoning tables, general notes) in nested <section> tags.
    3. Bounding Boxes: Include `data-bbox="[ymin, xmin, ymax, xmax]"` (scaled 0-1000) on all structural tags (<section>, <h1>-<h6>, <p>, <table>, <div>).
    4. Semantic Hierarchy: Use semantic tags (<h1>-<h6> for headers, <p> for text/notes, <table> for schedules/tables).
    5. Fidelity: Transcribe all text, numbers, dimensions, and codes EXACTLY as printed without summarization.
    6. Single Line Constraint: Do NOT insert line breaks or newlines anywhere in the string. Output everything on one continuous line.
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            },
        ]
    )

    response = llm.invoke([message])
    content = str(response.content).strip()

    # Clean codeblock wrapping if model still generates it
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]

    return content.strip()

def parse_vlm_html_to_documents(raw_html: str, file_name: str, page_num: int) -> List[Document]:
    """Splits raw Gemini VLM HTML into chunked LangChain Documents with parsed bounding box metadata."""
    chunks = html_splitter.split_text(raw_html)
    documents = []

    for idx, chunk in enumerate(chunks):
        # Extract highest-level bounding box found inside the chunk HTML snippet
        bbox_match = re.search(r'data-bbox="\[(.*?)\]"', chunk)
        bbox = [float(x.strip()) for x in bbox_match.group(1).split(",")] if bbox_match else []

        prov = [{
            "page_no": page_num,
            "type": "gemini_vlm_ocr",
            "bbox": bbox,
            "chunk_index": idx
        }]

        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": file_name,
                    "file_type": "pdf",
                    "page_number": page_num,
                    "page_numbers": str(page_num),
                    "extraction_method": "gemini_vlm_ocr",
                    "bbox": json.dumps(bbox),
                    "dl_prov": json.dumps(prov),
                }
            )
        )

    return documents


def _get_compressed_retriever(store: Chroma) -> ContextualCompressionRetriever:
    """Wraps a Chroma vector store with VertexAIRank contextual compression.

    Args:
        store (Chroma): The initialized Chroma vector store instance.

    Returns:
        ContextualCompressionRetriever: A retriever equipped with two-stage reranking.
    """
    base_retriever = store.as_retriever(search_kwargs={"k": FETCH_K})
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )
    


def _load_text_documents(file_path: Path) -> List[Document]:
    """Loads plain text or markdown files into LangChain Documents with standard metadata.

    Args:
        file_path (Path): Path object pointing to the text or markdown document.

    Returns:
        List[Document]: List of instantiated Document objects with source and page metadata.
    """
    docs = TextLoader(str(file_path), encoding="utf-8").load()
    for doc in docs:
        doc.metadata.update({
            "source": file_path.name,
            "file_type": file_path.suffix.lstrip("."),
            "page_number": 1,
            "page_numbers": "1",
        })
    return docs


def _extract_shx_text_thorough(annot: fitz.Annot) -> str:
    """Extracts text content from PyMuPDF annotation dictionary fields.

    Args:
        annot (fitz.Annot): PyMuPDF annotation object.

    Returns:
        str: Extracted string content from subject, title, content, or underlying text layers.
    """
    info = annot.info
    text = (info.get("content") or info.get("subject") or info.get("title") or "").strip()
    return text or (annot.get_text().strip() if hasattr(annot, "get_text") else "")



def load_documents_from_folder(
    property_name: str = "default",
    target_files: Optional[List[str]] = None
) -> Tuple[List[Document], Dict[str, str]]:
    
    # Target property-specific directory
    prop_dir = DOCS_DIR / property_name
    prop_dir.mkdir(parents=True, exist_ok=True)

    documents: List[Document] = []
    file_report: Dict[str, str] = {}

    candidate_paths = (
        [prop_dir / f if not Path(f).is_absolute() else Path(f) for f in target_files]
        if target_files else sorted(prop_dir.iterdir())
    )

    for file_path in candidate_paths:
        if not file_path.exists() or file_path.is_dir():
            continue

        if file_path.suffix.lower() in {".txt", ".md"}:
            txt_docs = _load_text_documents(file_path)
            for d in txt_docs:
                d.metadata["extraction_method"] = "raw_text"
            documents.extend(txt_docs)
            file_report[file_path.name] = "raw_text"

        elif file_path.suffix.lower() == ".pdf":
            doc = fitz.open(str(file_path))
            shx_pages_count, vlm_pages_count = 0, 0

            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1

                shx_docs = _chunk_shx_annotations(page, page_num, file_path.name)

                if shx_docs:
                    documents.extend(shx_docs)
                    documents.extend(_chunk_native_text_spatially(page, page_num, file_path.name))
                    shx_pages_count += 1
                else:
                    raw_html = _extract_page_vlm_html(page, page_num)
                    (HTML_OUTPUT_DIR / f"{Path(file_path).stem}_page_{page_num}.html").write_text(raw_html, encoding="utf-8")
                    documents.extend(parse_vlm_html_to_documents(raw_html, file_path.name, page_num))
                    vlm_pages_count += 1

            file_report[file_path.name] = f"shx_pages({shx_pages_count})_vlm_pages({vlm_pages_count})"

    # Tag every loaded document with the property name metadata
    for doc in documents:
        doc.metadata["property_name"] = property_name

    return documents, file_report


def build_or_reload_chain() -> None:
    """Initializes or reloads the active Chroma vector store and contextual compression RAG pipeline."""
    global vector_store, retriever, rag_chain

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings_model,
    )
    retriever = _get_compressed_retriever(vector_store)
    answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, answer_chain)


def ingest_documents(
    property_name: str = "default",
    target_files: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Processes, tags with property metadata, and appends chunks into ChromaDB."""
    global vector_store, retriever, rag_chain

    # 1. Load documents for the specific property
    raw_docs, _ = load_documents_from_folder(property_name=property_name, target_files=target_files)   
    final_chunks: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # 2. Split non-PDF documents while preserving property_name
    for doc in raw_docs:
        if doc.metadata.get("file_type") == "pdf":
            final_chunks.append(doc)
        else:
            split_docs = text_splitter.split_documents([doc])
            final_chunks.extend(split_docs)

    valid_chunks = [d for d in final_chunks if d.page_content and d.page_content.strip()]
    if not valid_chunks:
        return {"chunks_indexed": 0, "documents_loaded": len(raw_docs), "property": property_name}
    
    # 3. Ensure property_name is attached to every single chunk
    for chunk in valid_chunks:
        chunk.metadata["property_name"] = property_name

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 4. Initialize vector_store if needed, then append new documents (DO NOT use Chroma.from_documents)
    if vector_store is None:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings_model,
        )

    # 5. Append in batches so existing vector data for other properties remains intact
    batch_size = 50 
    for i in range(0, len(valid_chunks), batch_size):
        vector_store.add_documents(valid_chunks[i : i + batch_size])
    
    # 6. Refresh active RAG chain
    build_or_reload_chain()

    return {
        "chunks_indexed": len(valid_chunks), 
        "documents_loaded": len(raw_docs),
        "property": property_name
    }
    
def preview_chroma_chunks(
    limit: int = 5, 
    chars: int = 400, 
    filename: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves document previews directly from ChromaDB with clean metadata outputs.

    Args:
        limit (int): Maximum number of chunk records to return. Defaults to 5.
        chars (int): Character truncation limit for the text preview snippet. Defaults to 400.
        filename (Optional[str]): Optional filename filter for target document previews.

    Returns:
        List[Dict[str, Any]]: List of dictionary previews containing chunk text and sanitized metadata.
    """
    if vector_store is None:
        return []

    data = vector_store.get(
        limit=limit, 
        where={"source": filename} if filename else None,
        include=["documents", "metadatas"]
    )

    previews: List[Dict[str, Any]] = []
    for text, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
        clean_meta = {k: v for k, v in meta.items() if k != "dl_prov"}
        previews.append({"text": text[:chars], "metadata": clean_meta})

    return previews


def _extract_source_info(doc: Document, index: int) -> Dict[str, Any]:
    """Extracts citation metadata and deserializes bounding box values from a Document object.

    Args:
        doc (Document): The LangChain Document instance containing metadata.
        index (int): The 1-based index allocated to the document for citation tracking.

    Returns:
        Dict[str, Any]: Dictionary containing index, source name, page numbers, and parsed provenance lists.
    """
    meta = getattr(doc, "metadata", {}) or {}
    dl_prov = []
    if "dl_prov" in meta:
        try:
            dl_prov = json.loads(meta["dl_prov"])
        except Exception:
            dl_prov = []

    return {
        "citation_id": index,
        "source": meta.get("source", "Unknown"),
        "file_type": meta.get("file_type", "Unknown"),
        "page_number": meta.get("page_number", 1),
        "page_numbers": str(meta.get("page_numbers", "N/A")),
        "dl_prov": dl_prov,
    }



def ask_question(question: str, property_name: Optional[str] = None) -> Dict[str, Any]:
    """Executes a two-stage retrieval query with VertexAIRank compression and cited LLM generation.

    Args:
        question (str): The natural language query string.

    Returns:
        Dict[str, Any]: Payload containing the grounded LLM answer and associated citation source objects.
    """
    global vector_store

    # 1. Ensure vector_store is initialized FIRST
    if vector_store is None:
        build_or_reload_chain()

    # 2. Build ChromaDB filter
    filter_dict = None
    if property_name and property_name != "All":
        filter_dict = {"property_name": property_name}

    # 3. Retrieve raw candidates directly using the filter
    raw_candidates = vector_store.similarity_search(
        question, 
        k=FETCH_K, 
        filter=filter_dict
    )

    # 4. Guard against empty results before invoking Vertex AI Reranker
    if not raw_candidates:
        return {
            "answer": f"No relevant documents found for property: '{property_name or 'All'}'.",
            "sources": []
        }

    # 5. Rerank valid documents
    reranked_docs = reranker.compress_documents(documents=raw_candidates, query=question)
    content_to_meta = {doc.page_content: doc.metadata for doc in raw_candidates}
    print("\n=================== RERANKED RESULTS ===================")
    for rank, doc in enumerate(reranked_docs, start=1):
        meta = doc.metadata
        score = meta.get("relevance_score", "N/A")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        snippet = doc.page_content.strip().replace("\n", " ")[:120]
        print(f"Rank [{rank:02d}] | Score: {score_str} | File: {meta.get('source', 'Unknown')} | Page: {meta.get('page_number', 1)}")
        print(f"         Snippet: \"{snippet}...\"\n")
    print("===========================================================================\n")

    for i, doc in enumerate(reranked_docs, start=1):
        doc.metadata.update(content_to_meta.get(doc.page_content, {}))
        doc.metadata["source_index"] = i
        doc.metadata.setdefault("source", "Unknown")
        doc.metadata.setdefault("page_number", 1)

    answer_chain = create_stuff_documents_chain(
        llm=llm, prompt=prompt_template, document_prompt=document_prompt
    )
    response = answer_chain.invoke({"context": reranked_docs, "input": question})
    answer_text = (
        response.get("answer") or response.get("output_text") or str(response)
        if isinstance(response, dict) else str(response)
    )

    sources = [_extract_source_info(doc, i) for i, doc in enumerate(reranked_docs, start=1)]
    return {"answer": answer_text, "sources": sources}