import logging
import os
import base64
import json
import shutil
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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.messages import HumanMessage

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# --- PATH & STORAGE SETUP ---
if os.getenv("K_SERVICE"):
    GCS_MOUNT_DIR = Path("/mnt/rag_data")
    LOCAL_TMP_DIR = Path("/tmp")
else:
    GCS_MOUNT_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
    LOCAL_TMP_DIR = GCS_MOUNT_DIR

DOCS_DIR = GCS_MOUNT_DIR / "docs"
GCS_CHROMA_DIR = GCS_MOUNT_DIR / "chroma_db"
LOCAL_CHROMA_DIR = LOCAL_TMP_DIR / "chroma_db"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
GCS_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "internal_documents")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "construction-rag-505118")
GCP_LOCATION = os.getenv("GCP_LOCATION", "global")
HTML_OUTPUT_DIR = PROJECT_ROOT / "extracted_html"
HTML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FETCH_K = 50
FINAL_K = 10

# Load credentials locally only (Cloud Run uses native Service Account ADC)
key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if key_path and not os.getenv("K_SERVICE"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(PROJECT_ROOT / key_path)

# Cloud-native embedding model (removes local Ollama daemon dependency)
embeddings_model = HuggingFaceEmbeddings(
    model_name="perplexity-ai/pplx-embed-v1-0.6b",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    project=GCP_PROJECT_ID,
    location=GCP_LOCATION,
    temperature=0.0,
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
    chunk_overlap=150,
)

vector_store = None
retriever = None
rag_chain = None

logger = logging.getLogger(__name__)


# --- GCS & LOCAL CHROMA SYNC HELPERS ---
def get_active_chroma_dir() -> Path:
    """Sync ChromaDB files from GCS FUSE to fast local container storage (/tmp) on boot."""
    if os.getenv("K_SERVICE"):
        if GCS_CHROMA_DIR.exists() and any(GCS_CHROMA_DIR.iterdir()):
            if not any(LOCAL_CHROMA_DIR.iterdir()):
                shutil.copytree(GCS_CHROMA_DIR, LOCAL_CHROMA_DIR, dirs_exist_ok=True)
        return LOCAL_CHROMA_DIR
    return GCS_CHROMA_DIR


def sync_chroma_to_gcs() -> None:
    """Sync updated local /tmp database back to GCS FUSE after document ingestion."""
    if os.getenv("K_SERVICE") and LOCAL_CHROMA_DIR.exists():
        shutil.copytree(LOCAL_CHROMA_DIR, GCS_CHROMA_DIR, dirs_exist_ok=True)


def _make_cad_doc(
    text: str,
    bbox: List[float],
    page_num: int,
    file_name: str,
    method: str,
) -> Document:
    """Create a PDF document chunk with a single bounding box in its provenance metadata."""
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
        },
    )


def _box_distance(b1: List[float], b2: List[float]) -> float:
    """Return the minimum edge-to-edge distance between two bounding boxes."""
    dx = max(0.0, b1[0] - b2[2], b2[0] - b1[2])
    dy = max(0.0, b1[1] - b2[3], b2[1] - b1[3])
    return (dx**2 + dy**2) ** 0.5


def _make_cad_doc_multi_bbox(
    text: str,
    boxes: List[List[float]],
    page_num: int,
    file_name: str,
    method: str,
) -> Document:
    """Create a document chunk that retains multiple bounding boxes in provenance metadata."""
    prov = [{"page_no": page_num, "bbox": b, "type": method} for b in boxes]

    enclosing_bbox = [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
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
        },
    )


def _chunk_shx_annotations(
    page: fitz.Page,
    page_num: int,
    file_name: str,
    min_word_cutoff: int = 10,
) -> List[Document]:
    """Extract SHX annotation text and group nearby annotations into document chunks."""
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
                    extracted.append({"text": clean_text, "bbox": bbox})

    if not extracted:
        return []

    clusters = [{"texts": [item["text"]], "boxes": [item["bbox"]]} for item in extracted]

    def cluster_envelope(cluster: dict) -> List[float]:
        """Return the bounding envelope for a spatial cluster."""
        boxes = cluster["boxes"]
        return [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ]

    def cluster_word_count(cluster: dict) -> int:
        """Return the total word count across the texts within a cluster."""
        return len(" ".join(cluster["texts"]).split())

    while len(clusters) > 1:
        short_idx = None
        for idx, cl in enumerate(clusters):
            if cluster_word_count(cl) <= min_word_cutoff:
                short_idx = idx
                break

        if short_idx is None:
            break

        short_cluster = clusters[short_idx]
        short_env = cluster_envelope(short_cluster)

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
            target = clusters[best_neighbor_idx]
            target["texts"].extend(short_cluster["texts"])
            target["boxes"].extend(short_cluster["boxes"])
            clusters.pop(short_idx)
        else:
            break

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
    max_gap: float = 25.0,
) -> List[Document]:
    """Group native PDF text blocks by their spatial layout and create document chunks."""
    blocks = sorted(
        [{"text": b[4].strip(), "bbox": [float(x) for x in b[:4]]} for b in page.get_text("blocks") if b[4].strip()],
        key=lambda b: (b["bbox"][1], b["bbox"][0]),
    )
    if not blocks:
        return []

    docs, cur_txt, cur_box = [], [], []

    def flush():
        """Emit the current accumulated text block cluster as a document chunk."""
        if cur_txt:
            env = [
                min(b[0] for b in cur_box),
                min(b[1] for b in cur_box),
                max(b[2] for b in cur_box),
                max(b[3] for b in cur_box),
            ]
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
    """Render a page to an image and ask Gemini to return structured HTML with bounding boxes."""
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

    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]

    return content.strip()


def parse_vlm_html_to_documents(raw_html: str, file_name: str, page_num: int) -> List[Document]:
    """Split HTML returned by Gemini into chunked document records with bounding-box metadata."""
    chunks = html_splitter.split_text(raw_html)
    documents = []

    for idx, chunk in enumerate(chunks):
        bbox_match = re.search(r'data-bbox="\[(.*?)\]"', chunk)
        bbox = [float(x.strip()) for x in bbox_match.group(1).split(",")] if bbox_match else []

        prov = [{
            "page_no": page_num,
            "type": "gemini_vlm_ocr",
            "bbox": bbox,
            "chunk_index": idx,
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
                },
            )
        )

    return documents


def _get_compressed_retriever(store: Chroma) -> ContextualCompressionRetriever:
    """Wrap a Chroma store with a Vertex AI contextual compression retriever."""
    base_retriever = store.as_retriever(search_kwargs={"k": FETCH_K})
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )


def _load_text_documents(file_path: Path) -> List[Document]:
    """Load a text or markdown file into a list of LangChain documents with metadata."""
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
    """Pull text from the available fields of a PyMuPDF annotation object."""
    info = annot.info
    text = (info.get("content") or info.get("subject") or info.get("title") or "").strip()
    return text or (annot.get_text().strip() if hasattr(annot, "get_text") else "")


def load_documents_from_folder(
    property_name: str = "default",
    target_files: Optional[List[str]] = None,
) -> Tuple[List[Document], Dict[str, str]]:
    """Load every document in a property folder and return the records plus a file report."""
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

    for doc in documents:
        doc.metadata["property_name"] = property_name

    return documents, file_report


def build_or_reload_chain() -> None:
    """Reinitialize the vector store and retrieval chain from the current document set."""
    global vector_store, retriever, rag_chain

    active_dir = get_active_chroma_dir()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(active_dir),
        embedding_function=embeddings_model,
    )
    retriever = _get_compressed_retriever(vector_store)
    answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, answer_chain)


def ingest_documents(
    property_name: str = "default",
    target_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract content from files, split it into chunks, and add it to the Chroma index."""
    global vector_store, retriever, rag_chain

    raw_docs, _ = load_documents_from_folder(property_name=property_name, target_files=target_files)
    final_chunks: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for doc in raw_docs:
        if doc.metadata.get("file_type") == "pdf":
            final_chunks.append(doc)
        else:
            split_docs = text_splitter.split_documents([doc])
            final_chunks.extend(split_docs)

    valid_chunks = [d for d in final_chunks if d.page_content and d.page_content.strip()]
    if not valid_chunks:
        return {"chunks_indexed": 0, "documents_loaded": len(raw_docs), "property": property_name}

    for chunk in valid_chunks:
        chunk.metadata["property_name"] = property_name

    active_dir = get_active_chroma_dir()

    if vector_store is None:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(active_dir),
            embedding_function=embeddings_model,
        )

    batch_size = 50
    for i in range(0, len(valid_chunks), batch_size):
        vector_store.add_documents(valid_chunks[i : i + batch_size])

    build_or_reload_chain()
    sync_chroma_to_gcs()

    return {
        "chunks_indexed": len(valid_chunks),
        "documents_loaded": len(raw_docs),
        "property": property_name,
    }


def preview_chroma_chunks(
    limit: int = 5,
    chars: int = 400,
    filename: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return preview records for the most recent Chroma documents, optionally filtered by name."""
    if vector_store is None:
        return []

    data = vector_store.get(
        limit=limit,
        where={"source": filename} if filename else None,
        include=["documents", "metadatas"],
    )

    previews: List[Dict[str, Any]] = []
    for text, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
        clean_meta = {k: v for k, v in meta.items() if k != "dl_prov"}
        previews.append({"text": text[:chars], "metadata": clean_meta})

    return previews


def _extract_source_info(doc: Document, index: int) -> Dict[str, Any]:
    """Convert a retrieved document into the citation payload used in the frontend."""
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
        "property_name": meta.get("property_name"),  # Expose property_name to frontend
    }


def ask_question(question: str, property_name: Optional[str] = None) -> Dict[str, Any]:
    """Run a property-aware retrieval and reranking flow, then answer using the grounded context."""
    global vector_store

    if vector_store is None:
        build_or_reload_chain()

    filter_dict = None
    if property_name and property_name != "All":
        filter_dict = {"property_name": property_name}

    raw_candidates = vector_store.similarity_search(
        question,
        k=FETCH_K,
        filter=filter_dict,
    )

    if not raw_candidates:
        return {
            "answer": f"No relevant documents found for property: '{property_name or 'All'}'.",
            "sources": [],
        }

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
        llm=llm,
        prompt=prompt_template,
        document_prompt=document_prompt,
    )
    response = answer_chain.invoke({"context": reranked_docs, "input": question})
    answer_text = (
        response.get("answer") or response.get("output_text") or str(response)
        if isinstance(response, dict) else str(response)
    )

    sources = [_extract_source_info(doc, i) for i, doc in enumerate(reranked_docs, start=1)]
    return {"answer": answer_text, "sources": sources}