import logging
import os
import time
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
import fitz
import torch
import re 
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    AcceleratorOptions,
    AcceleratorDevice
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_google_community import VertexAIRank
from langchain_classic.retrievers import ContextualCompressionRetriever

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_docling import DoclingLoader
from langchain_docling.loader import BaseMetaExtractor, ExportType
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
CHROMA_DIR = PROJECT_ROOT / "chroma_db" / "langchain_document_intelligence"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "internal_documents")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "global")
HTML_OUTPUT_DIR = PROJECT_ROOT / "extracted_html"
HTML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["DOCLING_DEVICE"] = "cpu"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

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
    

class PageAwareMetaExtractor(BaseMetaExtractor):
    """Metadata extractor that parses page numbers and provenance details from Docling objects."""

    def extract_chunk_meta(self, file_path: str, chunk: Any) -> Dict[str, Any]:
        """Extracts chunk-level metadata including page numbers and bounding box provenance.

        Args:
            file_path (str): Path to the source file being processed.
            chunk (Any): The Docling chunk object containing document metadata and items.

        Returns:
            Dict[str, Any]: Extracted metadata containing page numbers, source filename, and serialized provenance JSON.
        """
        dl_meta = chunk.meta.export_json_dict() if hasattr(chunk, "meta") else {}
        page_numbers, prov_list = [], []

        if "doc_items" in dl_meta:
            for item in dl_meta["doc_items"]:
                for prov in item.get("prov", []):
                    if "page_no" in prov:
                        page_numbers.append(prov["page_no"])
                    elif "page_number" in prov:
                        page_numbers.append(prov["page_number"])
                    if "bbox" in prov:
                        prov_list.append(prov)

        unique_pages = sorted(list(set(page_numbers)))
        metadata: Dict[str, Any] = {
            "source": Path(file_path).name,
            "file_type": "pdf",
            "page_number": unique_pages[0] if unique_pages else 1,
            "page_numbers": ", ".join(map(str, unique_pages)) if unique_pages else "N/A",
            "dl_prov": json.dumps(prov_list)
        }
        return metadata

    def extract_dl_doc_meta(self, file_path: str, dl_doc: Any) -> Dict[str, Any]:
        """Extracts top-level metadata for a processed Docling document instance.

        Args:
            file_path (str): Path to the source file.
            dl_doc (Any): The top-level Docling document instance.

        Returns:
            Dict[str, Any]: Basic document metadata containing source filename and file type.
        """
        return {"source": Path(file_path).name, "file_type": "pdf"}


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


def load_documents_from_folder(target_files: Optional[List[str]] = None) -> Tuple[List[Document], Dict[str, str]]:
    """Loads documents using PyMuPDF for SHX annotations. Bypasses VLM only on pages with SHX data; 
    routes all other pages (native vector text or scanned) through Gemini VLM HTML extraction.
    """
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Document folder not found: {DOCS_DIR}")

    documents: List[Document] = []
    file_report: Dict[str, str] = {}

    candidate_paths = (
        [DOCS_DIR / f if not Path(f).is_absolute() else Path(f) for f in target_files]
        if target_files else sorted(DOCS_DIR.iterdir())
    )

    for file_path in candidate_paths:
        if not file_path.exists():
            continue

        if file_path.suffix.lower() in {".txt", ".md"}:
            txt_docs = _load_text_documents(file_path)
            for d in txt_docs:
                d.metadata["extraction_method"] = "raw_text"
            documents.extend(txt_docs)
            file_report[file_path.name] = "raw_text"

        elif file_path.suffix.lower() == ".pdf":
            doc = fitz.open(str(file_path))
            shx_pages_count = 0
            vlm_pages_count = 0

            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1
                shx_docs_on_page: List[Document] = []

                # 1. Collect SHX annotations on the page
                for annot in page.annots():
                    shx_text = _extract_shx_text_thorough(annot)
                    if shx_text:
                        rect = annot.rect
                        bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
                        prov = [{"page_no": page_num, "bbox": bbox, "type": "shx_annotation"}]

                        shx_docs_on_page.append(
                            Document(
                                page_content=shx_text,
                                metadata={
                                    "source": file_path.name,
                                    "file_type": "pdf",
                                    "page_number": page_num,
                                    "page_numbers": str(page_num),
                                    "extraction_method": "shx_annotation",
                                    "bbox": json.dumps(bbox),
                                    "dl_prov": json.dumps(prov)
                                }
                            )
                        )

                # 2. Routing logic
                if shx_docs_on_page:
                    # SHX annotations exist -> append SHX text and skip VLM
                    documents.extend(shx_docs_on_page)
                    shx_pages_count += 1
                else:
                    # No SHX annotations (native vector text OR scanned) -> process via Gemini VLM
                    raw_html = _extract_page_vlm_html(page, page_num)
                    
                    html_filename = f"{Path(file_path).stem}_page_{page_num}.html"
                    html_filepath = HTML_OUTPUT_DIR / html_filename
                    html_filepath.write_text(raw_html, encoding="utf-8")
                    
                    vlm_docs = parse_vlm_html_to_documents(raw_html, file_path.name, page_num)
                    documents.extend(vlm_docs)
                    vlm_pages_count += 1

            file_report[file_path.name] = f"shx_pages({shx_pages_count})_vlm_pages({vlm_pages_count})"

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


def ingest_documents(target_files: Optional[List[str]] = None) -> Dict[str, int]:
    """Processes, chunks, and writes documents into ChromaDB in batches before reloading the pipeline.

    Args:
        target_files (Optional[List[str]]): Optional list of specific file paths or names to ingest.

    Returns:
        Dict[str, int]: Summary dictionary containing total chunk and loaded document counts.
    """
    global vector_store, retriever, rag_chain

    raw_docs, _ = load_documents_from_folder(target_files=target_files)   
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
            final_chunks.extend(text_splitter.split_documents([doc]))

    valid_chunks = [d for d in final_chunks if d.page_content and d.page_content.strip()]
    if not valid_chunks:
        return {"chunks_indexed": 0, "documents_loaded": len(raw_docs)}
    
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    batch_size = 50 
    
    vector_store = Chroma.from_documents(
        documents=valid_chunks[:batch_size],
        embedding=embeddings_model,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    
    for i in range(batch_size, len(valid_chunks), batch_size):
        vector_store.add_documents(valid_chunks[i : i + batch_size])
    
    retriever = _get_compressed_retriever(vector_store)
    answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, answer_chain)

    return {"chunks_indexed": len(final_chunks), "documents_loaded": len(raw_docs)}


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


def ask_question(question: str) -> Dict[str, Any]:
    """Executes a two-stage retrieval query with VertexAIRank compression and cited LLM generation.

    Args:
        question (str): The natural language query string.

    Returns:
        Dict[str, Any]: Payload containing the grounded LLM answer and associated citation source objects.
    """
    global vector_store
    if vector_store is None:
        build_or_reload_chain()

    raw_candidates = vector_store.similarity_search(question, k=FETCH_K)
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