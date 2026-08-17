import logging
import os
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import fitz
import torch

# Docling Imports
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
# LangChain Integrations & Core

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_docling import DoclingLoader
from langchain_docling.loader import BaseMetaExtractor, ExportType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain Chains Import with Fallback Support
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

# ==========================================
# CONFIGURATION & PATHS
# ==========================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db" / "langchain_document_intelligence"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "internal_documents")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "global")

os.environ["DOCLING_DEVICE"] = "cpu"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

FETCH_K = 50
FINAL_K = 10

key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if key_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(PROJECT_ROOT / key_path)

# ==========================================
# MODELS & PROMPTS SETUP
# ==========================================
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
    model="semantic-ranker-fast@latest",  # or "semantic-ranker-default-004"
    top_n=FINAL_K,                         # Number of final documents to keep
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

# Global runtime state for DB and Chain
vector_store = None
retriever = None
rag_chain = None

# Reranker configuration
logger = logging.getLogger(__name__)


def _get_compressed_retriever(store: Chroma) -> ContextualCompressionRetriever:
    """Wraps Chroma vector store with VertexAIRank compression."""
    base_retriever = store.as_retriever(search_kwargs={"k": FETCH_K})
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )
    
class PageAwareMetaExtractor(BaseMetaExtractor):
    def extract_chunk_meta(self, file_path: str, chunk: Any) -> Dict[str, Any]:
        dl_meta = chunk.meta.export_json_dict() if hasattr(chunk, "meta") else {}

        page_numbers = []
        prov_list = []

        # Extract page numbers and bounding box objects from underlying doc items
        if "doc_items" in dl_meta:
            for item in dl_meta["doc_items"]:
                for prov in item.get("prov", []):
                    # Capture page index
                    if "page_no" in prov:
                        page_numbers.append(prov["page_no"])
                    elif "page_number" in prov:
                        page_numbers.append(prov["page_number"])

                    # Capture bounding box data
                    if "bbox" in prov:
                        prov_list.append(prov)

        unique_pages = sorted(list(set(page_numbers)))
        file_name = Path(file_path).name

        metadata: Dict[str, Any] = {
            "source": file_name,
            "file_type": "pdf",
        }

        if unique_pages:
            metadata["page_number"] = unique_pages[0]
            metadata["page_numbers"] = ", ".join(map(str, unique_pages))
        else:
            metadata["page_number"] = 1
            metadata["page_numbers"] = "N/A"

        # Serialize provenance/bbox array to a JSON string for ChromaDB compatibility
        metadata["dl_prov"] = json.dumps(prov_list)

        return metadata

    def extract_dl_doc_meta(self, file_path: str, dl_doc: Any) -> Dict[str, Any]:
        return {"source": Path(file_path).name, "file_type": "pdf"}


# ==========================================
# DOCUMENT LOADERS
# ==========================================
def _load_text_documents(file_path: Path) -> List[Document]:
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({
            "source": file_path.name,
            "file_type": file_path.suffix.lstrip("."),
            "page_number": 1,           # <--- REQUIRED for prompt template
            "page_numbers": "1",
        })
    return docs


def load_documents_from_folder(target_files: Optional[List[str]] = None) -> List[Document]:
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Document folder not found: {DOCS_DIR}")

    documents: List[Document] = []
    pdf_paths: List[Path] = []

    # 1. Determine candidate files
    candidate_paths = (
        [DOCS_DIR / f if not Path(f).is_absolute() else Path(f) for f in target_files]
        if target_files else sorted(DOCS_DIR.iterdir())
    )

    # 2. Separate text files and PDF paths
    for file_path in candidate_paths:
        if not file_path.exists():
            continue
        if file_path.suffix.lower() in {".txt", ".md"}:
            documents.extend(_load_text_documents(file_path))
        elif file_path.suffix.lower() == ".pdf":
            pdf_paths.append(file_path)

    # 3. Pass PDFs directly to Docling (No PyMuPDF rendering step)
    if pdf_paths:
        pipeline_options = PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU  # Bypasses MPS float64 crash entirely
    )
)
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = RapidOcrOptions(backend="paddle")

        custom_converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        hybrid_chunker = HybridChunker(max_tokens=216, merge_peers=True)

        pdf_loader = DoclingLoader(
            file_path=[str(p) for p in pdf_paths], # Absolute PDF paths
            export_type=ExportType.DOC_CHUNKS,
            chunker=hybrid_chunker,
            converter=custom_converter,
            meta_extractor=PageAwareMetaExtractor(),
        )
        documents.extend(pdf_loader.load())

    return documents


# ==========================================
# CORE RAG FUNCTIONS
# ==========================================
def build_or_reload_chain() -> None:
    """Initializes or reloads the Chroma vector store and compression retriever."""
    global vector_store, retriever, rag_chain

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings_model,
    )
    
    # Correctly attach compression retriever
    retriever = _get_compressed_retriever(vector_store)
    answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, answer_chain)

def ingest_documents(target_files: Optional[List[str]] = None) -> Dict[str, int]:
    """Ingests documents into ChromaDB and re-attaches the compressed retriever."""
    global vector_store, retriever, rag_chain

    raw_docs = load_documents_from_folder(target_files=target_files)
    
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

    valid_chunks = [
        doc for doc in final_chunks
        if doc.page_content and doc.page_content.strip()
    ]
    
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
        batch = valid_chunks[i : i + batch_size]
        vector_store.add_documents(batch)
    
    retriever = _get_compressed_retriever(vector_store)
    answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, answer_chain)

    return {"chunks_indexed": len(final_chunks), "documents_loaded": len(raw_docs)}

def preview_chroma_chunks(
    limit: int = 5, 
    chars: int = 400, 
    filename: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Returns previews of chunks directly from the Chroma vector store."""
    if vector_store is None:
        return []

    # Optional metadata filter for a specific filename
    where_clause = {"source": filename} if filename else None

    # Fetch stored chunks and metadata directly from Chroma
    data = vector_store.get(
        limit=limit, 
        where=where_clause,
        include=["documents", "metadatas"]
    )

    previews: List[Dict[str, Any]] = []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    for text, meta in zip(documents, metadatas):
        # Create a clean metadata copy excluding bounding box arrays (dl_prov)
        clean_meta = {k: v for k, v in meta.items() if k != "dl_prov"}
        
        previews.append(
            {
                "text": text[:chars],
                "metadata": clean_meta,
            }
        )

    return previews


def _extract_source_info(doc: Document, index: int) -> Dict[str, Any]:
    """Extracts clean source metadata including bounding boxes and citation index."""
    meta = getattr(doc, "metadata", {}) or {}
    
    # Load bounding boxes JSON string if available
    dl_prov = []
    if "dl_prov" in meta:
        try:
            dl_prov = json.loads(meta["dl_prov"])
        except Exception:
            dl_prov = []

    return {
        "citation_id": index,  # 1-based index matching [Source 1]
        "source": meta.get("source", "Unknown"),
        "file_type": meta.get("file_type", "Unknown"),
        "page_number": meta.get("page_number", 1),
        "page_numbers": str(meta.get("page_numbers", "N/A")),
        "dl_prov": dl_prov,    # Bounding boxes for drawing the highlight box
    }

def ask_question(question: str) -> Dict[str, Any]:
    global vector_store
    if vector_store is None:
        build_or_reload_chain()

    # 1. Fetch raw candidate documents (contains full metadata)
    raw_candidates = vector_store.similarity_search(question, k=FETCH_K)

    # 2. Rerank candidates using VertexAIRank
    reranked_docs = reranker.compress_documents(documents=raw_candidates, query=question)

    # 3. Restore lost metadata from raw_candidates by matching page_content
    content_to_meta = {doc.page_content: doc.metadata for doc in raw_candidates}
    
    
    print(f"\n=================== RERANKED RESULTS' ===================")
    for rank, doc in enumerate(reranked_docs, start=1):
        meta = doc.metadata
        score = meta.get("relevance_score", "N/A")
        # Format score as float if present
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        
        source = meta.get("source", "Unknown")
        page = meta.get("page_number", 1)
        snippet = doc.page_content.strip().replace("\n", " ")[:120]

        print(f"Rank [{rank:02d}] | Score: {score_str} | File: {source} | Page: {page}")
        print(f"         Snippet: \"{snippet}...\"\n")
    print("===========================================================================\n")
    
    
    for i, doc in enumerate(reranked_docs, start=1):
        # Merge original metadata back onto reranked document
        original_meta = content_to_meta.get(doc.page_content, {})
        doc.metadata.update(original_meta)
        
        # Ensure mandatory prompt template variables exist
        doc.metadata["source_index"] = i
        doc.metadata.setdefault("source", "Unknown")
        doc.metadata.setdefault("page_number", 1)

    # 4. Invoke LLM chain
    answer_chain = create_stuff_documents_chain(
        llm=llm, prompt=prompt_template, document_prompt=document_prompt
    )
    response = answer_chain.invoke({"context": reranked_docs, "input": question})
    answer_text = (
        response.get("answer") or response.get("output_text") or str(response)
        if isinstance(response, dict) else str(response)
    )

    # 4. Extract source metadata matching [Source 1], [Source 2] order
    sources = [
        _extract_source_info(doc, i) for i, doc in enumerate(reranked_docs, start=1)
    ]

    return {"answer": answer_text, "sources": sources}
