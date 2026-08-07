import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Docling Imports
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

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


# Lazy-loaded CrossEncoder instance to prevent process spawn crashes during reload
_reranker_model = None

# ==========================================
# CONFIGURATION & PATHS
# ==========================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db" / "langchain_document_intelligence"
KEYS_DIR = PROJECT_ROOT / "keys"

COLLECTION_NAME = "internal_documents"

GCP_PROJECT_ID = "text-email-digest-test"
GCP_LOCATION = "global"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    KEYS_DIR / "text-email-digest-test-d8e0e0d73e80.json"
)

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
    "[Source {source_index}]\nDocument Content:\n{page_content}\n"
)

# Global runtime state for DB and Chain
vector_store = None
retriever = None
rag_chain = None

# Reranker configuration
FETCH_N = 50
FINAL_K = 10
logger = logging.getLogger(__name__)

# Custom Metadata Extractor for LangChain Docling
import json
from pathlib import Path
from typing import Any, Dict
from langchain_docling.loader import BaseMetaExtractor


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
            "file_type": file_path.suffix.lstrip(".")
        })
    return docs


def load_documents_from_folder() -> List[Document]:
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Document folder not found: {DOCS_DIR}")

    documents: List[Document] = []
    pdf_paths: List[str] = []

    for file_path in sorted(DOCS_DIR.iterdir()):
        if file_path.suffix.lower() in {".txt", ".md"}:
            documents.extend(_load_text_documents(file_path))
        elif file_path.suffix.lower() == ".pdf":
            pdf_paths.append(str(file_path))

    if pdf_paths:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.images_scale = 3.0
        pipeline_options.ocr_options = RapidOcrOptions()

        custom_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # Define HybridChunker with target token size
        hybrid_chunker = HybridChunker(
            max_tokens=450,      # Target embedding window size
            merge_peers=True     # Combines small elements under the same header
        )

        pdf_loader = DoclingLoader(
            file_path=pdf_paths,
            export_type=ExportType.DOC_CHUNKS,
            chunker=hybrid_chunker,              # <--- PASS CHUNKER HERE
            converter=custom_converter,
            meta_extractor=PageAwareMetaExtractor(),
        )
        documents.extend(pdf_loader.load())

    if not documents:
        raise ValueError("No supported documents found in the docs folder.")

    return documents


# ==========================================
# CORE RAG FUNCTIONS
# ==========================================
def build_or_reload_chain() -> None:
    """Initializes or reloads the Chroma vector store and the RAG retrieval chain."""
    global vector_store, rag_chain

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings_model,
    )
    # expose a retriever globally so ask_question can fetch large candidate sets
    global retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, answer_chain)


def ingest_documents() -> Dict[str, int]:
    """Ingests documents into ChromaDB using Docling HybridChunker for PDFs."""
    global vector_store, rag_chain

    raw_docs = load_documents_from_folder()
    
    final_chunks: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for doc in raw_docs:
        # If it's a PDF, it is already chunked natively by HybridChunker
        if doc.metadata.get("file_type") == "pdf":
            final_chunks.append(doc)
        else:
            # Only split plain text/markdown files
            final_chunks.extend(text_splitter.split_documents([doc]))

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=final_chunks,
        embedding=embeddings_model,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    global retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.8},
    )
    answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, answer_chain)

    return {"chunks_indexed": len(final_chunks), "documents_loaded": len(raw_docs)}

def preview_chroma_chunks(limit: int = 5, chars: int = 400) -> List[Dict[str, Any]]:
    """Returns previews of chunks directly from the Chroma vector store."""
    if vector_store is None:
        return []

    # Fetch stored chunks and metadata directly from Chroma
    data = vector_store.get(limit=limit, include=["documents", "metadatas"])

    previews: List[Dict[str, Any]] = []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    for text, meta in zip(documents, metadatas):
        previews.append(
            {
                "text": text[:chars],
                "metadata": meta,
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
    global vector_store, retriever
    if vector_store is None:
        build_or_reload_chain()

    # 1. Fetch top candidate documents
    docs: List[Document] = []
    try:
        if retriever is not None and hasattr(retriever, "invoke"):
            docs = retriever.invoke(question)[:FINAL_K]
        elif vector_store is not None and hasattr(vector_store, "similarity_search"):
            docs = vector_store.similarity_search(question, k=FINAL_K)
    except Exception:
        docs = []

    if not docs:
        return {
            "answer": "I cannot find this information in the provided documents.",
            "sources": [],
        }

    # 2. Assign 1-based citation index to document metadata
    for i, doc in enumerate(docs, start=1):
        doc.metadata["source_index"] = i

    # 3. Create chain and pass "context": docs
    answer_chain = create_stuff_documents_chain(
        llm=llm, prompt=prompt_template, document_prompt=document_prompt
    )

    # FIXED: Changed "input_documents" to "context"
    response = answer_chain.invoke({"context": docs, "input": question})

    if isinstance(response, dict):
        answer_text = (
            response.get("answer")
            or response.get("output_text")
            or str(response)
        )
    else:
        answer_text = str(response)

    # 4. Extract sources array matching [Source 1], [Source 2] order
    sources = [
        _extract_source_info(doc, i) for i, doc in enumerate(docs, start=1)
    ]

    return {"answer": answer_text, "sources": sources}
