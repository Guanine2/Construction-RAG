import json
from pathlib import Path
import fitz  
from PIL import Image, ImageDraw
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db" / "langchain_document_intelligence"
COLLECTION_NAME = "internal_documents"

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

def highlight_first_chroma_chunk():
    # 1. Connect to Chroma DB
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings_model,
    )

    # 2. Fetch the very first chunk from Chroma
    data = vector_store.get(limit=1, include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    if not documents:
        print("No chunks found in Chroma DB. Please run ingest_documents() first.")
        return

    text = documents[0]
    meta = metadatas[0]
    source_file = meta.get("source")
    page_no = int(meta.get("page_number", 1))
    dl_prov = json.loads(meta.get("dl_prov", "[]"))

    print(f"--- Chunk 1 Preview ---")
    print(f"Text: {text[:120]}...")
    print(f"Source PDF: {source_file} (Page {page_no})")

    if not dl_prov:
        print("No bounding box coordinates found in metadata. Re-run ingest_documents() with updated extractor.")
        return

    # 3. Load PDF Page using PyMuPDF
    pdf_path = DOCS_DIR / source_file
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    page_idx = max(0, page_no - 1)  # 0-indexed page number
    page = doc.load_page(page_idx)

    # Render PDF page to an image (scale = 2.0 for high resolution)
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img, "RGBA")
    page_h = page.rect.height

    # 4. Draw bounding boxes
    for prov in dl_prov:
        bbox = prov.get("bbox")
        if not bbox:
            continue

        l, t, r, b = bbox["l"], bbox["t"], bbox["r"], bbox["b"]
        origin = bbox.get("coord_origin", "BOTTOMLEFT").upper()

        # Convert Docling PDF coordinates (Bottom-Left origin) to PyMuPDF image coordinates (Top-Left origin)
        if "BOTTOM" in origin:
            top_pt = page_h - t
            bottom_pt = page_h - b
        else:
            top_pt = t
            bottom_pt = b

        x0 = min(l, r) * zoom
        x1 = max(l, r) * zoom
        y0 = min(top_pt, bottom_pt) * zoom
        y1 = max(top_pt, bottom_pt) * zoom

        # Draw semi-transparent yellow overlay with orange border
        draw.rectangle(
            [x0, y0, x1, y1], 
            fill=(255, 235, 59, 80), 
            outline=(255, 152, 0, 255), 
            width=3
        )

    output_file = "first_chunk_highlighted.png"
    img.save(output_file)
    print(f"Success! Highlighted image saved to: {Path(output_file).resolve()}")

if __name__ == "__main__":
    highlight_first_chroma_chunk()