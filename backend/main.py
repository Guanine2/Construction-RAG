from typing import Dict, Tuple, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import io
from fastapi.responses import Response
import fitz 
from PIL import Image, ImageDraw
import json


# Import engine logic
from backend.rag_engine import (
    DOCS_DIR,
    build_or_reload_chain,
    ingest_documents,
    preview_chroma_chunks,
    ask_question,
)

app = FastAPI(title="Document Intelligence RAG")

PAGE_CACHE: Dict[Tuple[str, int], Tuple[Image.Image, float]] = {}

class QueryRequest(BaseModel):
    question: str


class PreviewRequest(BaseModel):
    limit: int = Field(default=5, ge=1, le=100)
    chars: int = Field(default=400, ge=1)
    filename: Optional[str] = Field(
        default=None, 
        description="Optional source filename to filter chunks (e.g., 'blueprint.pdf')"
    )
class HighlightRequest(BaseModel):
    source: str
    page_number: int
    dl_prov: list

class IngestRequest(BaseModel):
    files: Optional[List[str]] = None


def parse_and_normalize_bboxes(dl_prov_raw: Union[str, List, Dict]) -> List[List[float]]:
    """
    Parses 'dl_prov' and converts any bbox format (PyMuPDF list or Docling dict)
    into standard [x0, y0, x1, y1] float lists.
    """
    # 1. Deserialize string if passed as raw JSON from ChromaDB
    if isinstance(dl_prov_raw, str):
        try:
            dl_prov_raw = json.loads(dl_prov_raw)
        except Exception:
            return []

    # 2. Ensure we are working with a list of provenance items
    prov_list = dl_prov_raw if isinstance(dl_prov_raw, list) else [dl_prov_raw]

    normalized_bboxes = []

    for item in prov_list:
        if not isinstance(item, dict):
            continue
            
        bbox = item.get("bbox")
        if not bbox:
            continue

        # Case A: BBox is already a list [x0, y0, x1, y1] (PyMuPDF format)
        if isinstance(bbox, list) and len(bbox) >= 4:
            normalized_bboxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])

        # Case B: BBox is a dict (Docling / PDFKit format)
        elif isinstance(bbox, dict):
            # Handles {'l': x0, 't': y0, 'r': x1, 'b': y1} or {'x0': ..., 'y0': ...}
            x0 = bbox.get("l") or bbox.get("x0") or bbox.get("left", 0)
            y0 = bbox.get("t") or bbox.get("y0") or bbox.get("top", 0)
            x1 = bbox.get("r") or bbox.get("x1") or bbox.get("right", 0)
            y1 = bbox.get("b") or bbox.get("y1") or bbox.get("bottom", 0)
            normalized_bboxes.append([float(x0), float(y0), float(x1), float(y1)])

    return normalized_bboxes

@app.post("/render-highlight")
def render_highlight_endpoint(payload: HighlightRequest):
    try:
        pdf_path = DOCS_DIR / payload.source
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")

        doc = fitz.open(str(pdf_path))
        
        # Validate 1-based page index range
        if payload.page_number < 1 or payload.page_number > len(doc):
            raise HTTPException(status_code=400, detail="Page number out of range")

        page = doc.load_page(payload.page_number - 1)  # 0-based indexing for PyMuPDF

        # Parse and normalize bounding boxes safely from Pydantic payload
        bboxes = parse_and_normalize_bboxes(payload.dl_prov)

        shape = page.new_shape()
        
        # Draw highlight rectangles over image canvas
        for bbox in bboxes:
            rect = fitz.Rect(bbox)
            
            # Draw filled rectangle with translucent opacity
            shape.draw_rect(rect)
            shape.finish(
                fill=(1, 1, 0),        # RGB Yellow fill
                fill_opacity=0.35,     # 35% opacity (translucent marker effect)
                color=(1, 0.8, 0),     # Optional subtle border (RGB Gold/Orange)
                width=0.5
            )

        shape.commit()
        # Render 300 DPI image
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")

        return Response(content=img_bytes, media_type="image/png")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Highlight rendering error: {str(exc)}")
    
@app.on_event("startup")
def startup_event() -> None:
    build_or_reload_chain()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest_endpoint(request: Optional[IngestRequest] = None) -> dict:
    try:
        target_files = request.files if request else None
        result = ingest_documents(target_files=target_files)
        return {
            "message": "Documents ingested successfully.",
            **result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/preview-chunks")
def preview_chunks_endpoint(request: PreviewRequest) -> dict:
    try:
        return {
            "chunks": preview_chroma_chunks(
                limit=request.limit, 
                chars=request.chars, 
                filename=request.filename
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/ask")
def ask_endpoint(request: QueryRequest) -> dict:
    try:
        return ask_question(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)