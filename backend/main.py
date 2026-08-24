from typing import Dict, Tuple, List, Optional, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
import uvicorn
from fastapi.responses import Response
import pymupdf as fitz 
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
    property_name: Optional[str] = None


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

@app.get("/properties")
def get_properties():
    """Returns subdirectories in DOCS_DIR as available property names."""
    if not DOCS_DIR.exists():
        return {"properties": []}
    properties = [d.name for d in DOCS_DIR.iterdir() if d.is_dir()]
    return {"properties": properties}

@app.post("/upload-and-ingest")
async def upload_and_ingest(
    property_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    # Sanitize property name for directory paths
    clean_prop_name = property_name.strip().replace(" ", "_")
    target_dir = DOCS_DIR / clean_prop_name
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for file in files:
        file_path = target_dir / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        saved_files.append(file_path)

    # Trigger ingestion passing property metadata
    result = ingest_documents(property_name=clean_prop_name, target_files=saved_files)
    return {"status": "success", "property": clean_prop_name, **result}

# --- Helper Function: BBox Parser & Normalizer ---
def parse_and_normalize_bboxes(
    dl_prov_raw: Union[str, List, Dict],
    page_width: float = 0.0,
    page_height: float = 0.0
) -> List[List[float]]:
    """
    Parses 'dl_prov' and converts any bbox format (PyMuPDF, Gemini VLM, or Docling)
    into standard [x0, y0, x1, y1] float lists scaled to actual PDF points.
    """
    if isinstance(dl_prov_raw, str):
        try:
            dl_prov_raw = json.loads(dl_prov_raw)
        except Exception:
            return []

    prov_list = dl_prov_raw if isinstance(dl_prov_raw, list) else [dl_prov_raw]
    normalized_bboxes = []

    for item in prov_list:
        if not isinstance(item, dict):
            continue
            
        bbox = item.get("bbox")
        if not bbox:
            continue

        method = item.get("type") or item.get("extraction_method", "")

        # CASE 1: Gemini VLM Format ([ymin, xmin, ymax, xmax] normalized 0-1000)
        # Rely primarily on method name to prevent misclassifying small absolute PDF points
        if method == "gemini_vlm_ocr":
            ymin, xmin, ymax, xmax = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            
            # Swap axes to (x0, y0, x1, y1) and scale 0-1000 -> actual PDF points
            x0 = (xmin / 1000.0) * page_width if page_width else xmin
            y0 = (ymin / 1000.0) * page_height if page_height else ymin
            x1 = (xmax / 1000.0) * page_width if page_width else xmax
            y1 = (ymax / 1000.0) * page_height if page_height else ymax
            
            normalized_bboxes.append([x0, y0, x1, y1])

        # CASE 2: PyMuPDF / SHX / Native Text Format ([x0, y0, x1, y1] absolute points)
        elif method in {"shx_annotation", "pymupdf_native_spatial", "pymupdf_native_text"} or (
            isinstance(bbox, list) and len(bbox) >= 4
        ):
            # If no method specified, fallback: check if coordinates exceed 1000 or if x1 > x0
            # If all values <= 1000 AND method is untagged, we check if it looks like VLM
            if not method and all(v <= 1000 for v in bbox[:4]) and (page_width > 1000 or page_height > 1000):
                ymin, xmin, ymax, xmax = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                x0 = (xmin / 1000.0) * page_width if page_width else xmin
                y0 = (ymin / 1000.0) * page_height if page_height else ymin
                x1 = (xmax / 1000.0) * page_width if page_width else xmax
                y1 = (ymax / 1000.0) * page_height if page_height else ymax
                normalized_bboxes.append([x0, y0, x1, y1])
            else:
                normalized_bboxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])

        # CASE 3: Dict Format ({'l': x0, 't': y0, 'r': x1, 'b': y1})
        elif isinstance(bbox, dict):
            x0 = bbox.get("l") or bbox.get("x0") or bbox.get("left", 0)
            y0 = bbox.get("t") or bbox.get("y0") or bbox.get("top", 0)
            x1 = bbox.get("r") or bbox.get("x1") or bbox.get("right", 0)
            y1 = bbox.get("b") or bbox.get("y1") or bbox.get("bottom", 0)
            normalized_bboxes.append([float(x0), float(y0), float(x1), float(y1)])

    return normalized_bboxes

# --- FastAPI Endpoint ---
@app.post("/render-highlight")
def render_highlight_endpoint(payload: HighlightRequest):
    try:
        pdf_path = DOCS_DIR / payload.source
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")

        # Context manager guarantees doc.close() executes on C level
        with fitz.open(str(pdf_path)) as doc:
            if payload.page_number < 1 or payload.page_number > len(doc):
                raise HTTPException(status_code=400, detail="Page number out of range")

            page = doc.load_page(payload.page_number - 1)
            page_width = page.rect.width
            page_height = page.rect.height

            bboxes = parse_and_normalize_bboxes(
                dl_prov_raw=payload.dl_prov,
                page_width=page_width,
                page_height=page_height
            )

            shape = page.new_shape()
            for bbox in bboxes:
                rect = fitz.Rect(bbox)
                shape.draw_rect(rect)
                shape.finish(
                    fill=(1, 1, 0),
                    fill_opacity=0.35,
                    color=(1, 0.8, 0),
                    width=0.5
                )
            shape.commit()

            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")

        return Response(content=img_bytes, media_type="image/png")

    except HTTPException:
        raise
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