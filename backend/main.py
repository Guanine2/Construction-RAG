from typing import Dict, Tuple, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import io
from fastapi.responses import Response
import fitz 
from PIL import Image, ImageDraw


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
    

@app.post("/render-highlight")
def render_highlight_endpoint(req: HighlightRequest):
    cache_key = (req.source, req.page_number)
    
    # 1. Standardize 300 DPI target scale
    TARGET_DPI = 300
    scale_factor = TARGET_DPI / 72.0  # Scale multiplier (4.1667) for bounding box coordinates

    try:
        if cache_key in PAGE_CACHE:
            base_img, page_h = PAGE_CACHE[cache_key]
        else:
            pdf_path = DOCS_DIR / req.source
            if not pdf_path.exists():
                raise HTTPException(status_code=404, detail="PDF file not found")

            doc = fitz.open(pdf_path)
            page_idx = max(0, req.page_number - 1)
            page = doc.load_page(page_idx)

            # RENDER CRISP AT 300 DPI DIRECTLY
            pix = page.get_pixmap(dpi=TARGET_DPI)

            base_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_h = page.rect.height

            # Cache the pristine high-res 300 DPI base image
            PAGE_CACHE[cache_key] = (base_img, page_h)

        # 2. Draw high-precision boxes on crisp image copy
        img = base_img.copy()
        draw = ImageDraw.Draw(img, "RGBA")

        for prov in req.dl_prov:
            bbox = prov.get("bbox")
            if not bbox:
                continue

            l, t, r, b = bbox["l"], bbox["t"], bbox["r"], bbox["b"]
            origin = str(bbox.get("coord_origin", "BOTTOMLEFT")).upper()

            if "BOTTOM" in origin:
                top_pt = page_h - t
                bottom_pt = page_h - b
            else:
                top_pt = t
                bottom_pt = b

            # Scale PDF points (72 DPI base) to match 300 DPI image canvas
            x0, x1 = min(l, r) * scale_factor, max(l, r) * scale_factor
            y0, y1 = min(top_pt, bottom_pt) * scale_factor, max(top_pt, bottom_pt) * scale_factor

            draw.rectangle(
                [x0, y0, x1, y1],
                fill=(255, 235, 59, 90),
                outline=(255, 152, 0, 255),
                width=3,
            )

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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