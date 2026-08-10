from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import io
from fastapi.responses import Response
from pydantic import BaseModel
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


class QueryRequest(BaseModel):
    question: str


class PreviewRequest(BaseModel):
    limit: int = 5
    chars: int = 400

class HighlightRequest(BaseModel):
    source: str
    page_number: int
    dl_prov: list

@app.post("/render-highlight")
def render_highlight_endpoint(req: HighlightRequest):
    pdf_path = DOCS_DIR / req.source
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")

    try:
        doc = fitz.open(pdf_path)
        page_idx = max(0, req.page_number - 1)
        page = doc.load_page(page_idx)

        zoom = 4.17
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img, "RGBA")
        page_h = page.rect.height

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

            x0, x1 = min(l, r) * zoom, max(l, r) * zoom
            y0, y1 = min(top_pt, bottom_pt) * zoom, max(top_pt, bottom_pt) * zoom

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
def ingest_endpoint() -> dict:
    try:
        result = ingest_documents()
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
            "chunks": preview_chroma_chunks(limit=request.limit, chars=request.chars),
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