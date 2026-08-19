import json
from pathlib import Path
import fitz  # PyMuPDF
import pymupdf4llm  # Automatically activates PyMuPDF-Layout engine


def highlight_tables_with_pymupdf_layout(input_pdf_path: str, output_pdf_path: str) -> None:
    """Highlights tables detected by PyMuPDF-Layout using a yellow overlay."""
    doc = fitz.open(input_pdf_path)

    # 1. Run PyMuPDF-Layout analysis to retrieve structured layout JSON
    # This automatically uses the GNN layout engine to find tables & sections
    raw_json = pymupdf4llm.to_json(input_pdf_path)
    layout_data = json.loads(raw_json)

    total_tables_found = 0

    # 2. Iterate pages and filter for layout blocks classified as 'table'
    for page_idx, page_info in enumerate(layout_data):
        page_num = page_idx + 1
        page = doc[page_idx]

        # Extract items tagged as 'table' by PyMuPDF-Layout
        blocks = page_info.get("blocks", page_info.get("items", []))
        table_blocks = [b for b in blocks if b.get("type") == "table"]

        if not table_blocks:
            print(f"Page {page_num}: No tables detected by PyMuPDF-Layout.")
            continue

        count = len(table_blocks)
        total_tables_found += count
        print(f"Page {page_num}: Highlighting {count} table(s).")

        for block in table_blocks:
            bbox = block.get("bbox")
            if not bbox:
                continue

            # 3. Convert bbox tuple [x0, y0, x1, y1] directly to fitz.Rect
            table_rect = fitz.Rect(bbox)

            # Option A: Add native PDF yellow highlight annotation
            annot = page.add_highlight_annot(table_rect)
            annot.set_colors(stroke=(1, 1, 0))  # RGB Yellow
            annot.update()

            # Option B: Semi-transparent filled rectangle burned onto vector layer
            # page.draw_rect(table_rect, color=(1, 0.8, 0), fill=(1, 1, 0), fill_opacity=0.35)

    doc.save(output_pdf_path)
    doc.close()
    print(f"\nFinished! Total layout tables highlighted: {total_tables_found}")
    print(f"Annotated PDF saved to: {output_pdf_path}")


if __name__ == "__main__":
    target_pdf = "docs/FINAL FOR CONSTRUCTION-04.13.2026-2 Barchester Way, Westfield, NJ 07090.pdf"
    annotated_pdf = "docs/sample_drawing_tables_annotated.pdf"

    if Path(target_pdf).exists():
        highlight_tables_with_pymupdf_layout(target_pdf, annotated_pdf)
    else:
        print(f"Error: Could not find target file at '{target_pdf}'")