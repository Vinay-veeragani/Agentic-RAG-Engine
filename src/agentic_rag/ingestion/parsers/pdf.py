from __future__ import annotations

import statistics

import pymupdf as fitz

from agentic_rag.core.errors import InvalidDocumentError
from agentic_rag.ingestion.cleaners.text import clean_text
from agentic_rag.ingestion.parsed_document import DocumentElement, ElementType, ParsedDocument

# A block is treated as a heading when its max font size exceeds the
# document's body-text size by this factor. Heuristic, not exact — PDFs carry
# no semantic "this is a heading" marker the way DOCX styles or HTML tags do.
_HEADING_SIZE_RATIO = 1.15


class PdfParser:
    """Preserves page number, a heuristic heading/paragraph split (by relative
    font size — PDF has no native heading semantics), and tables (via
    PyMuPDF's table detection, best-effort per page).
    """

    def parse(self, *, filename: str, content: bytes) -> ParsedDocument:
        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise InvalidDocumentError(
                f"could not parse {filename!r} as a PDF file", details={"filename": filename}
            ) from exc

        try:
            body_size = _estimate_body_font_size(doc)
            elements: list[DocumentElement] = []
            current_heading: str | None = None
            order_index = 0

            for page_index in range(doc.page_count):
                page: fitz.Page = doc[page_index]
                page_number = page_index + 1

                table_bboxes: list[fitz.Rect] = []
                try:
                    tables = page.find_tables()
                except Exception:  # pragma: no cover - depends on PyMuPDF build
                    tables = None
                if tables is not None:
                    for table in tables.tables:
                        table_text = _render_table(table)
                        if table_text:
                            elements.append(
                                DocumentElement(
                                    element_type=ElementType.TABLE,
                                    text=table_text,
                                    order_index=order_index,
                                    page=page_number,
                                    heading=current_heading,
                                )
                            )
                            order_index += 1
                        table_bboxes.append(fitz.Rect(table.bbox))

                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    if block.get("type") != 0:  # 0 == text block
                        continue
                    block_rect = fitz.Rect(block["bbox"])
                    if any(block_rect.intersects(t) for t in table_bboxes):
                        continue  # already captured as a table above

                    lines_text = []
                    max_size = 0.0
                    for line in block.get("lines", []):
                        spans_text = "".join(span["text"] for span in line.get("spans", []))
                        if spans_text.strip():
                            lines_text.append(spans_text)
                        for span in line.get("spans", []):
                            max_size = max(max_size, span.get("size", 0.0))

                    text = clean_text(" ".join(lines_text))
                    if not text:
                        continue

                    is_heading = body_size > 0 and max_size >= body_size * _HEADING_SIZE_RATIO
                    if is_heading:
                        current_heading = text
                        elements.append(
                            DocumentElement(
                                element_type=ElementType.HEADING,
                                text=text,
                                order_index=order_index,
                                page=page_number,
                                heading=current_heading,
                                metadata={"font_size": max_size},
                            )
                        )
                    else:
                        elements.append(
                            DocumentElement(
                                element_type=ElementType.PARAGRAPH,
                                text=text,
                                order_index=order_index,
                                page=page_number,
                                heading=current_heading,
                            )
                        )
                    order_index += 1

            metadata = doc.metadata or {}
            title = (metadata.get("title") or "").strip() or None
            return ParsedDocument(
                filename=filename,
                document_type="pdf",
                elements=elements,
                title=title,
                page_count=doc.page_count,
            )
        finally:
            doc.close()


def _estimate_body_font_size(doc: fitz.Document) -> float:
    sizes: list[float] = []
    for page_index in range(doc.page_count):
        page: fitz.Page = doc[page_index]
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        sizes.append(span["size"])
    return statistics.median(sizes) if sizes else 0.0


def _render_table(table: fitz.table.Table) -> str:
    try:
        rows = table.extract()
    except Exception:  # pragma: no cover
        return ""
    lines = []
    for row in rows:
        cells = [clean_text(str(cell)) if cell is not None else "" for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)
