"""Parser tests for binary formats (PDF, DOCX) that need a real generated
file rather than hand-written bytes."""

import io

import docx
import pymupdf

from agentic_rag.ingestion.parsed_document import ElementType
from agentic_rag.ingestion.parsers.docx import DocxParser
from agentic_rag.ingestion.parsers.pdf import PdfParser


def _build_sample_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Document Heading", fontsize=20)
    page.insert_text((72, 110), "This is a normal paragraph of body text.", fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_parser_distinguishes_heading_by_font_size() -> None:
    parsed = PdfParser().parse(filename="a.pdf", content=_build_sample_pdf())
    assert parsed.page_count == 1

    headings = [e for e in parsed.elements if e.element_type == ElementType.HEADING]
    paragraphs = [e for e in parsed.elements if e.element_type == ElementType.PARAGRAPH]

    assert any("Document Heading" in h.text for h in headings)
    assert any("normal paragraph" in p.text for p in paragraphs)
    assert all(e.page == 1 for e in parsed.elements)


def _build_sample_docx() -> bytes:
    document = docx.Document()
    document.add_heading("Report Title", level=1)
    document.add_paragraph("Body paragraph under the heading.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Key"
    table.rows[0].cells[1].text = "Value"
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_docx_parser_extracts_headings_paragraphs_and_tables() -> None:
    parsed = DocxParser().parse(filename="a.docx", content=_build_sample_docx())

    headings = [e for e in parsed.elements if e.element_type == ElementType.HEADING]
    paragraphs = [e for e in parsed.elements if e.element_type == ElementType.PARAGRAPH]
    tables = [e for e in parsed.elements if e.element_type == ElementType.TABLE]

    assert headings[0].text == "Report Title"
    assert paragraphs[0].text == "Body paragraph under the heading."
    assert paragraphs[0].heading == "Report Title"
    assert "Key | Value" in tables[0].text
