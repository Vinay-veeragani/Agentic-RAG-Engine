"""Parser interface + registry.

`DocumentParser.parse(source) -> ParsedDocument` (spec §5). Each format gets
its own parser rather than forcing everything through one code path — a PDF's
notion of "page" and a CSV's notion of "row" are not the same shape, and
pretending otherwise is exactly the kind of toy abstraction the spec warns
against (§42).
"""

from __future__ import annotations

from typing import Protocol

from agentic_rag.core.errors import InvalidDocumentError
from agentic_rag.core.models import DocumentType
from agentic_rag.ingestion.parsed_document import ParsedDocument


class DocumentParser(Protocol):
    def parse(self, *, filename: str, content: bytes) -> ParsedDocument: ...


def get_parser(document_type: DocumentType) -> DocumentParser:
    # Imported lazily so e.g. importing the CSV parser never pulls in PyMuPDF.
    from agentic_rag.ingestion.parsers.csv_parser import CsvParser
    from agentic_rag.ingestion.parsers.docx import DocxParser
    from agentic_rag.ingestion.parsers.html import HtmlParser
    from agentic_rag.ingestion.parsers.json_parser import JsonParser
    from agentic_rag.ingestion.parsers.markdown import MarkdownParser
    from agentic_rag.ingestion.parsers.pdf import PdfParser
    from agentic_rag.ingestion.parsers.text import TextParser

    registry: dict[DocumentType, DocumentParser] = {
        DocumentType.PDF: PdfParser(),
        DocumentType.DOCX: DocxParser(),
        DocumentType.TXT: TextParser(),
        DocumentType.MARKDOWN: MarkdownParser(),
        DocumentType.HTML: HtmlParser(),
        DocumentType.CSV: CsvParser(),
        DocumentType.JSON: JsonParser(),
    }
    parser = registry.get(document_type)
    if parser is None:
        raise InvalidDocumentError(f"no parser registered for {document_type!r}")
    return parser
