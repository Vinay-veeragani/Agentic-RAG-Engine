from __future__ import annotations

from agentic_rag.core.errors import InvalidDocumentError
from agentic_rag.ingestion.cleaners.text import clean_text
from agentic_rag.ingestion.parsed_document import DocumentElement, ElementType, ParsedDocument


class TextParser:
    """Plain text: one paragraph per blank-line-separated block. No headings/
    pages/tables to preserve — this is the simplest possible parser and is
    intentionally not routed through any of the richer parsers below."""

    def parse(self, *, filename: str, content: bytes) -> ParsedDocument:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError(
                f"could not decode {filename!r} as UTF-8 text", details={"filename": filename}
            ) from exc

        text = clean_text(text)
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        elements = [
            DocumentElement(element_type=ElementType.PARAGRAPH, text=block, order_index=i)
            for i, block in enumerate(blocks)
        ]
        return ParsedDocument(filename=filename, document_type="txt", elements=elements)
