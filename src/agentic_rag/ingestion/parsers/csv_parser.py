from __future__ import annotations

import csv
import io

from agentic_rag.core.errors import InvalidDocumentError
from agentic_rag.ingestion.cleaners.text import clean_text
from agentic_rag.ingestion.parsed_document import DocumentElement, ElementType, ParsedDocument


class CsvParser:
    """One element per data row, rendered as `column: value` pairs rather
    than raw comma-separated text — this reads far better once embedded and
    retrieved as a standalone chunk than a bare CSV line would."""

    def parse(self, *, filename: str, content: bytes) -> ParsedDocument:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError(
                f"could not decode {filename!r} as UTF-8 text", details={"filename": filename}
            ) from exc

        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise InvalidDocumentError(
                f"{filename!r} has no header row", details={"filename": filename}
            )

        elements: list[DocumentElement] = []
        for row_index, row in enumerate(reader):
            rendered = ", ".join(f"{key}: {value}" for key, value in row.items() if key)
            rendered = clean_text(rendered)
            if not rendered:
                continue
            elements.append(
                DocumentElement(
                    element_type=ElementType.TABLE,
                    text=rendered,
                    order_index=row_index,
                    metadata={"row_index": row_index},
                )
            )

        return ParsedDocument(
            filename=filename,
            document_type="csv",
            elements=elements,
            metadata={"columns": reader.fieldnames},
        )
