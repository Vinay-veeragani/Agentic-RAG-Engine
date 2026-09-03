from __future__ import annotations

import json

from agentic_rag.core.errors import InvalidDocumentError
from agentic_rag.ingestion.cleaners.text import clean_text
from agentic_rag.ingestion.parsed_document import DocumentElement, ElementType, ParsedDocument


class JsonParser:
    """A top-level list of objects (the common "records" shape) becomes one
    element per record. Any other JSON shape (a single object, a scalar, a
    list of scalars) becomes one element holding the pretty-printed whole —
    there is no generally "correct" way to chunk arbitrary JSON structure
    without knowing its schema, so this does not attempt one.
    """

    def parse(self, *, filename: str, content: bytes) -> ParsedDocument:
        try:
            data = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidDocumentError(
                f"could not parse {filename!r} as JSON", details={"filename": filename}
            ) from exc

        elements: list[DocumentElement] = []
        if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
            for row_index, record in enumerate(data):
                rendered = clean_text(
                    ", ".join(f"{key}: {value}" for key, value in record.items())
                )
                if rendered:
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.TABLE,
                            text=rendered,
                            order_index=row_index,
                            metadata={"row_index": row_index},
                        )
                    )
        else:
            rendered = clean_text(json.dumps(data, indent=2, ensure_ascii=False))
            if rendered:
                elements.append(
                    DocumentElement(
                        element_type=ElementType.PARAGRAPH, text=rendered, order_index=0
                    )
                )

        return ParsedDocument(filename=filename, document_type="json", elements=elements)
