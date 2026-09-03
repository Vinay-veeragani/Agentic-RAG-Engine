"""The common internal representation every format-specific parser normalizes
into: Document -> elements -> metadata (spec §5).

Deliberately a plain dataclass model, not a SQLAlchemy/Pydantic one: parsing
is a pure, DB-independent transformation (bytes in, structure out), and
keeping it that way is what makes each parser unit-testable without a
database. `ingestion/pipeline.py` is the layer that turns a `ParsedDocument`
into DB rows; `chunking/` (Phase 3) is what turns it into `DocumentChunk`s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ElementType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CAPTION = "caption"
    LIST_ITEM = "list_item"
    CODE = "code"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class DocumentElement:
    """One structural unit of a parsed document.

    `heading`/`section` carry the *nearest enclosing* heading/section title at
    the time this element was parsed — later chunking (Phase 3) relies on this
    to preserve heading/section context per chunk without re-parsing.
    """

    element_type: ElementType
    text: str
    order_index: int
    page: int | None = None
    heading: str | None = None
    section: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    """Output of `DocumentParser.parse()` — one per source file."""

    filename: str
    document_type: str
    elements: list[DocumentElement]
    title: str | None = None
    page_count: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def text_elements(self) -> list[DocumentElement]:
        """Elements chunking should actually consider (excludes e.g. tables,
        which Phase 3 may handle specially rather than sentence-chunk)."""
        return [e for e in self.elements if e.text.strip()]
