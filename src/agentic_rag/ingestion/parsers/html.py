from __future__ import annotations

from bs4 import BeautifulSoup
from bs4.element import Tag

from agentic_rag.core.errors import InvalidDocumentError
from agentic_rag.ingestion.cleaners.text import clean_text
from agentic_rag.ingestion.parsed_document import DocumentElement, ElementType, ParsedDocument

_HEADING_TAGS = {f"h{i}": i for i in range(1, 7)}
_BLOCK_TAGS = (*_HEADING_TAGS.keys(), "p", "li", "pre", "table", "figcaption")


class HtmlParser:
    """Walks block-level tags in document order. Known limitation: a block
    tag nested inside another matched block tag (e.g. a `<p>` inside a
    `<li>`) is emitted as two separate elements rather than merged — good
    enough for retrieval chunking, not a full DOM-to-structure reconstruction.
    """

    def parse(self, *, filename: str, content: bytes) -> ParsedDocument:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError(
                f"could not decode {filename!r} as UTF-8 text", details={"filename": filename}
            ) from exc

        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else None

        elements: list[DocumentElement] = []
        current_heading: str | None = None
        order_index = 0

        for tag in soup.find_all(_BLOCK_TAGS):
            if not isinstance(tag, Tag):
                continue

            if tag.name in _HEADING_TAGS:
                heading_text = clean_text(tag.get_text(" ", strip=True))
                if not heading_text:
                    continue
                current_heading = heading_text
                elements.append(
                    DocumentElement(
                        element_type=ElementType.HEADING,
                        text=heading_text,
                        order_index=order_index,
                        heading=current_heading,
                        metadata={"level": _HEADING_TAGS[tag.name]},
                    )
                )
                order_index += 1
                continue

            if tag.name == "table":
                table_text = _render_table(tag)
                if table_text:
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.TABLE,
                            text=table_text,
                            order_index=order_index,
                            heading=current_heading,
                        )
                    )
                    order_index += 1
                continue

            element_type = {
                "p": ElementType.PARAGRAPH,
                "li": ElementType.LIST_ITEM,
                "pre": ElementType.CODE,
                "figcaption": ElementType.CAPTION,
            }[tag.name]
            block_text = clean_text(tag.get_text(" ", strip=True))
            if block_text:
                elements.append(
                    DocumentElement(
                        element_type=element_type,
                        text=block_text,
                        order_index=order_index,
                        heading=current_heading,
                    )
                )
                order_index += 1

        return ParsedDocument(
            filename=filename, document_type="html", elements=elements, title=title
        )


def _render_table(table: Tag) -> str:
    rows = []
    for row in table.find_all("tr"):
        cells = [clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["td", "th"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)
