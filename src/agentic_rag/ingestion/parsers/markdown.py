from __future__ import annotations

from markdown_it import MarkdownIt
from markdown_it.token import Token

from agentic_rag.core.errors import InvalidDocumentError
from agentic_rag.ingestion.cleaners.text import clean_text
from agentic_rag.ingestion.parsed_document import DocumentElement, ElementType, ParsedDocument

_HEADING_TAGS = {f"h{i}" for i in range(1, 7)}


class MarkdownParser:
    """Structure-aware: walks the CommonMark token stream to recover headings,
    paragraphs, list items, and fenced code blocks, tracking the current
    heading as context for later elements (spec §5 "preserve headings").

    Known limitation: GFM tables are not specially recognized (no table
    plugin is installed) — a markdown table parses as plain paragraph text
    rather than a TABLE element.
    """

    def __init__(self) -> None:
        self._md = MarkdownIt("commonmark")

    def parse(self, *, filename: str, content: bytes) -> ParsedDocument:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError(
                f"could not decode {filename!r} as UTF-8 text", details={"filename": filename}
            ) from exc

        tokens = self._md.parse(text)
        elements: list[DocumentElement] = []
        current_heading: str | None = None
        order_index = 0
        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token.type == "heading_open" and token.tag in _HEADING_TAGS:
                inline = tokens[i + 1] if i + 1 < len(tokens) else None
                heading_text = clean_text(inline.content) if inline else ""
                if heading_text:
                    current_heading = heading_text
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.HEADING,
                            text=heading_text,
                            order_index=order_index,
                            heading=current_heading,
                            metadata={"level": int(token.tag[1])},
                        )
                    )
                    order_index += 1
                i += 3  # heading_open, inline, heading_close
                continue

            if token.type == "paragraph_open":
                inline = tokens[i + 1] if i + 1 < len(tokens) else None
                para_text = clean_text(inline.content) if inline else ""
                if para_text:
                    # Tight list items wrap their text in a paragraph_open/close
                    # pair too (markdown-it marks it `hidden`), so a paragraph
                    # nested inside a list_item is a LIST_ITEM, not a PARAGRAPH.
                    element_type = (
                        ElementType.LIST_ITEM if _in_list_item(tokens, i) else ElementType.PARAGRAPH
                    )
                    elements.append(
                        DocumentElement(
                            element_type=element_type,
                            text=para_text,
                            order_index=order_index,
                            heading=current_heading,
                        )
                    )
                    order_index += 1
                i += 3
                continue

            if token.type == "fence":
                code_text = token.content.strip("\n")
                if code_text:
                    elements.append(
                        DocumentElement(
                            element_type=ElementType.CODE,
                            text=code_text,
                            order_index=order_index,
                            heading=current_heading,
                            metadata={"language": token.info.strip() or None},
                        )
                    )
                    order_index += 1
                i += 1
                continue

            i += 1

        return ParsedDocument(filename=filename, document_type="markdown", elements=elements)


def _in_list_item(tokens: list[Token], index: int) -> bool:
    """True if the token at `index` is nested inside an (possibly nested)
    `list_item_open` ... `list_item_close` pair, by scanning backward and
    tracking open/close depth (a closed sibling item's tokens must not be
    mistaken for an enclosing one)."""
    depth = 0
    for j in range(index - 1, -1, -1):
        t = tokens[j].type
        if t == "list_item_close":
            depth += 1
        elif t == "list_item_open":
            if depth == 0:
                return True
            depth -= 1
    return False
