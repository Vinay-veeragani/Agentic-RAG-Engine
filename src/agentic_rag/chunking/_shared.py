"""Helpers shared by the token-window-based chunkers (fixed, recursive)."""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from agentic_rag.chunking.tokenization import decode, encode
from agentic_rag.ingestion.parsed_document import DocumentElement

_SEPARATOR = "\n\n"


@dataclass(slots=True)
class ElementSpan:
    start: int  # inclusive token offset into the concatenated stream
    end: int  # exclusive
    element: DocumentElement


@dataclass(slots=True)
class TokenStream:
    token_ids: list[str]
    spans: list[ElementSpan]

    def element_at(self, token_offset: int) -> DocumentElement:
        starts = [span.start for span in self.spans]
        index = bisect.bisect_right(starts, token_offset) - 1
        index = max(0, min(index, len(self.spans) - 1))
        return self.spans[index].element


def build_token_stream(elements: list[DocumentElement]) -> TokenStream:
    separator_ids = encode(_SEPARATOR)
    token_ids: list[str] = []
    spans: list[ElementSpan] = []

    for i, element in enumerate(elements):
        if i > 0:
            token_ids.extend(separator_ids)
        element_ids = encode(element.text)
        start = len(token_ids)
        token_ids.extend(element_ids)
        spans.append(ElementSpan(start=start, end=len(token_ids), element=element))

    return TokenStream(token_ids=token_ids, spans=spans)


def decode_window(stream: TokenStream, start: int, end: int) -> str:
    return decode(stream.token_ids[start:end]).strip()


_DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


def recursive_split_text(
    text: str, max_tokens: int, separators: tuple[str, ...] = _DEFAULT_SEPARATORS
) -> list[str]:
    """Splits `text` into pieces each <= `max_tokens`, trying separators from
    coarsest ("\\n\\n") to finest (" ") and greedily re-merging adjacent
    pieces up to the budget, so a chunk is as large as it can be without
    exceeding it rather than being split at every separator occurrence."""
    from agentic_rag.chunking.tokenization import count_tokens

    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    if not separators:
        token_ids = encode(text)
        return [
            decode(token_ids[i : i + max_tokens]).strip()
            for i in range(0, len(token_ids), max_tokens)
        ]

    sep, rest = separators[0], separators[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) <= 1:
        return recursive_split_text(text, max_tokens, rest)

    chunks: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}{sep}{part}" if buffer else part
        if count_tokens(candidate) <= max_tokens:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        if count_tokens(part) > max_tokens:
            chunks.extend(recursive_split_text(part, max_tokens, rest))
            buffer = ""
        else:
            buffer = part
    if buffer:
        chunks.append(buffer)
    return chunks
