from __future__ import annotations

from agentic_rag.chunking._shared import recursive_split_text
from agentic_rag.chunking.base import ChunkCandidate, ChunkingConfig
from agentic_rag.chunking.tokenization import count_tokens
from agentic_rag.ingestion.parsed_document import DocumentElement, ParsedDocument


class RecursiveChunker:
    """Accumulates whole elements (paragraphs, list items, ...) into a chunk
    up to the token budget, carrying trailing elements forward as overlap.
    An element that alone exceeds the budget (a huge paragraph or table) is
    recursively split on a separator hierarchy rather than accumulated.

    Keeps the natural element boundaries the fixed-size chunker ignores,
    while still guaranteeing every chunk fits the configured budget.
    """

    async def chunk(self, parsed: ParsedDocument, config: ChunkingConfig) -> list[ChunkCandidate]:
        elements = parsed.text_elements
        if not elements:
            return []

        candidates: list[ChunkCandidate] = []
        buffer: list[DocumentElement] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            text = "\n\n".join(e.text for e in buffer)
            anchor = buffer[0]
            candidates.append(
                ChunkCandidate(
                    text=text,
                    order_index=len(candidates),
                    token_count=count_tokens(text),
                    character_count=len(text),
                    page=anchor.page,
                    section=anchor.section,
                    heading=anchor.heading,
                )
            )
            # Carry trailing elements forward as overlap for the next chunk.
            overlap: list[DocumentElement] = []
            overlap_tokens = 0
            for element in reversed(buffer):
                element_tokens = count_tokens(element.text)
                if overlap_tokens + element_tokens > config.chunk_overlap_tokens:
                    break
                overlap.insert(0, element)
                overlap_tokens += element_tokens
            buffer = overlap
            buffer_tokens = overlap_tokens

        for element in elements:
            element_tokens = count_tokens(element.text)

            if element_tokens > config.chunk_size_tokens:
                flush()
                for piece in recursive_split_text(element.text, config.chunk_size_tokens):
                    candidates.append(
                        ChunkCandidate(
                            text=piece,
                            order_index=len(candidates),
                            token_count=count_tokens(piece),
                            character_count=len(piece),
                            page=element.page,
                            section=element.section,
                            heading=element.heading,
                        )
                    )
                continue

            if buffer_tokens + element_tokens > config.chunk_size_tokens:
                flush()

            buffer.append(element)
            buffer_tokens += element_tokens

        flush()
        return [c for c in candidates if c.text.strip()]
