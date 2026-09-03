from __future__ import annotations

from agentic_rag.chunking._shared import build_token_stream, decode_window
from agentic_rag.chunking.base import ChunkCandidate, ChunkingConfig
from agentic_rag.chunking.tokenization import count_tokens
from agentic_rag.ingestion.parsed_document import ParsedDocument


class FixedSizeChunker:
    """Pure token-count sliding window over the whole document's text,
    completely ignoring structure. This is the simplest possible baseline —
    useful for comparison, not the default (see StructuralChunker)."""

    async def chunk(self, parsed: ParsedDocument, config: ChunkingConfig) -> list[ChunkCandidate]:
        elements = parsed.text_elements
        if not elements:
            return []

        stream = build_token_stream(elements)
        step = max(config.chunk_size_tokens - config.chunk_overlap_tokens, 1)
        total = len(stream.token_ids)

        windows: list[tuple[int, int]] = []
        start = 0
        while start < total:
            end = min(start + config.chunk_size_tokens, total)
            windows.append((start, end))
            if end == total:
                break
            start += step

        # Merge a too-small trailing window into the previous one rather than
        # emitting a near-empty final chunk.
        if len(windows) > 1:
            last_start, last_end = windows[-1]
            if last_end - last_start < config.min_chunk_size_tokens:
                windows.pop()
                prev_start, _ = windows[-1]
                windows[-1] = (prev_start, last_end)

        candidates: list[ChunkCandidate] = []
        for order_index, (start, end) in enumerate(windows):
            text = decode_window(stream, start, end)
            if not text:
                continue
            anchor_element = stream.element_at(start)
            candidates.append(
                ChunkCandidate(
                    text=text,
                    order_index=order_index,
                    token_count=count_tokens(text),
                    character_count=len(text),
                    page=anchor_element.page,
                    section=anchor_element.section,
                    heading=anchor_element.heading,
                )
            )
        return candidates
