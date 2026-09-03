from __future__ import annotations

from itertools import groupby

from agentic_rag.chunking.base import ChunkCandidate, ChunkingConfig
from agentic_rag.chunking.recursive import RecursiveChunker
from agentic_rag.chunking.tokenization import count_tokens
from agentic_rag.ingestion.parsed_document import DocumentElement, ParsedDocument


class StructuralChunker:
    """Groups elements into contiguous runs sharing the same heading (i.e.
    document sections) and never splits across a heading boundary. A section
    that fits the token budget becomes one chunk; a section that doesn't gets
    a *parent* chunk (the whole section, for expanded context — intentionally
    allowed to exceed the normal budget) plus *child* chunks produced by
    `RecursiveChunker` over just that section (spec §7: parent-child chunk
    relationships).

    This is the default chunker: it never destroys document structure the
    way FixedSizeChunker does, and it never blindly accumulates across
    section boundaries the way plain RecursiveChunker (run over the whole
    document) would.
    """

    def __init__(self) -> None:
        self._recursive = RecursiveChunker()

    async def chunk(self, parsed: ParsedDocument, config: ChunkingConfig) -> list[ChunkCandidate]:
        elements = parsed.text_elements
        if not elements:
            return []

        candidates: list[ChunkCandidate] = []
        for _heading, group_iter in groupby(elements, key=lambda e: e.heading):
            group = list(group_iter)
            await self._chunk_section(group, config, candidates)
        return candidates

    async def _chunk_section(
        self,
        group: list[DocumentElement],
        config: ChunkingConfig,
        candidates: list[ChunkCandidate],
    ) -> None:
        text = "\n\n".join(e.text for e in group)
        anchor = group[0]
        token_count = count_tokens(text)

        if token_count <= config.chunk_size_tokens:
            candidates.append(
                ChunkCandidate(
                    text=text,
                    order_index=len(candidates),
                    token_count=token_count,
                    character_count=len(text),
                    page=anchor.page,
                    section=anchor.section,
                    heading=anchor.heading,
                )
            )
            return

        parent_index = len(candidates)
        candidates.append(
            ChunkCandidate(
                text=text,
                order_index=parent_index,
                token_count=token_count,
                character_count=len(text),
                page=anchor.page,
                section=anchor.section,
                heading=anchor.heading,
                metadata={"is_parent": True},
            )
        )

        section_doc = ParsedDocument(
            filename="", document_type="", elements=group
        )
        for child in await self._recursive.chunk(section_doc, config):
            child.order_index = len(candidates)
            child.parent_index = parent_index
            candidates.append(child)
