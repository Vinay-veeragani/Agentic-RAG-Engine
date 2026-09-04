"""Chunking interface, config, and registry.

`Chunker.chunk()` operates purely on a `ParsedDocument` and returns
`ChunkCandidate`s — no DB/session dependency, so each strategy is unit
testable in isolation. `chunking/pipeline.py` is what turns candidates into
persisted `DocumentChunk` rows (with embeddings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field

from agentic_rag.ingestion.parsed_document import ParsedDocument


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"


class ChunkingConfig(BaseModel):
    """Persisted verbatim onto `DocumentVersion.chunking_config` so a given
    index can always be reproduced later ("record chunking configuration
    ... so experiments can be reproduced")."""

    strategy: ChunkingStrategy = ChunkingStrategy.STRUCTURAL
    chunk_size_tokens: int = Field(default=400, gt=0)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    min_chunk_size_tokens: int = Field(default=20, ge=0)
    semantic_similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


@dataclass(slots=True)
class ChunkCandidate:
    """Pre-persistence chunk. `parent_index` refers to another candidate's
    position in the same returned list (resolved to a real `parent_chunk_id`
    once parents are persisted — see chunking/pipeline.py), never to a
    database ID directly."""

    text: str
    order_index: int
    token_count: int
    character_count: int
    page: int | None = None
    section: str | None = None
    heading: str | None = None
    parent_index: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class Chunker(Protocol):
    async def chunk(
        self, parsed: ParsedDocument, config: ChunkingConfig
    ) -> list[ChunkCandidate]:
        """Async uniformly across strategies — only SemanticChunker actually
        awaits anything (embedding calls), but a single Protocol shape means
        callers never need to know which strategy is in use."""
        ...


def get_chunker(strategy: ChunkingStrategy, *, embedding_provider: object | None = None) -> Chunker:
    from agentic_rag.chunking.fixed import FixedSizeChunker
    from agentic_rag.chunking.recursive import RecursiveChunker
    from agentic_rag.chunking.semantic import SemanticChunker
    from agentic_rag.chunking.structural import StructuralChunker

    if strategy == ChunkingStrategy.FIXED:
        return FixedSizeChunker()
    if strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveChunker()
    if strategy == ChunkingStrategy.STRUCTURAL:
        return StructuralChunker()
    if strategy == ChunkingStrategy.SEMANTIC:
        if embedding_provider is None:
            raise ValueError("semantic chunking requires an embedding_provider")
        return SemanticChunker(embedding_provider)  # type: ignore[arg-type]
    raise ValueError(f"unknown chunking strategy: {strategy!r}")
