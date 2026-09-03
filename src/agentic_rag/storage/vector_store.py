"""Vector store interface.

Kept as an explicit Protocol — separate from `storage/models.py` — so retrieval
code (Phase 4) depends on this interface rather than directly on pgvector/
SQLAlchemy specifics. pgvector-via-Postgres is the only implementation for
now (spec §9/§28 principle: pgvector as the primary store), but this seam is
where a different backend would plug in without touching retrieval code.

Not implemented yet: the concrete PgVectorStore lands with dense retrieval in
Phase 4, once there's an embedding provider (Phase 3) to actually produce
vectors worth searching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: UUID
    score: float


class VectorStore(Protocol):
    async def upsert_embedding(self, chunk_id: UUID, embedding: list[float]) -> None: ...

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        collection_id: UUID | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]: ...
