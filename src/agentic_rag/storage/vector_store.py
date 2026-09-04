"""Vector store interface.

Kept as an explicit Protocol — separate from `storage/models.py` — as the
seam a different vector backend would plug into without touching
retrieval code. In practice, `retrieval/dense.py` queries pgvector
directly via SQLAlchemy rather than going through this Protocol: this
codebase only ever targets pgvector-via-Postgres, so a concrete
`PgVectorStore` implementing it hasn't actually been built — this type
exists to name the seam, not because it's wired up yet.
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
