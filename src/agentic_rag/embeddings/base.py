"""Embedding provider interface.

Every provider (local, remote, mock) implements this one Protocol so nothing
above this layer — chunking, indexing, retrieval — depends on which model
actually produced the vectors (spec §8 principle: don't hardcode one
embedding model throughout the codebase).

`dimensions` must match `storage.models.EMBEDDING_DIMENSIONS` for whatever
provider is actually configured, since pgvector columns have a fixed
dimension — see docs/architecture.md for that constraint.
"""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding per input text, same order, same length."""
        ...
