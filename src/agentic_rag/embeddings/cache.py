"""Embedding cache wrapper.

Wraps any `EmbeddingProvider` and caches per-text results behind the shared
`CacheClient` (Redis, or the in-memory fallback — see storage/cache.py).
Cache keys include the model name, so switching embedding models never
serves a stale vector computed by a different model for the same text.
"""

from __future__ import annotations

import hashlib
import json

from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.observability.metrics import CACHE_HITS, CACHE_MISSES
from agentic_rag.storage.cache import CacheClient

_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # embeddings for fixed text+model never go stale


class CachedEmbeddingProvider:
    def __init__(self, provider: EmbeddingProvider, cache: CacheClient) -> None:
        self._provider = provider
        self._cache = cache

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    def _cache_key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embedding:{self.model_name}:{text_hash}"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        keys = [self._cache_key(text) for text in texts]
        cached_values = [await self._cache.get(key) for key in keys]

        misses = [i for i, cached in enumerate(cached_values) if cached is None]
        CACHE_HITS.labels(cache="embedding").inc(len(texts) - len(misses))
        CACHE_MISSES.labels(cache="embedding").inc(len(misses))
        if misses:
            fresh = await self._provider.embed_texts([texts[i] for i in misses])
            for miss_index, embedding in zip(misses, fresh, strict=True):
                await self._cache.set(
                    keys[miss_index], json.dumps(embedding), ex=_CACHE_TTL_SECONDS
                )
                cached_values[miss_index] = json.dumps(embedding)

        return [json.loads(value) for value in cached_values]  # type: ignore[arg-type]
