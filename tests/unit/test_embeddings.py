import pytest

from agentic_rag.embeddings.cache import CachedEmbeddingProvider
from agentic_rag.embeddings.providers import MockEmbeddingProvider
from agentic_rag.storage.cache import InMemoryCache


@pytest.mark.asyncio
async def test_mock_embedding_provider_is_deterministic() -> None:
    provider = MockEmbeddingProvider()
    first = await provider.embed_texts(["hello world"])
    second = await provider.embed_texts(["hello world"])
    assert first == second


@pytest.mark.asyncio
async def test_mock_embedding_provider_differs_for_different_text() -> None:
    provider = MockEmbeddingProvider()
    vectors = await provider.embed_texts(["hello", "goodbye"])
    assert vectors[0] != vectors[1]


@pytest.mark.asyncio
async def test_mock_embedding_provider_produces_unit_vectors() -> None:
    import numpy as np

    provider = MockEmbeddingProvider(dimensions=384)
    [vector] = await provider.embed_texts(["some text"])
    assert len(vector) == 384
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_cached_embedding_provider_serves_repeat_calls_from_cache() -> None:
    calls: list[list[str]] = []

    class CountingProvider(MockEmbeddingProvider):
        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            calls.append(texts)
            return await super().embed_texts(texts)

    cache = InMemoryCache()
    provider = CachedEmbeddingProvider(CountingProvider(), cache)

    first = await provider.embed_texts(["hello", "world"])
    second = await provider.embed_texts(["hello", "world"])

    assert first == second
    assert len(calls) == 1  # second call served entirely from cache


@pytest.mark.asyncio
async def test_cached_embedding_provider_only_fetches_misses() -> None:
    calls: list[list[str]] = []

    class CountingProvider(MockEmbeddingProvider):
        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            calls.append(texts)
            return await super().embed_texts(texts)

    cache = InMemoryCache()
    provider = CachedEmbeddingProvider(CountingProvider(), cache)

    await provider.embed_texts(["hello"])
    await provider.embed_texts(["hello", "new text"])

    assert calls == [["hello"], ["new text"]]
