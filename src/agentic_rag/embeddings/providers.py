"""Mock + remote embedding providers, and the provider registry.

`MockEmbeddingProvider` is deterministic (same text -> same vector every
run) so tests and demos never require a real model or network access
(spec principle #8). `OpenAIEmbeddingProvider` is the one remote example —
implemented over plain httpx rather than the `openai` SDK, since this is the
only call site that needs it and a whole SDK dependency isn't worth it for
one REST call.
"""

from __future__ import annotations

import hashlib

import httpx
import numpy as np

from agentic_rag.core.config import ProviderName, Settings
from agentic_rag.core.errors import ModelProviderError
from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.embeddings.local import DEFAULT_DIMENSIONS


class MockEmbeddingProvider:
    """Hashes each text into a deterministic pseudo-random unit vector.

    Not semantically meaningful (two similar sentences do not get similar
    vectors) — this is a stand-in for testing the *plumbing* (storage,
    retrieval, ranking mechanics), not retrieval quality.
    """

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return "mock"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self._dimensions)
        vector /= np.linalg.norm(vector) or 1.0
        return [float(x) for x in vector]


class OpenAIEmbeddingProvider:
    """Calls OpenAI's embeddings endpoint directly, requesting a truncated
    `dimensions` output so the vector width matches our fixed pgvector
    column regardless of which provider is configured (OpenAI's v3 embedding
    models support Matryoshka-style dimension truncation via this param)."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = DEFAULT_DIMENSIONS,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": texts, "dimensions": self._dimensions},
            )
        if response.status_code != 200:
            raise ModelProviderError(
                f"OpenAI embeddings request failed: {response.status_code}",
                details={"body": response.text[:500]},
            )
        data = response.json()["data"]
        ordered = sorted(data, key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]


def get_embedding_provider(provider: ProviderName, settings: Settings) -> EmbeddingProvider:
    if provider == "mock":
        return MockEmbeddingProvider()
    if provider in ("local", "ollama"):
        from agentic_rag.embeddings.local import LocalEmbeddingProvider

        return LocalEmbeddingProvider()
    if provider == "openai":
        if not settings.openai_api_key:
            raise ModelProviderError("OPENAI_API_KEY is not configured")
        return OpenAIEmbeddingProvider(api_key=settings.openai_api_key)
    raise ModelProviderError(f"no embedding provider available for {provider!r}")
