"""Reranking (spec §14): takes a wider candidate pool from hybrid retrieval
down to a precise top-k, using a signal retrieval scores alone don't capture
(cross-encoders jointly attend over the query and each candidate, unlike
the independent query/document embeddings behind dense retrieval).

Both retrieval and reranker scores are always preserved on the same
`RetrievedCandidate` — spec §14 "never lose provenance."
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from agentic_rag.core.config import ProviderName
from agentic_rag.core.errors import ModelProviderError
from agentic_rag.retrieval.base import RetrievedCandidate

DEFAULT_LOCAL_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    async def rerank(
        self, query: str, candidates: list[RetrievedCandidate], *, top_k: int
    ) -> list[RetrievedCandidate]:
        """Returns at most `top_k` candidates, each with `rerank_score` set,
        sorted by that score descending. Input order/scores otherwise
        untouched — this only adds a score and truncates/reorders."""
        ...


class MockReranker:
    """Deterministic term-overlap scorer: no model, no network. Not
    semantically meaningful (this is not a real cross-encoder — it won't
    catch paraphrases or reorderings) — a stand-in for testing the
    reranking *plumbing*, matching MockEmbeddingProvider's role for
    embeddings."""

    async def rerank(
        self, query: str, candidates: list[RetrievedCandidate], *, top_k: int
    ) -> list[RetrievedCandidate]:
        query_terms = set(query.lower().split())
        scored = []
        for candidate in candidates:
            content_terms = set(candidate.content.lower().split())
            overlap = len(query_terms & content_terms)
            score = overlap / len(query_terms) if query_terms else 0.0
            candidate.rerank_score = score
            scored.append(candidate)
        scored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return scored[:top_k]


class LocalCrossEncoderReranker:
    """A real cross-encoder (sentence-transformers `CrossEncoder`), running
    on CPU, no API key. Weights download from Hugging Face Hub on first use
    and are cached locally after that."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def _get_model(self) -> Any:  # sentence-transformers ships no py.typed stubs
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device="cpu")
        return self._model

    async def rerank(
        self, query: str, candidates: list[RetrievedCandidate], *, top_k: int
    ) -> list[RetrievedCandidate]:
        if not candidates:
            return []
        scores = await asyncio.to_thread(self._score_sync, query, candidates)
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)
        ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return ranked[:top_k]

    def _score_sync(self, query: str, candidates: list[RetrievedCandidate]) -> list[float]:
        model = self._get_model()
        pairs = [(query, c.content) for c in candidates]
        return model.predict(pairs)


def get_reranker(provider: ProviderName) -> Reranker:
    if provider == "mock":
        return MockReranker()
    if provider in ("local", "ollama"):
        return LocalCrossEncoderReranker()
    raise ModelProviderError(f"no reranker available for {provider!r}")
