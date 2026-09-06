"""Reranking: takes a wider candidate pool from hybrid retrieval
down to a precise top-k, using a signal retrieval scores alone don't capture
(cross-encoders jointly attend over the query and each candidate, unlike
the independent query/document embeddings behind dense retrieval).

Both retrieval and reranker scores are always preserved on the same
`RetrievedCandidate` — never lose provenance.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from agentic_rag.core.config import ProviderName
from agentic_rag.core.errors import ModelProviderError
from agentic_rag.observability.metrics import RERANK_FAILURES
from agentic_rag.observability.tracing import get_logger
from agentic_rag.retrieval.base import RetrievedCandidate

logger = get_logger(__name__)

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

    async def warm_up(self) -> None:
        """Loads the model now (downloading it on first-ever use) instead
        of on the first real request — found to cost ~10+ seconds when
        that first request also happened to be a real user's, via a real
        end-to-end test. `api/main.py`'s startup calls this when the
        configured reranker supports it; not part of the `Reranker`
        Protocol since `MockReranker` has nothing to warm up."""
        await asyncio.to_thread(self._get_model)

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


async def rerank_with_fallback(
    reranker: Reranker, query: str, candidates: list[RetrievedCandidate], *, top_k: int
) -> list[RetrievedCandidate]:
    """Runs `reranker.rerank`, but never lets a reranker failure (e.g. a
    model that fails to load or errors mid-inference) fail the whole query.
    Falls back to the input order — already fusion/retrieval-scored —
    truncated to `top_k`, with `rerank_score` left unset so callers can tell
    reranking didn't actually run.
    """
    try:
        return await reranker.rerank(query, candidates, top_k=top_k)
    except Exception:
        logger.warning(
            "reranker.failed", reranker=type(reranker).__name__, exc_info=True
        )
        RERANK_FAILURES.inc()
        return candidates[:top_k]


def get_reranker(provider: ProviderName) -> Reranker:
    if provider == "mock":
        return MockReranker()
    if provider in ("local", "ollama"):
        return LocalCrossEncoderReranker()
    raise ModelProviderError(f"no reranker available for {provider!r}")
