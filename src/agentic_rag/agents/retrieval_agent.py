"""One retrieval-and-rerank pass within the agentic loop (the "retrieve" ->
"rerank" steps), supporting more than one query variant at once (used for
query expansion) by fusing per-variant result rankings with the same
`reciprocal_rank_fusion` used to combine dense+sparse in
`retrieval/hybrid.py` — applied one level up, across query variants instead
of across retrieval methods.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.agents.planner import RetrievalPlan
from agentic_rag.core.models import RetrievalStrategy
from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.observability.events import EventEmitter, EventType
from agentic_rag.observability.metrics import RERANK_LATENCY_SECONDS, RETRIEVAL_LATENCY_SECONDS
from agentic_rag.retrieval.base import RetrievedCandidate
from agentic_rag.retrieval.dense import DenseRetriever
from agentic_rag.retrieval.fusion import reciprocal_rank_fusion
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.reranking import Reranker, rerank_with_fallback
from agentic_rag.retrieval.sparse import SparseRetriever

DEFAULT_EVIDENCE_TOP_K = 8


@dataclass(slots=True)
class RetrievalOutcome:
    candidates: list[RetrievedCandidate]
    retrieval_latency_seconds: float
    rerank_latency_seconds: float


class RetrievalAgent:
    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker,
    ) -> None:
        self._session = session
        self._embeddings = embedding_provider
        self._reranker = reranker

    async def retrieve(
        self,
        queries: list[str],
        plan: RetrievalPlan,
        *,
        evidence_top_k: int = DEFAULT_EVIDENCE_TOP_K,
        emitter: EventEmitter | None = None,
    ) -> RetrievalOutcome:
        """`queries` is one or more variants of the same information need
        (query expansion) — a single query is the common case.

        Retrieved sequentially, not via `asyncio.gather`: all variants share
        one `AsyncSession`, and SQLAlchemy's async session does not support
        concurrent operations from multiple coroutines — "execute
        concurrently where safe" does not apply here; this is exactly the
        "where safe" carve-out, not an oversight.
        """
        if emitter:
            emitter.emit(EventType.RETRIEVAL_STARTED, queries=queries, strategy=plan.strategy.value)

        retrieval_start = time.perf_counter()
        variant_results = [await self._retrieve_single(q, plan) for q in queries]
        merged = (
            variant_results[0] if len(variant_results) == 1 else _fuse_variants(variant_results)
        )
        retrieval_latency = time.perf_counter() - retrieval_start
        RETRIEVAL_LATENCY_SECONDS.observe(retrieval_latency)

        if emitter:
            emitter.emit(EventType.RETRIEVAL_COMPLETED, candidate_count=len(merged))
            emitter.emit(EventType.RERANKING_STARTED, candidate_count=len(merged))

        rerank_start = time.perf_counter()
        reranked = await rerank_with_fallback(
            self._reranker, queries[0], merged, top_k=evidence_top_k
        )
        for rank, candidate in enumerate(reranked, start=1):
            candidate.rank = rank
        rerank_latency = time.perf_counter() - rerank_start
        RERANK_LATENCY_SECONDS.observe(rerank_latency)

        if emitter:
            emitter.emit(EventType.RERANKING_COMPLETED, evidence_count=len(reranked))

        return RetrievalOutcome(
            candidates=reranked,
            retrieval_latency_seconds=retrieval_latency,
            rerank_latency_seconds=rerank_latency,
        )

    async def _retrieve_single(
        self, query_text: str, plan: RetrievalPlan
    ) -> list[RetrievedCandidate]:
        if plan.strategy == RetrievalStrategy.DENSE:
            dense = DenseRetriever(self._session, self._embeddings)
            return await dense.retrieve(query_text, top_k=plan.top_k, filters=plan.filters)
        if plan.strategy == RetrievalStrategy.SPARSE:
            sparse = SparseRetriever(self._session)
            return await sparse.retrieve(query_text, top_k=plan.top_k, filters=plan.filters)
        hybrid = HybridRetriever(self._session, self._embeddings)
        return await hybrid.retrieve(
            query_text, top_k=plan.top_k, candidate_pool_size=plan.top_k, filters=plan.filters
        )


def _fuse_variants(variant_results: list[list[RetrievedCandidate]]) -> list[RetrievedCandidate]:
    rankings = [[c.chunk_id for c in variant] for variant in variant_results]
    fused = reciprocal_rank_fusion(rankings)

    by_id: dict[uuid.UUID, RetrievedCandidate] = {}
    for variant in variant_results:
        for candidate in variant:
            existing = by_id.get(candidate.chunk_id)
            if existing is None:
                by_id[candidate.chunk_id] = candidate
                continue
            # Keep the best score seen for this chunk across variants —
            # never lose provenance, even when merging.
            if candidate.dense_score is not None:
                if existing.dense_score is None or candidate.dense_score > existing.dense_score:
                    existing.dense_score = candidate.dense_score
            if candidate.sparse_score is not None:
                if existing.sparse_score is None or candidate.sparse_score > existing.sparse_score:
                    existing.sparse_score = candidate.sparse_score

    results: list[RetrievedCandidate] = []
    for chunk_id, fusion_score in fused:
        candidate = by_id[chunk_id]
        candidate.fusion_score = fusion_score
        results.append(candidate)
    return results
