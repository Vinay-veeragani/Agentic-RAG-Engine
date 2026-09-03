"""Hybrid retrieval: dense + sparse, combined via Reciprocal Rank Fusion
(spec §9)."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.retrieval.base import MetadataFilter, RetrievedCandidate
from agentic_rag.retrieval.dense import DenseRetriever
from agentic_rag.retrieval.fusion import reciprocal_rank_fusion
from agentic_rag.retrieval.sparse import SparseRetriever

DEFAULT_CANDIDATE_POOL_SIZE = 30


class HybridRetriever:
    def __init__(self, session: AsyncSession, embedding_provider: EmbeddingProvider) -> None:
        self._dense = DenseRetriever(session, embedding_provider)
        self._sparse = SparseRetriever(session)

    async def retrieve(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
        filters: MetadataFilter | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievedCandidate]:
        """Retrieves `candidate_pool_size` candidates from each method (wider
        than `top_k`, so fusion has enough signal to reorder), fuses them,
        and returns the top `top_k` — never losing either method's raw score
        (spec §14 provenance) even for a candidate only one method found.
        """
        dense_results = await self._dense.retrieve(
            query_text, top_k=candidate_pool_size, filters=filters
        )
        sparse_results = await self._sparse.retrieve(
            query_text, top_k=candidate_pool_size, filters=filters
        )

        dense_by_id = {c.chunk_id: c for c in dense_results}
        sparse_by_id = {c.chunk_id: c for c in sparse_results}

        dense_ranking = [c.chunk_id for c in dense_results]
        sparse_ranking = [c.chunk_id for c in sparse_results]
        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=rrf_k)

        results: list[RetrievedCandidate] = []
        for rank, (chunk_id, fusion_score) in enumerate(fused[:top_k], start=1):
            results.append(_merge(chunk_id, dense_by_id, sparse_by_id, fusion_score, rank))
        return results


def _merge(
    chunk_id: uuid.UUID,
    dense_by_id: dict[uuid.UUID, RetrievedCandidate],
    sparse_by_id: dict[uuid.UUID, RetrievedCandidate],
    fusion_score: float,
    rank: int,
) -> RetrievedCandidate:
    base = dense_by_id.get(chunk_id) or sparse_by_id[chunk_id]
    dense = dense_by_id.get(chunk_id)
    sparse = sparse_by_id.get(chunk_id)
    return RetrievedCandidate(
        chunk_id=base.chunk_id,
        document_id=base.document_id,
        content=base.content,
        page=base.page,
        section=base.section,
        heading=base.heading,
        document_filename=base.document_filename,
        document_title=base.document_title,
        dense_score=dense.dense_score if dense else None,
        sparse_score=sparse.sparse_score if sparse else None,
        fusion_score=fusion_score,
        rank=rank,
    )
