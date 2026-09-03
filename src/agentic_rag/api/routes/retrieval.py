from fastapi import APIRouter

from agentic_rag.api.dependencies.db import DbSession
from agentic_rag.api.dependencies.embeddings import EmbeddingProviderDep
from agentic_rag.api.schemas.retrieval import (
    RetrievedCandidateResponse,
    RetrieveRequest,
    RetrieveResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from agentic_rag.core.models import RetrievalStrategy
from agentic_rag.retrieval.base import RetrievedCandidate
from agentic_rag.retrieval.dense import DenseRetriever
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.sparse import SparseRetriever

router = APIRouter(tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest, db: DbSession, embedding_provider: EmbeddingProviderDep
) -> SearchResponse:
    """Simple hybrid search — one relevance score per result. For the full
    per-method score breakdown and strategy selection, see POST /retrieve."""
    retriever = HybridRetriever(db, embedding_provider)
    results = await retriever.retrieve(body.query, top_k=body.top_k, filters=body.filters)
    return SearchResponse(
        query=body.query,
        results=[
            SearchResultItem(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_filename=r.document_filename,
                document_title=r.document_title,
                page=r.page,
                section=r.section,
                heading=r.heading,
                content=r.content,
                score=r.fusion_score or 0.0,
            )
            for r in results
        ],
    )


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    body: RetrieveRequest, db: DbSession, embedding_provider: EmbeddingProviderDep
) -> RetrieveResponse:
    """Developer/debug view: exposes dense/sparse/fusion scores independently
    and lets the caller pick a specific retrieval strategy."""
    results: list[RetrievedCandidate]
    if body.strategy == RetrievalStrategy.DENSE:
        dense = DenseRetriever(db, embedding_provider)
        results = await dense.retrieve(
            body.query,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
            filters=body.filters,
        )
    elif body.strategy == RetrievalStrategy.SPARSE:
        sparse = SparseRetriever(db)
        results = await sparse.retrieve(body.query, top_k=body.top_k, filters=body.filters)
    else:
        hybrid = HybridRetriever(db, embedding_provider)
        results = await hybrid.retrieve(
            body.query,
            top_k=body.top_k,
            candidate_pool_size=body.candidate_pool_size,
            filters=body.filters,
        )

    return RetrieveResponse(
        query=body.query,
        strategy=body.strategy,
        results=[
            RetrievedCandidateResponse(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_filename=r.document_filename,
                document_title=r.document_title,
                page=r.page,
                section=r.section,
                heading=r.heading,
                content=r.content,
                dense_score=r.dense_score,
                sparse_score=r.sparse_score,
                fusion_score=r.fusion_score,
                rank=r.rank,
            )
            for r in results
        ],
    )
