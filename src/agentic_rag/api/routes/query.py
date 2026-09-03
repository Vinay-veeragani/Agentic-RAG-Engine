from fastapi import APIRouter

from agentic_rag.agents.planner import QueryDecomposer, RetrievalPlanner
from agentic_rag.agents.query_analyzer import QueryAnalyzer, QueryExpander
from agentic_rag.agents.research_agent import AgenticRetrievalLoop
from agentic_rag.api.dependencies.db import DbSession
from agentic_rag.api.dependencies.embeddings import EmbeddingProviderDep
from agentic_rag.api.dependencies.llm import LLMProviderDep
from agentic_rag.api.dependencies.reranker import RerankerDep
from agentic_rag.api.schemas.query import (
    AgenticRetrieveRequest,
    AgenticRetrieveResponse,
    IterationTraceResponse,
    QueryAnalyzeRequest,
    QueryAnalyzeResponse,
)
from agentic_rag.api.schemas.retrieval import RetrievedCandidateResponse
from agentic_rag.core.config import get_settings

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/analyze", response_model=QueryAnalyzeResponse)
async def analyze_query(
    body: QueryAnalyzeRequest, llm: LLMProviderDep
) -> QueryAnalyzeResponse:
    settings = get_settings()

    analysis = await QueryAnalyzer(llm).analyze(body.query)
    plan = await RetrievalPlanner(
        llm, max_iterations_ceiling=settings.max_retrieval_iterations
    ).plan(body.query, analysis)

    expanded_queries = None
    if plan.expand_query:
        expanded_queries = (await QueryExpander(llm).expand(body.query)).expanded_queries

    subqueries = None
    if plan.decompose:
        subqueries = (await QueryDecomposer(llm).decompose(body.query)).subqueries

    return QueryAnalyzeResponse(
        query=body.query,
        analysis=analysis,
        plan=plan,
        expanded_queries=expanded_queries,
        subqueries=subqueries,
    )


@router.post("/retrieve", response_model=AgenticRetrieveResponse)
async def agentic_retrieve(
    body: AgenticRetrieveRequest,
    db: DbSession,
    llm: LLMProviderDep,
    embedding_provider: EmbeddingProviderDep,
    reranker: RerankerDep,
) -> AgenticRetrieveResponse:
    """Runs the full bounded agentic retrieval loop (spec §16) and returns
    its structured trace + final evidence — no synthesized answer yet
    (Phase 9's POST /query will add that)."""
    loop = AgenticRetrievalLoop(
        session=db,
        llm=llm,
        embedding_provider=embedding_provider,
        reranker=reranker,
        settings=get_settings(),
    )
    result = await loop.run(body.query, collection_id=body.collection_id)

    return AgenticRetrieveResponse(
        query=result.query,
        trace_id=result.trace_id,
        analysis=result.analysis,
        plan=result.plan,
        iterations=[
            IterationTraceResponse(
                iteration=it.iteration,
                queries_used=it.queries_used,
                retrieval_strategy=it.retrieval_strategy,
                candidates_retrieved=it.candidates_retrieved,
                sufficient=it.sufficient,
                reason=it.reason,
                missing_information=it.missing_information,
            )
            for it in result.iterations
        ],
        termination_reason=result.termination_reason,
        evidence=[
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
                rerank_score=r.rerank_score,
                rank=r.rank,
            )
            for r in result.evidence
        ],
    )
