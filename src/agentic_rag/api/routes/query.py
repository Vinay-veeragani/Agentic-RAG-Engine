from fastapi import APIRouter

from agentic_rag.agents.citation_agent import CitationAgent
from agentic_rag.agents.planner import QueryDecomposer, RetrievalPlanner
from agentic_rag.agents.query_analyzer import QueryAnalyzer, QueryExpander
from agentic_rag.agents.research_agent import AgenticRetrievalLoop, IterationTrace
from agentic_rag.agents.synthesis_agent import SynthesisAgent
from agentic_rag.agents.verifier import AnswerVerifier
from agentic_rag.api.dependencies.db import DbSession
from agentic_rag.api.dependencies.embeddings import EmbeddingProviderDep
from agentic_rag.api.dependencies.llm import LLMProviderDep
from agentic_rag.api.dependencies.reranker import RerankerDep
from agentic_rag.api.schemas.query import (
    AgenticRetrieveRequest,
    AgenticRetrieveResponse,
    CitationResponse,
    ContradictionResponse,
    IterationTraceResponse,
    QueryAnalyzeRequest,
    QueryAnalyzeResponse,
    QueryRequest,
    QueryResponse,
)
from agentic_rag.api.schemas.retrieval import RetrievedCandidateResponse
from agentic_rag.citations.formatter import format_citation
from agentic_rag.core.config import get_settings
from agentic_rag.core.models import AnswerStatus, TerminationReason

router = APIRouter(prefix="/query", tags=["query"])

_LOOP_STATUS_OVERRIDE = {
    TerminationReason.CONFLICTING_EVIDENCE: AnswerStatus.CONFLICTING_EVIDENCE,
    TerminationReason.NO_EVIDENCE_FOUND: AnswerStatus.NO_EVIDENCE_FOUND,
}


def _iteration_trace_response(it: IterationTrace) -> IterationTraceResponse:
    return IterationTraceResponse(
        iteration=it.iteration,
        queries_used=it.queries_used,
        retrieval_strategy=it.retrieval_strategy,
        candidates_retrieved=it.candidates_retrieved,
        sufficient=it.sufficient,
        reason=it.reason,
        missing_information=it.missing_information,
        contradictions=[
            ContradictionResponse(
                claim_a=c.claim_a,
                claim_b=c.claim_b,
                document_a=c.document_a,
                document_b=c.document_b,
                chunk_id_a=c.chunk_id_a,
                chunk_id_b=c.chunk_id_b,
                resolution=c.resolution,
            )
            for c in it.contradictions
        ],
        years_referenced=it.years_referenced,
        spans_multiple_periods=it.spans_multiple_periods,
    )


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
    its structured trace + final evidence — no synthesized answer (see
    POST /query for that)."""
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
        iterations=[_iteration_trace_response(it) for it in result.iterations],
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


@router.post("", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    db: DbSession,
    llm: LLMProviderDep,
    embedding_provider: EmbeddingProviderDep,
    reranker: RerankerDep,
) -> QueryResponse:
    """The full pipeline (spec §29): agentic retrieval loop -> answer
    synthesis -> citation validation -> grounded answer. If the loop ended
    without usable evidence (conflicting or none found), synthesis is
    skipped entirely — there is nothing honest to synthesize from, and
    attempting it anyway would risk exactly the hallucination this system
    is built to avoid.
    """
    loop = AgenticRetrievalLoop(
        session=db,
        llm=llm,
        embedding_provider=embedding_provider,
        reranker=reranker,
        settings=get_settings(),
    )
    loop_result = await loop.run(body.query, collection_id=body.collection_id)

    status: AnswerStatus
    answer: str | None = None
    citations: list[CitationResponse] = []
    citation_completeness: float | None = None
    citation_precision: float | None = None

    override = _LOOP_STATUS_OVERRIDE.get(loop_result.termination_reason)
    if override is not None:
        status = override
    else:
        verifier = AnswerVerifier(SynthesisAgent(llm), CitationAgent(llm))
        verification = await verifier.generate(body.query, loop_result.evidence)
        status = verification.status
        answer = verification.answer
        citations = [
            CitationResponse(
                label=format_citation(citation, index),
                claim=citation.claim,
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                document_filename=citation.document_filename,
                page=citation.page,
                section=citation.section,
                source=citation.source,
                evidence_score=citation.evidence_score,
            )
            for index, citation in enumerate(verification.citations, start=1)
        ]
        if verification.citation_metrics is not None:
            citation_completeness = verification.citation_metrics.citation_completeness
            citation_precision = verification.citation_metrics.citation_precision

    return QueryResponse(
        query=loop_result.query,
        trace_id=loop_result.trace_id,
        status=status,
        answer=answer,
        citations=citations,
        citation_completeness=citation_completeness,
        citation_precision=citation_precision,
        termination_reason=loop_result.termination_reason,
        iterations=[_iteration_trace_response(it) for it in loop_result.iterations],
    )
