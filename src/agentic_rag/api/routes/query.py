import asyncio
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

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
    TraceResponse,
)
from agentic_rag.api.schemas.retrieval import RetrievedCandidateResponse
from agentic_rag.citations.formatter import format_citation
from agentic_rag.core.config import Settings, get_settings
from agentic_rag.core.errors import AgenticRAGError, QueryTimeoutError
from agentic_rag.core.models import AnswerStatus, TerminationReason
from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.generation.llm import LLMProvider
from agentic_rag.observability.events import EventEmitter, EventType, trace_store
from agentic_rag.observability.tracing import get_logger, get_trace_id
from agentic_rag.retrieval.reranking import Reranker

router = APIRouter(prefix="/query", tags=["query"])
queries_router = APIRouter(prefix="/queries", tags=["query"])
logger = get_logger(__name__)

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
        retrieval_latency_seconds=it.retrieval_latency_seconds,
        rerank_latency_seconds=it.rerank_latency_seconds,
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
    """Runs the full bounded agentic retrieval loop and returns
    its structured trace + final evidence — no synthesized answer (see
    POST /query for that)."""
    trace_id = get_trace_id() or "unbound"
    emitter = EventEmitter(trace_id)
    settings = get_settings()

    loop = AgenticRetrievalLoop(
        session=db,
        llm=llm,
        embedding_provider=embedding_provider,
        reranker=reranker,
        settings=settings,
    )
    try:
        result = await asyncio.wait_for(
            loop.run(body.query, collection_id=body.collection_id, emitter=emitter),
            timeout=settings.max_query_latency_seconds,
        )
    except TimeoutError as exc:
        raise QueryTimeoutError(
            f"Query exceeded the {settings.max_query_latency_seconds}s latency budget"
        ) from exc
    emitter.emit(EventType.QUERY_COMPLETED, termination_reason=result.termination_reason.value)
    trace_store.store(trace_id, emitter.events)

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


async def _run_query_pipeline(
    query_text: str,
    collection_id: uuid.UUID | None,
    *,
    db: AsyncSession,
    llm: LLMProvider,
    embedding_provider: EmbeddingProvider,
    reranker: Reranker,
    settings: Settings,
    emitter: EventEmitter,
) -> QueryResponse:
    """The full pipeline: agentic retrieval loop -> answer
    synthesis -> citation validation -> grounded answer. If the loop ended
    without usable evidence (conflicting or none found), synthesis is
    skipped entirely — there is nothing honest to synthesize from, and
    attempting it anyway would risk exactly the hallucination this system
    is built to avoid. Shared by both POST /query and POST /query/stream.
    """
    loop = AgenticRetrievalLoop(
        session=db,
        llm=llm,
        embedding_provider=embedding_provider,
        reranker=reranker,
        settings=settings,
    )
    loop_result = await loop.run(query_text, collection_id=collection_id, emitter=emitter)

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
        verification = await verifier.generate(query_text, loop_result.evidence, emitter=emitter)
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

    emitter.emit(
        EventType.QUERY_COMPLETED,
        status=status.value,
        termination_reason=loop_result.termination_reason.value,
    )

    return QueryResponse(
        query=loop_result.query,
        trace_id=loop_result.trace_id,
        analysis=loop_result.analysis,
        plan=loop_result.plan,
        status=status,
        answer=answer,
        citations=citations,
        citation_completeness=citation_completeness,
        citation_precision=citation_precision,
        termination_reason=loop_result.termination_reason,
        iterations=[_iteration_trace_response(it) for it in loop_result.iterations],
    )


@router.post("", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    db: DbSession,
    llm: LLMProviderDep,
    embedding_provider: EmbeddingProviderDep,
    reranker: RerankerDep,
) -> QueryResponse:
    trace_id = get_trace_id() or "unbound"
    emitter = EventEmitter(trace_id)
    settings = get_settings()
    try:
        response = await asyncio.wait_for(
            _run_query_pipeline(
                body.query,
                body.collection_id,
                db=db,
                llm=llm,
                embedding_provider=embedding_provider,
                reranker=reranker,
                settings=settings,
                emitter=emitter,
            ),
            timeout=settings.max_query_latency_seconds,
        )
    except TimeoutError as exc:
        trace_store.store(trace_id, emitter.events)
        raise QueryTimeoutError(
            f"Query exceeded the {settings.max_query_latency_seconds}s latency budget"
        ) from exc
    trace_store.store(trace_id, emitter.events)
    return response


@router.post("/stream")
async def query_stream(
    body: QueryRequest,
    db: DbSession,
    llm: LLMProviderDep,
    embedding_provider: EmbeddingProviderDep,
    reranker: RerankerDep,
) -> StreamingResponse:
    """SSE version of POST /query: the same pipeline, but each
    structured event is pushed to the client as it happens rather than only
    the final response. Reconnection: a client may send `Last-Event-ID`, but
    events aren't replayed from before a reconnect — only this process's
    `TraceStore` (in-memory, not persisted) can be queried afterward via
    GET /queries/{trace_id}/trace, and only for the trace ID a client
    learned from the stream's own `query.started` event.
    """
    trace_id = get_trace_id() or "unbound"
    emitter = EventEmitter(trace_id, queue=True)
    settings = get_settings()

    async def run_and_close() -> None:
        try:
            await asyncio.wait_for(
                _run_query_pipeline(
                    body.query,
                    body.collection_id,
                    db=db,
                    llm=llm,
                    embedding_provider=embedding_provider,
                    reranker=reranker,
                    settings=settings,
                    emitter=emitter,
                ),
                timeout=settings.max_query_latency_seconds,
            )
        except TimeoutError:
            emitter.emit(
                EventType.QUERY_FAILED,
                error="QueryTimeoutError",
                message=f"Query exceeded the {settings.max_query_latency_seconds}s latency budget",
            )
        except AgenticRAGError as exc:
            emitter.emit(EventType.QUERY_FAILED, error=exc.code.value, message=exc.message)
        except Exception as exc:
            logger.error("query_stream.unhandled_error", error_type=type(exc).__name__)
            emitter.emit(
                EventType.QUERY_FAILED,
                error="MODEL_ERROR",
                message="An internal error occurred.",
            )
        finally:
            trace_store.store(trace_id, emitter.events)
            emitter.close()

    async def event_stream() -> AsyncIterator[str]:
        task = asyncio.create_task(run_and_close())
        try:
            while True:
                event = await emitter.queue.get()
                if event is None:
                    break
                yield (
                    f"id: {event.event_id}\n"
                    f"event: {event.event_type.value}\n"
                    f"data: {event.model_dump_json()}\n\n"
                )
        finally:
            await task

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@queries_router.get("/{trace_id}/trace", response_model=TraceResponse)
async def get_query_trace(trace_id: str) -> TraceResponse:
    """GET /queries/{id}/trace. Process-local, in-memory only
    (see `observability.events.TraceStore`) — not persisted across restarts
    or shared across worker processes; a documented gap, not silently
    approximated."""
    events = trace_store.get(trace_id)
    if events is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return TraceResponse(trace_id=trace_id, events=events)
