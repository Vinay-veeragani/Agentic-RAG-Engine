import uuid

from pydantic import BaseModel, Field

from agentic_rag.agents.planner import RetrievalPlan
from agentic_rag.agents.query_analyzer import QueryAnalysis
from agentic_rag.api.schemas.retrieval import RetrievedCandidateResponse
from agentic_rag.core.models import AnswerStatus, TerminationReason
from agentic_rag.observability.events import Event


class QueryAnalyzeRequest(BaseModel):
    query: str = Field(min_length=1)


class QueryAnalyzeResponse(BaseModel):
    """A preview of query understanding + retrieval planning, ahead of the
    full agentic retrieval loop (Phase 7) that will actually execute a plan
    like this. Not one of spec §29's listed endpoints — added because
    "run it and see the real output" is how every prior phase in this repo
    was verified, and there is otherwise no way to exercise this phase over
    HTTP yet."""

    query: str
    analysis: QueryAnalysis
    plan: RetrievalPlan
    expanded_queries: list[str] | None = None
    subqueries: list[str] | None = None


class AgenticRetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    collection_id: uuid.UUID


class ContradictionResponse(BaseModel):
    claim_a: str
    claim_b: str
    document_a: str
    document_b: str
    chunk_id_a: uuid.UUID
    chunk_id_b: uuid.UUID
    resolution: str | None


class IterationTraceResponse(BaseModel):
    iteration: int
    queries_used: list[str]
    retrieval_strategy: str
    candidates_retrieved: int
    sufficient: bool
    reason: str
    missing_information: list[str]
    contradictions: list[ContradictionResponse]
    years_referenced: list[int]
    spans_multiple_periods: bool
    retrieval_latency_seconds: float
    rerank_latency_seconds: float


class AgenticRetrieveResponse(BaseModel):
    """The full agentic retrieval loop's output: plan, one trace entry per
    iteration, why it stopped, and the final evidence — no synthesized
    answer yet (that's Phase 9's POST /query, spec §29). Structured
    decisions and telemetry only, never hidden chain-of-thought (spec §16)."""

    query: str
    trace_id: str
    analysis: QueryAnalysis
    plan: RetrievalPlan
    iterations: list[IterationTraceResponse]
    termination_reason: TerminationReason
    evidence: list[RetrievedCandidateResponse]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    collection_id: uuid.UUID


class CitationResponse(BaseModel):
    label: str
    claim: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str
    page: int | None
    section: str | None
    source: str | None
    evidence_score: float | None


class QueryResponse(BaseModel):
    """The full pipeline (spec §29 POST /query): query -> plan -> retrieve ->
    rerank -> evidence evaluation -> (refine/retry) -> synthesis -> citation
    validation -> grounded answer. `status` is `grounded` only when at least
    one claim survived citation validation; otherwise `answer`/`citations`
    are empty and `status` explains why (insufficient/conflicting evidence,
    or none found) rather than ever guessing at an answer."""

    query: str
    trace_id: str
    analysis: QueryAnalysis
    plan: RetrievalPlan
    status: AnswerStatus
    answer: str | None
    citations: list[CitationResponse]
    citation_completeness: float | None
    citation_precision: float | None
    termination_reason: TerminationReason
    iterations: list[IterationTraceResponse]


class TraceResponse(BaseModel):
    """spec §29's GET /queries/{id}/trace — the same structured events (spec
    §30) a live SSE stream would have emitted for this query, replayed from
    this process's in-memory TraceStore."""

    trace_id: str
    events: list[Event]
