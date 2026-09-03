import uuid

from pydantic import BaseModel, Field

from agentic_rag.agents.planner import RetrievalPlan
from agentic_rag.agents.query_analyzer import QueryAnalysis
from agentic_rag.api.schemas.retrieval import RetrievedCandidateResponse
from agentic_rag.core.models import TerminationReason


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
    collection_id: uuid.UUID | None = None


class IterationTraceResponse(BaseModel):
    iteration: int
    queries_used: list[str]
    retrieval_strategy: str
    candidates_retrieved: int
    sufficient: bool
    reason: str
    missing_information: list[str]


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
