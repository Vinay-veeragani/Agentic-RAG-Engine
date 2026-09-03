from pydantic import BaseModel, Field

from agentic_rag.agents.planner import RetrievalPlan
from agentic_rag.agents.query_analyzer import QueryAnalysis


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
