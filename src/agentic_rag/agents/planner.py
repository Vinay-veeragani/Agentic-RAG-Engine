"""Retrieval planning and query decomposition.

The plan decides *whether* expansion/decomposition/iteration should happen;
it never executes them. `RetrievalPlanner` is also where the hard budget
ceilings from `core/config.py` actually get enforced — an LLM (or the mock)
proposing `max_iterations: 50` must never be trusted; it's clamped here
before the plan is used by anything downstream: the planner must be bounded
and validated.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_rag.agents.query_analyzer import QueryAnalysis
from agentic_rag.core.models import RetrievalStrategy
from agentic_rag.generation.llm import LLMProvider
from agentic_rag.retrieval.base import MetadataFilter

_PLANNER_SYSTEM_PROMPT = (
    "You are the retrieval planner for a knowledge retrieval system. Given a "
    "query and its classification, decide a retrieval strategy and whether "
    "query expansion, decomposition, and how many retrieval iterations are "
    "appropriate. Prefer hybrid retrieval unless there's a clear reason not "
    "to. Only enable decomposition for genuinely multi-part or comparison "
    "queries, and only enable expansion when the query is vague, short, or "
    "uses ambiguous terminology."
)

_DECOMPOSER_SYSTEM_PROMPT = (
    "You decompose a complex query into a minimal set of independent, "
    "standalone subqueries that together cover everything the original "
    "query asks for. Do not answer the query. Each subquery must be "
    "self-contained (resolve any pronouns/references to the original "
    "query's subject) and independently retrievable."
)


class RetrievalPlan(BaseModel):
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    expand_query: bool = False
    decompose: bool = False
    max_iterations: int = Field(default=1, ge=1)
    top_k: int = Field(default=10, ge=1)
    filters: MetadataFilter = Field(default_factory=MetadataFilter)


class QueryDecomposition(BaseModel):
    subqueries: list[str] = Field(min_length=1, max_length=8)


class RetrievalPlanner:
    def __init__(
        self,
        llm: LLMProvider,
        *,
        max_iterations_ceiling: int,
        top_k_ceiling: int = 50,
    ) -> None:
        self._llm = llm
        self._max_iterations_ceiling = max_iterations_ceiling
        self._top_k_ceiling = top_k_ceiling

    async def plan(self, query_text: str, analysis: QueryAnalysis) -> RetrievalPlan:
        plan = await self._llm.complete_structured(
            system_prompt=_PLANNER_SYSTEM_PROMPT,
            user_prompt=(
                f"Query: {query_text}\n"
                f"Classification: {analysis.model_dump_json()}"
            ),
            schema=RetrievalPlan,
        )
        # Hard ceilings from core/config.py — never trust the plan alone,
        # regardless of which provider (or the mock) produced it.
        plan.max_iterations = max(1, min(plan.max_iterations, self._max_iterations_ceiling))
        plan.top_k = max(1, min(plan.top_k, self._top_k_ceiling))
        return plan


class QueryDecomposer:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def decompose(self, query_text: str) -> QueryDecomposition:
        result = await self._llm.complete_structured(
            system_prompt=_DECOMPOSER_SYSTEM_PROMPT,
            user_prompt=f"Query: {query_text}",
            schema=QueryDecomposition,
        )
        result.subqueries = result.subqueries[:8] or [query_text]
        return result
