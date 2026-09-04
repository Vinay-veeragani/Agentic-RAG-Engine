"""Query understanding and query expansion.

Both return strict pydantic-validated output via
`LLMProvider.complete_structured` — never uncontrolled free-form planning
text. This is one of the few places an LLM call is used at all: classifying
a query into a category, and proposing phrasing variants, are genuinely
language-understanding tasks that deterministic rules handle poorly in the
general case (engineering principle #2), unlike e.g. RRF or chunk windowing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_rag.core.models import QueryType
from agentic_rag.generation.llm import LLMProvider

_ANALYZER_SYSTEM_PROMPT = (
    "You are the query analyzer for a knowledge retrieval system. Classify "
    "the user's query so the retrieval planner can choose an appropriate "
    "strategy. Do not answer the query — only classify it."
)

_EXPANDER_SYSTEM_PROMPT = (
    "You expand a search query into a small set of alternative phrasings "
    "and closely related terms that would help a hybrid (dense + sparse) "
    "retrieval system find relevant documents it might otherwise miss — "
    "e.g. domain synonyms, related metrics, or rephrasings. Do not answer "
    "the query. Keep each variant a short, standalone search query."
)


class QueryAnalysis(BaseModel):
    query_type: QueryType
    is_ambiguous: bool
    is_answerable: bool
    reasoning: str = Field(
        description="One short sentence: why this classification, not a chain of thought."
    )


class QueryExpansion(BaseModel):
    expanded_queries: list[str] = Field(min_length=1, max_length=5)


class QueryAnalyzer:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def analyze(self, query_text: str) -> QueryAnalysis:
        return await self._llm.complete_structured(
            system_prompt=_ANALYZER_SYSTEM_PROMPT,
            user_prompt=f"Query: {query_text}",
            schema=QueryAnalysis,
        )


class QueryExpander:
    """Only meaningful when the planner has decided expansion is useful
    ("The planner should decide when expansion is useful. Do not
    blindly expand every query.") — this class does the expansion itself,
    not the decide-whether-to step."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def expand(self, query_text: str) -> QueryExpansion:
        result = await self._llm.complete_structured(
            system_prompt=_EXPANDER_SYSTEM_PROMPT,
            user_prompt=f"Query: {query_text}",
            schema=QueryExpansion,
        )
        # Never trust a provider to respect the schema's bounds — clamp
        # defensively regardless of what actually validated.
        result.expanded_queries = result.expanded_queries[:5] or [query_text]
        return result
