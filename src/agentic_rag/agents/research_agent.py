"""The agentic retrieval loop (spec §16): plan -> retrieve -> rerank ->
evaluate evidence -> if insufficient, refine and retry -> stop when
sufficient or budget exhausted.

Hard-bounded by construction: the loop body runs inside
`for iteration in range(1, plan.max_iterations + 1)`, and `plan.max_iterations`
is itself clamped to `settings.max_retrieval_iterations` by
`RetrievalPlanner` before this ever runs — a `for` loop over a fixed range
cannot run forever regardless of what any LLM (or the mock) proposes. A
second, independent ceiling (`settings.max_retrieval_calls`) is checked every
iteration in case a single query expands into many retrieval calls (spec
§12's concurrent subquery execution). Every run ends in exactly one
`TerminationReason` — never a silent stop.

No hidden chain-of-thought is exposed: `AgenticRetrievalResult` carries
structured `IterationTrace`s (queries used, strategy, candidate count,
sufficiency verdict, reason, missing information) and nothing else.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.agents.evidence_agent import EvidenceAgent
from agentic_rag.agents.planner import QueryDecomposer, RetrievalPlan, RetrievalPlanner
from agentic_rag.agents.query_analyzer import QueryAnalysis, QueryAnalyzer, QueryExpander
from agentic_rag.agents.retrieval_agent import RetrievalAgent
from agentic_rag.core.config import Settings
from agentic_rag.core.models import TerminationReason
from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.generation.llm import LLMProvider
from agentic_rag.observability.tracing import bind_trace_id
from agentic_rag.retrieval.base import RetrievedCandidate
from agentic_rag.retrieval.reranking import Reranker


@dataclass(slots=True)
class IterationTrace:
    iteration: int
    queries_used: list[str]
    retrieval_strategy: str
    candidates_retrieved: int
    sufficient: bool
    reason: str
    missing_information: list[str]


@dataclass(slots=True)
class AgenticRetrievalResult:
    query: str
    trace_id: str
    analysis: QueryAnalysis
    plan: RetrievalPlan
    iterations: list[IterationTrace]
    termination_reason: TerminationReason
    evidence: list[RetrievedCandidate]


class AgenticRetrievalLoop:
    def __init__(
        self,
        *,
        session: AsyncSession,
        llm: LLMProvider,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self._settings = settings
        self._analyzer = QueryAnalyzer(llm)
        self._planner = RetrievalPlanner(
            llm, max_iterations_ceiling=settings.max_retrieval_iterations
        )
        self._expander = QueryExpander(llm)
        self._decomposer = QueryDecomposer(llm)
        self._evidence_agent = EvidenceAgent(llm)
        self._retrieval_agent = RetrievalAgent(session, embedding_provider, reranker)

    async def run(
        self, query_text: str, *, collection_id: uuid.UUID | None = None
    ) -> AgenticRetrievalResult:
        trace_id = bind_trace_id()

        analysis = await self._analyzer.analyze(query_text)
        plan = await self._planner.plan(query_text, analysis)
        if collection_id is not None and plan.filters.collection_id is None:
            # Caller-provided scoping always wins over whatever the
            # LLM/mock guessed (it has no way to know a collection_id).
            plan.filters.collection_id = collection_id

        current_queries = [query_text]
        if plan.expand_query:
            expansion = await self._expander.expand(query_text)
            current_queries = expansion.expanded_queries

        iterations: list[IterationTrace] = []
        evidence: list[RetrievedCandidate] = []
        retrieval_calls = 0
        termination_reason = TerminationReason.MAX_ITERATIONS_REACHED

        for iteration in range(1, plan.max_iterations + 1):
            if retrieval_calls >= self._settings.max_retrieval_calls:
                termination_reason = TerminationReason.MAX_RETRIEVAL_CALLS_REACHED
                break

            if plan.decompose and iteration == 1:
                decomposition = await self._decomposer.decompose(query_text)
                # A decomposition's subqueries each cost one retrieval call;
                # never let it alone blow through the remaining budget just
                # because the check above only runs once per outer iteration.
                remaining_calls = self._settings.max_retrieval_calls - retrieval_calls
                subqueries = decomposition.subqueries[: max(1, remaining_calls)]
                # Sequential, not gathered: subqueries share this loop's one
                # AsyncSession, which does not support concurrent operations
                # from multiple coroutines (see retrieval_agent.py).
                subquery_results = [
                    await self._retrieval_agent.retrieve([subquery], plan)
                    for subquery in subqueries
                ]
                retrieval_calls += len(subqueries)
                candidates = _dedupe_by_chunk_id(
                    [c for group in subquery_results for c in group]
                )
                queries_used = subqueries
            else:
                candidates = await self._retrieval_agent.retrieve(current_queries, plan)
                retrieval_calls += 1
                queries_used = current_queries

            evidence = candidates
            assessment = await self._evidence_agent.assess(query_text, candidates)
            iterations.append(
                IterationTrace(
                    iteration=iteration,
                    queries_used=queries_used,
                    retrieval_strategy=plan.strategy.value,
                    candidates_retrieved=len(candidates),
                    sufficient=assessment.sufficient,
                    reason=assessment.reason,
                    missing_information=assessment.missing_information,
                )
            )

            if assessment.sufficient:
                termination_reason = TerminationReason.SUFFICIENT_EVIDENCE
                break
            if not candidates:
                termination_reason = TerminationReason.NO_EVIDENCE_FOUND
                break
            if iteration == plan.max_iterations:
                termination_reason = TerminationReason.MAX_ITERATIONS_REACHED
                break

            current_queries = [_refine_query(query_text, assessment.missing_information)]

        return AgenticRetrievalResult(
            query=query_text,
            trace_id=trace_id,
            analysis=analysis,
            plan=plan,
            iterations=iterations,
            termination_reason=termination_reason,
            evidence=evidence,
        )


def _refine_query(original_query: str, missing_information: list[str]) -> str:
    """Deterministic refinement: fold the evidence gap back into the search
    query rather than calling an LLM for this — a bare keyword-augmented
    re-search is a defensible strategy (engineering principle #1: prefer
    deterministic logic where it's sufficient) and keeps each retry cheap."""
    if not missing_information:
        return f"{original_query} additional details"
    return f"{original_query} {' '.join(missing_information)}"


def _dedupe_by_chunk_id(candidates: list[RetrievedCandidate]) -> list[RetrievedCandidate]:
    """Keeps the first occurrence of each chunk across subquery result sets
    (spec §12 decomposition) rather than re-fusing many small rankings —
    a simplification documented in docs/architecture.md, not an oversight."""
    seen: dict[uuid.UUID, RetrievedCandidate] = {}
    for candidate in candidates:
        seen.setdefault(candidate.chunk_id, candidate)
    return list(seen.values())
