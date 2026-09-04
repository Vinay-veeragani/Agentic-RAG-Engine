"""The agentic retrieval loop (spec §16): plan -> retrieve -> rerank ->
evaluate evidence -> if insufficient, refine and retry -> stop when
sufficient, contradictory, or budget exhausted.

Hard-bounded by construction: the loop body runs inside
`for iteration in range(1, plan.max_iterations + 1)`, and `plan.max_iterations`
is itself clamped to `settings.max_retrieval_iterations` by
`RetrievalPlanner` before this ever runs — a `for` loop over a fixed range
cannot run forever regardless of what any LLM (or the mock) proposes. A
second, independent ceiling (`settings.max_retrieval_calls`) is checked every
iteration in case a single query expands into many retrieval calls (spec
§12's concurrent subquery execution). Every run ends in exactly one
`TerminationReason` — never a silent stop.

An unresolved contradiction between sources (spec §18) ends the run
immediately rather than looping — refining the search query cannot fix two
sources genuinely disagreeing, so retrying would just waste budget pretending
the problem is retrieval quality.

No hidden chain-of-thought is exposed: `AgenticRetrievalResult` carries
structured `IterationTrace`s (queries used, strategy, candidate count,
sufficiency verdict, reason, missing information, contradictions, temporal
spread, per-phase latency) and nothing else. An optional `EventEmitter`
(spec §30) publishes the same structured decisions as they happen, for a
live SSE stream or later trace replay — never the reasoning behind them.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.agents.evidence_agent import Contradiction, EvidenceAgent
from agentic_rag.agents.multi_hop import MultiHopResolver, resolve_second_hop_query
from agentic_rag.agents.planner import QueryDecomposer, RetrievalPlan, RetrievalPlanner
from agentic_rag.agents.query_analyzer import QueryAnalysis, QueryAnalyzer, QueryExpander
from agentic_rag.agents.retrieval_agent import (
    DEFAULT_EVIDENCE_TOP_K,
    RetrievalAgent,
    RetrievalOutcome,
)
from agentic_rag.core.config import Settings
from agentic_rag.core.models import QueryType, TerminationReason
from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.generation.llm import LLMProvider
from agentic_rag.observability.events import EventEmitter, EventType
from agentic_rag.observability.metrics import (
    QUERY_FAILURES,
    QUERY_LATENCY_SECONDS,
    RETRIEVAL_ITERATIONS,
)
from agentic_rag.observability.tracing import bind_trace_id, get_trace_id
from agentic_rag.retrieval.base import RetrievedCandidate
from agentic_rag.retrieval.reranking import Reranker
from agentic_rag.storage.models import Collection


@dataclass(slots=True)
class IterationTrace:
    iteration: int
    queries_used: list[str]
    retrieval_strategy: str
    candidates_retrieved: int
    sufficient: bool
    reason: str
    missing_information: list[str]
    contradictions: list[Contradiction] = field(default_factory=list)
    years_referenced: list[int] = field(default_factory=list)
    spans_multiple_periods: bool = False
    retrieval_latency_seconds: float = 0.0
    rerank_latency_seconds: float = 0.0


@dataclass(slots=True)
class AgenticRetrievalResult:
    query: str
    trace_id: str
    analysis: QueryAnalysis
    plan: RetrievalPlan
    iterations: list[IterationTrace]
    termination_reason: TerminationReason
    evidence: list[RetrievedCandidate]
    total_latency_seconds: float = 0.0
    retrieval_latency_seconds: float = 0.0
    rerank_latency_seconds: float = 0.0
    retrieval_calls: int = 0


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
        self._session = session
        self._llm = llm
        self._settings = settings
        self._analyzer = QueryAnalyzer(llm)
        self._planner = RetrievalPlanner(
            llm, max_iterations_ceiling=settings.max_retrieval_iterations
        )
        self._expander = QueryExpander(llm)
        self._decomposer = QueryDecomposer(llm)
        self._retrieval_agent = RetrievalAgent(session, embedding_provider, reranker)
        self._multi_hop_resolver = MultiHopResolver(llm)

    async def run(
        self,
        query_text: str,
        *,
        collection_id: uuid.UUID | None = None,
        emitter: EventEmitter | None = None,
    ) -> AgenticRetrievalResult:
        get_trace_id() or bind_trace_id()
        run_start = time.perf_counter()
        if emitter:
            emitter.emit(EventType.QUERY_STARTED, query=query_text)

        try:
            result = await self._run(query_text, collection_id=collection_id, emitter=emitter)
        except Exception as exc:
            QUERY_FAILURES.labels(reason=type(exc).__name__).inc()
            if emitter:
                emitter.emit(EventType.QUERY_FAILED, error=type(exc).__name__, message=str(exc))
            raise

        result.total_latency_seconds = time.perf_counter() - run_start
        QUERY_LATENCY_SECONDS.labels(status=result.termination_reason.value).observe(
            result.total_latency_seconds
        )
        RETRIEVAL_ITERATIONS.observe(len(result.iterations))
        # Deliberately does NOT emit QUERY_COMPLETED here: this loop is also
        # used as just the first stage of the full POST /query pipeline
        # (synthesis + citation validation follow it there), and
        # "query.completed" is meant to mark the end of the *whole* pipeline
        # a caller invoked — the caller emits it once everything it's doing
        # is actually done (see api/routes/query.py).
        return result

    async def _run(
        self,
        query_text: str,
        *,
        collection_id: uuid.UUID | None,
        emitter: EventEmitter | None,
    ) -> AgenticRetrievalResult:
        trace_id = get_trace_id() or bind_trace_id()
        evidence_agent = await self._build_evidence_agent(collection_id)

        analysis = await self._analyzer.analyze(query_text)
        if emitter:
            emitter.emit(EventType.QUERY_ANALYZED, query_type=analysis.query_type.value)

        plan = await self._planner.plan(query_text, analysis)
        if collection_id is not None and plan.filters.collection_id is None:
            # Caller-provided scoping always wins over whatever the
            # LLM/mock guessed (it has no way to know a collection_id).
            plan.filters.collection_id = collection_id
        if emitter:
            emitter.emit(
                EventType.PLAN_CREATED,
                strategy=plan.strategy.value,
                expand_query=plan.expand_query,
                decompose=plan.decompose,
                max_iterations=plan.max_iterations,
            )

        current_queries = [query_text]
        if plan.expand_query:
            expansion = await self._expander.expand(query_text)
            current_queries = expansion.expanded_queries

        iterations: list[IterationTrace] = []
        evidence: list[RetrievedCandidate] = []
        retrieval_calls = 0
        total_retrieval_latency = 0.0
        total_rerank_latency = 0.0
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

                if (
                    analysis.query_type == QueryType.MULTI_HOP
                    and len(subqueries) >= 2
                    and remaining_calls >= 2
                ):
                    (
                        candidates,
                        iteration_retrieval_latency,
                        iteration_rerank_latency,
                        queries_used,
                    ) = await self._retrieve_chained(subqueries[0], subqueries[1], plan, emitter)
                    retrieval_calls += 2
                else:
                    # Sequential, not gathered: subqueries share this loop's
                    # one AsyncSession, which does not support concurrent
                    # operations from multiple coroutines (see
                    # retrieval_agent.py).
                    subquery_outcomes = [
                        await self._retrieval_agent.retrieve([subquery], plan, emitter=emitter)
                        for subquery in subqueries
                    ]
                    retrieval_calls += len(subqueries)
                    candidates = _dedupe_by_chunk_id(
                        [c for outcome in subquery_outcomes for c in outcome.candidates]
                    )
                    iteration_retrieval_latency = sum(
                        o.retrieval_latency_seconds for o in subquery_outcomes
                    )
                    iteration_rerank_latency = sum(
                        o.rerank_latency_seconds for o in subquery_outcomes
                    )
                    queries_used = subqueries
            else:
                outcome = await self._retrieval_agent.retrieve(
                    current_queries, plan, emitter=emitter
                )
                retrieval_calls += 1
                candidates = outcome.candidates
                iteration_retrieval_latency = outcome.retrieval_latency_seconds
                iteration_rerank_latency = outcome.rerank_latency_seconds
                queries_used = current_queries

            total_retrieval_latency += iteration_retrieval_latency
            total_rerank_latency += iteration_rerank_latency

            evidence = candidates
            evaluation = await evidence_agent.evaluate(query_text, candidates)
            assessment = evaluation.assessment
            if emitter:
                emitter.emit(
                    EventType.EVIDENCE_EVALUATED,
                    sufficient=assessment.sufficient,
                    reason=assessment.reason,
                    contradictions=len(evaluation.contradictions),
                )
            iterations.append(
                IterationTrace(
                    iteration=iteration,
                    queries_used=queries_used,
                    retrieval_strategy=plan.strategy.value,
                    candidates_retrieved=len(candidates),
                    sufficient=assessment.sufficient,
                    reason=assessment.reason,
                    missing_information=assessment.missing_information,
                    contradictions=evaluation.contradictions,
                    years_referenced=evaluation.years_referenced,
                    spans_multiple_periods=evaluation.spans_multiple_periods,
                    retrieval_latency_seconds=iteration_retrieval_latency,
                    rerank_latency_seconds=iteration_rerank_latency,
                )
            )

            if any(c.resolution is None for c in evaluation.contradictions):
                # An unresolved disagreement between sources — refining the
                # query won't fix it, so stop and surface it rather than
                # burning the remaining iteration budget (spec §18).
                termination_reason = TerminationReason.CONFLICTING_EVIDENCE
                break
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
            if emitter:
                emitter.emit(EventType.RETRIEVAL_REFINED, next_queries=current_queries)

        return AgenticRetrievalResult(
            query=query_text,
            trace_id=trace_id,
            analysis=analysis,
            plan=plan,
            iterations=iterations,
            termination_reason=termination_reason,
            evidence=evidence,
            retrieval_latency_seconds=total_retrieval_latency,
            rerank_latency_seconds=total_rerank_latency,
            retrieval_calls=retrieval_calls,
        )

    async def _retrieve_chained(
        self,
        first_hop_query: str,
        second_hop_query: str,
        plan: RetrievalPlan,
        emitter: EventEmitter | None,
    ) -> tuple[list[RetrievedCandidate], float, float, list[str]]:
        """Real dependency-chained multi-hop retrieval (see agents/multi_hop
        .py): hop two's query is only resolved *after* hop one's evidence
        comes back, using an entity extracted from it — not run
        independently and merged by ranking alone, which is what plain
        decomposition (the branch above) does and which cannot find
        evidence keyed by an entity the original query never named.
        Sequential for the same reason plain decomposition is: both hops
        share this loop's one AsyncSession.
        """
        hop1: RetrievalOutcome = await self._retrieval_agent.retrieve(
            [first_hop_query], plan, emitter=emitter
        )
        entity = await self._multi_hop_resolver.extract_bridge_entity(
            first_hop_query, hop1.candidates
        )
        resolved_second_hop_query = resolve_second_hop_query(second_hop_query, entity)
        hop2: RetrievalOutcome = await self._retrieval_agent.retrieve(
            [resolved_second_hop_query], plan, emitter=emitter
        )

        # Each hop's own RetrievalAgent.retrieve() already reranked and
        # truncated *within* that hop — but a bare concatenation would
        # still always rank every hop-one chunk ahead of every hop-two
        # chunk regardless of actual relevance, silently starving
        # recall@k for the second hop's evidence. Re-rank the merged set
        # by whatever score each candidate actually has before truncating.
        candidates = _rank_and_truncate(
            _dedupe_by_chunk_id(hop1.candidates + hop2.candidates), DEFAULT_EVIDENCE_TOP_K
        )
        retrieval_latency = hop1.retrieval_latency_seconds + hop2.retrieval_latency_seconds
        rerank_latency = hop1.rerank_latency_seconds + hop2.rerank_latency_seconds
        return candidates, retrieval_latency, rerank_latency, [
            first_hop_query,
            resolved_second_hop_query,
        ]

    async def _build_evidence_agent(self, collection_id: uuid.UUID | None) -> EvidenceAgent:
        """Reads the collection's configured source authority order (spec
        §20), if any, so the evidence agent never hardcodes one hierarchy as
        universally correct."""
        if collection_id is None:
            return EvidenceAgent(self._llm)
        collection = await self._session.get(Collection, collection_id)
        raw_order = (collection.source_authority_config or {}).get("order") if collection else None
        order = [str(item) for item in raw_order] if isinstance(raw_order, list) else None
        return EvidenceAgent(self._llm, source_authority_order=order)


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


def _rank_and_truncate(
    candidates: list[RetrievedCandidate], top_k: int
) -> list[RetrievedCandidate]:
    """Re-sorts a merged candidate set (from more than one retrieval call)
    by whatever score each candidate actually carries — rerank score where
    reranking ran, fusion score otherwise — and truncates to `top_k`,
    reassigning `rank` to reflect the merged order. Used where a bare
    concatenation would silently bias toward whichever call happened
    first (see `_retrieve_chained` above)."""

    def _score(candidate: RetrievedCandidate) -> float:
        if candidate.rerank_score is not None:
            return candidate.rerank_score
        return candidate.fusion_score or 0.0

    ranked = sorted(candidates, key=_score, reverse=True)[:top_k]
    for rank, candidate in enumerate(ranked, start=1):
        candidate.rank = rank
    return ranked
