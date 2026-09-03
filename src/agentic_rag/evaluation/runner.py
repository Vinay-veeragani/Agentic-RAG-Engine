"""Runs the baseline and agentic pipelines over the benchmark corpus (spec
§34) and computes real metrics from the actual results — never hardcoded
numbers. This is the one place that ties together retrieval/generation/
citation metrics plus system metrics (latency, iterations, estimated
tokens) into a single comparable report.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.agents.citation_agent import CitationAgent
from agentic_rag.agents.research_agent import AgenticRetrievalLoop
from agentic_rag.agents.synthesis_agent import SynthesisAgent
from agentic_rag.agents.verifier import AnswerVerifier
from agentic_rag.chunking.tokenization import count_tokens
from agentic_rag.citations.validator import CitationQualityMetrics
from agentic_rag.core.config import Settings
from agentic_rag.core.models import TerminationReason
from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.evaluation.baseline import BaselinePipeline
from agentic_rag.evaluation.citations import AggregatedCitationMetrics, aggregate_citation_metrics
from agentic_rag.evaluation.datasets import BuiltEvalCase, build_benchmark_corpus
from agentic_rag.evaluation.generation import GenerationJudge
from agentic_rag.evaluation.retrieval import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from agentic_rag.generation.llm import LLMProvider
from agentic_rag.retrieval.reranking import Reranker
from agentic_rag.storage.object_store import ObjectStore

EVAL_K = 5


@dataclass(slots=True)
class RetrievalMetrics:
    recall: float
    precision: float
    mrr: float
    ndcg: float
    hit_rate: float


def _compute_retrieval_metrics(
    retrieved_document_ids: list[uuid.UUID], relevant_document_ids: set[uuid.UUID], k: int = EVAL_K
) -> RetrievalMetrics:
    return RetrievalMetrics(
        recall=recall_at_k(retrieved_document_ids, relevant_document_ids, k),
        precision=precision_at_k(retrieved_document_ids, relevant_document_ids, k),
        mrr=mean_reciprocal_rank(retrieved_document_ids, relevant_document_ids),
        ndcg=ndcg_at_k(retrieved_document_ids, relevant_document_ids, k),
        hit_rate=hit_rate_at_k(retrieved_document_ids, relevant_document_ids, k),
    )


@dataclass(slots=True)
class PipelineCaseResult:
    answer: str | None
    latency_seconds: float
    estimated_tokens: int
    retrieval: RetrievalMetrics
    answer_relevance: float | None = None
    status: str | None = None
    iterations: int | None = None
    citation_metrics: CitationQualityMetrics | None = None


@dataclass(slots=True)
class CaseComparison:
    query: str
    category: str
    relevant_document_count: int
    baseline: PipelineCaseResult
    agentic: PipelineCaseResult


@dataclass(slots=True)
class PipelineSummary:
    mean_recall: float
    mean_precision: float
    mean_mrr: float
    mean_ndcg: float
    mean_hit_rate: float
    mean_latency_seconds: float
    mean_estimated_tokens: float
    mean_answer_relevance: float | None
    citation_metrics: AggregatedCitationMetrics | None


@dataclass(slots=True)
class BenchmarkReport:
    cases: list[CaseComparison]
    baseline_summary: PipelineSummary
    agentic_summary: PipelineSummary


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize(
    results: list[PipelineCaseResult], cases: list[BuiltEvalCase], *, include_citations: bool
) -> PipelineSummary:
    # Ambiguous/unanswerable cases have no relevant documents by design —
    # they test correct abstention, not ranking quality, so they're
    # excluded from the retrieval-metric averages (each case's own numbers
    # are still visible in `BenchmarkReport.cases`, just not folded into
    # this mean).
    scored = [
        (r, c) for r, c in zip(results, cases, strict=True) if c.relevant_document_ids
    ]
    relevances = [r.answer_relevance for r in results if r.answer_relevance is not None]
    return PipelineSummary(
        mean_recall=_mean([r.retrieval.recall for r, _ in scored]),
        mean_precision=_mean([r.retrieval.precision for r, _ in scored]),
        mean_mrr=_mean([r.retrieval.mrr for r, _ in scored]),
        mean_ndcg=_mean([r.retrieval.ndcg for r, _ in scored]),
        mean_hit_rate=_mean([r.retrieval.hit_rate for r, _ in scored]),
        mean_latency_seconds=_mean([r.latency_seconds for r in results]),
        mean_estimated_tokens=_mean([float(r.estimated_tokens) for r in results]),
        mean_answer_relevance=_mean(relevances) if relevances else None,
        citation_metrics=(
            aggregate_citation_metrics([r.citation_metrics for r in results])
            if include_citations
            else None
        ),
    )


async def run_benchmark(
    *,
    session: AsyncSession,
    object_store: ObjectStore,
    embedding_provider: EmbeddingProvider,
    llm: LLMProvider,
    reranker: Reranker,
    settings: Settings,
) -> BenchmarkReport:
    collection_id, cases = await build_benchmark_corpus(
        session=session, object_store=object_store, embedding_provider=embedding_provider
    )
    await session.commit()

    baseline_pipeline = BaselinePipeline(session, embedding_provider, llm)
    judge = GenerationJudge(llm)

    comparisons: list[CaseComparison] = []
    for case in cases:
        baseline_result = await baseline_pipeline.run(case.query, collection_id=collection_id)
        baseline_relevance = await judge.judge_answer_relevance(case.query, baseline_result.answer)
        baseline_case_result = PipelineCaseResult(
            answer=baseline_result.answer,
            latency_seconds=baseline_result.latency_seconds,
            estimated_tokens=count_tokens(baseline_result.answer or ""),
            retrieval=_compute_retrieval_metrics(
                baseline_result.document_ids, case.relevant_document_ids
            ),
            answer_relevance=baseline_relevance,
        )

        start = time.perf_counter()
        loop = AgenticRetrievalLoop(
            session=session,
            llm=llm,
            embedding_provider=embedding_provider,
            reranker=reranker,
            settings=settings,
        )
        loop_result = await loop.run(case.query, collection_id=collection_id)
        agentic_latency = time.perf_counter() - start
        agentic_document_ids = [c.document_id for c in loop_result.evidence]

        status: str | None
        answer: str | None = None
        agentic_relevance: float | None = None
        citation_metrics = None
        if loop_result.termination_reason in (
            TerminationReason.CONFLICTING_EVIDENCE,
            TerminationReason.NO_EVIDENCE_FOUND,
        ):
            status = loop_result.termination_reason.value
        else:
            verifier = AnswerVerifier(SynthesisAgent(llm), CitationAgent(llm))
            verification = await verifier.generate(case.query, loop_result.evidence)
            status = verification.status.value
            answer = verification.answer
            citation_metrics = verification.citation_metrics
            agentic_relevance = await judge.judge_answer_relevance(case.query, answer)

        agentic_case_result = PipelineCaseResult(
            answer=answer,
            latency_seconds=agentic_latency,
            estimated_tokens=count_tokens(answer or ""),
            retrieval=_compute_retrieval_metrics(agentic_document_ids, case.relevant_document_ids),
            answer_relevance=agentic_relevance,
            status=status,
            iterations=len(loop_result.iterations),
            citation_metrics=citation_metrics,
        )

        comparisons.append(
            CaseComparison(
                query=case.query,
                category=case.category.value,
                relevant_document_count=len(case.relevant_document_ids),
                baseline=baseline_case_result,
                agentic=agentic_case_result,
            )
        )

    return BenchmarkReport(
        cases=comparisons,
        baseline_summary=_summarize(
            [c.baseline for c in comparisons], cases, include_citations=False
        ),
        agentic_summary=_summarize(
            [c.agentic for c in comparisons], cases, include_citations=True
        ),
    )
