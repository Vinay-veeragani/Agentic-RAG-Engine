"""Runs the baseline-vs-agentic benchmark (spec §33/§34) against the real
local Postgres and prints a comparison table plus a JSON report.

Usage (from repo root, with the venv active):
    python benchmarks/run_evaluation.py [--embedding local|mock] [--llm mock]

Defaults to the local sentence-transformers embedding provider (real,
semantically meaningful — not mock) so retrieval metrics are actually
measuring retrieval quality, not noise; and the mock LLM provider, since
this environment has no Ollama install or API key configured. Every number
printed is computed from an actual run against the corpus built fresh by
`evaluation/datasets.py` — nothing here is a hardcoded/fabricated figure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from agentic_rag.core.config import Settings
from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.evaluation.runner import BenchmarkReport, run_benchmark
from agentic_rag.generation.llm import LLMProvider
from agentic_rag.retrieval.reranking import MockReranker
from agentic_rag.storage.object_store import LocalFileObjectStore
from agentic_rag.storage.postgres import get_session_factory


def _build_embedding_provider(name: str) -> EmbeddingProvider:
    if name == "local":
        from agentic_rag.embeddings.local import LocalEmbeddingProvider

        return LocalEmbeddingProvider()
    from agentic_rag.embeddings.providers import MockEmbeddingProvider

    return MockEmbeddingProvider()


def _build_llm_provider(name: str) -> LLMProvider:
    if name == "mock":
        from agentic_rag.generation.mock import MockLLMProvider

        return MockLLMProvider()
    from agentic_rag.core.config import get_settings
    from agentic_rag.generation.providers import get_llm_provider

    return get_llm_provider(name, get_settings())  # type: ignore[arg-type]


def _print_table(report: BenchmarkReport) -> None:
    header = f"{'category':24s} | {'recall b/a':>11s} | {'ndcg b/a':>11s} | agentic status"
    print(header)
    print("-" * len(header))
    for case in report.cases:
        b, a = case.baseline.retrieval, case.agentic.retrieval
        print(
            f"{case.category:24s} | {b.recall:.2f}/{a.recall:.2f}      "
            f"| {b.ndcg:.2f}/{a.ndcg:.2f}      | {case.agentic.status}"
        )
    print()
    print("Summary (means; retrieval metrics exclude ambiguous/unanswerable cases):")
    summaries = (("baseline", report.baseline_summary), ("agentic", report.agentic_summary))
    for label, summary in summaries:
        print(f"  {label}:")
        print(
            f"    recall={summary.mean_recall:.3f} precision={summary.mean_precision:.3f} "
            f"mrr={summary.mean_mrr:.3f} ndcg={summary.mean_ndcg:.3f} "
            f"hit_rate={summary.mean_hit_rate:.3f}"
        )
        print(
            f"    latency={summary.mean_latency_seconds:.4f}s "
            f"est_tokens={summary.mean_estimated_tokens:.1f} "
            f"answer_relevance={summary.mean_answer_relevance}"
        )
        if summary.citation_metrics is not None:
            print(
                f"    citation_precision={summary.citation_metrics.mean_precision:.3f} "
                f"citation_completeness={summary.citation_metrics.mean_completeness:.3f}"
            )


async def _main(embedding: str, llm_name: str, output: Path) -> None:
    settings = Settings()
    session_factory = get_session_factory(settings.database_url)
    object_store = LocalFileObjectStore(settings.object_store_root)
    embedding_provider = _build_embedding_provider(embedding)
    llm = _build_llm_provider(llm_name)

    async with session_factory() as session:
        report = await run_benchmark(
            session=session,
            object_store=object_store,
            embedding_provider=embedding_provider,
            llm=llm,
            reranker=MockReranker(),
            settings=settings,
        )

    _print_table(report)

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "embedding_provider": embedding,
        "llm_provider": llm_name,
        "cases": [asdict(c) for c in report.cases],
        "baseline_summary": asdict(report.baseline_summary),
        "agentic_summary": asdict(report.agentic_summary),
    }
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nFull report written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", choices=["local", "mock"], default="local")
    parser.add_argument("--llm", default="mock")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/latest.json")
    )
    args = parser.parse_args()
    asyncio.run(_main(args.embedding, args.llm, args.output))
