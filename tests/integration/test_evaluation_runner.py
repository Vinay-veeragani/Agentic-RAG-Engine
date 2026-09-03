import pytest

from agentic_rag.core.config import Settings
from agentic_rag.embeddings.providers import MockEmbeddingProvider
from agentic_rag.evaluation.baseline import BaselinePipeline
from agentic_rag.evaluation.datasets import BenchmarkCategory, build_benchmark_corpus
from agentic_rag.evaluation.runner import run_benchmark
from agentic_rag.generation.mock import MockLLMProvider
from agentic_rag.retrieval.reranking import MockReranker


@pytest.fixture
def object_store(tmp_path):
    from agentic_rag.storage.object_store import LocalFileObjectStore

    return LocalFileObjectStore(tmp_path)


@pytest.mark.asyncio
async def test_build_benchmark_corpus_resolves_real_document_ids(db_session, object_store) -> None:
    collection_id, cases = await build_benchmark_corpus(
        session=db_session, object_store=object_store, embedding_provider=MockEmbeddingProvider()
    )
    assert collection_id is not None
    assert len(cases) >= 8

    categories = {c.category for c in cases}
    assert BenchmarkCategory.SIMPLE_FACTUAL in categories
    assert BenchmarkCategory.UNANSWERABLE in categories
    assert BenchmarkCategory.CONTRADICTORY_EVIDENCE in categories

    simple_case = next(c for c in cases if c.category == BenchmarkCategory.SIMPLE_FACTUAL)
    assert len(simple_case.relevant_document_ids) >= 1

    unanswerable_case = next(c for c in cases if c.category == BenchmarkCategory.UNANSWERABLE)
    assert unanswerable_case.relevant_document_ids == set()


@pytest.mark.asyncio
async def test_baseline_pipeline_returns_none_for_empty_collection(db_session) -> None:
    import uuid

    pipeline = BaselinePipeline(db_session, MockEmbeddingProvider(), MockLLMProvider())
    result = await pipeline.run("anything", collection_id=uuid.uuid4())
    assert result.answer is None
    assert result.document_ids == []


@pytest.mark.asyncio
async def test_baseline_pipeline_answers_from_real_corpus(db_session, object_store) -> None:
    collection_id, cases = await build_benchmark_corpus(
        session=db_session, object_store=object_store, embedding_provider=MockEmbeddingProvider()
    )
    await db_session.commit()

    pipeline = BaselinePipeline(db_session, MockEmbeddingProvider(), MockLLMProvider())
    simple_case = next(c for c in cases if c.category == BenchmarkCategory.SIMPLE_FACTUAL)
    result = await pipeline.run(simple_case.query, collection_id=collection_id)

    assert result.answer is not None
    assert len(result.document_ids) >= 1


@pytest.mark.asyncio
async def test_run_benchmark_produces_a_report_for_every_case(db_session, object_store) -> None:
    settings = Settings()
    report = await run_benchmark(
        session=db_session,
        object_store=object_store,
        embedding_provider=MockEmbeddingProvider(),
        llm=MockLLMProvider(),
        reranker=MockReranker(),
        settings=settings,
    )

    assert len(report.cases) >= 8
    for case in report.cases:
        assert case.baseline.retrieval is not None
        assert case.agentic.retrieval is not None
        assert case.agentic.iterations is not None and case.agentic.iterations >= 1

    # System metrics are real measurements, not fabricated numbers.
    assert report.baseline_summary.mean_latency_seconds >= 0.0
    assert report.agentic_summary.mean_latency_seconds >= 0.0
    assert report.agentic_summary.citation_metrics is not None
    assert report.baseline_summary.citation_metrics is None  # baseline has no citations at all


@pytest.mark.asyncio
async def test_run_benchmark_unanswerable_case_never_grounds_an_answer(
    db_session, object_store
) -> None:
    settings = Settings()
    report = await run_benchmark(
        session=db_session,
        object_store=object_store,
        embedding_provider=MockEmbeddingProvider(),
        llm=MockLLMProvider(),
        reranker=MockReranker(),
        settings=settings,
    )
    unanswerable = next(
        c for c in report.cases if c.category == BenchmarkCategory.UNANSWERABLE.value
    )
    assert unanswerable.agentic.status != "grounded"
