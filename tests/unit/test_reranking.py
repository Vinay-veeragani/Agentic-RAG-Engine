import uuid

import pytest

from agentic_rag.retrieval.base import RetrievedCandidate
from agentic_rag.retrieval.reranking import (
    LocalCrossEncoderReranker,
    MockReranker,
    get_reranker,
    rerank_with_fallback,
)


def _candidate(content: str) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        page=None,
        section=None,
        heading=None,
        document_filename="doc.txt",
        document_title=None,
    )


@pytest.mark.asyncio
async def test_mock_reranker_ranks_higher_overlap_first() -> None:
    candidates = [
        _candidate("completely unrelated weather forecast content"),
        _candidate("quarterly revenue and profit margin details"),
    ]

    ranked = await MockReranker().rerank("revenue profit margin", candidates, top_k=10)

    assert ranked[0].content.startswith("quarterly revenue")
    assert all(c.rerank_score is not None for c in ranked)


@pytest.mark.asyncio
async def test_mock_reranker_respects_top_k() -> None:
    candidates = [_candidate(f"item {i} revenue") for i in range(5)]
    ranked = await MockReranker().rerank("revenue", candidates, top_k=2)
    assert len(ranked) == 2


@pytest.mark.asyncio
async def test_mock_reranker_handles_empty_candidates() -> None:
    ranked = await MockReranker().rerank("revenue", [], top_k=5)
    assert ranked == []


@pytest.mark.asyncio
async def test_mock_reranker_handles_empty_query() -> None:
    candidates = [_candidate("some content")]
    ranked = await MockReranker().rerank("", candidates, top_k=5)
    assert ranked[0].rerank_score == 0.0


def test_get_reranker_returns_mock() -> None:
    assert isinstance(get_reranker("mock"), MockReranker)


def test_get_reranker_raises_for_unknown_provider() -> None:
    from agentic_rag.core.errors import ModelProviderError

    with pytest.raises(ModelProviderError):
        get_reranker("openai")


@pytest.mark.slow
@pytest.mark.asyncio
async def test_local_cross_encoder_reranker_ranks_relevant_content_first() -> None:
    """Exercises the real cross-encoder model (`ms-marco-MiniLM-L-6-v2`),
    not the mock — every other reranking test only proves the plumbing
    works, not that reranking actually improves ranking quality."""
    candidates = [
        _candidate("Heavy rainfall is forecast across the region this week."),
        _candidate("Quarterly revenue increased significantly due to strong demand."),
    ]

    ranked = await LocalCrossEncoderReranker().rerank(
        "why did revenue grow this quarter", candidates, top_k=2
    )

    assert ranked[0].content.startswith("Quarterly revenue")
    assert ranked[0].rerank_score is not None
    assert ranked[1].rerank_score is not None
    assert ranked[0].rerank_score > ranked[1].rerank_score


@pytest.mark.asyncio
async def test_rerank_with_fallback_returns_unreranked_candidates_on_error() -> None:
    class BrokenReranker:
        async def rerank(self, query, candidates, *, top_k):
            raise RuntimeError("model failed to load")

    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
    result = await rerank_with_fallback(BrokenReranker(), "query", candidates, top_k=2)

    assert result == candidates[:2]
    assert all(c.rerank_score is None for c in result)


@pytest.mark.asyncio
async def test_rerank_with_fallback_passes_through_on_success() -> None:
    candidates = [_candidate("unrelated"), _candidate("revenue profit margin")]
    result = await rerank_with_fallback(
        MockReranker(), "revenue profit margin", candidates, top_k=1
    )

    assert len(result) == 1
    assert result[0].content == "revenue profit margin"
