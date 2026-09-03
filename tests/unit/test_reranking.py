import uuid

import pytest

from agentic_rag.retrieval.base import RetrievedCandidate
from agentic_rag.retrieval.reranking import MockReranker, get_reranker


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
