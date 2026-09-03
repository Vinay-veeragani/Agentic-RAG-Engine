import pytest

from agentic_rag.evaluation.generation import GenerationJudge
from agentic_rag.generation.mock import MockLLMProvider


@pytest.mark.asyncio
async def test_judge_returns_zero_for_empty_answer() -> None:
    judge = GenerationJudge(MockLLMProvider())
    assert await judge.judge_answer_relevance("some query", None) == 0.0
    assert await judge.judge_answer_relevance("some query", "") == 0.0


@pytest.mark.asyncio
async def test_judge_scores_relevant_answer_higher_than_unrelated() -> None:
    judge = MockLLMProvider()
    j = GenerationJudge(judge)
    relevant = await j.judge_answer_relevance(
        "why did revenue decline", "Revenue declined due to weaker demand."
    )
    unrelated = await j.judge_answer_relevance(
        "why did revenue decline", "Heavy rainfall is forecast this week."
    )
    assert relevant > unrelated
