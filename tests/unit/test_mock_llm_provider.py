import pytest

from agentic_rag.core.models import QueryType, RetrievalStrategy
from agentic_rag.generation.mock import MockLLMProvider


@pytest.mark.asyncio
async def test_mock_llm_is_deterministic() -> None:
    llm = MockLLMProvider()
    from agentic_rag.agents.query_analyzer import QueryAnalysis

    first = await llm.complete_structured(
        system_prompt="sys", user_prompt="Query: What is X?", schema=QueryAnalysis
    )
    second = await llm.complete_structured(
        system_prompt="sys", user_prompt="Query: What is X?", schema=QueryAnalysis
    )
    assert first == second


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_type"),
    [
        ("Compare Apple and Microsoft revenue", QueryType.COMPARISON),
        ("What changed between 2023 and 2025?", QueryType.TEMPORAL),
        ("Summarize the annual report", QueryType.SUMMARIZATION),
        ("Why did revenue decline?", QueryType.ANALYTICAL),
        ("hi", QueryType.AMBIGUOUS),
        ("What was Apple's revenue in 2025?", QueryType.SIMPLE_FACTUAL),
    ],
)
async def test_mock_llm_classifies_query_type_by_heuristic(query, expected_type) -> None:
    from agentic_rag.agents.query_analyzer import QueryAnalysis

    llm = MockLLMProvider()
    result = await llm.complete_structured(
        system_prompt="sys", user_prompt=f"Query: {query}", schema=QueryAnalysis
    )
    assert result.query_type == expected_type


@pytest.mark.asyncio
async def test_mock_llm_fills_retrieval_plan_with_valid_defaults() -> None:
    from agentic_rag.agents.planner import RetrievalPlan

    llm = MockLLMProvider()
    plan = await llm.complete_structured(
        system_prompt="sys", user_prompt="Query: anything", schema=RetrievalPlan
    )
    assert plan.strategy == RetrievalStrategy.HYBRID
    assert plan.max_iterations >= 1
    assert plan.top_k >= 1
    assert plan.filters.collection_id is None  # nested MetadataFilter stays empty, not deep-filled


@pytest.mark.asyncio
async def test_mock_llm_produces_multiple_subqueries_for_conjunctive_query() -> None:
    from agentic_rag.agents.planner import QueryDecomposition

    llm = MockLLMProvider()
    result = await llm.complete_structured(
        system_prompt="sys",
        user_prompt="Query: revenue in 2023 and revenue in 2024 and revenue in 2025",
        schema=QueryDecomposition,
    )
    assert len(result.subqueries) >= 2
