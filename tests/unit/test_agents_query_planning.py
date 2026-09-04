import pytest

from agentic_rag.agents.planner import QueryDecomposer, RetrievalPlanner
from agentic_rag.agents.query_analyzer import QueryAnalyzer, QueryExpander
from agentic_rag.generation.mock import MockLLMProvider


@pytest.mark.asyncio
async def test_query_analyzer_returns_structured_analysis() -> None:
    analyzer = QueryAnalyzer(MockLLMProvider())
    analysis = await analyzer.analyze("Compare Apple and Microsoft revenue")
    assert analysis.query_type is not None
    assert isinstance(analysis.is_ambiguous, bool)
    assert isinstance(analysis.reasoning, str) and analysis.reasoning


@pytest.mark.asyncio
async def test_planner_clamps_max_iterations_to_ceiling() -> None:
    """The mock defaults max_iterations to 3; a ceiling of 1 must still win —
    the plan is never trusted to self-limit."""
    planner = RetrievalPlanner(MockLLMProvider(), max_iterations_ceiling=1)
    analyzer = QueryAnalyzer(MockLLMProvider())
    analysis = await analyzer.analyze("Compare Apple and Microsoft revenue")

    plan = await planner.plan("Compare Apple and Microsoft revenue", analysis)
    assert plan.max_iterations == 1


@pytest.mark.asyncio
async def test_planner_clamps_top_k_to_ceiling() -> None:
    planner = RetrievalPlanner(MockLLMProvider(), max_iterations_ceiling=3, top_k_ceiling=5)
    analyzer = QueryAnalyzer(MockLLMProvider())
    analysis = await analyzer.analyze("What was Apple's revenue?")

    plan = await planner.plan("What was Apple's revenue?", analysis)
    assert plan.top_k <= 5


@pytest.mark.asyncio
async def test_planner_enables_decompose_for_comparison_query() -> None:
    planner = RetrievalPlanner(MockLLMProvider(), max_iterations_ceiling=3)
    analyzer = QueryAnalyzer(MockLLMProvider())
    query = "Compare revenue and R&D spending of Microsoft and Google"
    analysis = await analyzer.analyze(query)

    plan = await planner.plan(query, analysis)
    assert plan.decompose is True


@pytest.mark.asyncio
async def test_query_expander_returns_at_least_one_variant_and_respects_bound() -> None:
    expander = QueryExpander(MockLLMProvider())
    result = await expander.expand("revenue")
    assert 1 <= len(result.expanded_queries) <= 5


@pytest.mark.asyncio
async def test_query_decomposer_splits_conjunctive_query() -> None:
    decomposer = QueryDecomposer(MockLLMProvider())
    result = await decomposer.decompose("Microsoft revenue 2023 and Google revenue 2023")
    assert len(result.subqueries) >= 2


@pytest.mark.asyncio
async def test_query_decomposer_returns_original_query_when_not_splittable() -> None:
    decomposer = QueryDecomposer(MockLLMProvider())
    result = await decomposer.decompose("revenue")
    assert result.subqueries == ["revenue"]
