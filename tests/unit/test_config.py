from agentic_rag.core.config import Settings


def test_default_budgets_are_finite_and_positive() -> None:
    """Guards spec §1's "never create an infinite agent loop" — these must
    never default to 0/unbounded."""
    settings = Settings()
    assert 0 < settings.max_retrieval_iterations <= 10
    assert 0 < settings.max_retrieval_calls <= 50
    assert settings.max_tokens_per_query > 0
    assert settings.max_query_latency_seconds > 0


def test_default_providers_require_no_api_key() -> None:
    settings = Settings()
    assert settings.llm_provider == "mock"
    assert settings.embedding_provider == "mock"
    assert settings.reranker_provider == "mock"
