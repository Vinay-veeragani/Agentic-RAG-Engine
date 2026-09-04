from pydantic import BaseModel


class SettingsResponse(BaseModel):
    """Non-secret configuration only — never API keys or other internals
    that could leak to consumers. Powers the frontend's Settings page."""

    app_env: str
    llm_provider: str
    embedding_provider: str
    reranker_provider: str
    max_retrieval_iterations: int
    max_retrieval_calls: int
    max_tokens_per_query: int
    max_query_latency_seconds: float
    max_upload_size_bytes: int
    auth_enabled: bool
    rate_limit_enabled: bool
    rate_limit_requests_per_window: int
    rate_limit_window_seconds: int
    workers: int
    cache_backend: str
