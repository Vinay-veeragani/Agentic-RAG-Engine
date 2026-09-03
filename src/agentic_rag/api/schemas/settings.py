from pydantic import BaseModel


class SettingsResponse(BaseModel):
    """Non-secret configuration only — never API keys (spec §36: don't leak
    internals to consumers). Powers the frontend's Settings page."""

    app_env: str
    llm_provider: str
    embedding_provider: str
    reranker_provider: str
    max_retrieval_iterations: int
    max_retrieval_calls: int
    max_tokens_per_query: int
    max_query_latency_seconds: float
    max_upload_size_bytes: int
