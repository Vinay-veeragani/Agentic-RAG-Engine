"""Application configuration.

All configuration is environment-driven (12-factor style) so the same code runs
unmodified against native-local infra (Windows Postgres + managed Redis) or the
docker-compose stack — see docs/architecture.md.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["mock", "openai", "anthropic", "gemini", "ollama", "local"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "test", "ci", "production"] = "local"
    log_level: str = "INFO"

    database_url: str = Field(
        default="postgresql+asyncpg://agentic_rag_app:agentic_rag_dev_pw@localhost:5432/agentic_rag"
    )
    redis_url: str = Field(default="redis://localhost:6379/0")

    llm_provider: ProviderName = "mock"
    embedding_provider: ProviderName = "mock"
    reranker_provider: ProviderName = "mock"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # Agentic retrieval loop budgets — hard ceilings, never exceeded regardless of
    # what a planner/agent "wants". See agents/planner.py and the retrieval loop.
    max_retrieval_iterations: int = 3
    max_retrieval_calls: int = 8
    max_tokens_per_query: int = 8000
    max_query_latency_seconds: float = 60.0

    # Ingestion (spec §36: file size limits).
    max_upload_size_bytes: int = 25 * 1024 * 1024
    object_store_root: str = "./data/objects"

    # Frontend dev server origin(s) allowed to call this API (spec §37).
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # Phase 13 hardening.
    #
    # API-key auth: empty (the default) means auth is disabled — every route
    # stays open, matching every other provider's "mock by default, nothing
    # requires a credential to run locally" pattern. Set one or more keys to
    # require `Authorization: Bearer <key>` or `X-API-Key: <key>` on every
    # route except /health and /metrics.
    api_keys: list[str] = Field(default_factory=list)

    # Off by default — like `api_keys` above, this matches every other
    # provider setting's "nothing extra required to run locally" default.
    # Set true explicitly for a shared/production deployment.
    rate_limit_enabled: bool = False
    rate_limit_requests_per_window: int = 120
    rate_limit_window_seconds: int = 60

    # SQLAlchemy async engine pool sizing — small defaults suit a single local
    # dev process; production deployments behind multiple workers should size
    # these to (expected concurrent requests / worker count).
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # Must match the actual `--workers N` uvicorn is started with (or the
    # process manager's equivalent) — used only to validate that rate
    # limiting has a real, cross-process cache backend before allowing more
    # than one worker to start (see api/main.py's create_app()). Each
    # worker is a separate process with its own InMemoryCache when no real
    # Redis is configured, so a naive multi-worker deployment would
    # otherwise silently multiply the effective rate limit by the worker
    # count instead of enforcing it — found during an engineering audit.
    workers: int = 1


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings instance.

    Cached deliberately: config should not change mid-process. Tests that need
    different settings should construct `Settings(...)` directly rather than
    mutating environment variables after this has been called once.
    """
    return Settings()
