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


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings instance.

    Cached deliberately: config should not change mid-process. Tests that need
    different settings should construct `Settings(...)` directly rather than
    mutating environment variables after this has been called once.
    """
    return Settings()
