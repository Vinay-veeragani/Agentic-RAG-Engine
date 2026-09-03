from fastapi import APIRouter

from agentic_rag.api.schemas.settings import SettingsResponse
from agentic_rag.core.config import get_settings
from agentic_rag.security.auth import auth_required

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
async def read_settings() -> SettingsResponse:
    settings = get_settings()
    return SettingsResponse(
        app_env=settings.app_env,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        reranker_provider=settings.reranker_provider,
        max_retrieval_iterations=settings.max_retrieval_iterations,
        max_retrieval_calls=settings.max_retrieval_calls,
        max_tokens_per_query=settings.max_tokens_per_query,
        max_query_latency_seconds=settings.max_query_latency_seconds,
        max_upload_size_bytes=settings.max_upload_size_bytes,
        auth_enabled=auth_required(settings),
        rate_limit_enabled=settings.rate_limit_enabled,
        rate_limit_requests_per_window=settings.rate_limit_requests_per_window,
        rate_limit_window_seconds=settings.rate_limit_window_seconds,
    )
