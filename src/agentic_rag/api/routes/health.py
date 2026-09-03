from fastapi import APIRouter
from sqlalchemy import text

from agentic_rag.api.dependencies.db import DbSession
from agentic_rag.api.schemas.health import HealthResponse
from agentic_rag.core.config import get_settings
from agentic_rag.storage.cache import get_cache

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(db: DbSession) -> HealthResponse:
    db_status = "ok"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "unavailable"

    cache_status = "ok"
    try:
        cache = get_cache(get_settings().redis_url)
        await cache.ping()
    except Exception:
        cache_status = "unavailable"

    overall = "ok" if db_status == "ok" and cache_status == "ok" else "degraded"
    return HealthResponse(status=overall, database=db_status, cache=cache_status)
