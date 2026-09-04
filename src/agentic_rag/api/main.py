"""FastAPI application entrypoint.

Wires: structured logging, a per-request trace ID, domain-error -> HTTP
translation (never leaking stack traces), and route registration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from agentic_rag.api.routes.collections import router as collections_router
from agentic_rag.api.routes.documents import router as documents_router
from agentic_rag.api.routes.evaluations import router as evaluations_router
from agentic_rag.api.routes.health import router as health_router
from agentic_rag.api.routes.metrics import router as metrics_router
from agentic_rag.api.routes.query import queries_router
from agentic_rag.api.routes.query import router as query_router
from agentic_rag.api.routes.retrieval import router as retrieval_router
from agentic_rag.api.routes.settings import router as settings_router
from agentic_rag.api.schemas.errors import ErrorResponse
from agentic_rag.core.config import Settings, get_settings
from agentic_rag.core.errors import AgenticRAGError
from agentic_rag.core.models import FailureMode
from agentic_rag.observability.tracing import (
    bind_trace_id,
    configure_logging,
    get_logger,
)
from agentic_rag.security.auth import auth_required, extract_api_key, is_valid_api_key
from agentic_rag.security.rate_limit import RateLimiter
from agentic_rag.storage.cache import close_cache, get_cache, is_real_redis_configured
from agentic_rag.storage.postgres import dispose_engine

logger = get_logger(__name__)

_ERROR_STATUS: dict[FailureMode, int] = {
    FailureMode.NO_KNOWLEDGE: 200,
    FailureMode.INSUFFICIENT_EVIDENCE: 200,
    FailureMode.CONFLICTING_EVIDENCE: 200,
    FailureMode.RETRIEVAL_ERROR: 502,
    FailureMode.MODEL_ERROR: 502,
    FailureMode.TIMEOUT: 504,
    FailureMode.BUDGET_EXCEEDED: 429,
    FailureMode.INVALID_DOCUMENT: 400,
    FailureMode.UNSUPPORTED_FILE_TYPE: 415,
    FailureMode.PROMPT_INJECTION_DETECTED: 400,
    FailureMode.RATE_LIMITED: 429,
    FailureMode.UNAUTHORIZED: 401,
}

# Infra probes stay reachable with no credential and outside rate limiting
# regardless of configuration — a misconfigured API key must never be able
# to take down health checks or metrics scraping.
_AUTH_EXEMPT_PATHS = {"/health", "/metrics"}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    configure_logging(get_settings().log_level)
    logger.info("app.startup")
    yield
    await dispose_engine()
    await close_cache()
    logger.info("app.shutdown")


def validate_runtime_config(settings: Settings) -> None:
    """Fails loudly at startup rather than silently misbehaving at request
    time. Rate limiting's cache-backed counters only coordinate correctly
    across multiple worker processes when backed by a real Redis instance —
    each worker gets its own independent InMemoryCache otherwise, silently
    multiplying the effective rate limit by the worker count instead of
    enforcing it. A single worker is unaffected (its in-memory cache *is*
    the whole process), so this only blocks the specific combination that's
    actually broken, not every no-Redis deployment."""
    if (
        settings.rate_limit_enabled
        and settings.workers > 1
        and not is_real_redis_configured(settings.redis_url)
    ):
        raise RuntimeError(
            f"RATE_LIMIT_ENABLED=true with WORKERS={settings.workers} requires a "
            "real REDIS_URL. The in-memory cache fallback does not coordinate "
            "rate-limit counters across worker processes, which would silently "
            f"turn a {settings.rate_limit_requests_per_window}-request limit into "
            f"roughly {settings.rate_limit_requests_per_window * settings.workers} "
            "requests actually allowed. Configure a real Redis instance, run with "
            "WORKERS=1, or set RATE_LIMIT_ENABLED=false."
        )


def create_app() -> FastAPI:
    validate_runtime_config(get_settings())
    app = FastAPI(title="Agentic RAG Platform", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-trace-id"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def trace_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-trace-id")
        trace_id = bind_trace_id(incoming)
        response = await call_next(request)
        response.headers["x-trace-id"] = trace_id
        return response

    @app.middleware("http")
    async def security_headers_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if get_settings().app_env == "production":
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
            )
        return response

    @app.middleware("http")
    async def auth_and_rate_limit_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        settings = get_settings()
        if request.method == "OPTIONS" or request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)

        api_key = extract_api_key(
            authorization=request.headers.get("authorization"),
            x_api_key=request.headers.get("x-api-key"),
        )

        if auth_required(settings) and not is_valid_api_key(settings, api_key):
            logger.warning("request.unauthorized", path=request.url.path)
            body = ErrorResponse(
                code=FailureMode.UNAUTHORIZED,
                message="A valid API key is required (Authorization: Bearer <key> or X-API-Key).",
                trace_id=request.headers.get("x-trace-id"),
            )
            return JSONResponse(status_code=401, content=body.model_dump())

        if settings.rate_limit_enabled:
            limiter = RateLimiter(
                get_cache(settings.redis_url),
                requests_per_window=settings.rate_limit_requests_per_window,
                window_seconds=settings.rate_limit_window_seconds,
            )
            client_key = api_key or (request.client.host if request.client else "unknown")
            allowed, retry_after = await limiter.check(client_key)
            if not allowed:
                logger.warning("request.rate_limited", client_key=client_key, path=request.url.path)
                body = ErrorResponse(
                    code=FailureMode.RATE_LIMITED,
                    message="Rate limit exceeded.",
                    trace_id=request.headers.get("x-trace-id"),
                )
                return JSONResponse(
                    status_code=429,
                    content=body.model_dump(),
                    headers={"Retry-After": str(retry_after)},
                )

        return await call_next(request)

    @app.exception_handler(AgenticRAGError)
    async def handle_domain_error(request: Request, exc: AgenticRAGError) -> JSONResponse:
        logger.warning("request.domain_error", code=exc.code, message=exc.message)
        status_code = _ERROR_STATUS.get(exc.code, 500)
        body = ErrorResponse(
            code=exc.code,
            message=exc.message,
            trace_id=request.headers.get("x-trace-id"),
        )
        return JSONResponse(status_code=status_code, content=body.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("request.unhandled_error", error_type=type(exc).__name__)
        body = ErrorResponse(
            code=FailureMode.MODEL_ERROR,
            message="An internal error occurred.",
            trace_id=request.headers.get("x-trace-id"),
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    app.include_router(health_router)
    app.include_router(collections_router)
    app.include_router(documents_router)
    app.include_router(retrieval_router)
    app.include_router(query_router)
    app.include_router(queries_router)
    app.include_router(metrics_router)
    app.include_router(settings_router)
    app.include_router(evaluations_router)
    return app


app = create_app()
