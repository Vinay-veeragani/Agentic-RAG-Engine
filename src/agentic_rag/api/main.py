"""FastAPI application entrypoint.

Wires: structured logging, a per-request trace ID, domain-error -> HTTP
translation (never leaking stack traces, spec §29/§36), and route registration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from agentic_rag.api.routes.collections import router as collections_router
from agentic_rag.api.routes.documents import router as documents_router
from agentic_rag.api.routes.health import router as health_router
from agentic_rag.api.routes.metrics import router as metrics_router
from agentic_rag.api.routes.query import queries_router
from agentic_rag.api.routes.query import router as query_router
from agentic_rag.api.routes.retrieval import router as retrieval_router
from agentic_rag.api.schemas.errors import ErrorResponse
from agentic_rag.core.config import get_settings
from agentic_rag.core.errors import AgenticRAGError
from agentic_rag.core.models import FailureMode
from agentic_rag.observability.tracing import (
    bind_trace_id,
    configure_logging,
    get_logger,
)
from agentic_rag.storage.cache import close_cache
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
}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    configure_logging(get_settings().log_level)
    logger.info("app.startup")
    yield
    await dispose_engine()
    await close_cache()
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic RAG Platform", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def trace_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-trace-id")
        trace_id = bind_trace_id(incoming)
        response = await call_next(request)
        response.headers["x-trace-id"] = trace_id
        return response

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
    return app


app = create_app()
