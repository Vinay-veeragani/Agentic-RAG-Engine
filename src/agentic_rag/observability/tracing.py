"""Structured logging + trace ID propagation.

Every request/query gets a `trace_id` that flows through logs, DB rows
(`queries.trace_id`, `events.trace_id`), and SSE events.
This module owns generating/holding that ID; it is not a full OpenTelemetry
setup yet — that's added in the observability phase once there's an
actual pipeline to trace spans across. Structured JSON logging is real today
because it costs nothing extra and every later phase benefits from it.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


def new_trace_id() -> str:
    return uuid.uuid4().hex


def bind_trace_id(trace_id: str | None = None) -> str:
    """Set the trace ID for the current async context; returns the ID used."""
    trace_id = trace_id or new_trace_id()
    _trace_id_var.set(trace_id)
    return trace_id


def get_trace_id() -> str | None:
    return _trace_id_var.get()


def _add_trace_id(
    _logger: object, _method_name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    trace_id = get_trace_id()
    if trace_id is not None:
        event_dict["trace_id"] = trace_id
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Call once at process startup (see api/main.py)."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_trace_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
