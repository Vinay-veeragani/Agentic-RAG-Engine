"""Structured agentic-execution events.

Every event carries an event ID, query (trace) ID, timestamp, event type,
and a structured payload — nothing else. This is deliberately the surface
exposed instead of hidden chain-of-thought: an external caller (SSE stream,
trace replay) only ever sees these structured decisions, never raw model
reasoning.

`EventEmitter` is dual-purpose: it always appends to an in-memory list
(consumed synchronously by `POST /query`) and, when constructed with
`queue=True`, also pushes onto an `asyncio.Queue` a concurrent SSE endpoint
can drain live while the pipeline is still running.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from agentic_rag.observability.tracing import get_logger

logger = get_logger(__name__)


class EventType(StrEnum):
    QUERY_STARTED = "query.started"
    QUERY_ANALYZED = "query.analyzed"
    PLAN_CREATED = "plan.created"
    RETRIEVAL_STARTED = "retrieval.started"
    RETRIEVAL_COMPLETED = "retrieval.completed"
    RERANKING_STARTED = "reranking.started"
    RERANKING_COMPLETED = "reranking.completed"
    EVIDENCE_EVALUATED = "evidence.evaluated"
    RETRIEVAL_REFINED = "retrieval.refined"
    GENERATION_STARTED = "generation.started"
    CITATION_VALIDATION_STARTED = "citation.validation.started"
    QUERY_COMPLETED = "query.completed"
    QUERY_FAILED = "query.failed"


class Event(BaseModel):
    event_id: str
    query_id: str
    timestamp: datetime
    event_type: EventType
    payload: dict[str, object] = Field(default_factory=dict)


class EventEmitter:
    def __init__(self, trace_id: str, *, queue: bool = False) -> None:
        self._trace_id = trace_id
        self._events: list[Event] = []
        self._queue: asyncio.Queue[Event | None] | None = asyncio.Queue() if queue else None

    def emit(self, event_type: EventType, **payload: object) -> Event:
        event = Event(
            event_id=uuid.uuid4().hex,
            query_id=self._trace_id,
            timestamp=datetime.now(UTC),
            event_type=event_type,
            payload=payload,
        )
        self._events.append(event)
        logger.info(
            "agentic.event", event_type=event_type.value, query_id=self._trace_id, **payload
        )
        if self._queue is not None:
            self._queue.put_nowait(event)
        return event

    def close(self) -> None:
        """Signals a streaming consumer (see `stream()`) that no more events
        are coming. A no-op if this emitter wasn't constructed with a queue."""
        if self._queue is not None:
            self._queue.put_nowait(None)

    @property
    def queue(self) -> asyncio.Queue[Event | None]:
        """The live queue a concurrent SSE consumer drains (see
        `api/routes/query.py`). Only present when constructed with
        `queue=True` — the non-streaming `POST /query` path never touches it."""
        if self._queue is None:
            raise RuntimeError("EventEmitter was not constructed with queue=True")
        return self._queue

    @property
    def events(self) -> list[Event]:
        return list(self._events)


_MAX_STORED_TRACES = 200


class TraceStore:
    """Process-local, in-memory only — not persisted across restarts and not
    shared across worker processes. A real trace store backed by the
    `events` table (already present in the schema, but unused) is a documented
    gap; this exists so `GET /queries/{id}/trace` and reconnecting SSE
    clients have *something* to read within one process's lifetime, rather
    than the endpoint not existing at all.
    """

    def __init__(self, max_traces: int = _MAX_STORED_TRACES) -> None:
        self._max_traces = max_traces
        self._traces: OrderedDict[str, list[Event]] = OrderedDict()

    def store(self, trace_id: str, events: list[Event]) -> None:
        self._traces[trace_id] = events
        self._traces.move_to_end(trace_id)
        while len(self._traces) > self._max_traces:
            self._traces.popitem(last=False)

    def get(self, trace_id: str) -> list[Event] | None:
        return self._traces.get(trace_id)


trace_store = TraceStore()
