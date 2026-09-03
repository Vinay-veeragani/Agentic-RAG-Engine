import asyncio

import pytest

from agentic_rag.observability.events import EventEmitter, EventType, TraceStore


def test_emit_returns_a_fully_populated_event() -> None:
    emitter = EventEmitter("trace-123")
    event = emitter.emit(EventType.QUERY_STARTED, query="hello")
    assert event.query_id == "trace-123"
    assert event.event_type == EventType.QUERY_STARTED
    assert event.payload == {"query": "hello"}
    assert event.event_id


def test_events_property_accumulates_in_order() -> None:
    emitter = EventEmitter("trace-123")
    emitter.emit(EventType.QUERY_STARTED)
    emitter.emit(EventType.QUERY_COMPLETED)
    assert [e.event_type for e in emitter.events] == [
        EventType.QUERY_STARTED,
        EventType.QUERY_COMPLETED,
    ]


def test_queue_access_without_queue_true_raises() -> None:
    emitter = EventEmitter("trace-123")
    with pytest.raises(RuntimeError):
        _ = emitter.queue


@pytest.mark.asyncio
async def test_queue_receives_emitted_events_and_close_sends_sentinel() -> None:
    emitter = EventEmitter("trace-123", queue=True)
    emitter.emit(EventType.QUERY_STARTED)
    emitter.close()

    first = await asyncio.wait_for(emitter.queue.get(), timeout=1)
    assert first is not None and first.event_type == EventType.QUERY_STARTED

    sentinel = await asyncio.wait_for(emitter.queue.get(), timeout=1)
    assert sentinel is None


def test_trace_store_round_trip() -> None:
    store = TraceStore(max_traces=2)
    emitter = EventEmitter("trace-a")
    emitter.emit(EventType.QUERY_STARTED)
    store.store("trace-a", emitter.events)

    assert store.get("trace-a") == emitter.events
    assert store.get("unknown") is None


def test_trace_store_evicts_oldest_beyond_capacity() -> None:
    store = TraceStore(max_traces=2)
    store.store("a", [])
    store.store("b", [])
    store.store("c", [])  # evicts "a"

    assert store.get("a") is None
    assert store.get("b") is not None
    assert store.get("c") is not None
