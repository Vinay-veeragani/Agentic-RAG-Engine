import uuid

import pytest


async def _upload_and_index(client, collection_id: str, filename: str, content: bytes) -> None:
    upload_resp = await client.post(
        "/documents",
        data={"collection_id": collection_id},
        files={"file": (filename, content, "text/plain")},
    )
    document_id = upload_resp.json()["document"]["id"]
    index_resp = await client.post(f"/documents/{document_id}/ingest", json={})
    assert index_resp.status_code == 200


def _parse_sse_events(body: str) -> list[dict]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        data_line = next((line for line in lines if line.startswith("data: ")), None)
        if data_line:
            import json

            events.append(json.loads(data_line.removeprefix("data: ")))
    return events


@pytest.mark.asyncio
async def test_query_stream_emits_expected_event_sequence(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]
    await _upload_and_index(
        client, collection_id, "revenue.txt", b"Revenue declined due to weaker demand."
    )

    async with client.stream(
        "POST",
        "/query/stream",
        json={"query": "why did revenue decline", "collection_id": collection_id},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = ""
        async for chunk in response.aiter_text():
            body += chunk

    events = _parse_sse_events(body)
    event_types = [e["event_type"] for e in events]

    assert event_types[0] == "query.started"
    assert event_types[-1] == "query.completed"
    assert "plan.created" in event_types
    assert "retrieval.completed" in event_types
    assert "reranking.completed" in event_types
    assert "evidence.evaluated" in event_types
    assert "generation.started" in event_types
    assert "citation.validation.started" in event_types
    # Every event shares the same query/trace ID.
    assert len({e["query_id"] for e in events}) == 1


@pytest.mark.asyncio
async def test_query_stream_reports_failure_event_on_no_evidence(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    async with client.stream(
        "POST",
        "/query/stream",
        json={"query": "anything at all", "collection_id": collection_id},
    ) as response:
        body = ""
        async for chunk in response.aiter_text():
            body += chunk

    events = _parse_sse_events(body)
    completed = next(e for e in events if e["event_type"] == "query.completed")
    assert completed["payload"]["termination_reason"] == "no_evidence_found"


@pytest.mark.asyncio
async def test_query_then_get_trace_returns_the_same_events(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]
    await _upload_and_index(
        client, collection_id, "revenue.txt", b"Revenue declined due to weaker demand."
    )

    query_resp = await client.post(
        "/query", json={"query": "why did revenue decline", "collection_id": collection_id}
    )
    trace_id = query_resp.json()["trace_id"]

    trace_resp = await client.get(f"/queries/{trace_id}/trace")
    assert trace_resp.status_code == 200
    body = trace_resp.json()
    assert body["trace_id"] == trace_id
    event_types = [e["event_type"] for e in body["events"]]
    assert event_types[0] == "query.started"
    assert event_types[-1] == "query.completed"


@pytest.mark.asyncio
async def test_get_trace_404_for_unknown_trace_id(client) -> None:
    response = await client.get(f"/queries/{uuid.uuid4().hex}/trace")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_text(client) -> None:
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
