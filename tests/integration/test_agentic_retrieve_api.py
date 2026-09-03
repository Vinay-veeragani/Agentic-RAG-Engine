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


@pytest.mark.asyncio
async def test_agentic_retrieve_endpoint_returns_full_trace(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]
    await _upload_and_index(
        client,
        collection_id,
        "revenue.txt",
        b"Revenue declined due to weaker enterprise demand and pricing pressure.",
    )

    response = await client.post(
        "/query/retrieve",
        json={"query": "why did revenue decline", "collection_id": collection_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["termination_reason"] == "sufficient_evidence"
    assert len(body["iterations"]) >= 1
    assert body["iterations"][0]["sufficient"] is True
    assert len(body["evidence"]) >= 1
    assert body["trace_id"]


@pytest.mark.asyncio
async def test_agentic_retrieve_endpoint_no_evidence_for_empty_collection(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    response = await client.post(
        "/query/retrieve",
        json={"query": "anything at all", "collection_id": collection_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["termination_reason"] == "no_evidence_found"
    assert body["evidence"] == []


@pytest.mark.asyncio
async def test_agentic_retrieve_endpoint_rejects_empty_query(client) -> None:
    response = await client.post("/query/retrieve", json={"query": ""})
    assert response.status_code == 422
