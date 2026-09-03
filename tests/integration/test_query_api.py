import uuid

import pytest


async def _upload_and_index(
    client, collection_id: str, filename: str, content: bytes, *, source: str | None = None
) -> None:
    data = {"collection_id": collection_id}
    if source is not None:
        data["source"] = source
    upload_resp = await client.post(
        "/documents",
        data=data,
        files={"file": (filename, content, "text/plain")},
    )
    document_id = upload_resp.json()["document"]["id"]
    index_resp = await client.post(f"/documents/{document_id}/ingest", json={})
    assert index_resp.status_code == 200


@pytest.mark.asyncio
async def test_query_endpoint_returns_grounded_answer_with_citations(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]
    await _upload_and_index(
        client,
        collection_id,
        "revenue.txt",
        b"Revenue declined due to weaker enterprise demand and pricing pressure.",
        source="Annual Report",
    )

    response = await client.post(
        "/query", json={"query": "why did revenue decline", "collection_id": collection_id}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "grounded"
    assert body["answer"]
    assert len(body["citations"]) >= 1
    citation = body["citations"][0]
    assert citation["label"].startswith("[1]")
    assert citation["document_filename"] == "revenue.txt"
    assert body["citation_completeness"] is not None
    assert body["citation_precision"] is not None


@pytest.mark.asyncio
async def test_query_endpoint_returns_no_evidence_found_for_empty_collection(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    response = await client.post(
        "/query", json={"query": "anything at all", "collection_id": collection_id}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_evidence_found"
    assert body["answer"] is None
    assert body["citations"] == []


@pytest.mark.asyncio
async def test_query_endpoint_returns_conflicting_evidence_without_synthesizing(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]
    await _upload_and_index(client, collection_id, "a.txt", b"Revenue declined 4% in Q3.")
    await _upload_and_index(client, collection_id, "b.txt", b"Revenue declined 9% in Q3.")

    response = await client.post(
        "/query",
        json={"query": "what happened to revenue", "collection_id": collection_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "conflicting_evidence"
    assert body["answer"] is None
    assert body["citations"] == []
    # the contradiction itself is still visible in the trace
    assert body["iterations"][0]["contradictions"]


@pytest.mark.asyncio
async def test_query_endpoint_rejects_empty_query(client) -> None:
    response = await client.post("/query", json={"query": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_endpoint_insufficient_evidence_for_unrelated_content(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]
    await _upload_and_index(
        client, collection_id, "weather.txt", b"Heavy rainfall is forecast this week."
    )

    response = await client.post(
        "/query",
        json={"query": "why did quarterly revenue decline", "collection_id": collection_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["answer"] is None
