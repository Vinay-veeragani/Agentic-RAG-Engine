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


@pytest.mark.asyncio
async def test_agentic_retrieve_endpoint_reports_unresolved_conflict(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]
    await _upload_and_index(client, collection_id, "a.txt", b"Revenue declined 4% in Q3.")
    await _upload_and_index(client, collection_id, "b.txt", b"Revenue declined 9% in Q3.")

    response = await client.post(
        "/query/retrieve",
        json={"query": "what happened to revenue", "collection_id": collection_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["termination_reason"] == "conflicting_evidence"
    contradictions = body["iterations"][0]["contradictions"]
    assert len(contradictions) == 1
    assert contradictions[0]["resolution"] is None


@pytest.mark.asyncio
async def test_agentic_retrieve_endpoint_resolves_conflict_via_collection_authority(
    client,
) -> None:
    collection_resp = await client.post(
        "/collections",
        json={
            "name": f"col-{uuid.uuid4().hex[:8]}",
            "source_authority_order": ["annual report", "press release"],
        },
    )
    assert collection_resp.json()["source_authority_config"] == {
        "order": ["annual report", "press release"]
    }
    collection_id = collection_resp.json()["id"]

    await _upload_and_index(
        client, collection_id, "annual.txt", b"Revenue declined 4% in Q3.", source="Annual Report"
    )
    await _upload_and_index(
        client, collection_id, "press.txt", b"Revenue declined 9% in Q3.", source="Press Release"
    )

    response = await client.post(
        "/query/retrieve",
        json={"query": "what happened to revenue", "collection_id": collection_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["termination_reason"] != "conflicting_evidence"
    contradictions = body["iterations"][0]["contradictions"]
    assert len(contradictions) == 1
    assert contradictions[0]["resolution"] is not None


@pytest.mark.asyncio
async def test_upload_document_persists_source(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    upload_resp = await client.post(
        "/documents",
        data={"collection_id": collection_id, "source": "Annual Report"},
        files={"file": ("report.txt", b"Some content.", "text/plain")},
    )
    assert upload_resp.json()["document"]["source"] == "Annual Report"
