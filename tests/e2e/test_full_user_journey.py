"""A single test walking the whole user journey through the real HTTP API
in sequence, the way a person actually uses this system — not one endpoint
in isolation, which is what every test in tests/integration already covers.
This is what distinguishes an e2e test from an integration test here:
cross-endpoint state consistency (a collection created in one request is
usable in the next; a document ingested in one request is searchable,
askable, and traceable in later ones) rather than per-endpoint correctness.

Found missing entirely during an engineering audit — tests/e2e/ previously
contained only an empty __init__.py."""

import uuid

import pytest


@pytest.mark.asyncio
async def test_full_user_journey_from_upload_to_grounded_answer(client) -> None:
    # 1. Create a collection, the way the Collections page does.
    collection_resp = await client.post(
        "/collections",
        json={
            "name": f"e2e-{uuid.uuid4().hex[:8]}",
            "source_authority_order": ["annual report", "press release"],
        },
    )
    assert collection_resp.status_code == 201
    collection_id = collection_resp.json()["id"]

    # 2. Upload two documents, the way the Documents page does.
    upload_resp = await client.post(
        "/documents",
        data={"collection_id": collection_id, "source": "Annual Report"},
        files={
            "file": (
                "revenue.txt",
                b"Revenue declined in Q3 due to weaker enterprise demand and pricing pressure.",
                "text/plain",
            )
        },
    )
    assert upload_resp.status_code == 201
    document_id = upload_resp.json()["document"]["id"]

    unrelated_resp = await client.post(
        "/documents",
        data={"collection_id": collection_id},
        files={
            "file": (
                "weather.txt",
                b"Heavy rainfall is forecast across the region this week.",
                "text/plain",
            )
        },
    )
    assert unrelated_resp.status_code == 201

    # 3. Confirm both documents are listed, scoped to the collection.
    list_resp = await client.get("/documents", params={"collection_id": collection_id})
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 2

    # 4. Index (chunk + embed) both documents.
    index_resp = await client.post(f"/documents/{document_id}/ingest", json={})
    assert index_resp.status_code == 200
    assert index_resp.json()["chunk_count"] >= 1

    unrelated_document_id = unrelated_resp.json()["document"]["id"]
    index_unrelated_resp = await client.post(
        f"/documents/{unrelated_document_id}/ingest", json={}
    )
    assert index_unrelated_resp.status_code == 200

    # 5. Direct search, the way the Search page does — must find the
    # relevant document, scoped to this collection, not the unrelated one.
    search_resp = await client.post(
        "/search",
        json={"query": "revenue decline", "collection_id": collection_id, "top_k": 5},
    )
    assert search_resp.status_code == 200
    search_results = search_resp.json()["results"]
    assert len(search_results) >= 1
    assert search_results[0]["document_filename"] == "revenue.txt"

    # 6. Ask a grounded question, the way the Ask page does — the full
    # agentic pipeline: plan -> retrieve -> rerank -> evidence -> synthesis
    # -> citation validation.
    ask_resp = await client.post(
        "/query",
        json={"query": "why did revenue decline", "collection_id": collection_id},
    )
    assert ask_resp.status_code == 200
    ask_body = ask_resp.json()
    assert ask_body["status"] == "grounded"
    assert ask_body["answer"]
    assert len(ask_body["citations"]) >= 1
    citation = ask_body["citations"][0]
    assert citation["document_filename"] == "revenue.txt"
    trace_id = ask_body["trace_id"]

    # 7. The trace from that exact query is independently retrievable, the
    # way the Traces page does — proving trace_id isn't just an opaque
    # field in the response but a real, queryable handle.
    trace_resp = await client.get(f"/queries/{trace_id}/trace")
    assert trace_resp.status_code == 200
    event_types = [e["event_type"] for e in trace_resp.json()["events"]]
    assert event_types[0] == "query.started"
    assert event_types[-1] == "query.completed"

    # 8. A second collection never sees the first collection's documents —
    # the same isolation guarantee the Ask/Search pages depend on.
    other_collection_resp = await client.post(
        "/collections", json={"name": f"e2e-other-{uuid.uuid4().hex[:8]}"}
    )
    other_collection_id = other_collection_resp.json()["id"]
    isolated_ask_resp = await client.post(
        "/query",
        json={"query": "why did revenue decline", "collection_id": other_collection_id},
    )
    assert isolated_ask_resp.status_code == 200
    assert isolated_ask_resp.json()["status"] == "no_evidence_found"

    # 9. Server settings, the way the Settings page does.
    settings_resp = await client.get("/settings")
    assert settings_resp.status_code == 200
    assert "llm_provider" in settings_resp.json()
