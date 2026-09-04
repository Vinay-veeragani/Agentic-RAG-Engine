import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from agentic_rag.agents.research_agent import AgenticRetrievalLoop
from agentic_rag.api.main import create_app
from agentic_rag.core.errors import ModelProviderError


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
async def test_query_endpoint_returns_502_when_llm_provider_fails(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider failure that survives retries (core/retry.py) surfaces as
    ModelProviderError, an AgenticRAGError — it must reach the client as a
    clean 502/MODEL_ERROR via the global domain-error handler, never a raw
    500 or a hung request."""

    async def raises(self, *args, **kwargs):
        raise ModelProviderError("upstream LLM provider is unreachable")

    monkeypatch.setattr(AgenticRetrievalLoop, "run", raises)

    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    response = await client.post(
        "/query", json={"query": "why did revenue decline", "collection_id": collection_id}
    )

    assert response.status_code == 502
    body = response.json()
    assert body["code"] == "MODEL_ERROR"


@pytest.mark.asyncio
async def test_query_endpoint_returns_500_and_hides_details_on_unexpected_error(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuinely unexpected failure (e.g. the database connection itself
    breaking mid-query) is not a domain error — it must still produce a
    clean response, and must never leak the raw exception message to the
    client (api/main.py's handle_unexpected_error discipline)."""

    async def raises(self, *args, **kwargs):
        raise RuntimeError("dsn=postgresql://app:supersecret@db-internal:5432/agentic_rag")

    monkeypatch.setattr(AgenticRetrievalLoop, "run", raises)

    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    # Starlette's ServerErrorMiddleware re-raises after handling an
    # unhandled exception (so a real ASGI server logs it) — httpx's
    # ASGITransport re-raises that into the test by default too, so this
    # one request needs raise_app_exceptions=False to observe the actual
    # HTTP response the client would receive in production.
    transport = ASGITransport(app=create_app(), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as unsafe_client:
        response = await unsafe_client.post(
            "/query",
            json={"query": "why did revenue decline", "collection_id": collection_id},
        )

    assert response.status_code == 500
    body = response.json()
    assert body["message"] == "An internal error occurred."
    assert "supersecret" not in response.text


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
