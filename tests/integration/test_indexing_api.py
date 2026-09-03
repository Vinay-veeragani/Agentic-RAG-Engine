import uuid

import pytest


@pytest.mark.asyncio
async def test_ingest_endpoint_indexes_uploaded_document(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    content = b"# Overview\n\nRevenue grew 12 percent in 2025.\n\n## Risks\n\nSupply chain risk."
    upload_resp = await client.post(
        "/documents",
        data={"collection_id": collection_id},
        files={"file": ("report.md", content, "text/markdown")},
    )
    document_id = upload_resp.json()["document"]["id"]

    index_resp = await client.post(f"/documents/{document_id}/ingest", json={})
    assert index_resp.status_code == 200
    body = index_resp.json()
    assert body["strategy"] == "structural"
    assert body["embedding_model"] == "mock"
    assert body["embedding_dimensions"] == 384
    assert body["chunk_count"] >= 2


@pytest.mark.asyncio
async def test_ingest_endpoint_accepts_strategy_override(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    content = b"Paragraph one.\n\nParagraph two.\n\nParagraph three."
    upload_resp = await client.post(
        "/documents",
        data={"collection_id": collection_id},
        files={"file": ("notes.txt", content, "text/plain")},
    )
    document_id = upload_resp.json()["document"]["id"]

    index_resp = await client.post(
        f"/documents/{document_id}/ingest",
        json={"strategy": "fixed", "chunk_size_tokens": 5, "chunk_overlap_tokens": 0},
    )
    assert index_resp.status_code == 200
    assert index_resp.json()["strategy"] == "fixed"
    assert index_resp.json()["chunk_count"] >= 1


@pytest.mark.asyncio
async def test_ingest_endpoint_404_for_unknown_document(client) -> None:
    response = await client.post(f"/documents/{uuid.uuid4()}/ingest", json={})
    assert response.status_code == 404
