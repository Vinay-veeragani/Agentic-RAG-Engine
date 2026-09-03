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
async def test_retrieve_with_rerank_returns_rerank_scores_and_truncates(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    await _upload_and_index(
        client, collection_id, "revenue.txt", b"Quarterly revenue increased due to demand."
    )
    await _upload_and_index(
        client, collection_id, "weather.txt", b"Heavy rainfall is forecast this week."
    )

    response = await client.post(
        "/retrieve",
        json={
            "query": "revenue",
            "strategy": "hybrid",
            "candidate_pool_size": 10,
            "rerank": True,
            "rerank_top_k": 1,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["rerank_score"] is not None
    assert body["results"][0]["rank"] == 1


@pytest.mark.asyncio
async def test_retrieve_without_rerank_has_null_rerank_score(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]
    await _upload_and_index(client, collection_id, "notes.txt", b"Some arbitrary content.")

    response = await client.post("/retrieve", json={"query": "arbitrary", "rerank": False})
    assert response.status_code == 200
    assert all(r["rerank_score"] is None for r in response.json()["results"])
