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
async def test_search_endpoint_returns_ranked_results(client) -> None:
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

    response = await client.post("/search", json={"query": "revenue", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) >= 1
    assert "revenue" in body["results"][0]["content"].lower()


@pytest.mark.asyncio
async def test_retrieve_endpoint_exposes_per_method_scores(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]
    await _upload_and_index(
        client, collection_id, "revenue.txt", b"Quarterly revenue increased due to demand."
    )

    response = await client.post(
        "/retrieve", json={"query": "revenue", "strategy": "hybrid", "top_k": 5}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "hybrid"
    assert len(body["results"]) >= 1
    result = body["results"][0]
    assert result["fusion_score"] is not None
    assert result["rank"] == 1


@pytest.mark.asyncio
async def test_retrieve_endpoint_supports_dense_only_strategy(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]
    await _upload_and_index(client, collection_id, "notes.txt", b"Some arbitrary content here.")

    response = await client.post(
        "/retrieve", json={"query": "arbitrary", "strategy": "dense", "top_k": 5}
    )
    assert response.status_code == 200
    body = response.json()
    assert all(r["sparse_score"] is None for r in body["results"])
    assert all(r["dense_score"] is not None for r in body["results"])


@pytest.mark.asyncio
async def test_search_endpoint_scopes_to_collection_filter(client) -> None:
    collection_a = (
        await client.post("/collections", json={"name": f"col-a-{uuid.uuid4().hex[:8]}"})
    ).json()["id"]
    collection_b = (
        await client.post("/collections", json={"name": f"col-b-{uuid.uuid4().hex[:8]}"})
    ).json()["id"]

    await _upload_and_index(client, collection_a, "a.txt", b"Unique marker term alpha content.")
    await _upload_and_index(client, collection_b, "b.txt", b"Unique marker term alpha content.")

    response = await client.post(
        "/search",
        json={"query": "alpha", "top_k": 10, "filters": {"collection_id": collection_a}},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
