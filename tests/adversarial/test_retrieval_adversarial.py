"""Adversarial retrieval inputs (spec §38): queries containing characters
that would be meaningful tsquery/SQL syntax if mishandled. `plainto_tsquery`
(rather than `to_tsquery`) treats the whole input as plain text rather than
query syntax, and SQLAlchemy always parameterizes values — this asserts that
holds rather than assuming it."""

import uuid

import pytest

from agentic_rag.retrieval.sparse import SparseRetriever


@pytest.mark.parametrize(
    "query_text",
    [
        "revenue & OR NOT ()",
        "'; DROP TABLE document_chunks; --",
        "\"unterminated quote",
        "",
        "   ",
        "a" * 5000,
    ],
)
@pytest.mark.asyncio
async def test_sparse_retriever_never_raises_on_adversarial_query(db_session, query_text) -> None:
    retriever = SparseRetriever(db_session)
    results = await retriever.retrieve(query_text, top_k=5)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_endpoint_rejects_empty_query(client) -> None:
    response = await client.post("/search", json={"query": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_retrieve_endpoint_returns_empty_for_nonexistent_collection(client) -> None:
    response = await client.post(
        "/retrieve",
        json={
            "query": "anything",
            "filters": {"collection_id": str(uuid.uuid4())},
        },
    )
    assert response.status_code == 200
    assert response.json()["results"] == []
