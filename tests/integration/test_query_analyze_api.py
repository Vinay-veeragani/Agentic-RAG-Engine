import pytest


@pytest.mark.asyncio
async def test_analyze_endpoint_returns_analysis_and_plan(client) -> None:
    response = await client.post(
        "/query/analyze", json={"query": "Compare Apple and Microsoft revenue"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["analysis"]["query_type"] == "comparison"
    assert body["plan"]["strategy"] == "hybrid"
    assert body["plan"]["max_iterations"] >= 1


@pytest.mark.asyncio
async def test_analyze_endpoint_includes_subqueries_when_decompose_enabled(client) -> None:
    response = await client.post(
        "/query/analyze",
        json={"query": "Microsoft revenue 2023 and Google revenue 2023 and Apple revenue 2023"},
    )
    assert response.status_code == 200
    body = response.json()
    if body["plan"]["decompose"]:
        assert body["subqueries"] is not None
        assert len(body["subqueries"]) >= 2


@pytest.mark.asyncio
async def test_analyze_endpoint_rejects_empty_query(client) -> None:
    response = await client.post("/query/analyze", json={"query": ""})
    assert response.status_code == 422
