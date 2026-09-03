import pytest


@pytest.mark.asyncio
async def test_health_endpoint_reports_ok(client) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["cache"] == "ok"


@pytest.mark.asyncio
async def test_health_response_carries_trace_id_header(client) -> None:
    response = await client.get("/health")
    assert "x-trace-id" in response.headers
