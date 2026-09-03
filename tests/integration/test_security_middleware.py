"""End-to-end checks for the auth + rate-limit middleware wired up in
`api/main.py` (Phase 13). Each test monkeypatches `api.main.get_settings`
(not the process-wide `core.config.get_settings` lru_cache) so the override
never leaks into other tests in the same session."""

import pytest
from httpx import ASGITransport, AsyncClient

from agentic_rag.api.main import create_app
from agentic_rag.core.config import Settings


async def _client_for(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> AsyncClient:
    monkeypatch.setattr("agentic_rag.api.main.get_settings", lambda: settings)
    app = create_app()
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_request_without_api_key_is_rejected_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(api_keys=["expected-key"], rate_limit_enabled=False)
    async with await _client_for(settings, monkeypatch) as client:
        response = await client.get("/collections")

    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


@pytest.mark.asyncio
async def test_request_with_correct_api_key_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(api_keys=["expected-key"], rate_limit_enabled=False)
    async with await _client_for(settings, monkeypatch) as client:
        response = await client.get(
            "/collections", headers={"X-API-Key": "expected-key"}
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_request_with_wrong_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(api_keys=["expected-key"], rate_limit_enabled=False)
    async with await _client_for(settings, monkeypatch) as client:
        response = await client.get(
            "/collections", headers={"Authorization": "Bearer wrong-key"}
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_health_and_metrics_are_exempt_from_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(api_keys=["expected-key"], rate_limit_enabled=False)
    async with await _client_for(settings, monkeypatch) as client:
        health_response = await client.get("/health")
        metrics_response = await client.get("/metrics")

    assert health_response.status_code == 200
    assert metrics_response.status_code == 200


@pytest.mark.asyncio
async def test_auth_disabled_by_default_allows_unauthenticated_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(api_keys=[], rate_limit_enabled=False)
    async with await _client_for(settings, monkeypatch) as client:
        response = await client.get("/collections")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_returns_429_once_window_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        api_keys=[],
        rate_limit_enabled=True,
        rate_limit_requests_per_window=2,
        rate_limit_window_seconds=60,
    )
    async with await _client_for(settings, monkeypatch) as client:
        first = await client.get("/health")  # exempt, doesn't consume budget
        second = await client.get("/collections")
        third = await client.get("/collections")
        fourth = await client.get("/collections")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert fourth.status_code == 429
    assert fourth.headers.get("retry-after") is not None
    assert fourth.json()["code"] == "RATE_LIMITED"
