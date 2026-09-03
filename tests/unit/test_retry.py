import httpx
import pytest

from agentic_rag.core.retry import request_with_retry


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("POST", "http://test"))


@pytest.mark.asyncio
async def test_returns_immediately_on_success() -> None:
    calls = {"n": 0}

    async def send() -> httpx.Response:
        calls["n"] += 1
        return _response(200)

    response = await request_with_retry(send, max_attempts=3, base_delay_seconds=0.0)
    assert response.status_code == 200
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retries_on_retryable_status_then_succeeds() -> None:
    calls = {"n": 0}

    async def send() -> httpx.Response:
        calls["n"] += 1
        return _response(503) if calls["n"] < 3 else _response(200)

    response = await request_with_retry(send, max_attempts=3, base_delay_seconds=0.0)
    assert response.status_code == 200
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_does_not_retry_on_client_error() -> None:
    calls = {"n": 0}

    async def send() -> httpx.Response:
        calls["n"] += 1
        return _response(400)

    response = await request_with_retry(send, max_attempts=3, base_delay_seconds=0.0)
    assert response.status_code == 400
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retries_on_transport_error_then_raises_after_exhausting_attempts() -> None:
    calls = {"n": 0}

    async def send() -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("boom", request=httpx.Request("POST", "http://test"))

    with pytest.raises(httpx.ConnectError):
        await request_with_retry(send, max_attempts=3, base_delay_seconds=0.0)
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_returns_last_response_after_exhausting_retryable_status_retries() -> None:
    calls = {"n": 0}

    async def send() -> httpx.Response:
        calls["n"] += 1
        return _response(503)

    response = await request_with_retry(send, max_attempts=3, base_delay_seconds=0.0)
    assert response.status_code == 503
    assert calls["n"] == 3
