import pytest

from agentic_rag.security.rate_limit import RateLimiter
from agentic_rag.storage.cache import InMemoryCache


@pytest.mark.asyncio
async def test_allows_requests_under_the_limit() -> None:
    limiter = RateLimiter(InMemoryCache(), requests_per_window=3, window_seconds=60)
    for _ in range(3):
        allowed, _ = await limiter.check("client-a")
        assert allowed is True


@pytest.mark.asyncio
async def test_blocks_requests_over_the_limit() -> None:
    limiter = RateLimiter(InMemoryCache(), requests_per_window=2, window_seconds=60)
    assert (await limiter.check("client-a"))[0] is True
    assert (await limiter.check("client-a"))[0] is True
    allowed, retry_after = await limiter.check("client-a")
    assert allowed is False
    assert retry_after > 0


@pytest.mark.asyncio
async def test_different_clients_have_independent_budgets() -> None:
    limiter = RateLimiter(InMemoryCache(), requests_per_window=1, window_seconds=60)
    assert (await limiter.check("client-a"))[0] is True
    assert (await limiter.check("client-b"))[0] is True
    assert (await limiter.check("client-a"))[0] is False
