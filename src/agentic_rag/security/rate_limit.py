"""Fixed-window rate limiting on top of the existing cache abstraction.

A fixed window (not a sliding one) is a deliberate simplicity tradeoff:
it can allow up to 2x the configured rate across a window boundary (a
burst at the end of one window followed by a burst at the start of the
next), which a sliding-window or token-bucket algorithm would prevent.
That's an accepted, documented gap — a real sliding-window limiter needs
either a sorted-set-per-request (expensive under `InMemoryCache`, which
only implements the Redis subset this codebase already uses elsewhere) or
a Lua script (only meaningful against a real Redis, not the in-memory
local-dev fallback). Fixed-window works identically against both.
"""

from __future__ import annotations

import time

from agentic_rag.storage.cache import CacheClient


class RateLimiter:
    def __init__(
        self,
        cache: CacheClient,
        *,
        requests_per_window: int,
        window_seconds: int,
    ) -> None:
        self._cache = cache
        self._limit = requests_per_window
        self._window_seconds = window_seconds

    async def check(self, client_key: str) -> tuple[bool, int]:
        """Returns `(allowed, retry_after_seconds)`. `retry_after_seconds` is
        always the time left in the current window, useful as a `Retry-After`
        header value whether or not the request was allowed."""
        now = int(time.time())
        window_start = now - (now % self._window_seconds)
        key = f"ratelimit:{client_key}:{window_start}"

        count = await self._cache.incr(key)
        if count == 1:
            await self._cache.expire(key, self._window_seconds)

        retry_after = self._window_seconds - (now - window_start)
        return count <= self._limit, retry_after
