"""Cache client factory.

Thin wrapper only — this is not the caching *policy* (key construction,
invalidation on document/index version changes) described in spec §32, which
lands with the embedding/retrieval caching work in later phases. This module
just gives the rest of the codebase one place to get a cache client behind a
Redis-compatible interface.

Backend selection: a real `redis.asyncio.Redis` when `redis_url` is set to an
actual instance (native-Windows-dev pointing at a managed Redis like Upstash,
or the docker-compose `redis` service), or `InMemoryCache` when no managed
Redis is available yet. `InMemoryCache` implements only the subset of the
Redis API this codebase actually calls — it is a stand-in for local dev
without external accounts, not a Redis reimplementation, and does not persist
across process restarts or coordinate across multiple processes the way real
Redis does.
"""

from __future__ import annotations

import time as time_module
from typing import Protocol

from redis.asyncio import Redis


class CacheClient(Protocol):
    """Mirrors the subset of `redis.asyncio.Redis`'s public signature this
    codebase calls (param names included) so `InMemoryCache` structurally
    satisfies the same Protocol as the real Redis client."""

    async def get(self, name: str) -> str | bytes | None: ...
    async def set(self, name: str, value: str, *, ex: int | None = None) -> bool | None: ...
    async def delete(self, *names: str) -> int: ...
    async def incr(self, name: str) -> int: ...
    async def expire(self, name: str, time: int) -> bool: ...
    async def ping(self) -> bool: ...
    async def aclose(self) -> None: ...


class InMemoryCache:
    """Single-process, in-memory stand-in for Redis. See module docstring."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float | None]] = {}

    async def get(self, name: str) -> str | None:
        entry = self._store.get(name)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time_module.monotonic() > expires_at:
            del self._store[name]
            return None
        return value

    async def set(self, name: str, value: str, *, ex: int | None = None) -> bool | None:
        expires_at = time_module.monotonic() + ex if ex is not None else None
        self._store[name] = (value, expires_at)
        return True

    async def delete(self, *names: str) -> int:
        return sum(1 for name in names if self._store.pop(name, None) is not None)

    async def incr(self, name: str) -> int:
        entry = self._store.get(name)
        if entry is not None:
            value, expires_at = entry
            if expires_at is not None and time_module.monotonic() > expires_at:
                entry = None
        if entry is None:
            self._store[name] = ("1", None)
            return 1
        value, expires_at = entry
        new_value = int(value) + 1
        self._store[name] = (str(new_value), expires_at)
        return new_value

    async def expire(self, name: str, time: int) -> bool:
        entry = self._store.get(name)
        if entry is None:
            return False
        value, _ = entry
        self._store[name] = (value, time_module.monotonic() + time)
        return True

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        self._store.clear()


_client: CacheClient | None = None

# Any redis_url still carrying the .env.example placeholder is treated as
# "no managed Redis configured yet" rather than attempted as a real endpoint.
_UNSET_MARKERS = ("<your-upstash-endpoint>", "<password>")


def _is_configured(redis_url: str) -> bool:
    return not any(marker in redis_url for marker in _UNSET_MARKERS)


def get_cache(redis_url: str) -> CacheClient:
    global _client
    if _client is None:
        client: CacheClient
        if _is_configured(redis_url):
            # redis-py's typeshed stubs declare async methods returning
            # `Awaitable[T]` rather than `Coroutine[Any, Any, T]`, which mypy
            # treats as a structural mismatch against our Protocol even though
            # the real runtime behavior is identical.
            client = Redis.from_url(redis_url, decode_responses=True)  # type: ignore[assignment]
        else:
            client = InMemoryCache()
        _client = client
    return _client


async def close_cache() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
    _client = None
