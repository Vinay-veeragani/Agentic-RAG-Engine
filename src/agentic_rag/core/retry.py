"""Bounded retry with exponential backoff for transient network failures.

Retries only transport-level failures (connection errors, timeouts) and
retryable server errors (5xx) — never a 4xx, since retrying a client error
(bad request, auth failure) can't possibly succeed differently on a second
attempt. Bounded by `max_attempts` like every other loop in this codebase
(the agentic retrieval loop's iteration ceiling, the structured-output
retry in `generation/llm.py`): a flaky provider can slow a request down,
never hang it forever.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx

_RETRYABLE_STATUS = {500, 502, 503, 504}
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)


async def request_with_retry(
    send: Callable[[], Awaitable[httpx.Response]],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
) -> httpx.Response:
    """Calls `send()` up to `max_attempts` times, retrying on connection
    failures/timeouts or a retryable 5xx status, with exponential backoff
    between attempts. Returns the last response (which the caller still
    checks for a non-200 status) once retries are exhausted, or re-raises
    the last transport exception if every attempt failed to connect at all.
    """
    last_transport_error: Exception | None = None
    last_response: httpx.Response | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = await send()
        except _RETRYABLE_TRANSPORT_ERRORS as exc:
            last_transport_error = exc
        else:
            last_transport_error = None
            last_response = response
            if response.status_code not in _RETRYABLE_STATUS:
                return response

        if attempt < max_attempts:
            await asyncio.sleep(base_delay_seconds * (2 ** (attempt - 1)))

    if last_transport_error is not None:
        raise last_transport_error
    assert last_response is not None
    return last_response
