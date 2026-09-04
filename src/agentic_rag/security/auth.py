"""Optional API-key authentication.

`Settings.api_keys` empty (the default) means auth is disabled — every route
stays open, matching the rest of this codebase's "mock/local by default,
nothing requires a credential to run locally" pattern (see `docs/
architecture.md`'s "no auth implemented" gap, now closed but still opt-in).
Configure one or more keys to require a credential on every route except
`/health` and `/metrics` (see `api/main.py`'s auth middleware).
"""

from __future__ import annotations

import hmac

from agentic_rag.core.config import Settings

_BEARER_PREFIX = "Bearer "


def auth_required(settings: Settings) -> bool:
    return len(settings.api_keys) > 0


def extract_api_key(*, authorization: str | None, x_api_key: str | None) -> str | None:
    """Accepts either `Authorization: Bearer <key>` or `X-API-Key: <key>` —
    the former is the HTTP-standard convention, the latter is what many
    simple API clients (and curl one-liners) reach for first."""
    if authorization and authorization.startswith(_BEARER_PREFIX):
        return authorization[len(_BEARER_PREFIX) :].strip() or None
    if x_api_key:
        return x_api_key.strip() or None
    return None


def is_valid_api_key(settings: Settings, key: str | None) -> bool:
    """Constant-time per candidate — a plain `in` check leaks timing
    information proportional to how many leading characters a guess
    shares with a real key. `settings.api_keys` is small (a handful of
    keys at most), so comparing against every one costs nothing."""
    if key is None:
        return False
    return any(hmac.compare_digest(key, candidate) for candidate in settings.api_keys)
