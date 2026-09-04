"""validate_runtime_config (api/main.py) fails app startup loudly for the
one configuration that's actually broken — rate limiting enabled with more
than one worker and no real Redis — rather than letting it silently
misbehave at request time (each worker would get its own independent
in-memory rate-limit counter). Found during an engineering audit."""

import pytest

from agentic_rag.api.main import validate_runtime_config
from agentic_rag.core.config import Settings


def test_rejects_rate_limiting_with_multiple_workers_and_no_real_redis() -> None:
    settings = Settings(rate_limit_enabled=True, workers=4)
    with pytest.raises(RuntimeError, match="REDIS_URL"):
        validate_runtime_config(settings)


def test_allows_rate_limiting_with_a_single_worker_and_no_real_redis() -> None:
    settings = Settings(rate_limit_enabled=True, workers=1)
    validate_runtime_config(settings)  # must not raise


def test_allows_multiple_workers_when_a_real_redis_is_configured() -> None:
    settings = Settings(
        rate_limit_enabled=True,
        workers=4,
        redis_url="redis://real-host:6379",
    )
    validate_runtime_config(settings)  # must not raise


def test_allows_multiple_workers_when_rate_limiting_is_disabled() -> None:
    settings = Settings(rate_limit_enabled=False, workers=4)
    validate_runtime_config(settings)  # must not raise
