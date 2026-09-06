from __future__ import annotations

import os

# Must run before any `agentic_rag` import: Settings()/get_settings() reads
# DATABASE_URL at construction time, and get_settings() is process-wide
# lru_cache'd, so whatever it sees on its first real call (inside a test,
# e.g. via create_app()) is locked in for the rest of the session. Every
# integration test that goes through the `client` fixture hits real routes
# that call db.commit() for real — there is no per-test rollback for that
# path — so without this override, running the suite permanently writes
# thousands of collections/documents/chunks into whatever database
# DATABASE_URL points to. Found the hard way: a real engineering audit
# session left over 2,300 leftover test collections in the actual local
# dev database. `setdefault` so an explicit CI/CD override still wins.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://agentic_rag_app:agentic_rag_dev_pw@localhost:5432/agentic_rag_test",
)
# Same reasoning: the test suite must never depend on whatever a developer's
# local .env happens to have configured for manual live testing. Found the
# hard way in the same session — .env had LLM_PROVIDER=groq set for a real
# end-to-end test, and running pytest right after burned real Groq API
# quota (and failed on rate limits) instead of using the free, instant mock.
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("RERANKER_PROVIDER", "mock")

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.api.main import create_app
from agentic_rag.core.config import Settings
from agentic_rag.storage.postgres import get_session_factory

# Refuses to run against anything that doesn't look like a disposable test
# database — the one hard safety rail against ever truncating real data if
# DATABASE_URL is ever misconfigured to point somewhere else.
_REQUIRED_TEST_DB_MARKER = "test"


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Reads DATABASE_URL from the environment (overridden above to a
    dedicated, disposable test database — see that override for why),
    everything else from real env/.env: tests run against real local
    Postgres + managed Redis, not mocks, since those are cheap and already
    available (see docs/architecture.md)."""
    return Settings()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _reset_test_database(settings: Settings) -> None:
    """Truncates every table before the test session starts, so leftover
    rows from a previous crashed/interrupted run never leak into this
    one — a fresh, empty database every time, not an ever-growing pile.
    Session-scoped + autouse: runs exactly once, before any test. Async
    (not `asyncio.run()` in a sync fixture) deliberately — the engine/
    session-factory this reuses are process-level singletons bound to
    whichever event loop first created them, and `storage/postgres.py`
    already documents why a per-fixture loop would break every later
    test's connection pool.
    """
    if _REQUIRED_TEST_DB_MARKER not in settings.database_url:
        raise RuntimeError(
            f"Refusing to run tests against {settings.database_url!r} — it "
            f"doesn't look like a disposable test database (expected "
            f"{_REQUIRED_TEST_DB_MARKER!r} in the name). This check exists "
            "specifically to prevent ever truncating a real database."
        )

    factory = get_session_factory(settings.database_url)
    async with factory() as session:
        tables = (
            await session.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' AND tablename != 'alembic_version'"
                )
            )
        ).scalars().all()
        if tables:
            quoted = ", ".join(f'"{t}"' for t in tables)
            await session.execute(text(f"TRUNCATE {quoted} RESTART IDENTITY CASCADE"))
            await session.commit()


@pytest_asyncio.fixture
async def db_session(settings: Settings) -> AsyncSession:
    factory = get_session_factory(settings.database_url)
    async with factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
