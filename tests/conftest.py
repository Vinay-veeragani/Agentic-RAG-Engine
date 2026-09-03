from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.api.main import create_app
from agentic_rag.core.config import Settings
from agentic_rag.storage.postgres import get_session_factory


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Reads real env/.env — tests run against the local dev Postgres +
    managed Redis, not mocks, since those are cheap and already available
    (see docs/architecture.md). Testcontainers-based isolation is added if/when
    CI needs a disposable Postgres."""
    return Settings()


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
