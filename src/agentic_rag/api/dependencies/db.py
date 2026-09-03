"""FastAPI dependency for a per-request DB session."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.core.config import Settings, get_settings
from agentic_rag.storage.postgres import get_session_factory


async def get_db_session(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[AsyncSession]:
    factory = get_session_factory(settings.database_url)
    async with factory() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(get_db_session)]
