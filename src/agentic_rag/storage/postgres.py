"""Async SQLAlchemy engine/session setup.

Single source of truth for the declarative `Base` and the async session
factory. Everything under storage/models.py declares tables against this
`Base`; Alembic's migrations/env.py imports it for autogeneration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(
    database_url: str, *, pool_size: int = 5, max_overflow: int = 10
) -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=max_overflow,
        )
    return _engine


def get_session_factory(
    database_url: str, *, pool_size: int = 5, max_overflow: int = 10
) -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(database_url, pool_size=pool_size, max_overflow=max_overflow),
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def session_scope(database_url: str) -> AsyncIterator[AsyncSession]:
    """One transaction per `async with` block; rolls back on exception."""
    factory = get_session_factory(database_url)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Call on application shutdown to close the pool cleanly."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
