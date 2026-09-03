from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from agentic_rag.core.config import get_settings
from agentic_rag.storage.models import Base  # noqa: F401  (registers all tables on Base.metadata)
from agentic_rag.storage.postgres import Base as DeclarativeBase

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = DeclarativeBase.metadata

# NOTE: alembic autogenerate does not know how to import `pgvector.sqlalchemy`
# for columns using the `Vector` type — any future `alembic revision
# --autogenerate` that adds/changes a vector column will need
# `import pgvector.sqlalchemy` added by hand to the generated file (see the
# initial_schema migration for the pattern) before it will run.


def _sync_database_url() -> str:
    """Alembic runs synchronously; the app's DATABASE_URL uses the async
    asyncpg driver, so swap in the sync psycopg driver just for migrations."""
    url = get_settings().database_url
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg://")


def run_migrations_offline() -> None:
    url = _sync_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _sync_database_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
