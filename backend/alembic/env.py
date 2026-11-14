"""
Alembic environment configuration for async SQLAlchemy migrations.

This module configures Alembic to work with async SQLAlchemy 2.0+.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from src.core.config import settings
from src.db import migration_safety

# One definition of what autogenerate and `alembic check` compare against, shared with
# tests/unit/test_schema_guards.py so the two cannot drift apart. It registers EVERY
# model (importing a single model class here left task_queue out). There is
# deliberately no include_object filter: a filter hides real drift.
from src.db.alembic_support import COMPARE_OPTIONS
from src.db.alembic_support import target_metadata as load_target_metadata

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set database URL from settings
config.set_main_option("sqlalchemy.url", str(settings.database_url))

# add your model's MetaData object here for 'autogenerate' support
target_metadata = load_target_metadata()

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **COMPARE_OPTIONS,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Run migrations with an established connection — bounded, and explained when blocked.

    The session gets lock_timeout / statement_timeout / idle_in_transaction_session_timeout /
    client_connection_check_interval first (src/db/migration_safety.py; incident 2026-10-21:
    an ALTER queued forever behind a hung COPY, and every startup-probe kill left another
    orphaned ALTER in the lock queue). While the migrations run, a second connection watches
    who blocks this session, so a lock timeout names the blocker instead of just failing.

    Args:
        connection: SQLAlchemy connection
    """
    guards = migration_safety.apply_session_guards(connection)
    with migration_safety.BlockerWatch(
        url=config.get_main_option("sqlalchemy.url"),
        backend_pid=guards.backend_pid,
        deadline_at=migration_safety.deadline_from_env(),
    ) as watch:
        try:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                **COMPARE_OPTIONS,
            )

            with context.begin_transaction():
                context.run_migrations()
        except Exception as exc:
            explained = migration_safety.explain(exc, guards, watch)
            if explained is not None:
                raise explained from exc
            raise


async def run_async_migrations() -> None:
    """
    Run migrations in async mode.

    Creates an async engine and runs migrations within an async context.
    """
    # Create async engine from config
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
