"""
Database connection and session management for MechInterp Studio.

This module provides both async and sync SQLAlchemy engines and session factories.
- Async sessions for FastAPI endpoints (asyncpg driver)
- Sync sessions for Celery workers (psycopg2 driver)
"""

from typing import AsyncGenerator, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool

from .config import settings

# Base class for all SQLAlchemy models
Base = declarative_base()


def create_engine() -> AsyncEngine:
    """
    Create async SQLAlchemy engine with connection pooling.

    Returns:
        AsyncEngine: Configured async database engine

    Configuration:
        - Uses asyncpg driver for PostgreSQL
        - NullPool for async engines (async-safe pooling)
        - Echo SQL in development mode
        - Pool pre-ping to check connection health
    """
    # Create engine with appropriate settings
    # Note: Async engines use NullPool by default, which is async-safe
    engine = create_async_engine(
        str(settings.database_url),
        echo=settings.is_development,  # Log SQL in development
        pool_pre_ping=True,  # Verify connections before using
        poolclass=NullPool,  # Required for async engines
        connect_args={
            "server_settings": {
                "application_name": "mistudio_backend",
            },
        },
    )

    return engine


# Global engine instance
engine: AsyncEngine = create_engine()

# Session factory for creating new sessions
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit
    autocommit=False,  # Manual commit control
    autoflush=False,  # Manual flush control
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function for FastAPI to get database sessions.

    Yields:
        AsyncSession: Database session

    Usage:
        ```python
        from fastapi import Depends
        from app.core.database import get_db

        @app.get("/datasets")
        async def list_datasets(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Dataset))
            return result.scalars().all()
        ```

    Notes:
        - Session is automatically closed after request
        - Exceptions trigger rollback
        - Successful completion triggers commit
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database by creating all tables.

    This function should be called on application startup in development
    or for testing. In production, use Alembic migrations instead.

    Usage:
        ```python
        from app.core.database import init_db

        @app.on_event("startup")
        async def startup():
            await init_db()
        ```

    Notes:
        - Only use in development/testing
        - Use Alembic migrations for production
        - Creates all tables defined in Base metadata
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db() -> None:
    """
    Dispose database engine and close all connections.

    This function should be called on application shutdown to cleanly
    close all database connections.

    Usage:
        ```python
        from app.core.database import dispose_db

        @app.on_event("shutdown")
        async def shutdown():
            await dispose_db()
        ```
    """
    await engine.dispose()


async def check_db_connection() -> bool:
    """
    Check if database connection is healthy.

    Returns:
        bool: True if connection is healthy, False otherwise

    Usage:
        ```python
        from app.core.database import check_db_connection

        @app.get("/health")
        async def health_check():
            db_healthy = await check_db_connection()
            return {"database": "connected" if db_healthy else "disconnected"}
        ```
    """
    try:
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception:
        return False


# ============================================================================
# Synchronous Database Sessions (for Celery Workers)
# ============================================================================


def create_sync_engine_instance():
    """
    Create synchronous SQLAlchemy engine for Celery workers.

    Returns:
        Engine: Configured sync database engine

    Configuration:
        - Uses psycopg2 driver for PostgreSQL
        - QueuePool for connection pooling (pool_size=5, max_overflow=10)
        - Echo SQL in development mode
        - Pool pre-ping to check connection health

    Notes:
        - Celery workers run in separate processes and need sync sessions
        - Connection pooling is safe for sync engines
    """
    # Convert async URL (asyncpg) to sync URL (psycopg2)
    sync_url = str(settings.database_url).replace('+asyncpg', '')

    engine = create_sync_engine(
        sync_url,
        echo=settings.is_development,  # Log SQL in development
        pool_pre_ping=True,  # Verify connections before using
        poolclass=QueuePool,  # Standard connection pooling for sync
        pool_size=5,  # Number of connections to keep in pool
        max_overflow=10,  # Max connections beyond pool_size
        connect_args={
            "application_name": "mistudio_celery",
        },
    )

    return engine


# Global sync engine instance for Celery workers
sync_engine = create_sync_engine_instance()

# Sync session factory for creating new sessions
SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    class_=Session,
    expire_on_commit=False,  # Don't expire objects after commit
    autocommit=False,  # Manual commit control
    autoflush=False,  # Manual flush control
)


@contextmanager
def get_sync_db() -> Generator[Session, None, None]:
    """
    Context manager for Celery workers to get sync database sessions.

    Yields:
        Session: Database session

    Usage:
        ```python
        from app.core.database import get_sync_db

        @celery_app.task
        def process_dataset(dataset_id: str):
            with get_sync_db() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                # ... process dataset ...
                db.commit()
        ```

    Notes:
        - Session is automatically closed after use
        - Exceptions trigger rollback
        - Successful completion triggers commit
        - Use this instead of async sessions in Celery tasks
    """
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
