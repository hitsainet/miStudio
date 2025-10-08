"""
Database connection and session management for MechInterp Studio.

This module provides async SQLAlchemy engine and session factories using
SQLAlchemy 2.0+ with asyncpg driver for PostgreSQL.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base
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
