"""
FastAPI dependencies.

This module provides dependency injection functions for FastAPI routes.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from .database import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session to FastAPI routes.

    Yields:
        AsyncSession: SQLAlchemy async database session

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            # Use db session here
            pass
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
