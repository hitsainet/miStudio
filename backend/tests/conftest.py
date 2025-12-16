"""
Pytest configuration and shared fixtures.

This module configures pytest for async testing and provides
common fixtures used across the test suite.
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from src.core.config import settings
from src.core.database import Base
from src.main import app


# pytest-asyncio is auto-configured via pyproject.toml asyncio_mode="auto"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create an event loop for the test session.

    This fixture ensures all async tests share the same event loop
    throughout the test session, preventing event loop conflicts.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """
    Create an async database engine for testing.

    Uses NullPool to prevent connection pooling issues during tests.
    Each test gets a fresh engine instance.
    """
    from sqlalchemy import text

    # Use test database URL if available, otherwise use main database
    database_url = str(settings.database_url)
    if "postgresql" in database_url and "test" not in database_url:
        # Append '_test' to database name for testing
        database_url = database_url.rsplit("/", 1)[0] + "/mistudio_test"

    engine = create_async_engine(
        database_url,
        echo=settings.is_development,
        poolclass=NullPool,  # Disable connection pooling for tests
    )

    # Create PostgreSQL enum types before creating tables
    # These are required by models that use create_type=False
    enum_definitions = [
        ("export_status", ["pending", "computing", "packaging", "completed", "failed", "cancelled"]),
        ("label_source_enum", ["auto", "user", "llm", "local_llm", "openai"]),
        ("analysis_type_enum", ["logit_lens", "correlations", "ablation", "nlp_analysis"]),
        ("extraction_status_enum", ["queued", "loading", "extracting", "saving", "completed", "failed", "cancelled"]),
    ]

    async with engine.begin() as conn:
        # Create enum types (IF NOT EXISTS to handle reruns)
        for enum_name, values in enum_definitions:
            values_str = ", ".join(f"'{v}'" for v in values)
            await conn.execute(text(
                f"DO $$ BEGIN "
                f"CREATE TYPE {enum_name} AS ENUM ({values_str}); "
                f"EXCEPTION WHEN duplicate_object THEN NULL; END $$;"
            ))

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop all tables and enum types after test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        # Drop enum types
        for enum_name, _ in enum_definitions:
            await conn.execute(text(f"DROP TYPE IF EXISTS {enum_name} CASCADE;"))

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Create an async database session for testing.

    Provides a clean database session for each test function.
    Automatically rolls back after each test to maintain isolation.

    Usage:
        async def test_something(async_session):
            result = await async_session.execute(...)
    """
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(scope="function")
async def client(async_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Create an async HTTP client for testing API endpoints.

    Provides a test client that uses the test database session
    via dependency override.

    Usage:
        async def test_something(client: AsyncClient):
            response = await client.get("/api/v1/health")
    """
    from src.core.deps import get_db

    async def override_get_db():
        yield async_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def anyio_backend():
    """
    Configure anyio backend for pytest-asyncio.

    Returns 'asyncio' to ensure all async tests use asyncio.
    """
    return "asyncio"


# Mock fixtures for external dependencies

@pytest.fixture
def mock_redis(mocker):
    """
    Mock Redis client for testing.

    Provides a mock Redis client that simulates Redis operations
    without requiring an actual Redis server.

    Usage:
        def test_something(mock_redis):
            mock_redis.get.return_value = b"test_value"
    """
    return mocker.MagicMock()


@pytest.fixture
def mock_celery(mocker):
    """
    Mock Celery app for testing.

    Provides a mock Celery app that simulates task queueing
    without requiring an actual Celery worker.

    Usage:
        def test_something(mock_celery):
            mock_celery.send_task.return_value.id = "task-123"
    """
    return mocker.MagicMock()


@pytest.fixture
def mock_websocket_manager(mocker):
    """
    Mock WebSocket manager for testing.

    Provides a mock WebSocketManager that simulates WebSocket
    operations without requiring actual WebSocket connections.

    Usage:
        async def test_something(mock_websocket_manager):
            await mock_websocket_manager.emit_event(...)
    """
    manager = mocker.AsyncMock()
    manager.emit_event = mocker.AsyncMock()
    manager.broadcast = mocker.AsyncMock()
    return manager


# Utility fixtures

@pytest.fixture
def sample_dataset_data():
    """
    Provide sample dataset data for testing.

    Returns a dictionary with valid dataset fields.
    """
    return {
        "name": "Test Dataset",
        "source": "HuggingFace",
        "repo_id": "test/dataset",
        "size_bytes": 1000000,
        "status": "ready",
        "metadata": {
            "splits": ["train", "validation", "test"],
            "features": {"text": "string"},
        }
    }


@pytest.fixture
def sample_model_data():
    """
    Provide sample model data for testing.

    Returns a dictionary with valid model fields.
    """
    return {
        "name": "Test Model",
        "repo_id": "test/model",
        "architecture": "GPT-2",
        "params_count": 124000000,
        "quantization": "FP16",
        "status": "ready",
        "num_layers": 12,
        "hidden_dim": 768,
        "num_heads": 12,
        "metadata": {
            "vocab_size": 50257,
            "max_position_embeddings": 1024,
        }
    }


@pytest.fixture
def sample_training_data():
    """
    Provide sample training data for testing.

    Returns a dictionary with valid training configuration.
    """
    return {
        "encoder_type": "sparse",
        "hyperparameters": {
            "learningRate": 0.0001,
            "batchSize": 256,
            "l1Coefficient": 0.001,
            "expansionFactor": 8,
            "trainingSteps": 10000,
            "trainingLayers": [6],
            "optimizer": "AdamW",
            "lrSchedule": "cosine",
            "ghostGradPenalty": True,
        },
        "status": "initializing",
        "current_step": 0,
        "total_steps": 10000,
        "progress": 0.0,
    }
