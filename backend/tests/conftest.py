"""
Pytest configuration and shared fixtures.

This module configures pytest for async testing and provides
common fixtures used across the test suite.
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
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

    # PostgreSQL enum types required by models that use create_type=False.
    # These are NOT created by Base.metadata.create_all (create_type=False), so
    # the fixture must manage their lifecycle explicitly. Values must match the
    # model definitions exactly (label_source_enum includes 'enhanced_llm').
    enum_definitions = [
        ("export_status", ["pending", "computing", "packaging", "completed", "failed", "cancelled"]),
        ("label_source_enum", ["auto", "user", "llm", "local_llm", "openai", "enhanced_llm", "mcp_agent"]),
        ("analysis_type_enum", ["logit_lens", "correlations", "ablation", "nlp_analysis"]),
        ("extraction_status_enum", ["queued", "loading", "extracting", "saving", "completed", "failed", "cancelled"]),
    ]

    async def _drop_enums(conn):
        # Drop WITHOUT CASCADE: tables are dropped separately by drop_all, and a
        # CASCADE here would drop dependent tables out from under drop_all,
        # causing "table does not exist" on the subsequent drop_all/create_all.
        for enum_name, _ in enum_definitions:
            await conn.execute(text(f"DROP TYPE IF EXISTS {enum_name};"))

    async with engine.begin() as conn:
        # Start from a clean slate: drop any leftovers from a previous test whose
        # teardown was interrupted. Tables first (they depend on the enums), then
        # the enums. This prevents the pg_type unique-index race on CREATE TYPE.
        await conn.run_sync(Base.metadata.drop_all)
        await _drop_enums(conn)

        # Create enum types, then tables.
        for enum_name, values in enum_definitions:
            values_str = ", ".join(f"'{v}'" for v in values)
            await conn.execute(text(f"CREATE TYPE {enum_name} AS ENUM ({values_str});"))
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Teardown: tables first (depend on enums), then enums (no CASCADE).
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await _drop_enums(conn)

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

    # Use ASGITransport for httpx 0.26+ compatibility
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
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

@pytest.fixture(autouse=True)
def _no_live_redis_for_progress_markers(monkeypatch):
    """Unit tests never reach a real Redis for the janitor progress markers.

    `job_progress.progress_stalled_seconds` is supporting evidence for the
    stuck-job reapers. Left unstubbed it connects to whatever Redis the
    environment points at, so markers written by one test run leak into the
    next and a janitor test's verdict depends on run order — three NLP and
    three tokenization tests flipped exactly that way.

    Returning None means "no evidence", which is precisely the fallback the
    gate is designed for, so every janitor test written before the gate keeps
    its original clock-based semantics. A test that wants to exercise the gate
    patches `progress_stalled_seconds` in its own module, as
    test_cleanup_stuck_extractions.py does.
    """
    monkeypatch.setattr(
        "src.workers.job_progress._client", lambda: None, raising=False
    )
