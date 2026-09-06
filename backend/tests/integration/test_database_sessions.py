"""
Integration tests for database session management.

Tests the sync and async session factories, context managers,
and the DatabaseTask base class.
"""

import pytest
from uuid import uuid4

from src.core.database import (
    get_sync_db,
    AsyncSessionLocal,
    sync_engine,
    engine as async_engine,
)
from src.models.dataset import Dataset, DatasetStatus
from src.workers.base_task import DatabaseTask


class TestSyncSessions:
    """Test suite for synchronous database sessions (Celery workers)."""

    def test_sync_session_context_manager(self):
        """Test that sync session context manager works correctly."""
        # Create a test dataset using sync session
        test_id = uuid4()
        test_name = f"test_sync_dataset_{uuid4().hex[:8]}"

        with get_sync_db() as db:
            dataset = Dataset(
                id=test_id,
                name=test_name,
                source="test",
                status=DatasetStatus.READY,
            )
            db.add(dataset)
            db.commit()

        # Verify dataset was created (using new session)
        with get_sync_db() as db:
            retrieved = db.query(Dataset).filter_by(id=test_id).first()
            assert retrieved is not None
            assert retrieved.name == test_name
            assert retrieved.status == DatasetStatus.READY

        # Cleanup
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=test_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()

    def test_sync_session_rollback_on_error(self):
        """Test that sync session rolls back on exception."""
        test_id = uuid4()
        test_name = f"test_rollback_{uuid4().hex[:8]}"

        try:
            with get_sync_db() as db:
                dataset = Dataset(
                    id=test_id,
                    name=test_name,
                    source="test",
                    status=DatasetStatus.READY,
                )
                db.add(dataset)
                db.flush()  # Write to DB but don't commit

                # Raise an error to trigger rollback
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify dataset was NOT created (rollback occurred)
        with get_sync_db() as db:
            retrieved = db.query(Dataset).filter_by(id=test_id).first()
            assert retrieved is None

    def test_sync_session_multiple_operations(self):
        """Test multiple database operations in single sync session."""
        test_ids = [uuid4(), uuid4(), uuid4()]
        test_names = [f"test_multi_{i}_{uuid4().hex[:8]}" for i in range(3)]

        # Create multiple datasets in one transaction
        with get_sync_db() as db:
            for test_id, test_name in zip(test_ids, test_names):
                dataset = Dataset(
                    id=test_id,
                    name=test_name,
                    source="test",
                    status=DatasetStatus.READY,
                )
                db.add(dataset)
            db.commit()

        # Verify all were created
        with get_sync_db() as db:
            for test_id, test_name in zip(test_ids, test_names):
                retrieved = db.query(Dataset).filter_by(id=test_id).first()
                assert retrieved is not None
                assert retrieved.name == test_name

        # Cleanup
        with get_sync_db() as db:
            for test_id in test_ids:
                dataset = db.query(Dataset).filter_by(id=test_id).first()
                if dataset:
                    db.delete(dataset)
            db.commit()

    def test_sync_engine_connection_pool(self):
        """Test that sync engine has proper connection pooling configured."""
        assert sync_engine.pool.size() == 5  # pool_size=5
        assert sync_engine.pool._max_overflow == 10  # max_overflow=10

    def test_sync_session_isolation(self):
        """Test that sync sessions are properly isolated."""
        test_id = uuid4()
        test_name = f"test_isolation_{uuid4().hex[:8]}"

        # Session 1: Create and commit
        with get_sync_db() as db1:
            dataset = Dataset(
                id=test_id,
                name=test_name,
                source="test",
                status=DatasetStatus.READY,
            )
            db1.add(dataset)
            db1.commit()

        # Session 2: Read (should see committed data)
        with get_sync_db() as db2:
            retrieved = db2.query(Dataset).filter_by(id=test_id).first()
            assert retrieved is not None
            assert retrieved.name == test_name

        # Session 3: Update
        with get_sync_db() as db3:
            dataset = db3.query(Dataset).filter_by(id=test_id).first()
            dataset.status = DatasetStatus.PROCESSING
            db3.commit()

        # Session 4: Verify update
        with get_sync_db() as db4:
            retrieved = db4.query(Dataset).filter_by(id=test_id).first()
            assert retrieved.status == DatasetStatus.PROCESSING

        # Cleanup
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=test_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()


class TestAsyncSessions:
    """Test suite for async database sessions (FastAPI endpoints)."""

    @pytest.mark.asyncio
    async def test_async_session_context_manager(self):
        """Test that async session context manager works correctly."""
        test_id = uuid4()
        test_name = f"test_async_dataset_{uuid4().hex[:8]}"

        # Create dataset using async session
        async with AsyncSessionLocal() as db:
            dataset = Dataset(
                id=test_id,
                name=test_name,
                source="test",
                status=DatasetStatus.READY,
            )
            db.add(dataset)
            await db.commit()

        # Verify dataset was created
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            result = await db.execute(select(Dataset).filter_by(id=test_id))
            retrieved = result.scalar_one_or_none()
            assert retrieved is not None
            assert retrieved.name == test_name

        # Cleanup
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Dataset).filter_by(id=test_id))
            dataset = result.scalar_one_or_none()
            if dataset:
                await db.delete(dataset)
                await db.commit()

    @pytest.mark.asyncio
    async def test_async_session_rollback_on_error(self):
        """Test that async session rolls back on exception."""
        test_id = uuid4()
        test_name = f"test_async_rollback_{uuid4().hex[:8]}"

        try:
            async with AsyncSessionLocal() as db:
                dataset = Dataset(
                    id=test_id,
                    name=test_name,
                    source="test",
                    status=DatasetStatus.READY,
                )
                db.add(dataset)
                await db.flush()

                # Raise an error to trigger rollback
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify dataset was NOT created
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            result = await db.execute(select(Dataset).filter_by(id=test_id))
            retrieved = result.scalar_one_or_none()
            assert retrieved is None

    @pytest.mark.asyncio
    async def test_async_engine_no_pool(self):
        """Test that async engine uses NullPool (async-safe)."""
        # Async engines should use NullPool for async safety
        assert async_engine.pool.__class__.__name__ == "NullPool"


class TestDatabaseTaskBase:
    """Test suite for DatabaseTask base class."""

    def test_database_task_get_db(self):
        """Test that DatabaseTask.get_db() returns sync session context manager."""
        # Create a test task instance
        task = DatabaseTask()

        # Test get_db returns working context manager
        test_id = uuid4()
        test_name = f"test_base_task_{uuid4().hex[:8]}"

        with task.get_db() as db:
            dataset = Dataset(
                id=test_id,
                name=test_name,
                source="test",
                status=DatasetStatus.READY,
            )
            db.add(dataset)
            db.commit()

        # Verify dataset was created
        with task.get_db() as db:
            retrieved = db.query(Dataset).filter_by(id=test_id).first()
            assert retrieved is not None
            assert retrieved.name == test_name

        # Cleanup
        with task.get_db() as db:
            dataset = db.query(Dataset).filter_by(id=test_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()

    def test_database_task_update_progress(self):
        """Test DatabaseTask.update_progress() utility method."""
        task = DatabaseTask()
        test_id = uuid4()
        test_name = f"test_progress_{uuid4().hex[:8]}"

        # Create test dataset
        with task.get_db() as db:
            dataset = Dataset(
                id=test_id,
                name=test_name,
                source="test",
                status=DatasetStatus.DOWNLOADING,
                progress=0.0,
            )
            db.add(dataset)
            db.commit()

        # Update progress using utility method
        with task.get_db() as db:
            task.update_progress(
                db=db,
                model_class=Dataset,
                record_id=str(test_id),
                progress=50.0,
                status=DatasetStatus.PROCESSING.value,
            )

        # Verify update
        with task.get_db() as db:
            retrieved = db.query(Dataset).filter_by(id=test_id).first()
            assert retrieved.progress == 50.0
            assert retrieved.status == DatasetStatus.PROCESSING

        # Cleanup
        with task.get_db() as db:
            dataset = db.query(Dataset).filter_by(id=test_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()

    def test_database_task_update_progress_extra_fields(self):
        """Test DatabaseTask.update_progress() with extra fields."""
        task = DatabaseTask()
        test_id = uuid4()
        test_name = f"test_extra_fields_{uuid4().hex[:8]}"

        # Create test dataset
        with task.get_db() as db:
            dataset = Dataset(
                id=test_id,
                name=test_name,
                source="test",
                status=DatasetStatus.DOWNLOADING,
                progress=0.0,
            )
            db.add(dataset)
            db.commit()

        # Update with extra fields
        with task.get_db() as db:
            task.update_progress(
                db=db,
                model_class=Dataset,
                record_id=str(test_id),
                progress=100.0,
                status=DatasetStatus.READY.value,
                num_samples=1000,
                size_bytes=5000000,
            )

        # Verify all fields updated
        with task.get_db() as db:
            retrieved = db.query(Dataset).filter_by(id=test_id).first()
            assert retrieved.progress == 100.0
            assert retrieved.status == DatasetStatus.READY
            assert retrieved.num_samples == 1000
            assert retrieved.size_bytes == 5000000

        # Cleanup
        with task.get_db() as db:
            dataset = db.query(Dataset).filter_by(id=test_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()

    def test_database_task_update_progress_nonexistent_record(self):
        """Test DatabaseTask.update_progress() with nonexistent record (should not crash)."""
        task = DatabaseTask()
        fake_id = str(uuid4())

        # Should handle gracefully
        with task.get_db() as db:
            task.update_progress(
                db=db,
                model_class=Dataset,
                record_id=fake_id,
                progress=50.0,
                status=DatasetStatus.PROCESSING.value,
            )
            # Should not raise exception


class TestSessionInteroperability:
    """Test suite for sync/async session interoperability."""

    @pytest.mark.asyncio
    async def test_sync_to_async_visibility(self):
        """Test that data written with sync session is visible to async session."""
        test_id = uuid4()
        test_name = f"test_interop_sync_to_async_{uuid4().hex[:8]}"

        # Write with sync session
        with get_sync_db() as db:
            dataset = Dataset(
                id=test_id,
                name=test_name,
                source="test",
                status=DatasetStatus.READY,
            )
            db.add(dataset)
            db.commit()

        # Read with async session
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            result = await db.execute(select(Dataset).filter_by(id=test_id))
            retrieved = result.scalar_one_or_none()
            assert retrieved is not None
            assert retrieved.name == test_name

        # Cleanup with sync
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=test_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()

    @pytest.mark.asyncio
    async def test_async_to_sync_visibility(self):
        """Test that data written with async session is visible to sync session."""
        test_id = uuid4()
        test_name = f"test_interop_async_to_sync_{uuid4().hex[:8]}"

        # Write with async session
        async with AsyncSessionLocal() as db:
            dataset = Dataset(
                id=test_id,
                name=test_name,
                source="test",
                status=DatasetStatus.READY,
            )
            db.add(dataset)
            await db.commit()

        # Read with sync session
        with get_sync_db() as db:
            retrieved = db.query(Dataset).filter_by(id=test_id).first()
            assert retrieved is not None
            assert retrieved.name == test_name

        # Cleanup with sync
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=test_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()
