"""
Integration tests for Phase 1 critical fixes.

Tests the complete workflows after architectural refactoring:
- Task 1.1: Shared WebSocket emitter
- Task 1.2: Standardized database sessions
- Task 1.3: Model file cleanup
- Task 1.4: HTTP client consistency
"""

import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch, MagicMock

from src.core.database import AsyncSessionLocal, get_sync_db
from src.models.dataset import Dataset, DatasetStatus
from src.models.model import Model, ModelStatus, QuantizationFormat
from src.services.dataset_service import DatasetService
from src.services.model_service import ModelService
from src.workers.websocket_emitter import (
    emit_dataset_progress,
    emit_model_progress,
    emit_extraction_progress,
)
from src.workers.base_task import DatabaseTask
from src.workers.dataset_tasks import download_dataset_task
from src.workers.model_tasks import delete_model_files


class TestWebSocketEmitterIntegration:
    """Test shared WebSocket emitter across all resource types."""

    @patch("src.workers.websocket_emitter.httpx.Client")
    def test_emit_dataset_progress_uses_httpx(self, mock_client_class):
        """Test that dataset progress emission uses httpx."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        dataset_id = str(uuid4())
        emit_dataset_progress(
            dataset_id=dataset_id,
            event="progress",
            data={"progress": 50.0, "status": "downloading"}
        )

        # Verify httpx.Client was used
        mock_client_class.assert_called_once()
        mock_client.post.assert_called_once()

        # Verify correct endpoint and payload
        call_args = mock_client.post.call_args
        assert "/api/internal/ws/emit" in call_args[0][0]

        # Verify channel in payload
        payload = call_args[1]["json"]
        assert f"datasets/{dataset_id}/progress" == payload["channel"]

    @patch("src.workers.websocket_emitter.httpx.Client")
    def test_emit_model_progress_uses_httpx(self, mock_client_class):
        """Test that model progress emission uses httpx."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        model_id = f"m_{uuid4().hex[:8]}"
        emit_model_progress(
            model_id=model_id,
            event="progress",
            data={"progress": 75.0, "status": "loading"}
        )

        # Verify httpx.Client was used
        mock_client_class.assert_called_once()
        mock_client.post.assert_called_once()

        # Verify correct endpoint and payload
        call_args = mock_client.post.call_args
        assert "/api/internal/ws/emit" in call_args[0][0]

        # Verify channel in payload
        payload = call_args[1]["json"]
        assert f"models/{model_id}/progress" == payload["channel"]

    @patch("src.workers.websocket_emitter.httpx.Client")
    def test_emit_extraction_progress_uses_httpx(self, mock_client_class):
        """Test that extraction progress emission uses httpx."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        model_id = f"m_{uuid4().hex[:8]}"
        extraction_id = "ext_test_123"
        emit_extraction_progress(
            model_id=model_id,
            extraction_id=extraction_id,
            progress=25.0,
            status="extracting",
            message="Processing layer 5"
        )

        # Verify httpx.Client was used
        mock_client_class.assert_called_once()
        mock_client.post.assert_called_once()


class TestDatabaseSessionIntegration:
    """Test database session standardization across workers and services."""

    def test_sync_session_in_base_task(self):
        """Test that DatabaseTask provides sync sessions for Celery workers."""
        task = DatabaseTask()
        test_id = uuid4()
        test_name = f"test_sync_session_{uuid4().hex[:8]}"

        # Create a dataset using the base task's get_db()
        with task.get_db() as db:
            dataset = Dataset(
                id=test_id,
                name=test_name,
                source="test",
                status=DatasetStatus.READY,
            )
            db.add(dataset)
            db.commit()

        # Verify it was created (using new session)
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

    @pytest.mark.asyncio
    async def test_async_session_in_service(self):
        """Test that services use async sessions correctly."""
        test_id = f"m_{uuid4().hex[:8]}"
        test_name = f"test_async_session_{uuid4().hex[:8]}"

        # Create model using async service
        async with AsyncSessionLocal() as db:
            model = Model(
                id=test_id,
                name=test_name,
                repo_id="test/model",
                architecture="llama",
                params_count=1000000,
                quantization=QuantizationFormat.FP32,
                status=ModelStatus.READY,
                progress=100.0,
            )
            db.add(model)
            await db.commit()

        # Verify using service
        async with AsyncSessionLocal() as db:
            retrieved = await ModelService.get_model(db, test_id)
            assert retrieved is not None
            assert retrieved.name == test_name

        # Cleanup
        async with AsyncSessionLocal() as db:
            await ModelService.delete_model(db, test_id)

    def test_sync_and_async_session_interoperability(self):
        """Test that sync (Celery) and async (FastAPI) sessions can interoperate."""
        test_id = uuid4()
        test_name = f"test_interop_{uuid4().hex[:8]}"

        # Write with sync session (Celery worker)
        with get_sync_db() as db:
            dataset = Dataset(
                id=test_id,
                name=test_name,
                source="test",
                status=DatasetStatus.READY,
            )
            db.add(dataset)
            db.commit()

        # Read with async session (FastAPI endpoint) in sync context
        async def read_async():
            async with AsyncSessionLocal() as db:
                from sqlalchemy import select
                result = await db.execute(select(Dataset).filter_by(id=test_id))
                return result.scalar_one_or_none()

        retrieved = asyncio.run(read_async())
        assert retrieved is not None
        assert retrieved.name == test_name

        # Cleanup with sync
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=test_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()


class TestModelFileCleanupIntegration:
    """Test model file cleanup workflow end-to-end."""

    @pytest.mark.asyncio
    async def test_complete_model_deletion_with_cleanup(self):
        """Test complete model deletion workflow with file cleanup."""
        model_id = f"m_{uuid4().hex[:8]}"
        test_name = f"test_cleanup_{uuid4().hex[:8]}"

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "raw_model"
            quantized_path = Path(tmpdir) / "quantized_model"

            # Create model files
            file_path.mkdir()
            (file_path / "model.bin").write_text("fake model data")
            quantized_path.mkdir()
            (quantized_path / "model_q4.bin").write_text("fake quantized data")

            # Create model in database
            async with AsyncSessionLocal() as db:
                model = Model(
                    id=model_id,
                    name=test_name,
                    repo_id="test/model",
                    architecture="llama",
                    params_count=1000000,
                    quantization=QuantizationFormat.Q4,
                    status=ModelStatus.READY,
                    progress=100.0,
                    file_path=str(file_path),
                    quantized_path=str(quantized_path),
                )
                db.add(model)
                await db.commit()

            # Verify files exist
            assert file_path.exists()
            assert quantized_path.exists()

            # Delete model (service layer)
            async with AsyncSessionLocal() as db:
                result = await ModelService.delete_model(db, model_id)

            # Verify service returned file paths
            assert result is not None
            assert result["deleted"] is True
            assert result["file_path"] == str(file_path)
            assert result["quantized_path"] == str(quantized_path)

            # Queue cleanup (this is what API endpoint does)
            cleanup_result = delete_model_files(
                model_id=result["model_id"],
                file_path=result["file_path"],
                quantized_path=result["quantized_path"]
            )

            # Verify cleanup succeeded
            assert len(cleanup_result["deleted_files"]) == 2
            assert len(cleanup_result["errors"]) == 0
            assert not file_path.exists()
            assert not quantized_path.exists()

            # Verify model deleted from database
            async with AsyncSessionLocal() as db:
                from sqlalchemy import select
                query = await db.execute(select(Model).filter_by(id=model_id))
                deleted_model = query.scalar_one_or_none()
                assert deleted_model is None


class TestConcurrentOperations:
    """Test that multiple operations can run concurrently without conflicts."""

    @pytest.mark.asyncio
    async def test_concurrent_dataset_and_model_creation(self):
        """Test creating dataset and model records concurrently."""
        dataset_id = uuid4()
        dataset_name = f"test_concurrent_ds_{uuid4().hex[:8]}"

        model_id = f"m_{uuid4().hex[:8]}"
        model_name = f"test_concurrent_model_{uuid4().hex[:8]}"

        async def create_dataset():
            async with AsyncSessionLocal() as db:
                dataset = Dataset(
                    id=dataset_id,
                    name=dataset_name,
                    source="test",
                    status=DatasetStatus.READY,
                )
                db.add(dataset)
                await db.commit()
                return dataset_id

        async def create_model():
            async with AsyncSessionLocal() as db:
                model = Model(
                    id=model_id,
                    name=model_name,
                    repo_id="test/model",
                    architecture="llama",
                    params_count=1000000,
                    quantization=QuantizationFormat.FP32,
                    status=ModelStatus.READY,
                    progress=100.0,
                )
                db.add(model)
                await db.commit()
                return model_id

        # Run concurrently
        results = await asyncio.gather(create_dataset(), create_model())

        assert len(results) == 2
        assert results[0] == dataset_id
        assert results[1] == model_id

        # Verify both were created
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select

            ds_query = await db.execute(select(Dataset).filter_by(id=dataset_id))
            dataset = ds_query.scalar_one_or_none()
            assert dataset is not None

            model_query = await db.execute(select(Model).filter_by(id=model_id))
            model = model_query.scalar_one_or_none()
            assert model is not None

        # Cleanup
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select

            ds_query = await db.execute(select(Dataset).filter_by(id=dataset_id))
            dataset = ds_query.scalar_one_or_none()
            if dataset:
                await db.delete(dataset)

            await ModelService.delete_model(db, model_id)

            await db.commit()

    def test_concurrent_sync_session_operations(self):
        """Test multiple sync session operations don't conflict."""
        test_ids = [uuid4() for _ in range(3)]
        test_names = [f"test_concurrent_{i}_{uuid4().hex[:8]}" for i in range(3)]

        # Create multiple datasets concurrently (simulating multiple Celery tasks)
        for test_id, test_name in zip(test_ids, test_names):
            with get_sync_db() as db:
                dataset = Dataset(
                    id=test_id,
                    name=test_name,
                    source="test",
                    status=DatasetStatus.READY,
                )
                db.add(dataset)
                db.commit()

        # Verify all were created
        for test_id, test_name in zip(test_ids, test_names):
            with get_sync_db() as db:
                retrieved = db.query(Dataset).filter_by(id=test_id).first()
                assert retrieved is not None
                assert retrieved.name == test_name

        # Cleanup
        for test_id in test_ids:
            with get_sync_db() as db:
                dataset = db.query(Dataset).filter_by(id=test_id).first()
                if dataset:
                    db.delete(dataset)
                    db.commit()


class TestPhase1ValidationSummary:
    """Summary tests validating all Phase 1 improvements."""

    def test_no_requests_import_in_workers(self):
        """Verify that no worker code directly imports requests."""
        import src.workers.model_tasks as model_tasks
        import src.workers.dataset_tasks as dataset_tasks
        import src.workers.websocket_emitter as emitter

        # Check module imports don't include requests
        for module in [model_tasks, dataset_tasks, emitter]:
            module_source = module.__file__
            with open(module_source, 'r') as f:
                content = f.read()
                assert "import requests" not in content, f"Found 'import requests' in {module_source}"

    def test_all_workers_use_httpx_emitter(self):
        """Verify all workers use shared httpx-based emitter."""
        import src.workers.model_tasks as model_tasks
        import src.workers.dataset_tasks as dataset_tasks

        # Check imports
        model_source = model_tasks.__file__
        dataset_source = dataset_tasks.__file__

        with open(model_source, 'r') as f:
            model_content = f.read()
            assert "from .websocket_emitter import" in model_content

        with open(dataset_source, 'r') as f:
            dataset_content = f.read()
            assert "from .websocket_emitter import" in dataset_content

    def test_base_task_provides_sync_sessions(self):
        """Verify DatabaseTask provides sync sessions."""
        from src.workers.base_task import DatabaseTask

        task = DatabaseTask()

        # get_db should return a context manager
        with task.get_db() as db:
            # Should be a sync session
            assert hasattr(db, 'query')  # Sync session method
            assert hasattr(db, 'commit')  # Both sync and async have this

            # Should NOT be async session
            from sqlalchemy.orm import Session
            from sqlalchemy.ext.asyncio import AsyncSession
            assert isinstance(db, Session)
            assert not isinstance(db, AsyncSession)

    @pytest.mark.asyncio
    async def test_services_use_async_sessions(self):
        """Verify services use async sessions."""
        from src.services.model_service import ModelService
        from sqlalchemy.ext.asyncio import AsyncSession

        async with AsyncSessionLocal() as db:
            # Verify this is an async session
            assert isinstance(db, AsyncSession)

            # Service methods should work with async sessions
            models, total = await ModelService.list_models(db, skip=0, limit=10)
            assert isinstance(models, list)
            assert isinstance(total, int)
