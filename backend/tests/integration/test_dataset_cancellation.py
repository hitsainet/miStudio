"""
Integration tests for dataset cancellation feature.

Tests the end-to-end dataset cancellation workflow, including:
- Database record updates
- File cleanup
- WebSocket notifications
- API endpoint behavior
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch, MagicMock

from src.models.dataset import Dataset, DatasetStatus
from src.services.dataset_service import DatasetService
from src.workers.dataset_tasks import cancel_dataset_download
from src.core.database import AsyncSessionLocal, get_sync_db


class TestDatasetCancellationTask:
    """Test suite for cancel_dataset_download task."""

    def test_cancel_dataset_with_both_paths(self):
        """Test cancelling dataset with both raw and tokenized paths."""
        dataset_id = uuid4()

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw_data"
            tokenized_path = Path(tmpdir) / "tokenized_data"

            # Create directories with files
            raw_path.mkdir()
            (raw_path / "data.arrow").write_text("fake raw data")

            tokenized_path.mkdir()
            (tokenized_path / "tokenized.arrow").write_text("fake tokenized data")

            # Create dataset in database with DOWNLOADING status
            with get_sync_db() as db:
                dataset = Dataset(
                    id=dataset_id,
                    name=f"test_cancel_{uuid4().hex[:8]}",
                    source="HuggingFace",
                    status=DatasetStatus.DOWNLOADING,
                    raw_path=str(raw_path),
                    tokenized_path=str(tokenized_path),
                )
                db.add(dataset)
                db.commit()

            # Verify files exist
            assert raw_path.exists()
            assert tokenized_path.exists()

            # Cancel the dataset
            result = cancel_dataset_download(dataset_id=str(dataset_id))

            # Verify result
            assert result["dataset_id"] == str(dataset_id)
            assert result["status"] == "cancelled"

            # Verify files were deleted
            assert not raw_path.exists()
            assert not tokenized_path.exists()

            # Verify database was updated
            with get_sync_db() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                assert dataset.status == DatasetStatus.ERROR
                assert dataset.error_message == "Cancelled by user"
                assert dataset.progress == 0.0

            # Cleanup
            with get_sync_db() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                if dataset:
                    db.delete(dataset)
                    db.commit()

    def test_cancel_dataset_with_only_raw_path(self):
        """Test cancelling dataset with only raw_path (no tokenized_path)."""
        dataset_id = uuid4()

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw_data"
            raw_path.mkdir()
            (raw_path / "data.arrow").write_text("fake raw data")

            # Create dataset in database
            with get_sync_db() as db:
                dataset = Dataset(
                    id=dataset_id,
                    name=f"test_cancel_{uuid4().hex[:8]}",
                    source="HuggingFace",
                    status=DatasetStatus.DOWNLOADING,
                    raw_path=str(raw_path),
                    tokenized_path=None,
                )
                db.add(dataset)
                db.commit()

            assert raw_path.exists()

            # Cancel the dataset
            result = cancel_dataset_download(dataset_id=str(dataset_id))

            # Verify result
            assert result["dataset_id"] == str(dataset_id)
            assert result["status"] == "cancelled"

            # Verify file was deleted
            assert not raw_path.exists()

            # Cleanup
            with get_sync_db() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                if dataset:
                    db.delete(dataset)
                    db.commit()

    def test_cancel_dataset_processing_status(self):
        """Test cancelling dataset with PROCESSING status."""
        dataset_id = uuid4()

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw_data"
            tokenized_path = Path(tmpdir) / "tokenized_data"

            raw_path.mkdir()
            (raw_path / "data.arrow").write_text("fake raw data")

            tokenized_path.mkdir()
            (tokenized_path / "tokenized.arrow").write_text("fake tokenized data")

            # Create dataset with PROCESSING status
            with get_sync_db() as db:
                dataset = Dataset(
                    id=dataset_id,
                    name=f"test_cancel_processing_{uuid4().hex[:8]}",
                    source="HuggingFace",
                    status=DatasetStatus.PROCESSING,
                    raw_path=str(raw_path),
                    tokenized_path=str(tokenized_path),
                )
                db.add(dataset)
                db.commit()

            # Cancel the dataset
            result = cancel_dataset_download(dataset_id=str(dataset_id))

            # Verify cancellation succeeded
            assert result["dataset_id"] == str(dataset_id)
            assert result["status"] == "cancelled"

            # Verify files were deleted
            assert not raw_path.exists()
            assert not tokenized_path.exists()

            # Cleanup
            with get_sync_db() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                if dataset:
                    db.delete(dataset)
                    db.commit()

    def test_cancel_dataset_handles_missing_dataset(self):
        """Test that cancel returns error for nonexistent dataset."""
        fake_id = str(uuid4())

        result = cancel_dataset_download(dataset_id=fake_id)

        assert "error" in result
        assert fake_id in result["error"]

    def test_cancel_dataset_handles_wrong_status(self):
        """Test that cancel returns error for dataset not in cancellable state."""
        dataset_id = uuid4()

        # Create dataset with READY status (not cancellable)
        with get_sync_db() as db:
            dataset = Dataset(
                id=dataset_id,
                name=f"test_wrong_status_{uuid4().hex[:8]}",
                source="HuggingFace",
                status=DatasetStatus.READY,
            )
            db.add(dataset)
            db.commit()

        # Try to cancel
        result = cancel_dataset_download(dataset_id=str(dataset_id))

        # Verify error returned
        assert "error" in result
        assert "not in a cancellable state" in result["error"]

        # Cleanup
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()

    def test_cancel_dataset_handles_missing_paths(self):
        """Test cancel handles missing file paths gracefully."""
        dataset_id = uuid4()

        # Create dataset with nonexistent paths
        with get_sync_db() as db:
            dataset = Dataset(
                id=dataset_id,
                name=f"test_missing_paths_{uuid4().hex[:8]}",
                source="HuggingFace",
                status=DatasetStatus.DOWNLOADING,
                raw_path="/nonexistent/path/to/raw",
                tokenized_path="/nonexistent/path/to/tokenized",
            )
            db.add(dataset)
            db.commit()

        # Cancel the dataset
        result = cancel_dataset_download(dataset_id=str(dataset_id))

        # Should succeed (no crash) even though files don't exist
        assert result["dataset_id"] == str(dataset_id)
        assert result["status"] == "cancelled"

        # Verify database was updated
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            assert dataset.status == DatasetStatus.ERROR
            assert dataset.error_message == "Cancelled by user"

        # Cleanup
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()

    def test_cancel_dataset_with_none_paths(self):
        """Test cancel handles None paths."""
        dataset_id = uuid4()

        # Create dataset with None paths
        with get_sync_db() as db:
            dataset = Dataset(
                id=dataset_id,
                name=f"test_none_paths_{uuid4().hex[:8]}",
                source="HuggingFace",
                status=DatasetStatus.DOWNLOADING,
                raw_path=None,
                tokenized_path=None,
            )
            db.add(dataset)
            db.commit()

        # Cancel the dataset
        result = cancel_dataset_download(dataset_id=str(dataset_id))

        # Should succeed with no deletions
        assert result["dataset_id"] == str(dataset_id)
        assert result["status"] == "cancelled"

        # Cleanup
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()

    @patch("src.workers.dataset_tasks.emit_dataset_progress")
    def test_cancel_dataset_sends_websocket_notification(self, mock_emit):
        """Test that cancellation sends WebSocket notification."""
        dataset_id = uuid4()

        # Create dataset
        with get_sync_db() as db:
            dataset = Dataset(
                id=dataset_id,
                name=f"test_websocket_{uuid4().hex[:8]}",
                source="HuggingFace",
                status=DatasetStatus.DOWNLOADING,
            )
            db.add(dataset)
            db.commit()

        # Cancel the dataset
        cancel_dataset_download(dataset_id=str(dataset_id))

        # Verify WebSocket emission was called
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args

        # Verify arguments
        assert call_args[0][0] == str(dataset_id)  # dataset_id
        assert call_args[0][1] == "error"  # event
        data = call_args[0][2]
        assert data["dataset_id"] == str(dataset_id)
        assert data["progress"] == 0.0
        assert data["status"] == "error"
        assert "cancelled" in data["message"].lower()

        # Cleanup
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()


class TestDatasetCancellationAPI:
    """Test suite for dataset cancellation API endpoint."""

    @pytest.mark.asyncio
    async def test_cancel_endpoint_success(self):
        """Test successful cancellation via API endpoint."""
        from fastapi.testclient import TestClient
        from src.main import app

        dataset_id = uuid4()

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw_data"
            raw_path.mkdir()
            (raw_path / "data.arrow").write_text("fake raw data")

            # Create dataset in database
            async with AsyncSessionLocal() as db:
                dataset = Dataset(
                    id=dataset_id,
                    name=f"test_api_cancel_{uuid4().hex[:8]}",
                    source="HuggingFace",
                    status=DatasetStatus.DOWNLOADING,
                    raw_path=str(raw_path),
                )
                db.add(dataset)
                await db.commit()

            # Call API endpoint
            with TestClient(app) as client:
                response = client.delete(f"/api/v1/datasets/{dataset_id}/cancel")

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["dataset_id"] == str(dataset_id)
            assert data["status"] == "cancelled"

            # Verify database was updated
            async with AsyncSessionLocal() as db:
                from sqlalchemy import select
                result = await db.execute(select(Dataset).filter_by(id=dataset_id))
                dataset = result.scalar_one_or_none()
                assert dataset.status == DatasetStatus.ERROR
                assert dataset.error_message == "Cancelled by user"

            # Cleanup
            async with AsyncSessionLocal() as db:
                from sqlalchemy import select
                result = await db.execute(select(Dataset).filter_by(id=dataset_id))
                dataset = result.scalar_one_or_none()
                if dataset:
                    await db.delete(dataset)
                    await db.commit()

    @pytest.mark.asyncio
    async def test_cancel_endpoint_not_found(self):
        """Test cancellation endpoint with nonexistent dataset."""
        from fastapi.testclient import TestClient
        from src.main import app

        fake_id = uuid4()

        with TestClient(app) as client:
            response = client.delete(f"/api/v1/datasets/{fake_id}/cancel")

        # Verify 404 error
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_cancel_endpoint_wrong_status(self):
        """Test cancellation endpoint with dataset in non-cancellable state."""
        from fastapi.testclient import TestClient
        from src.main import app

        dataset_id = uuid4()

        # Create dataset with READY status
        async with AsyncSessionLocal() as db:
            dataset = Dataset(
                id=dataset_id,
                name=f"test_api_wrong_status_{uuid4().hex[:8]}",
                source="HuggingFace",
                status=DatasetStatus.READY,
            )
            db.add(dataset)
            await db.commit()

        try:
            with TestClient(app) as client:
                response = client.delete(f"/api/v1/datasets/{dataset_id}/cancel")

            # Verify 400 error
            assert response.status_code == 400
            assert "cannot be cancelled" in response.json()["detail"].lower()

        finally:
            # Cleanup
            async with AsyncSessionLocal() as db:
                from sqlalchemy import select
                result = await db.execute(select(Dataset).filter_by(id=dataset_id))
                dataset = result.scalar_one_or_none()
                if dataset:
                    await db.delete(dataset)
                    await db.commit()


class TestDatasetCancellationIntegration:
    """End-to-end integration tests for dataset cancellation."""

    @pytest.mark.asyncio
    async def test_complete_cancellation_workflow(self):
        """Test complete workflow: create dataset, cancel, verify cleanup."""
        dataset_id = uuid4()

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw_data"
            tokenized_path = Path(tmpdir) / "tokenized_data"

            # Create files
            raw_path.mkdir()
            (raw_path / "data.arrow").write_text("fake raw data")

            tokenized_path.mkdir()
            (tokenized_path / "tokenized.arrow").write_text("fake tokenized data")

            # Create dataset in database
            async with AsyncSessionLocal() as db:
                dataset = Dataset(
                    id=dataset_id,
                    name=f"test_e2e_cancel_{uuid4().hex[:8]}",
                    source="HuggingFace",
                    status=DatasetStatus.PROCESSING,
                    raw_path=str(raw_path),
                    tokenized_path=str(tokenized_path),
                    progress=50.0,
                )
                db.add(dataset)
                await db.commit()

            # Verify files exist
            assert raw_path.exists()
            assert tokenized_path.exists()

            # Cancel via task (simulating API call)
            result = cancel_dataset_download(dataset_id=str(dataset_id))

            # Verify result
            assert result["status"] == "cancelled"

            # Verify files were deleted
            assert not raw_path.exists()
            assert not tokenized_path.exists()

            # Verify database state
            async with AsyncSessionLocal() as db:
                from sqlalchemy import select
                query = await db.execute(select(Dataset).filter_by(id=dataset_id))
                dataset = query.scalar_one_or_none()

                assert dataset is not None
                assert dataset.status == DatasetStatus.ERROR
                assert dataset.error_message == "Cancelled by user"
                assert dataset.progress == 0.0

            # Cleanup
            async with AsyncSessionLocal() as db:
                from sqlalchemy import select
                result = await db.execute(select(Dataset).filter_by(id=dataset_id))
                dataset = result.scalar_one_or_none()
                if dataset:
                    await db.delete(dataset)
                    await db.commit()
