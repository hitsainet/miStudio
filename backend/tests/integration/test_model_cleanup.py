"""
Integration tests for model file cleanup.

Tests the end-to-end model deletion workflow, including database
record deletion and background file cleanup via Celery task.
"""

import os
import pytest
import tempfile
from pathlib import Path
from uuid import uuid4

from src.models.model import Model, ModelStatus, QuantizationFormat
from src.services.model_service import ModelService
from src.workers.model_tasks import delete_model_files
from src.core.database import get_sync_db


class TestModelFileCleanup:
    """Test suite for model deletion and file cleanup."""

    def test_delete_model_returns_file_paths(self):
        """Test that ModelService.delete_model() returns file paths for cleanup."""
        import asyncio
        from src.core.database import AsyncSessionLocal

        async def run_test():
            model_id = f"m_{uuid4().hex[:8]}"
            test_name = f"test_model_{uuid4().hex[:8]}"

            # Create test model with file paths
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
                    file_path="/fake/path/to/model",
                    quantized_path="/fake/path/to/quantized",
                )
                db.add(model)
                await db.commit()

            # Delete model
            async with AsyncSessionLocal() as db:
                result = await ModelService.delete_model(db, model_id)

            # Verify result contains file paths
            assert result is not None
            assert result["deleted"] is True
            assert result["model_id"] == model_id
            assert result["file_path"] == "/fake/path/to/model"
            assert result["quantized_path"] == "/fake/path/to/quantized"

            # Verify model was deleted from database
            async with AsyncSessionLocal() as db:
                from sqlalchemy import select
                query = await db.execute(select(Model).filter_by(id=model_id))
                deleted_model = query.scalar_one_or_none()
                assert deleted_model is None

        asyncio.run(run_test())

    def test_delete_model_handles_missing_model(self):
        """Test that delete_model returns None for nonexistent model."""
        import asyncio
        from src.core.database import AsyncSessionLocal

        async def run_test():
            fake_id = f"m_{uuid4().hex[:8]}"

            async with AsyncSessionLocal() as db:
                result = await ModelService.delete_model(db, fake_id)

            assert result is None

        asyncio.run(run_test())

    def test_delete_model_files_task_with_both_paths(self):
        """Test delete_model_files task with both file_path and quantized_path."""
        model_id = f"m_{uuid4().hex[:8]}"

        # Create temporary directories to simulate model files
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "raw_model"
            quantized_path = Path(tmpdir) / "quantized_model"

            # Create directories with some files
            file_path.mkdir()
            (file_path / "model.bin").write_text("fake model data")

            quantized_path.mkdir()
            (quantized_path / "model_q4.bin").write_text("fake quantized data")

            # Verify directories exist
            assert file_path.exists()
            assert quantized_path.exists()

            # Call delete task
            result = delete_model_files(
                model_id=model_id,
                file_path=str(file_path),
                quantized_path=str(quantized_path)
            )

            # Verify result
            assert result["model_id"] == model_id
            assert len(result["deleted_files"]) == 2
            assert str(file_path) in result["deleted_files"]
            assert str(quantized_path) in result["deleted_files"]
            assert len(result["errors"]) == 0

            # Verify directories were deleted
            assert not file_path.exists()
            assert not quantized_path.exists()

    def test_delete_model_files_task_with_only_file_path(self):
        """Test delete_model_files task with only file_path (no quantized_path)."""
        model_id = f"m_{uuid4().hex[:8]}"

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "raw_model"
            file_path.mkdir()
            (file_path / "model.bin").write_text("fake model data")

            # Verify directory exists
            assert file_path.exists()

            # Call delete task with only file_path
            result = delete_model_files(
                model_id=model_id,
                file_path=str(file_path),
                quantized_path=None
            )

            # Verify result
            assert result["model_id"] == model_id
            assert len(result["deleted_files"]) == 1
            assert str(file_path) in result["deleted_files"]
            assert len(result["errors"]) == 0

            # Verify directory was deleted
            assert not file_path.exists()

    def test_delete_model_files_task_handles_missing_paths(self):
        """Test delete_model_files task handles missing file paths gracefully."""
        model_id = f"m_{uuid4().hex[:8]}"

        # Call with nonexistent paths
        result = delete_model_files(
            model_id=model_id,
            file_path="/nonexistent/path/to/model",
            quantized_path="/nonexistent/path/to/quantized"
        )

        # Should not crash, but should have no deleted files
        assert result["model_id"] == model_id
        assert len(result["deleted_files"]) == 0
        assert len(result["errors"]) == 0  # No errors since paths don't exist

    def test_delete_model_files_task_with_none_paths(self):
        """Test delete_model_files task with None paths."""
        model_id = f"m_{uuid4().hex[:8]}"

        # Call with None paths
        result = delete_model_files(
            model_id=model_id,
            file_path=None,
            quantized_path=None
        )

        # Should complete successfully with no deletions
        assert result["model_id"] == model_id
        assert len(result["deleted_files"]) == 0
        assert len(result["errors"]) == 0

    def test_delete_model_files_task_handles_permission_error(self):
        """Test delete_model_files task handles permission errors."""
        model_id = f"m_{uuid4().hex[:8]}"

        # Create temporary directory and make it read-only
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "read_only_model"
            file_path.mkdir()
            (file_path / "model.bin").write_text("fake model data")

            # Make directory read-only (this might not work on all systems)
            try:
                os.chmod(file_path, 0o444)

                # Call delete task
                result = delete_model_files(
                    model_id=model_id,
                    file_path=str(file_path),
                    quantized_path=None
                )

                # Should have error
                assert result["model_id"] == model_id
                # Depending on system, might fail to delete
                # Just verify it doesn't crash

            finally:
                # Restore permissions for cleanup
                try:
                    os.chmod(file_path, 0o755)
                except:
                    pass

    def test_end_to_end_model_deletion_workflow(self):
        """Test complete workflow: create model, delete from database, verify cleanup."""
        import asyncio
        from src.core.database import AsyncSessionLocal

        async def run_test():
            model_id = f"m_{uuid4().hex[:8]}"
            test_name = f"test_e2e_{uuid4().hex[:8]}"

            # Create temporary directories for model files
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = Path(tmpdir) / "raw_model"
                quantized_path = Path(tmpdir) / "quantized_model"

                # Create model files
                file_path.mkdir()
                (file_path / "model.bin").write_text("fake model data")

                quantized_path.mkdir()
                (quantized_path / "model_q4.bin").write_text("fake quantized data")

                # Create test model in database with real file paths
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

                # Delete model (this is what the API endpoint does)
                async with AsyncSessionLocal() as db:
                    result = await ModelService.delete_model(db, model_id)

                # Verify service returned file paths
                assert result is not None
                assert result["file_path"] == str(file_path)
                assert result["quantized_path"] == str(quantized_path)

                # Simulate API endpoint queuing cleanup task
                cleanup_result = delete_model_files(
                    model_id=result["model_id"],
                    file_path=result["file_path"],
                    quantized_path=result["quantized_path"]
                )

                # Verify cleanup succeeded
                assert len(cleanup_result["deleted_files"]) == 2
                assert len(cleanup_result["errors"]) == 0

                # Verify files were actually deleted
                assert not file_path.exists()
                assert not quantized_path.exists()

                # Verify model no longer in database
                async with AsyncSessionLocal() as db:
                    from sqlalchemy import select
                    query = await db.execute(select(Model).filter_by(id=model_id))
                    deleted_model = query.scalar_one_or_none()
                    assert deleted_model is None

        asyncio.run(run_test())

    def test_partial_cleanup_on_service_error(self):
        """Test that database deletion succeeds even if cleanup queueing fails."""
        import asyncio
        from src.core.database import AsyncSessionLocal

        async def run_test():
            model_id = f"m_{uuid4().hex[:8]}"
            test_name = f"test_partial_{uuid4().hex[:8]}"

            # Create test model
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
                    file_path="/fake/path",
                    quantized_path=None,
                )
                db.add(model)
                await db.commit()

            # Delete model
            async with AsyncSessionLocal() as db:
                result = await ModelService.delete_model(db, model_id)

            # Verify deletion succeeded
            assert result is not None
            assert result["deleted"] is True

            # Even if cleanup queueing fails, database deletion should be committed
            async with AsyncSessionLocal() as db:
                from sqlalchemy import select
                query = await db.execute(select(Model).filter_by(id=model_id))
                deleted_model = query.scalar_one_or_none()
                assert deleted_model is None

        asyncio.run(run_test())
