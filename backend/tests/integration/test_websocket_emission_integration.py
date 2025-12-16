"""
Integration tests for WebSocket emission flows.

Tests the end-to-end flow of progress emissions from worker tasks through
the WebSocket emitter, including database state synchronization.

These tests verify:
1. Training progress emission flow (training task → database → websocket)
2. Extraction progress emission flow (extraction task → database → websocket)
3. Model download progress emission flow (model task → database → websocket)
4. Dataset progress emission flow (dataset task → database → websocket)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4
from datetime import datetime

from src.core.database import get_sync_db
from src.models.training import Training, TrainingStatus
from src.models.model import Model, ModelStatus, QuantizationFormat
from src.models.dataset import Dataset, DatasetStatus
from src.workers.websocket_emitter import (
    emit_training_progress,
    emit_extraction_progress,
    emit_extraction_failed,
    emit_model_progress,
    emit_dataset_progress,
    emit_checkpoint_created,
)


class TestTrainingProgressEmissionFlow:
    """Test training progress WebSocket emission flow."""

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_training_creation_emission_with_database(self, mock_settings, mock_client_class):
        """Test training creation with database record and WebSocket emission."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        training_id = f"train_{uuid4().hex[:8]}"
        model_id = f"m_{uuid4().hex[:8]}"
        dataset_id = str(uuid4())

        # Create prerequisite model record (foreign key constraint)
        with get_sync_db() as db:
            model = Model(
                id=model_id,
                name=f"test_model_{uuid4().hex[:8]}",
                architecture="gpt2",
                params_count=124000000,
                quantization=QuantizationFormat.FP16.value,
                status=ModelStatus.READY.value,
                progress=100.0,
            )
            db.add(model)
            db.commit()

        # Create training record in database
        with get_sync_db() as db:
            training = Training(
                id=training_id,
                dataset_id=dataset_id,
                model_id=model_id,
                status=TrainingStatus.PENDING.value,
                progress=0.0,
                current_step=0,
                total_steps=1000,
                hyperparameters={},
            )
            db.add(training)
            db.commit()

        # Emit training creation event
        result = emit_training_progress(
            training_id,
            "created",
            {
                "training_id": training_id,
                "status": "pending",
                "progress": 0.0,
                "current_step": 0,
                "total_steps": 1000,
            }
        )

        # Verify emission succeeded
        assert result is True
        mock_client.__enter__.return_value.post.assert_called_once()

        # Verify database record exists
        with get_sync_db() as db:
            retrieved = db.query(Training).filter_by(id=training_id).first()
            assert retrieved is not None
            assert retrieved.status == TrainingStatus.PENDING.value

        # Cleanup (delete training first due to foreign key constraint)
        with get_sync_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            if training:
                db.delete(training)
                db.commit()

        with get_sync_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if model:
                db.delete(model)
                db.commit()

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_training_progress_updates_with_database(self, mock_settings, mock_client_class):
        """Test training progress updates with database synchronization."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        training_id = f"train_{uuid4().hex[:8]}"
        model_id = f"m_{uuid4().hex[:8]}"
        dataset_id = str(uuid4())

        # Create prerequisite model record (foreign key constraint)
        with get_sync_db() as db:
            model = Model(
                id=model_id,
                name=f"test_model_{uuid4().hex[:8]}",
                architecture="gpt2",
                params_count=124000000,
                quantization=QuantizationFormat.FP16.value,
                status=ModelStatus.READY.value,
                progress=100.0,
            )
            db.add(model)
            db.commit()

        # Create initial training record
        with get_sync_db() as db:
            training = Training(
                id=training_id,
                dataset_id=dataset_id,
                model_id=model_id,
                status=TrainingStatus.RUNNING.value,
                progress=0.0,
                current_step=0,
                total_steps=1000,
                hyperparameters={},
            )
            db.add(training)
            db.commit()

        # Simulate training progress at 25%
        result1 = emit_training_progress(
            training_id,
            "progress",
            {
                "training_id": training_id,
                "current_step": 250,
                "total_steps": 1000,
                "progress": 25.0,
                "loss": 0.5,
                "l0_sparsity": 10.0,
            }
        )

        # Update database
        with get_sync_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            training.current_step = 250
            training.progress = 25.0
            training.current_loss = 0.5
            training.current_l0_sparsity = 10.0
            db.commit()

        # Simulate training progress at 50%
        result2 = emit_training_progress(
            training_id,
            "progress",
            {
                "training_id": training_id,
                "current_step": 500,
                "total_steps": 1000,
                "progress": 50.0,
                "loss": 0.3,
                "l0_sparsity": 8.0,
            }
        )

        # Update database again
        with get_sync_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            training.current_step = 500
            training.progress = 50.0
            training.current_loss = 0.3
            training.current_l0_sparsity = 8.0
            db.commit()

        # Verify both emissions succeeded
        assert result1 is True
        assert result2 is True
        assert mock_client.__enter__.return_value.post.call_count == 2

        # Verify final database state
        with get_sync_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            assert training.current_step == 500
            assert training.progress == 50.0
            assert training.current_loss == 0.3
            assert training.current_l0_sparsity == 8.0

        # Cleanup (delete training first due to foreign key constraint)
        with get_sync_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            if training:
                db.delete(training)
                db.commit()

        with get_sync_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if model:
                db.delete(model)
                db.commit()

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_training_completion_emission_with_database(self, mock_settings, mock_client_class):
        """Test training completion with final database state."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        training_id = f"train_{uuid4().hex[:8]}"
        model_id = f"m_{uuid4().hex[:8]}"
        dataset_id = str(uuid4())

        # Create prerequisite model record (foreign key constraint)
        with get_sync_db() as db:
            model = Model(
                id=model_id,
                name=f"test_model_{uuid4().hex[:8]}",
                architecture="gpt2",
                params_count=124000000,
                quantization=QuantizationFormat.FP16.value,
                status=ModelStatus.READY.value,
                progress=100.0,
            )
            db.add(model)
            db.commit()

        # Create training record
        with get_sync_db() as db:
            training = Training(
                id=training_id,
                dataset_id=dataset_id,
                model_id=model_id,
                status=TrainingStatus.RUNNING.value,
                progress=99.0,
                current_step=990,
                total_steps=1000,
                hyperparameters={},
            )
            db.add(training)
            db.commit()

        # Emit completion event
        result = emit_training_progress(
            training_id,
            "completed",
            {
                "training_id": training_id,
                "status": "completed",
                "progress": 100.0,
                "current_step": 1000,
                "total_steps": 1000,
                "final_loss": 0.15,
            }
        )

        # Update database to completed state
        with get_sync_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            training.status = TrainingStatus.COMPLETED.value
            training.progress = 100.0
            training.current_step = 1000
            training.completed_at = datetime.utcnow()
            db.commit()

        # Verify emission and database state
        assert result is True
        with get_sync_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            assert training.status == TrainingStatus.COMPLETED.value
            assert training.progress == 100.0
            assert training.completed_at is not None

        # Cleanup (delete training first due to foreign key constraint)
        with get_sync_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            if training:
                db.delete(training)
                db.commit()

        with get_sync_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if model:
                db.delete(model)
                db.commit()

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_checkpoint_creation_emission_with_training(self, mock_settings, mock_client_class):
        """Test checkpoint creation emission alongside training progress."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        training_id = f"train_{uuid4().hex[:8]}"
        checkpoint_id = f"ckpt_{uuid4().hex[:8]}"
        model_id = f"m_{uuid4().hex[:8]}"
        dataset_id = str(uuid4())

        # Create prerequisite model record (foreign key constraint)
        with get_sync_db() as db:
            model = Model(
                id=model_id,
                name=f"test_model_{uuid4().hex[:8]}",
                architecture="gpt2",
                params_count=124000000,
                quantization=QuantizationFormat.FP16.value,
                status=ModelStatus.READY.value,
                progress=100.0,
            )
            db.add(model)
            db.commit()

        # Create training record
        with get_sync_db() as db:
            training = Training(
                id=training_id,
                dataset_id=dataset_id,
                model_id=model_id,
                status=TrainingStatus.RUNNING.value,
                progress=50.0,
                current_step=500,
                total_steps=1000,
                hyperparameters={},
            )
            db.add(training)
            db.commit()

        # Emit checkpoint creation
        result = emit_checkpoint_created(
            training_id=training_id,
            checkpoint_id=checkpoint_id,
            step=500,
            loss=0.25,
            is_best=True,
            storage_path="/data/checkpoints/test.safetensors"
        )

        # Verify emission succeeded
        assert result is True
        mock_client.__enter__.return_value.post.assert_called_once()

        # Verify channel and event structure
        call_args = mock_client.__enter__.return_value.post.call_args
        payload = call_args[1]["json"]
        assert payload["channel"] == f"trainings/{training_id}/checkpoints"
        assert payload["event"] == "checkpoint_created"
        assert payload["data"]["checkpoint_id"] == checkpoint_id
        assert payload["data"]["step"] == 500
        assert payload["data"]["is_best"] is True

        # Cleanup (delete training first due to foreign key constraint)
        with get_sync_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            if training:
                db.delete(training)
                db.commit()

        with get_sync_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if model:
                db.delete(model)
                db.commit()


class TestExtractionProgressEmissionFlow:
    """Test activation extraction progress WebSocket emission flow."""

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_extraction_progress_emission_with_model(self, mock_settings, mock_client_class):
        """Test extraction progress updates with model database record."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        model_id = f"m_{uuid4().hex[:8]}"
        extraction_id = f"ext_{uuid4().hex[:8]}"

        # Create model record
        with get_sync_db() as db:
            model = Model(
                id=model_id,
                name=f"test_extraction_model_{uuid4().hex[:8]}",
                architecture="gpt2",
                params_count=124000000,
                quantization=QuantizationFormat.Q4.value,
                status=ModelStatus.READY.value,
                progress=100.0,
            )
            db.add(model)
            db.commit()

        # Emit extraction start
        result1 = emit_extraction_progress(
            model_id,
            extraction_id,
            0.0,
            "starting",
            "Initializing extraction..."
        )

        # Emit extraction progress
        result2 = emit_extraction_progress(
            model_id,
            extraction_id,
            50.0,
            "extracting",
            "Processing batch 5/10"
        )

        # Emit extraction completion
        result3 = emit_extraction_progress(
            model_id,
            extraction_id,
            100.0,
            "complete",
            "Extraction complete"
        )

        # Verify all emissions succeeded
        assert all([result1, result2, result3])
        assert mock_client.__enter__.return_value.post.call_count == 3

        # Verify channel structure
        for call_args in mock_client.__enter__.return_value.post.call_args_list:
            payload = call_args[1]["json"]
            assert payload["channel"] == f"models/{model_id}/extraction"
            assert payload["event"] == "extraction:progress"  # Namespaced event
            assert payload["data"]["type"] == "extraction_progress"
            assert payload["data"]["extraction_id"] == extraction_id

        # Cleanup
        with get_sync_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if model:
                db.delete(model)
                db.commit()

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_extraction_failure_emission_with_retry_params(self, mock_settings, mock_client_class):
        """Test extraction failure with error classification and retry params."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        model_id = f"m_{uuid4().hex[:8]}"
        extraction_id = f"ext_{uuid4().hex[:8]}"

        # Create model record
        with get_sync_db() as db:
            model = Model(
                id=model_id,
                name=f"test_extraction_fail_{uuid4().hex[:8]}",
                architecture="gpt2",
                params_count=124000000,
                quantization=QuantizationFormat.Q4.value,
                status=ModelStatus.READY.value,
                progress=100.0,
            )
            db.add(model)
            db.commit()

        # Emit extraction failure with OOM error
        result = emit_extraction_failed(
            model_id,
            extraction_id,
            "CUDA out of memory. Tried to allocate 2.00 GiB",
            error_type="OOM",
            suggested_retry_params={"batch_size": 4}
        )

        # Verify emission succeeded
        assert result is True
        mock_client.__enter__.return_value.post.assert_called_once()

        # Verify failure payload structure
        call_args = mock_client.__enter__.return_value.post.call_args
        payload = call_args[1]["json"]
        assert payload["channel"] == f"models/{model_id}/extraction"
        assert payload["event"] == "extraction:failed"  # Namespaced event
        assert payload["data"]["type"] == "extraction_failed"
        assert payload["data"]["error_type"] == "OOM"
        assert payload["data"]["suggested_retry_params"] == {"batch_size": 4}
        assert payload["data"]["retry_available"] is True

        # Cleanup
        with get_sync_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if model:
                db.delete(model)
                db.commit()


class TestModelDownloadProgressEmissionFlow:
    """Test model download progress WebSocket emission flow."""

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_model_download_progress_with_database(self, mock_settings, mock_client_class):
        """Test model download progress with database state updates."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        model_id = f"m_{uuid4().hex[:8]}"
        model_name = f"test_model_download_{uuid4().hex[:8]}"

        # Create model record in downloading state
        with get_sync_db() as db:
            model = Model(
                id=model_id,
                name=model_name,
                architecture="gpt2",
                params_count=124000000,
                quantization=QuantizationFormat.FP16.value,
                status=ModelStatus.DOWNLOADING.value,
                progress=0.0,
            )
            db.add(model)
            db.commit()

        # Emit download start
        result1 = emit_model_progress(
            model_id,
            "progress",
            {
                "type": "model_progress",
                "model_id": model_id,
                "progress": 10.0,
                "status": "downloading",
                "message": "Downloading model files from HuggingFace..."
            }
        )

        # Update database
        with get_sync_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            model.progress = 10.0
            db.commit()

        # Emit download progress
        result2 = emit_model_progress(
            model_id,
            "progress",
            {
                "type": "model_progress",
                "model_id": model_id,
                "progress": 70.0,
                "status": "downloading",
                "message": "Downloaded 3.5GB / 5GB"
            }
        )

        # Update database
        with get_sync_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            model.progress = 70.0
            db.commit()

        # Emit model ready
        result3 = emit_model_progress(
            model_id,
            "progress",
            {
                "type": "model_progress",
                "model_id": model_id,
                "progress": 100.0,
                "status": "ready",
                "message": "Model ready"
            }
        )

        # Update database to ready state
        with get_sync_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            model.status = ModelStatus.READY.value
            model.progress = 100.0
            db.commit()

        # Verify all emissions succeeded
        assert all([result1, result2, result3])
        assert mock_client.__enter__.return_value.post.call_count == 3

        # Verify final database state
        with get_sync_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            assert model.status == ModelStatus.READY.value
            assert model.progress == 100.0

        # Cleanup
        with get_sync_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if model:
                db.delete(model)
                db.commit()


class TestDatasetProgressEmissionFlow:
    """Test dataset download/processing progress WebSocket emission flow."""

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_dataset_download_progress_with_database(self, mock_settings, mock_client_class):
        """Test dataset download progress with database state updates."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        dataset_id = uuid4()
        dataset_name = f"test_dataset_download_{uuid4().hex[:8]}"

        # Create dataset record
        with get_sync_db() as db:
            dataset = Dataset(
                id=dataset_id,
                name=dataset_name,
                source="huggingface:test/dataset",
                status=DatasetStatus.DOWNLOADING,
                progress=0.0,
            )
            db.add(dataset)
            db.commit()

        # Emit download start
        result1 = emit_dataset_progress(
            str(dataset_id),
            "progress",
            {
                "dataset_id": str(dataset_id),
                "progress": 0.0,
                "status": "downloading",
                "message": "Starting download..."
            }
        )

        # Emit download progress (10%)
        result2 = emit_dataset_progress(
            str(dataset_id),
            "progress",
            {
                "dataset_id": str(dataset_id),
                "progress": 10.0,
                "status": "downloading",
                "message": "Downloading from HuggingFace Hub..."
            }
        )

        # Update database
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            dataset.progress = 10.0
            db.commit()

        # Emit download progress (70%)
        result3 = emit_dataset_progress(
            str(dataset_id),
            "progress",
            {
                "dataset_id": str(dataset_id),
                "progress": 70.0,
                "status": "downloading",
                "message": "Saving dataset to disk..."
            }
        )

        # Update database
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            dataset.progress = 70.0
            db.commit()

        # Emit completion
        result4 = emit_dataset_progress(
            str(dataset_id),
            "completed",
            {
                "dataset_id": str(dataset_id),
                "progress": 100.0,
                "status": "ready",
                "message": "Dataset downloaded successfully"
            }
        )

        # Update database to ready state
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            dataset.status = DatasetStatus.READY
            dataset.progress = 100.0
            db.commit()

        # Verify all emissions succeeded
        assert all([result1, result2, result3, result4])
        assert mock_client.__enter__.return_value.post.call_count == 4

        # Verify final database state
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            assert dataset.status == DatasetStatus.READY
            assert dataset.progress == 100.0

        # Cleanup
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_dataset_tokenization_progress_with_database(self, mock_settings, mock_client_class):
        """Test dataset tokenization progress with database state updates."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        dataset_id = uuid4()
        dataset_name = f"test_dataset_tokenize_{uuid4().hex[:8]}"

        # Create dataset record (already downloaded)
        with get_sync_db() as db:
            dataset = Dataset(
                id=dataset_id,
                name=dataset_name,
                source="huggingface:test/dataset",
                status=DatasetStatus.PROCESSING,
                progress=0.0,
            )
            db.add(dataset)
            db.commit()

        # Emit tokenization progress milestones
        milestones = [
            (10.0, "Loading tokenizer..."),
            (20.0, "Loading dataset..."),
            (40.0, "Tokenizing samples..."),
            (80.0, "Calculating statistics..."),
            (95.0, "Saving results..."),
            (100.0, "Tokenization complete"),
        ]

        results = []
        for progress, message in milestones:
            result = emit_dataset_progress(
                str(dataset_id),
                "progress" if progress < 100 else "completed",
                {
                    "dataset_id": str(dataset_id),
                    "progress": progress,
                    "status": "processing" if progress < 100 else "ready",
                    "message": message
                }
            )
            results.append(result)

            # Update database
            with get_sync_db() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                dataset.progress = progress
                if progress == 100.0:
                    dataset.status = DatasetStatus.READY
                db.commit()

        # Verify all emissions succeeded
        assert all(results)
        assert mock_client.__enter__.return_value.post.call_count == len(milestones)

        # Verify final database state
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            assert dataset.status == DatasetStatus.READY
            assert dataset.progress == 100.0

        # Cleanup
        with get_sync_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if dataset:
                db.delete(dataset)
                db.commit()


class TestWebSocketEmissionErrorHandling:
    """Test error handling in WebSocket emission flows."""

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_emission_continues_on_websocket_failure(self, mock_settings, mock_client_class):
        """Test that database updates continue even if WebSocket emission fails."""
        # Setup - WebSocket emission fails
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client

        training_id = f"train_{uuid4().hex[:8]}"
        model_id = f"m_{uuid4().hex[:8]}"
        dataset_id = str(uuid4())

        # Create prerequisite model record (foreign key constraint)
        with get_sync_db() as db:
            model = Model(
                id=model_id,
                name=f"test_model_{uuid4().hex[:8]}",
                architecture="gpt2",
                params_count=124000000,
                quantization=QuantizationFormat.FP16.value,
                status=ModelStatus.READY.value,
                progress=100.0,
            )
            db.add(model)
            db.commit()

        # Create training record
        with get_sync_db() as db:
            training = Training(
                id=training_id,
                dataset_id=dataset_id,
                model_id=model_id,
                status=TrainingStatus.RUNNING.value,
                progress=0.0,
                current_step=0,
                total_steps=1000,
                hyperparameters={},
            )
            db.add(training)
            db.commit()

        # Attempt emission (will fail)
        result = emit_training_progress(
            training_id,
            "progress",
            {
                "training_id": training_id,
                "current_step": 500,
                "progress": 50.0,
            }
        )

        # Verify emission failed
        assert result is False

        # But database can still be updated independently
        with get_sync_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            training.current_step = 500
            training.progress = 50.0
            db.commit()

        # Verify database was updated successfully
        with get_sync_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            assert training.current_step == 500
            assert training.progress == 50.0

        # Cleanup (delete training first due to foreign key constraint)
        with get_sync_db() as db:
            training = db.query(Training).filter_by(id=training_id).first()
            if training:
                db.delete(training)
                db.commit()

        with get_sync_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if model:
                db.delete(model)
                db.commit()

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_emission_handles_non_200_status(self, mock_settings, mock_client_class):
        """Test that emissions return False for non-200 status codes."""
        # Setup - WebSocket returns 500
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 500
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        model_id = f"m_{uuid4().hex[:8]}"
        extraction_id = f"ext_{uuid4().hex[:8]}"

        # Attempt emission (will return False due to 500 status)
        result = emit_extraction_progress(
            model_id,
            extraction_id,
            50.0,
            "extracting",
            "Processing..."
        )

        # Verify emission failed gracefully
        assert result is False
        mock_client.__enter__.return_value.post.assert_called_once()
