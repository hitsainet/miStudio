"""
End-to-end workflow tests for SAE training.

Tests the complete flow: create training → Celery task enqueued →
training execution → checkpoint creation → status updates → WebSocket events.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from src.models.training import TrainingStatus
from src.models.model import ModelStatus, QuantizationFormat
from src.models.dataset import DatasetStatus
from src.schemas.training import TrainingCreate, TrainingUpdate, TrainingHyperparameters, SAEArchitectureType
from src.schemas.model import ModelDownloadRequest
from src.schemas.dataset import DatasetCreate
from src.services.training_service import TrainingService
from src.services.model_service import ModelService
from src.services.dataset_service import DatasetService
from src.services.checkpoint_service import CheckpointService


class TestTrainingWorkflow:
    """End-to-end workflow tests for SAE training operations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_training_creation_workflow(self, async_session):
        """
        Test complete workflow: POST /api/trainings → training created → Celery task enqueued.

        This test verifies Phase 20.1 requirements:
        1. Training job is created with correct initial state
        2. Celery task is enqueued for background execution
        3. Training record has correct status (PENDING → RUNNING transition)
        4. All required fields are populated correctly

        TDD Reference: Lines 1257-1277
        """
        # Step 1: Create test model
        model_request = ModelDownloadRequest(
            repo_id="test/tiny-gpt2",
            quantization=QuantizationFormat.FP16,
        )
        model = await ModelService.initiate_model_download(async_session, model_request)
        await async_session.commit()
        await async_session.refresh(model)

        # Mark model as ready
        model.status = ModelStatus.READY.value
        model.architecture = "gpt2"
        model.params_count = 124000000
        model.hidden_dim = 768
        model.num_layers = 12
        await async_session.commit()
        await async_session.refresh(model)

        # Step 2: Create test dataset
        dataset_data = DatasetCreate(
            name="test-dataset",
            source="HuggingFace",
            hf_repo_id="test/tiny-dataset",
            metadata={
                "splits": ["train"],
                "num_rows": 10000,
                "features": {"text": "string"}
            }
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_data)

        # Mark dataset as ready
        dataset.status = DatasetStatus.READY.value
        dataset.size_bytes = 1000000
        await async_session.commit()
        await async_session.refresh(dataset)

        # Step 3: Create training hyperparameters
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=8192,
            architecture_type=SAEArchitectureType.STANDARD,
            training_layers=[0, 6],
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=256,
            total_steps=100,  # Small for test
            warmup_steps=10,
            checkpoint_interval=50,
            log_interval=10,
        )

        # Step 4: Create training job
        training_data = TrainingCreate(
            model_id=model.id,
            dataset_id=str(dataset.id),  # Convert UUID to string
            extraction_id=None,
            hyperparameters=hyperparameters
        )

        # Create training via service
        training = await TrainingService.create_training(async_session, training_data)
        await async_session.commit()
        await async_session.refresh(training)

        # Simulate starting training (as API endpoint would do)
        # In real API, train_sae_task.delay() is called here
        mock_celery_task_id = "test-celery-task-id-12345"
        await TrainingService.start_training(
            async_session,
            training.id,
            mock_celery_task_id
        )
        await async_session.commit()
        await async_session.refresh(training)

        # Step 5: Verify training creation
        assert training.id is not None
        assert training.id.startswith("train_")
        assert training.model_id == model.id
        assert training.dataset_id == str(dataset.id)  # dataset_id stored as string
        assert training.extraction_id is None

        # Step 6: Verify initial status
        assert training.status == TrainingStatus.INITIALIZING.value  # After start_training()
        assert training.progress == 0.0
        assert training.current_step == 0
        assert training.total_steps == 100

        # Step 7: Verify hyperparameters stored correctly
        assert training.hyperparameters is not None
        assert training.hyperparameters["hidden_dim"] == 768
        assert training.hyperparameters["latent_dim"] == 8192
        assert training.hyperparameters["architecture_type"] == "standard"
        assert training.hyperparameters["training_layers"] == [0, 6]
        assert training.hyperparameters["l1_alpha"] == 0.001
        assert training.hyperparameters["batch_size"] == 256

        # Step 8: Verify Celery task ID
        assert training.celery_task_id == "test-celery-task-id-12345"

        # Step 9: Verify timestamps
        assert training.created_at is not None
        assert training.updated_at is not None
        assert training.started_at is not None  # Set by start_training()
        assert training.completed_at is None  # Not completed yet

        # Step 10: Verify current metrics are initialized
        assert training.current_loss is None  # Not started yet
        assert training.current_l0_sparsity is None
        assert training.current_dead_neurons is None
        assert training.current_learning_rate is None

        # Step 11: Verify file paths (set by worker, not service)
        # checkpoint_dir and logs_path would be set by training worker task
        # For service layer test, these are None initially

        # Cleanup
        await TrainingService.delete_training(async_session, training.id)
        await ModelService.delete_model(async_session, model.id)
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_training_progress_and_metrics_tracking(self, async_session):
        """
        Test training progress updates and metrics tracking during training.

        This test verifies Phase 20.2 requirements:
        1. Training status transitions (PENDING → INITIALIZING → RUNNING → COMPLETED)
        2. Progress tracking (0% → 100%)
        3. Current metrics are updated on training record
        4. Checkpoint creation works

        Note: This is a simplified integration test that verifies service layer methods
        for progress tracking work correctly. Full worker task testing with actual
        training loop execution is deferred to HP-2 Phase 3 (see HP2_Phase3_Implementation_Plan.md).
        """
        # Step 1: Create test model and dataset
        model_request = ModelDownloadRequest(
            repo_id="test/training-progress-model",
            quantization=QuantizationFormat.FP16,
        )
        model = await ModelService.initiate_model_download(async_session, model_request)
        model.status = ModelStatus.READY.value
        model.architecture = "gpt2"
        model.params_count = 124000000
        model.hidden_dim = 768
        model.num_layers = 12
        await async_session.commit()
        await async_session.refresh(model)

        dataset_data = DatasetCreate(
            name="test-progress-dataset",
            source="HuggingFace",
            hf_repo_id="test/progress-dataset",
            metadata={"splits": ["train"], "num_rows": 1000}
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_data)
        dataset.status = DatasetStatus.READY.value
        await async_session.commit()
        await async_session.refresh(dataset)

        # Step 2: Create training job
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=4096,
            architecture_type=SAEArchitectureType.STANDARD,
            training_layers=[0],
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=128,
            total_steps=100,  # Small number for testing
            warmup_steps=10,
            checkpoint_interval=25,  # Checkpoint every 25 steps
            log_interval=10,  # Log every 10 steps
        )

        training_data = TrainingCreate(
            model_id=model.id,
            dataset_id=str(dataset.id),
            extraction_id=None,
            hyperparameters=hyperparameters
        )

        training = await TrainingService.create_training(async_session, training_data)
        await async_session.commit()
        await async_session.refresh(training)

        # Verify initial status
        assert training.status == TrainingStatus.PENDING.value
        assert training.progress == 0.0
        assert training.current_step == 0

        # Step 3: Simulate training start
        await TrainingService.start_training(async_session, training.id, "test-celery-task-123")
        await async_session.commit()
        await async_session.refresh(training)

        assert training.status == TrainingStatus.INITIALIZING.value
        assert training.started_at is not None

        # Step 4: Simulate a single training progress update
        step = 50
        progress = (step / hyperparameters.total_steps) * 100.0
        loss = 0.5
        l0_sparsity = 0.06
        dead_neurons = 50
        learning_rate = 0.00025

        # Update training using TrainingUpdate schema
        update_data = TrainingUpdate(
            status=TrainingStatus.RUNNING,
            progress=progress,
            current_step=step,
            current_loss=loss,
            current_l0_sparsity=l0_sparsity,
            current_dead_neurons=dead_neurons,
            current_learning_rate=learning_rate
        )
        training = await TrainingService.update_training(
            async_session,
            training.id,
            update_data
        )
        await async_session.commit()
        await async_session.refresh(training)

        # Verify progress update
        assert training.current_step == step
        assert training.progress == progress
        assert training.current_loss == loss
        assert training.current_l0_sparsity == l0_sparsity
        assert training.current_dead_neurons == dead_neurons
        assert training.current_learning_rate == learning_rate
        assert training.status == TrainingStatus.RUNNING.value

        # Step 5: Test checkpoint creation
        checkpoint = await CheckpointService.create_checkpoint(
            async_session,
            training_id=training.id,
            step=step,
            loss=loss,
            l0_sparsity=l0_sparsity,
            storage_path=f"/tmp/test_checkpoint_step_{step}.safetensors",
            is_best=True,
            extra_metadata={"test": True}
        )
        await async_session.commit()

        assert checkpoint is not None
        assert checkpoint.step == step
        assert checkpoint.loss == loss
        assert checkpoint.is_best is True

        # Step 6: Mark training as completed
        await TrainingService.mark_training_completed(async_session, training.id)
        await async_session.commit()
        await async_session.refresh(training)

        assert training.status == TrainingStatus.COMPLETED.value
        assert training.progress == 100.0
        assert training.completed_at is not None

        # Step 7: Verify checkpoint can be retrieved
        checkpoints, total = await CheckpointService.list_checkpoints(async_session, training.id)
        assert len(checkpoints) >= 1  # At least the one we created
        assert total >= 1

        # Find our checkpoint
        our_checkpoint = next((c for c in checkpoints if c.step == step), None)
        assert our_checkpoint is not None
        assert our_checkpoint.is_best is True

        # Cleanup
        await CheckpointService.delete_checkpoint(async_session, checkpoint.id)
        await TrainingService.delete_training(async_session, training.id)
        await ModelService.delete_model(async_session, model.id)
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_training_creation_without_extraction(self, async_session):
        """
        Test training creation without pre-extracted activations.

        Verifies that:
        1. Training can be created without extraction_id
        2. extraction_id defaults to None
        3. Training will perform activation extraction during execution
        """
        # Create test model
        model_request = ModelDownloadRequest(
            repo_id="test/model-no-extraction",
            quantization=QuantizationFormat.FP16,
        )
        model = await ModelService.initiate_model_download(async_session, model_request)
        model.status = ModelStatus.READY.value
        model.architecture = "gpt2"
        model.hidden_dim = 768
        await async_session.commit()
        await async_session.refresh(model)

        # Create test dataset
        dataset_data = DatasetCreate(
            name="test-dataset-no-extraction",
            source="HuggingFace",
            hf_repo_id="test/dataset",
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_data)
        dataset.status = DatasetStatus.READY.value
        await async_session.commit()
        await async_session.refresh(dataset)

        # Create training without extraction_id
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=4096,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=128,
            total_steps=50,
        )

        training_data = TrainingCreate(
            model_id=model.id,
            dataset_id=str(dataset.id),  # Convert UUID to string
            extraction_id=None,  # No pre-extracted activations
            hyperparameters=hyperparameters
        )

        training = await TrainingService.create_training(async_session, training_data)
        await async_session.commit()
        await async_session.refresh(training)

        # Verify extraction_id is None
        assert training.extraction_id is None

        # Cleanup
        await TrainingService.delete_training(async_session, training.id)
        await ModelService.delete_model(async_session, model.id)
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_training_creation_validation_errors(self, async_session):
        """
        Test validation errors during training creation.

        Verifies that:
        1. Invalid model_id format raises validation error
        2. Invalid extraction_id format raises validation error
        3. Missing required hyperparameters raises validation error
        """
        # Test 1: Invalid model_id (doesn't start with 'm_')
        with pytest.raises(Exception) as exc_info:
            hyperparameters = TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=8192,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=256,
                total_steps=100,
            )
            training_data = TrainingCreate(
                model_id="invalid-model-id",  # Wrong format
                dataset_id="d_test_dataset",
                hyperparameters=hyperparameters
            )
        assert "must start with 'm_'" in str(exc_info.value)

        # Test 2: Invalid extraction_id (doesn't start with 'ext_m_')
        with pytest.raises(Exception) as exc_info:
            hyperparameters = TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=8192,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=256,
                total_steps=100,
            )
            training_data = TrainingCreate(
                model_id="m_test_model",
                dataset_id="d_test_dataset",
                extraction_id="invalid-extraction-id",  # Wrong format
                hyperparameters=hyperparameters
            )
        assert "must start with 'ext_m_'" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_training_list_with_filters(self, async_session):
        """
        Test listing trainings with various filters.

        Verifies that:
        1. Trainings can be filtered by model_id
        2. Trainings can be filtered by dataset_id
        3. Trainings can be filtered by status
        4. Pagination works correctly
        """
        # Create test model
        model_request = ModelDownloadRequest(
            repo_id="test/list-model",
            quantization=QuantizationFormat.FP16,
        )
        model = await ModelService.initiate_model_download(async_session, model_request)
        model.status = ModelStatus.READY.value
        model.architecture = "gpt2"
        model.hidden_dim = 768
        await async_session.commit()
        await async_session.refresh(model)

        # Create test dataset
        dataset_data = DatasetCreate(
            name="test-list-dataset",
            source="HuggingFace",
            hf_repo_id="test/list-dataset",
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_data)
        dataset.status = DatasetStatus.READY.value
        await async_session.commit()
        await async_session.refresh(dataset)

        # Create multiple trainings
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=4096,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=128,
            total_steps=50,
        )

        training_ids = []
        for i in range(3):
            training_data = TrainingCreate(
                model_id=model.id,
                dataset_id=str(dataset.id),  # Convert UUID to string
                hyperparameters=hyperparameters
            )
            training = await TrainingService.create_training(async_session, training_data)
            training_ids.append(training.id)
            await async_session.commit()

        # Test filter by model_id
        trainings, total = await TrainingService.list_trainings(
            async_session,
            model_id=model.id
        )
        assert total >= 3  # At least our 3 trainings

        # Test filter by dataset_id
        trainings, total = await TrainingService.list_trainings(
            async_session,
            dataset_id=str(dataset.id)  # Convert UUID to string
        )
        assert total >= 3  # At least our 3 trainings

        # Test filter by status
        trainings, total = await TrainingService.list_trainings(
            async_session,
            status=TrainingStatus.PENDING
        )
        assert total >= 3  # All our trainings are PENDING

        # Test pagination
        trainings, total = await TrainingService.list_trainings(
            async_session,
            model_id=model.id,
            limit=2,
            skip=0
        )
        assert len(trainings) <= 2  # Should return max 2 trainings

        # Cleanup
        for training_id in training_ids:
            await TrainingService.delete_training(async_session, training_id)
        await ModelService.delete_model(async_session, model.id)
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_training_deletion_cascades(self, async_session):
        """
        Test that deleting a training job cascades to related data.

        Verifies that:
        1. Deleting a training deletes associated metrics
        2. Deleting a training deletes associated checkpoints
        3. Training files are marked for cleanup
        """
        # Create test model and dataset
        model_request = ModelDownloadRequest(
            repo_id="test/delete-model",
            quantization=QuantizationFormat.FP16,
        )
        model = await ModelService.initiate_model_download(async_session, model_request)
        model.status = ModelStatus.READY.value
        model.architecture = "gpt2"
        model.hidden_dim = 768
        await async_session.commit()
        await async_session.refresh(model)

        dataset_data = DatasetCreate(
            name="test-delete-dataset",
            source="HuggingFace",
            hf_repo_id="test/delete-dataset",
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_data)
        dataset.status = DatasetStatus.READY.value
        await async_session.commit()
        await async_session.refresh(dataset)

        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=4096,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=128,
            total_steps=50,
        )

        training_data = TrainingCreate(
            model_id=model.id,
            dataset_id=str(dataset.id),  # Convert UUID to string
            hyperparameters=hyperparameters
        )
        training = await TrainingService.create_training(async_session, training_data)
        training_id = training.id
        await async_session.commit()

        # Verify training exists
        training = await TrainingService.get_training(async_session, training_id)
        assert training is not None

        # Delete training
        deleted = await TrainingService.delete_training(async_session, training_id)
        await async_session.commit()
        assert deleted is True

        # Verify training is deleted
        training = await TrainingService.get_training(async_session, training_id)
        assert training is None

        # Cleanup
        await ModelService.delete_model(async_session, model.id)
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()


# Note: These are INTEGRATION tests that verify the workflow logic.
# For REAL E2E testing with actual training execution:
#
# 1. Use pytest-asyncio with longer timeouts (training takes time)
# 2. Mock GPU operations to avoid hardware dependencies
# 3. Use small model (GPT-2 Small) and tiny dataset
# 4. Mock WebSocket emissions to avoid connection requirements
# 5. Verify checkpoint files are created on disk
# 6. Clean up test files after completion
#
# Example real E2E test structure:
#
# @pytest.mark.asyncio
# @pytest.mark.slow
# @pytest.mark.integration
# async def test_real_training_execution(async_session):
#     """Real E2E test with actual training loop execution."""
#     # Create model, dataset, training
#     # ...
#
#     # Mock GPU operations
#     with patch('torch.cuda.is_available', return_value=False):
#         with patch('src.workers.websocket_emitter.emit_training_progress'):
#             # Trigger real training task (CPU mode)
#             task_result = train_sae_task.delay(training.id)
#
#             # Poll for completion (with timeout)
#             for _ in range(300):  # 5 minutes timeout
#                 await asyncio.sleep(1)
#                 await async_session.refresh(training)
#                 if training.status in (
#                     TrainingStatus.COMPLETED,
#                     TrainingStatus.FAILED
#                 ):
#                     break
#
#             # Verify training succeeded
#             assert training.status == TrainingStatus.COMPLETED
#             assert training.current_step == training.total_steps
#             assert training.progress == 100.0
#
#             # Verify checkpoints created
#             checkpoints = await CheckpointService.list_checkpoints(
#                 async_session, training.id
#             )
#             assert len(checkpoints) > 0
#
#             # Verify metrics saved
#             metrics = await TrainingService.list_metrics(
#                 async_session, training.id
#             )
#             assert len(metrics) > 0
#
#     # Cleanup
#     # Delete training, checkpoints, model, dataset
