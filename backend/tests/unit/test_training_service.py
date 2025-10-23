"""
Unit tests for TrainingService.

Tests training job CRUD operations, status updates, filtering,
and pagination using async database sessions.
"""

import pytest
import pytest_asyncio
from datetime import datetime, UTC
from unittest.mock import patch, MagicMock

from src.services.training_service import TrainingService
from src.models.training import Training, TrainingStatus
from src.models.model import Model, ModelStatus, QuantizationFormat
from src.models.dataset import Dataset, DatasetStatus
from src.schemas.training import TrainingCreate, TrainingUpdate, TrainingHyperparameters, SAEArchitectureType


@pytest_asyncio.fixture
async def test_model(async_session):
    """Create a test model for training tests."""
    model = Model(
        id="m_test123",
        name="Test Model",
        repo_id="test/model",
        status=ModelStatus.READY.value,
        quantization=QuantizationFormat.FP16.value,
        architecture="gpt2",
        params_count=117000000,
        architecture_config={"num_hidden_layers": 12, "hidden_size": 768},
    )
    async_session.add(model)
    await async_session.commit()
    return model


@pytest_asyncio.fixture
async def test_dataset(async_session):
    """Create a test dataset for training tests."""
    from uuid import UUID
    dataset = Dataset(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        name="Test Dataset",
        source="HuggingFace",
        hf_repo_id="test/dataset",
        status=DatasetStatus.READY.value,
        num_samples=1000,
    )
    async_session.add(dataset)
    await async_session.commit()
    return dataset


@pytest.mark.asyncio
class TestTrainingServiceCreate:
    """Test TrainingService.create_training()."""

    async def test_create_training_success(self, async_session, test_model, test_dataset):
        """Test creating a training job."""
        # Prepare test data
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        # Mock WebSocket emitter to avoid actual emission
        with patch('src.services.training_service._emit_training_event_sync'):
            # Create training
            training = await TrainingService.create_training(async_session, training_data)

        # Verify training created
        assert training is not None
        assert training.id.startswith("train_")
        assert training.model_id == "m_test123"
        assert training.dataset_id == "12345678-1234-5678-1234-567812345678"
        assert training.status == TrainingStatus.PENDING.value
        assert training.progress == 0.0
        assert training.current_step == 0
        assert training.total_steps == 100000
        assert training.hyperparameters['hidden_dim'] == 768
        assert training.hyperparameters['latent_dim'] == 16384

    async def test_create_training_without_extraction_id(self, async_session, test_model, test_dataset):
        """Test creating training without extraction_id (optional field)."""
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        # extraction_id is optional, test creating without it
        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        assert training.extraction_id is None

    async def test_create_training_generates_unique_id(self, async_session, test_model, test_dataset):
        """Test that each training gets a unique ID."""
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training1 = await TrainingService.create_training(async_session, training_data)
            training2 = await TrainingService.create_training(async_session, training_data)

        assert training1.id != training2.id
        assert training1.id.startswith("train_")
        assert training2.id.startswith("train_")

    async def test_create_training_stores_hyperparameters(self, async_session, test_model, test_dataset):
        """Test that hyperparameters are correctly stored as JSONB."""
        hyperparameters = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=8192,
            architecture_type=SAEArchitectureType.SKIP,
            l1_alpha=0.005,
            learning_rate=0.0005,
            batch_size=2048,
            total_steps=50000,
            warmup_steps=500,
            weight_decay=0.01,
            checkpoint_interval=2500,
            log_interval=50,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Verify all hyperparameters stored
        hp = training.hyperparameters
        assert hp['hidden_dim'] == 512
        assert hp['latent_dim'] == 8192
        assert hp['architecture_type'] == 'skip'
        assert hp['l1_alpha'] == 0.005
        assert hp['learning_rate'] == 0.0005
        assert hp['batch_size'] == 2048
        assert hp['total_steps'] == 50000
        assert hp['warmup_steps'] == 500
        assert hp['weight_decay'] == 0.01


@pytest.mark.asyncio
class TestTrainingServiceGet:
    """Test TrainingService.get_training()."""

    async def test_get_training_success(self, async_session, test_model, test_dataset):
        """Test getting a training by ID."""
        # Create a training first
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            created_training = await TrainingService.create_training(async_session, training_data)

        # Get the training
        fetched_training = await TrainingService.get_training(async_session, created_training.id)

        assert fetched_training is not None
        assert fetched_training.id == created_training.id
        assert fetched_training.model_id == "m_test123"
        assert fetched_training.dataset_id == "12345678-1234-5678-1234-567812345678"

    async def test_get_training_not_found(self, async_session, test_model, test_dataset):
        """Test getting a non-existent training returns None."""
        training = await TrainingService.get_training(async_session, "train_nonexistent")

        assert training is None


@pytest.mark.asyncio
class TestTrainingServiceList:
    """Test TrainingService.list_trainings()."""

    async def test_list_all_trainings(self, async_session, test_model, test_dataset):
        """Test listing all trainings."""
        # Create multiple trainings using existing test fixtures
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            for i in range(3):
                training_data = TrainingCreate(
                    model_id="m_test123",  # Use existing test_model fixture
                    dataset_id="12345678-1234-5678-1234-567812345678",  # Use existing test_dataset fixture
                    hyperparameters=hyperparameters,
                )
                await TrainingService.create_training(async_session, training_data)

        # List trainings
        trainings, total = await TrainingService.list_trainings(async_session)

        assert len(trainings) == 3
        assert total == 3

    async def test_list_trainings_filter_by_model_id(self, async_session, test_model, test_dataset):
        """Test filtering trainings by model_id."""
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            # Create trainings with model m_test123
            for i in range(2):
                training_data = TrainingCreate(
                    model_id="m_test123",
                    dataset_id="12345678-1234-5678-1234-567812345678",
                    hyperparameters=hyperparameters,
                )
                await TrainingService.create_training(async_session, training_data)

        # Filter by model_id
        trainings, total = await TrainingService.list_trainings(
            async_session,
            model_id="m_test123"
        )

        assert len(trainings) == 2
        assert total == 2
        assert all(t.model_id == "m_test123" for t in trainings)

    async def test_list_trainings_filter_by_dataset_id(self, async_session, test_model, test_dataset):
        """Test filtering trainings by dataset_id."""
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            # Create trainings with the test dataset
            for i in range(2):
                training_data = TrainingCreate(
                    model_id="m_test123",
                    dataset_id="12345678-1234-5678-1234-567812345678",
                    hyperparameters=hyperparameters,
                )
                await TrainingService.create_training(async_session, training_data)

        # Filter by dataset_id
        trainings, total = await TrainingService.list_trainings(
            async_session,
            dataset_id="12345678-1234-5678-1234-567812345678"
        )

        assert len(trainings) == 2
        assert total == 2
        assert all(t.dataset_id == "12345678-1234-5678-1234-567812345678" for t in trainings)

    async def test_list_trainings_filter_by_status(self, async_session, test_model, test_dataset):
        """Test filtering trainings by status."""
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            # Create trainings
            for i in range(3):
                training_data = TrainingCreate(
                    model_id="m_test123",
                    dataset_id="12345678-1234-5678-1234-567812345678",
                    hyperparameters=hyperparameters,
                )
                await TrainingService.create_training(async_session, training_data)

        # All should be PENDING initially
        trainings, total = await TrainingService.list_trainings(
            async_session,
            status=TrainingStatus.PENDING
        )

        assert len(trainings) == 3
        assert total == 3
        assert all(t.status == TrainingStatus.PENDING.value for t in trainings)

    async def test_list_trainings_pagination(self, async_session, test_model, test_dataset):
        """Test pagination of training list."""
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            # Create 5 trainings
            for i in range(5):
                training_data = TrainingCreate(
                    model_id="m_test123",
                    dataset_id="12345678-1234-5678-1234-567812345678",
                    hyperparameters=hyperparameters,
                )
                await TrainingService.create_training(async_session, training_data)

        # Get first page (2 items)
        trainings_page1, total = await TrainingService.list_trainings(
            async_session,
            skip=0,
            limit=2
        )

        assert len(trainings_page1) == 2
        assert total == 5

        # Get second page (2 items)
        trainings_page2, total = await TrainingService.list_trainings(
            async_session,
            skip=2,
            limit=2
        )

        assert len(trainings_page2) == 2
        assert total == 5

        # Verify different trainings on different pages
        page1_ids = {t.id for t in trainings_page1}
        page2_ids = {t.id for t in trainings_page2}
        assert page1_ids.isdisjoint(page2_ids)

    async def test_list_trainings_ordered_by_created_at_desc(self, async_session, test_model, test_dataset):
        """Test that trainings are ordered by created_at descending (newest first)."""
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        created_ids = []
        with patch('src.services.training_service._emit_training_event_sync'):
            for i in range(3):
                training_data = TrainingCreate(
                    model_id="m_test123",  # Use existing test_model fixture
                    dataset_id="12345678-1234-5678-1234-567812345678",
                    hyperparameters=hyperparameters,
                )
                training = await TrainingService.create_training(async_session, training_data)
                created_ids.append(training.id)

        # List trainings
        trainings, _ = await TrainingService.list_trainings(async_session)

        # Should be in reverse order (newest first)
        fetched_ids = [t.id for t in trainings]
        assert fetched_ids == list(reversed(created_ids))


@pytest.mark.asyncio
class TestTrainingServiceUpdate:
    """Test TrainingService.update_training()."""

    async def test_update_training_status(self, async_session, test_model, test_dataset):
        """Test updating training status."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Update status
        update_data = TrainingUpdate(status=TrainingStatus.RUNNING)
        updated_training = await TrainingService.update_training(
            async_session,
            training.id,
            update_data
        )

        assert updated_training is not None
        assert updated_training.status == TrainingStatus.RUNNING.value

    async def test_update_training_progress(self, async_session, test_model, test_dataset):
        """Test updating training progress."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Update progress
        update_data = TrainingUpdate(
            current_step=5000,
            progress=5.0,
            current_loss=0.123,
        )
        updated_training = await TrainingService.update_training(
            async_session,
            training.id,
            update_data
        )

        assert updated_training.current_step == 5000
        assert updated_training.progress == 5.0
        assert updated_training.current_loss == 0.123

    async def test_update_training_not_found(self, async_session, test_model, test_dataset):
        """Test updating non-existent training returns None."""
        update_data = TrainingUpdate(status=TrainingStatus.RUNNING)
        updated_training = await TrainingService.update_training(
            async_session,
            "train_nonexistent",
            update_data
        )

        assert updated_training is None


@pytest.mark.asyncio
class TestTrainingServiceDelete:
    """Test TrainingService.delete_training()."""

    async def test_delete_training_success(self, async_session, test_model, test_dataset):
        """Test deleting a training."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Delete training
        deleted = await TrainingService.delete_training(async_session, training.id)

        assert deleted is True

        # Verify training deleted
        fetched = await TrainingService.get_training(async_session, training.id)
        assert fetched is None

    async def test_delete_training_not_found(self, async_session, test_model, test_dataset):
        """Test deleting non-existent training returns False."""
        deleted = await TrainingService.delete_training(async_session, "train_nonexistent")

        assert deleted is False


@pytest.mark.asyncio
class TestTrainingServiceWebSocketEvents:
    """Test WebSocket event emission."""

    async def test_create_training_emits_event(self, async_session, test_model, test_dataset):
        """Test that creating a training emits WebSocket event."""
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        # Mock and verify WebSocket event emission
        with patch('src.services.training_service._emit_training_event_sync') as mock_emit:
            training = await TrainingService.create_training(async_session, training_data)

            # Verify emit was called
            mock_emit.assert_called_once()
            call_args = mock_emit.call_args

            # Verify event data
            assert call_args[1]['training_id'] == training.id
            assert call_args[1]['event'] == "created"
            assert call_args[1]['data']['model_id'] == "m_test123"
            assert call_args[1]['data']['dataset_id'] == "12345678-1234-5678-1234-567812345678"
            assert call_args[1]['data']['status'] == TrainingStatus.PENDING.value


@pytest.mark.asyncio
class TestTrainingServiceStateManagement:
    """Test training state management methods."""

    async def test_start_training(self, async_session, test_model, test_dataset):
        """Test starting a training job."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Start training
        started_training = await TrainingService.start_training(
            async_session,
            training.id,
            "celery_task_123"
        )

        assert started_training is not None
        assert started_training.status == TrainingStatus.INITIALIZING.value
        assert started_training.celery_task_id == "celery_task_123"
        assert started_training.started_at is not None

    async def test_start_training_not_found(self, async_session):
        """Test starting non-existent training returns None."""
        result = await TrainingService.start_training(
            async_session,
            "train_nonexistent",
            "celery_task_123"
        )

        assert result is None

    async def test_pause_training(self, async_session, test_model, test_dataset):
        """Test pausing a running training job."""
        # Create and start training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)
            await TrainingService.start_training(async_session, training.id, "task_123")

        # Update to running
        update_data = TrainingUpdate(status=TrainingStatus.RUNNING)
        await TrainingService.update_training(async_session, training.id, update_data)

        # Pause training
        with patch('src.services.training_service._emit_training_event_sync') as mock_emit:
            paused_training = await TrainingService.pause_training(async_session, training.id)

        assert paused_training is not None
        assert paused_training.status == TrainingStatus.PAUSED.value
        mock_emit.assert_called_once()

    async def test_pause_training_not_running(self, async_session, test_model, test_dataset):
        """Test pausing a non-running training returns None."""
        # Create training in PENDING state
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Try to pause PENDING training (should fail)
        paused_training = await TrainingService.pause_training(async_session, training.id)

        assert paused_training is None

    async def test_resume_training(self, async_session, test_model, test_dataset):
        """Test resuming a paused training job."""
        # Create, start, and pause training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)
            await TrainingService.start_training(async_session, training.id, "task_123")

        # Update to running then pause
        update_data = TrainingUpdate(status=TrainingStatus.RUNNING)
        await TrainingService.update_training(async_session, training.id, update_data)

        with patch('src.services.training_service._emit_training_event_sync'):
            await TrainingService.pause_training(async_session, training.id)

        # Resume training
        with patch('src.services.training_service._emit_training_event_sync') as mock_emit:
            resumed_training = await TrainingService.resume_training(
                async_session,
                training.id,
                "new_task_456"
            )

        assert resumed_training is not None
        assert resumed_training.status == TrainingStatus.RUNNING.value
        assert resumed_training.celery_task_id == "new_task_456"
        mock_emit.assert_called_once()

    async def test_resume_training_not_paused(self, async_session, test_model, test_dataset):
        """Test resuming a non-paused training returns None."""
        # Create training in PENDING state
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Try to resume PENDING training (should fail)
        resumed_training = await TrainingService.resume_training(async_session, training.id)

        assert resumed_training is None

    async def test_stop_training(self, async_session, test_model, test_dataset):
        """Test stopping (cancelling) a training job."""
        # Create and start training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)
            await TrainingService.start_training(async_session, training.id, "task_123")

        # Stop training
        with patch('src.services.training_service._emit_training_event_sync') as mock_emit:
            stopped_training = await TrainingService.stop_training(async_session, training.id)

        assert stopped_training is not None
        assert stopped_training.status == TrainingStatus.CANCELLED.value
        assert stopped_training.completed_at is not None
        mock_emit.assert_called_once()

    async def test_stop_training_already_terminal(self, async_session, test_model, test_dataset):
        """Test stopping an already-completed training returns None."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)
            # Mark as completed
            await TrainingService.mark_training_completed(async_session, training.id)

        # Try to stop completed training (should fail)
        stopped_training = await TrainingService.stop_training(async_session, training.id)

        assert stopped_training is None

    async def test_mark_training_failed(self, async_session, test_model, test_dataset):
        """Test marking a training as failed."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Mark as failed
        with patch('src.services.training_service._emit_training_event_sync') as mock_emit:
            failed_training = await TrainingService.mark_training_failed(
                async_session,
                training.id,
                "CUDA out of memory",
                "Traceback: ..."
            )

        assert failed_training is not None
        assert failed_training.status == TrainingStatus.FAILED.value
        assert failed_training.error_message == "CUDA out of memory"
        assert failed_training.error_traceback == "Traceback: ..."
        assert failed_training.completed_at is not None
        mock_emit.assert_called_once()

    async def test_mark_training_failed_not_found(self, async_session):
        """Test marking non-existent training as failed returns None."""
        result = await TrainingService.mark_training_failed(
            async_session,
            "train_nonexistent",
            "Error message"
        )

        assert result is None

    async def test_mark_training_completed(self, async_session, test_model, test_dataset):
        """Test marking a training as completed."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Mark as completed
        with patch('src.services.training_service._emit_training_event_sync') as mock_emit:
            completed_training = await TrainingService.mark_training_completed(
                async_session,
                training.id
            )

        assert completed_training is not None
        assert completed_training.status == TrainingStatus.COMPLETED.value
        assert completed_training.progress == 100.0
        assert completed_training.completed_at is not None
        mock_emit.assert_called_once()

    async def test_mark_training_completed_not_found(self, async_session):
        """Test marking non-existent training as completed returns None."""
        result = await TrainingService.mark_training_completed(
            async_session,
            "train_nonexistent"
        )

        assert result is None


@pytest.mark.asyncio
class TestTrainingServiceMetrics:
    """Test training metrics management."""

    async def test_add_metric(self, async_session, test_model, test_dataset):
        """Test adding a training metric."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Add metric
        metric = await TrainingService.add_metric(
            async_session,
            training.id,
            step=1000,
            loss=0.234,
            l0_sparsity=15.5,
            learning_rate=0.0003,
        )

        assert metric is not None
        assert metric.training_id == training.id
        assert metric.step == 1000
        assert metric.loss == 0.234
        assert metric.l0_sparsity == 15.5
        assert metric.learning_rate == 0.0003

    async def test_get_metrics_all(self, async_session, test_model, test_dataset):
        """Test getting all metrics for a training."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Add multiple metrics
        for step in [1000, 2000, 3000]:
            await TrainingService.add_metric(
                async_session,
                training.id,
                step=step,
                loss=0.5 - (step / 10000),
            )

        # Get all metrics
        metrics = await TrainingService.get_metrics(async_session, training.id)

        assert len(metrics) == 3
        assert metrics[0].step == 1000
        assert metrics[1].step == 2000
        assert metrics[2].step == 3000

    async def test_get_metrics_with_start_step(self, async_session, test_model, test_dataset):
        """Test getting metrics with start_step filter."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Add multiple metrics
        for step in [1000, 2000, 3000, 4000]:
            await TrainingService.add_metric(
                async_session,
                training.id,
                step=step,
                loss=0.5,
            )

        # Get metrics from step 2000 onwards
        metrics = await TrainingService.get_metrics(
            async_session,
            training.id,
            start_step=2000
        )

        assert len(metrics) == 3  # Steps 2000, 3000, 4000
        assert all(m.step >= 2000 for m in metrics)

    async def test_get_metrics_with_end_step(self, async_session, test_model, test_dataset):
        """Test getting metrics with end_step filter."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Add multiple metrics
        for step in [1000, 2000, 3000, 4000]:
            await TrainingService.add_metric(
                async_session,
                training.id,
                step=step,
                loss=0.5,
            )

        # Get metrics up to step 3000
        metrics = await TrainingService.get_metrics(
            async_session,
            training.id,
            end_step=3000
        )

        assert len(metrics) == 3  # Steps 1000, 2000, 3000
        assert all(m.step <= 3000 for m in metrics)

    async def test_get_metrics_with_range(self, async_session, test_model, test_dataset):
        """Test getting metrics with both start and end step filters."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Add multiple metrics
        for step in [1000, 2000, 3000, 4000, 5000]:
            await TrainingService.add_metric(
                async_session,
                training.id,
                step=step,
                loss=0.5,
            )

        # Get metrics in range [2000, 4000]
        metrics = await TrainingService.get_metrics(
            async_session,
            training.id,
            start_step=2000,
            end_step=4000
        )

        assert len(metrics) == 3  # Steps 2000, 3000, 4000
        assert all(2000 <= m.step <= 4000 for m in metrics)

    async def test_get_metrics_with_limit(self, async_session, test_model, test_dataset):
        """Test getting metrics with limit."""
        # Create training
        hyperparameters = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training_data = TrainingCreate(
            model_id="m_test123",
            dataset_id="12345678-1234-5678-1234-567812345678",
            hyperparameters=hyperparameters,
        )

        with patch('src.services.training_service._emit_training_event_sync'):
            training = await TrainingService.create_training(async_session, training_data)

        # Add multiple metrics
        for step in range(1000, 6000, 1000):
            await TrainingService.add_metric(
                async_session,
                training.id,
                step=step,
                loss=0.5,
            )

        # Get only first 2 metrics
        metrics = await TrainingService.get_metrics(
            async_session,
            training.id,
            limit=2
        )

        assert len(metrics) == 2
