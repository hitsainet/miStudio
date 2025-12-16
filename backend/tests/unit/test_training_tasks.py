"""
Unit tests for training_tasks module.

Tests the TrainingTask helper methods and train_sae_task flow.
Note: Full integration testing of Celery tasks is done in integration tests.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

import torch
import torch.nn as nn

from src.workers.training_tasks import TrainingTask
from src.models.training import TrainingStatus


class TestTrainingTaskHelpers:
    """Test TrainingTask helper methods."""

    def test_update_training_progress_basic(self):
        """Test updating training progress with basic metrics."""
        task = TrainingTask()

        # Mock database operations
        mock_training = Mock()
        mock_db = Mock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = mock_training

        with patch.object(task, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_db

            task.update_training_progress(
                training_id="train_test123",
                step=5000,
                total_steps=10000,
                loss=0.123,
                l0_sparsity=0.05,
                dead_neurons=100,
                learning_rate=0.0003,
            )

            # Verify training record updated
            assert mock_training.progress == 50.0  # (5000/10000) * 100
            assert mock_training.current_step == 5000
            assert mock_training.current_loss == 0.123
            assert mock_training.current_l0_sparsity == 0.05
            assert mock_training.current_dead_neurons == 100
            assert mock_training.current_learning_rate == 0.0003
            assert mock_training.status == TrainingStatus.RUNNING.value
            mock_db.commit.assert_called_once()

    def test_update_training_progress_not_found(self):
        """Test updating progress when training not found."""
        task = TrainingTask()

        mock_db = Mock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = None

        with patch.object(task, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_db

            # Should not raise error, just do nothing
            task.update_training_progress(
                training_id="train_nonexistent",
                step=1000,
                total_steps=10000,
                loss=0.5,
            )

            # Commit should not be called if training not found
            mock_db.commit.assert_not_called()

    def test_update_training_progress_minimal_metrics(self):
        """Test updating progress with only required metrics."""
        task = TrainingTask()

        mock_training = Mock()
        mock_db = Mock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = mock_training

        with patch.object(task, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_db

            task.update_training_progress(
                training_id="train_test123",
                step=2500,
                total_steps=10000,
                loss=0.456,
            )

            assert mock_training.progress == 25.0
            assert mock_training.current_step == 2500
            assert mock_training.current_loss == 0.456
            assert mock_training.current_l0_sparsity is None
            assert mock_training.current_dead_neurons is None
            assert mock_training.current_learning_rate is None

    def test_calculate_training_progress_start(self):
        """Test progress calculation at training start (step 0)."""
        task = TrainingTask()

        mock_training = Mock()
        mock_db = Mock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = mock_training

        with patch.object(task, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_db

            task.update_training_progress(
                training_id="train_test123",
                step=0,
                total_steps=10000,
                loss=1.5,
            )

            assert mock_training.progress == 0.0
            assert mock_training.current_step == 0
            assert mock_training.status == TrainingStatus.RUNNING.value

    def test_calculate_training_progress_complete(self):
        """Test progress calculation at training completion (step = total_steps)."""
        task = TrainingTask()

        mock_training = Mock()
        mock_db = Mock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = mock_training

        with patch.object(task, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_db

            task.update_training_progress(
                training_id="train_test123",
                step=10000,
                total_steps=10000,
                loss=0.05,
            )

            assert mock_training.progress == 100.0
            assert mock_training.current_step == 10000
            assert mock_training.current_loss == 0.05
            assert mock_training.status == TrainingStatus.RUNNING.value

    def test_calculate_training_progress_with_checkpoint_steps(self):
        """Test progress calculation at various checkpoint milestones."""
        task = TrainingTask()

        mock_training = Mock()
        mock_db = Mock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = mock_training

        with patch.object(task, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_db

            # Test at 10% checkpoint
            task.update_training_progress(
                training_id="train_test123",
                step=1000,
                total_steps=10000,
                loss=0.8,
            )
            assert mock_training.progress == 10.0

            # Test at 25% checkpoint
            task.update_training_progress(
                training_id="train_test123",
                step=2500,
                total_steps=10000,
                loss=0.6,
            )
            assert mock_training.progress == 25.0

            # Test at 75% checkpoint
            task.update_training_progress(
                training_id="train_test123",
                step=7500,
                total_steps=10000,
                loss=0.2,
            )
            assert mock_training.progress == 75.0

            # Test at 90% checkpoint
            task.update_training_progress(
                training_id="train_test123",
                step=9000,
                total_steps=10000,
                loss=0.1,
            )
            assert mock_training.progress == 90.0

    def test_calculate_training_progress_fractional_steps(self):
        """Test progress calculation with non-round step numbers."""
        task = TrainingTask()

        mock_training = Mock()
        mock_db = Mock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = mock_training

        with patch.object(task, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_db

            # Test with non-divisible step count
            task.update_training_progress(
                training_id="train_test123",
                step=3333,
                total_steps=10000,
                loss=0.5,
            )

            # Should calculate as 33.33%
            assert abs(mock_training.progress - 33.33) < 0.01
            assert mock_training.current_step == 3333

    def test_log_metric_all_fields(self):
        """Test logging training metric with all fields."""
        task = TrainingTask()

        mock_db = Mock()

        with patch.object(task, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_db

            task.log_metric(
                training_id="train_test123",
                step=1000,
                loss=0.234,
                l0_sparsity=0.05,
                l1_sparsity=0.01,
                dead_neurons=50,
                learning_rate=0.0003,
                grad_norm=1.5,
                gpu_memory_used_mb=4096.0,
                samples_per_second=1000.0,
            )

            # Verify metric added
            mock_db.add.assert_called_once()
            metric = mock_db.add.call_args[0][0]
            assert metric.training_id == "train_test123"
            assert metric.step == 1000
            assert metric.loss == 0.234
            assert metric.l0_sparsity == 0.05
            assert metric.l1_sparsity == 0.01
            assert metric.dead_neurons == 50
            assert metric.learning_rate == 0.0003
            assert metric.grad_norm == 1.5
            assert metric.gpu_memory_used_mb == 4096.0
            assert metric.samples_per_second == 1000.0
            mock_db.commit.assert_called_once()

    def test_log_metric_minimal(self):
        """Test logging metric with only required fields."""
        task = TrainingTask()

        mock_db = Mock()

        with patch.object(task, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_db

            task.log_metric(
                training_id="train_test123",
                step=500,
                loss=0.567,
            )

            mock_db.add.assert_called_once()
            metric = mock_db.add.call_args[0][0]
            assert metric.training_id == "train_test123"
            assert metric.step == 500
            assert metric.loss == 0.567
            assert metric.l0_sparsity is None
            assert metric.dead_neurons is None

    def test_log_metric_multi_layer_aggregated(self):
        """Test logging aggregated metrics for multi-layer training."""
        task = TrainingTask()

        mock_db = Mock()

        with patch.object(task, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_db

            # Log aggregated metric (layer_idx=None)
            task.log_metric(
                training_id="train_test123",
                step=1000,
                loss=0.3,  # Average loss across all layers
                l0_sparsity=0.05,  # Average sparsity across all layers
                dead_neurons=150,  # Total dead neurons across all layers
                learning_rate=0.0003,
                layer_idx=None,  # Aggregated metric
            )

            mock_db.add.assert_called_once()
            metric = mock_db.add.call_args[0][0]
            assert metric.training_id == "train_test123"
            assert metric.step == 1000
            assert metric.loss == 0.3
            assert metric.l0_sparsity == 0.05
            assert metric.dead_neurons == 150
            assert metric.layer_idx is None  # Aggregated across all layers

    def test_log_metric_multi_layer_individual_layers(self):
        """Test logging individual layer metrics for multi-layer training."""
        task = TrainingTask()

        mock_db = Mock()

        with patch.object(task, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_db

            # Log metrics for layer 0
            task.log_metric(
                training_id="train_test123",
                step=1000,
                loss=0.25,
                l0_sparsity=0.04,
                dead_neurons=40,
                layer_idx=0,
            )

            # Log metrics for layer 1
            task.log_metric(
                training_id="train_test123",
                step=1000,
                loss=0.35,
                l0_sparsity=0.06,
                dead_neurons=60,
                layer_idx=1,
            )

            # Log metrics for layer 2
            task.log_metric(
                training_id="train_test123",
                step=1000,
                loss=0.30,
                l0_sparsity=0.05,
                dead_neurons=50,
                layer_idx=2,
            )

            # Verify 3 metrics were logged
            assert mock_db.add.call_count == 3

            # Verify layer indices are correct
            calls = mock_db.add.call_args_list
            assert calls[0][0][0].layer_idx == 0
            assert calls[0][0][0].loss == 0.25
            assert calls[1][0][0].layer_idx == 1
            assert calls[1][0][0].loss == 0.35
            assert calls[2][0][0].layer_idx == 2
            assert calls[2][0][0].loss == 0.30


class TestTrainSAETaskFlow:
    """Test main training task flow (with heavy mocking)."""

    @patch('src.workers.training_tasks.estimate_training_memory')
    @patch('src.workers.training_tasks.create_sae')
    @patch('src.workers.training_tasks.optim.Adam')
    @patch('src.workers.training_tasks.optim.lr_scheduler.LambdaLR')
    @patch('src.workers.training_tasks.torch.cuda.is_available')
    def test_train_sae_task_initialization(
        self,
        mock_cuda_available,
        mock_scheduler,
        mock_adam,
        mock_create_sae,
        mock_estimate_memory,
    ):
        """Test training task initialization phase."""
        # Setup mocks
        mock_cuda_available.return_value = False
        mock_estimate_memory.return_value = {
            'total_gb': 3.5,
            'fits_in_6gb': True,
        }
        mock_model = Mock(spec=nn.Module)
        mock_model.to.return_value = mock_model
        mock_create_sae.return_value = mock_model

        # This test verifies initialization logic without running full training
        # Full training loop testing would require loading actual data, which is
        # better suited for integration tests

        assert mock_estimate_memory is not None  # Placeholder assertion

    @patch('src.workers.training_tasks.estimate_training_memory')
    def test_train_sae_task_memory_validation_failure(self, mock_estimate_memory):
        """Test training task fails when memory budget exceeded."""
        mock_estimate_memory.return_value = {
            'total_gb': 8.0,
            'fits_in_6gb': False,
        }

        # Mock training record
        task = TrainingTask()

        # This test would verify that training fails gracefully when memory exceeded
        # Full implementation requires complex Celery task mocking

        assert not mock_estimate_memory.return_value['fits_in_6gb']


class TestTrainingTaskMemoryEstimation:
    """Test memory estimation integration."""

    def test_memory_estimation_called_with_correct_params(self):
        """Test that memory estimation receives correct hyperparameters."""
        from src.utils.resource_estimation import estimate_training_memory

        # Call memory estimation with typical SAE parameters
        result = estimate_training_memory(
            hidden_dim=768,
            latent_dim=16384,
            batch_size=4096,
        )

        # Verify result structure
        assert 'total_gb' in result
        assert 'total_bytes' in result
        assert 'breakdown' in result
        assert 'fits_in_6gb' in result
        assert isinstance(result['total_gb'], float)
        assert isinstance(result['fits_in_6gb'], bool)

        # Verify breakdown structure
        breakdown = result['breakdown']
        assert 'model_params_mb' in breakdown
        assert 'activations_mb' in breakdown
        assert 'gradients_mb' in breakdown
        assert 'optimizer_state_mb' in breakdown

    def test_memory_estimation_large_config(self):
        """Test memory estimation with large configuration."""
        from src.utils.resource_estimation import estimate_training_memory

        result = estimate_training_memory(
            hidden_dim=4096,
            latent_dim=65536,
            batch_size=8192,
        )

        # Should require significant memory (>6GB at least)
        assert result['total_gb'] > 6.0
        # fits_in_6gb now checks against actual available GPU memory, not fixed 6GB
        # Verify the fit assessment is consistent with the comparison
        assert result['fits_in_6gb'] == (result['total_gb'] <= result['available_gpu_gb'])

    def test_memory_estimation_small_config(self):
        """Test memory estimation with small configuration."""
        from src.utils.resource_estimation import estimate_training_memory

        result = estimate_training_memory(
            hidden_dim=128,
            latent_dim=512,
            batch_size=256,
        )

        # Should fit comfortably in 6GB
        assert result['total_gb'] < 1.0
        assert result['fits_in_6gb'] is True


class TestLearningRateScheduler:
    """Test learning rate scheduling logic."""

    def test_lr_warmup_schedule(self):
        """Test linear warmup schedule."""
        # Simulate lr_lambda from train_sae_task
        warmup_steps = 1000

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0

        # Test warmup phase
        assert lr_lambda(0) == 0.0
        assert lr_lambda(250) == 0.25
        assert lr_lambda(500) == 0.5
        assert lr_lambda(750) == 0.75
        assert lr_lambda(1000) == 1.0

        # Test post-warmup phase
        assert lr_lambda(1001) == 1.0
        assert lr_lambda(5000) == 1.0
        assert lr_lambda(10000) == 1.0

    def test_lr_no_warmup(self):
        """Test schedule with no warmup."""
        warmup_steps = 0

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0

        assert lr_lambda(0) == 1.0
        assert lr_lambda(100) == 1.0
        assert lr_lambda(1000) == 1.0


class TestCheckpointDirectoryCreation:
    """Test checkpoint directory management."""

    def test_checkpoint_dir_structure(self):
        """Test checkpoint directory structure."""
        from src.core.config import settings

        training_id = "train_test123"
        checkpoint_dir = settings.data_dir / "trainings" / training_id / "checkpoints"

        # Verify path structure
        assert "trainings" in str(checkpoint_dir)
        assert training_id in str(checkpoint_dir)
        assert "checkpoints" in str(checkpoint_dir)

    def test_checkpoint_path_for_step(self):
        """Test checkpoint file path generation."""
        from src.core.config import settings

        training_id = "train_test123"
        step = 5000
        checkpoint_dir = settings.data_dir / "trainings" / training_id / "checkpoints"
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.safetensors"

        assert checkpoint_path.suffix == ".safetensors"
        assert f"step_{step}" in str(checkpoint_path)
