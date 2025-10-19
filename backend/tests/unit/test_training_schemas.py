"""
Unit tests for Training Pydantic schemas.

Tests TrainingHyperparameters, TrainingCreate, and validation logic
for SAE training configuration.
"""

import pytest
from pydantic import ValidationError

from src.schemas.training import (
    TrainingHyperparameters,
    TrainingCreate,
    SAEArchitectureType,
)


class TestTrainingHyperparametersValidation:
    """Test TrainingHyperparameters validation rules."""

    def test_valid_hyperparameters(self):
        """Test that valid hyperparameters pass validation."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            architecture_type=SAEArchitectureType.STANDARD,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
            warmup_steps=1000,
            checkpoint_interval=5000,
            log_interval=100,
        )

        assert hp.hidden_dim == 768
        assert hp.latent_dim == 16384
        assert hp.architecture_type == SAEArchitectureType.STANDARD
        assert hp.l1_alpha == 0.001
        assert hp.learning_rate == 0.0003
        assert hp.batch_size == 4096
        assert hp.total_steps == 100000

    def test_hidden_dim_must_be_positive(self):
        """Test that hidden_dim must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=0,  # Invalid
                latent_dim=16384,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('hidden_dim',)
        assert 'greater than 0' in error['msg']

    def test_hidden_dim_cannot_be_negative(self):
        """Test that hidden_dim cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=-100,  # Invalid
                latent_dim=16384,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('hidden_dim',)

    def test_latent_dim_must_be_positive(self):
        """Test that latent_dim must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=0,  # Invalid
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('latent_dim',)
        assert 'greater than 0' in error['msg']

    def test_l1_alpha_must_be_positive(self):
        """Test that l1_alpha must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                l1_alpha=0.0,  # Invalid
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('l1_alpha',)
        assert 'greater than 0' in error['msg']

    def test_l1_alpha_cannot_be_negative(self):
        """Test that l1_alpha cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                l1_alpha=-0.001,  # Invalid
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('l1_alpha',)

    def test_learning_rate_must_be_positive(self):
        """Test that learning_rate must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                l1_alpha=0.001,
                learning_rate=0.0,  # Invalid
                batch_size=4096,
                total_steps=100000,
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('learning_rate',)
        assert 'greater than 0' in error['msg']

    def test_batch_size_must_be_positive(self):
        """Test that batch_size must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=0,  # Invalid
                total_steps=100000,
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('batch_size',)
        assert 'greater than 0' in error['msg']

    def test_total_steps_must_be_positive(self):
        """Test that total_steps must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=0,  # Invalid
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('total_steps',)
        assert 'greater than 0' in error['msg']

    def test_warmup_steps_can_be_zero(self):
        """Test that warmup_steps can be 0."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
            warmup_steps=0,  # Valid
        )

        assert hp.warmup_steps == 0

    def test_warmup_steps_cannot_be_negative(self):
        """Test that warmup_steps cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
                warmup_steps=-100,  # Invalid
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('warmup_steps',)

    def test_weight_decay_can_be_zero(self):
        """Test that weight_decay can be 0."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
            weight_decay=0.0,  # Valid
        )

        assert hp.weight_decay == 0.0

    def test_weight_decay_cannot_be_negative(self):
        """Test that weight_decay cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
                weight_decay=-0.01,  # Invalid
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('weight_decay',)

    def test_checkpoint_interval_must_be_positive(self):
        """Test that checkpoint_interval must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
                checkpoint_interval=0,  # Invalid
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('checkpoint_interval',)
        assert 'greater than 0' in error['msg']

    def test_log_interval_must_be_positive(self):
        """Test that log_interval must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
                log_interval=0,  # Invalid
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('log_interval',)
        assert 'greater than 0' in error['msg']

    def test_target_l0_must_be_positive_if_provided(self):
        """Test that target_l0 must be > 0 if provided."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
                target_l0=0.0,  # Invalid
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('target_l0',)

    def test_target_l0_cannot_exceed_one(self):
        """Test that target_l0 must be <= 1."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
                target_l0=1.5,  # Invalid (>1)
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('target_l0',)
        assert 'less than or equal to 1' in error['msg']

    def test_target_l0_can_be_none(self):
        """Test that target_l0 can be None."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
            target_l0=None,  # Valid
        )

        assert hp.target_l0 is None

    def test_grad_clip_norm_must_be_positive_if_provided(self):
        """Test that grad_clip_norm must be > 0 if provided."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
                grad_clip_norm=0.0,  # Invalid
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('grad_clip_norm',)

    def test_grad_clip_norm_can_be_none(self):
        """Test that grad_clip_norm can be None."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
            grad_clip_norm=None,  # Valid
        )

        assert hp.grad_clip_norm is None


class TestSAEArchitectureType:
    """Test SAE architecture type enum."""

    def test_architecture_type_standard(self):
        """Test standard architecture type."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            architecture_type=SAEArchitectureType.STANDARD,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        assert hp.architecture_type == SAEArchitectureType.STANDARD
        assert hp.architecture_type.value == "standard"

    def test_architecture_type_skip(self):
        """Test skip architecture type."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            architecture_type=SAEArchitectureType.SKIP,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        assert hp.architecture_type == SAEArchitectureType.SKIP
        assert hp.architecture_type.value == "skip"

    def test_architecture_type_transcoder(self):
        """Test transcoder architecture type."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            architecture_type=SAEArchitectureType.TRANSCODER,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        assert hp.architecture_type == SAEArchitectureType.TRANSCODER
        assert hp.architecture_type.value == "transcoder"

    def test_architecture_type_from_string(self):
        """Test creating architecture type from string."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            architecture_type="standard",  # String
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        assert hp.architecture_type == SAEArchitectureType.STANDARD

    def test_invalid_architecture_type(self):
        """Test that invalid architecture type raises error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingHyperparameters(
                hidden_dim=768,
                latent_dim=16384,
                architecture_type="invalid",  # Invalid
                l1_alpha=0.001,
                learning_rate=0.0003,
                batch_size=4096,
                total_steps=100000,
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('architecture_type',)


class TestTrainingCreateValidation:
    """Test TrainingCreate schema validation."""

    def test_valid_training_create(self):
        """Test that valid TrainingCreate passes validation."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training = TrainingCreate(
            model_id="m_abc123",
            dataset_id="ds_xyz789",
            hyperparameters=hp,
        )

        assert training.model_id == "m_abc123"
        assert training.dataset_id == "ds_xyz789"
        assert training.extraction_id is None
        assert training.hyperparameters == hp

    def test_model_id_must_start_with_m_prefix(self):
        """Test that model_id must start with 'm_'."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        with pytest.raises(ValidationError) as exc_info:
            TrainingCreate(
                model_id="invalid_id",  # Missing 'm_' prefix
                dataset_id="ds_xyz789",
                hyperparameters=hp,
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('model_id',)
        assert "must start with 'm_'" in str(exc_info.value)

    def test_model_id_cannot_be_empty(self):
        """Test that model_id cannot be empty."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        with pytest.raises(ValidationError) as exc_info:
            TrainingCreate(
                model_id="",  # Empty
                dataset_id="ds_xyz789",
                hyperparameters=hp,
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('model_id',)

    def test_dataset_id_cannot_be_empty(self):
        """Test that dataset_id cannot be empty."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        with pytest.raises(ValidationError) as exc_info:
            TrainingCreate(
                model_id="m_abc123",
                dataset_id="",  # Empty
                hyperparameters=hp,
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('dataset_id',)

    def test_extraction_id_must_start_with_ext_m_prefix(self):
        """Test that extraction_id must start with 'ext_m_'."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        with pytest.raises(ValidationError) as exc_info:
            TrainingCreate(
                model_id="m_abc123",
                dataset_id="ds_xyz789",
                extraction_id="invalid_id",  # Missing 'ext_m_' prefix
                hyperparameters=hp,
            )

        error = exc_info.value.errors()[0]
        assert error['loc'] == ('extraction_id',)
        assert "must start with 'ext_m_'" in str(exc_info.value)

    def test_extraction_id_can_be_none(self):
        """Test that extraction_id can be None."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training = TrainingCreate(
            model_id="m_abc123",
            dataset_id="ds_xyz789",
            extraction_id=None,  # Valid
            hyperparameters=hp,
        )

        assert training.extraction_id is None

    def test_training_create_with_extraction_id(self):
        """Test TrainingCreate with valid extraction_id."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        training = TrainingCreate(
            model_id="m_abc123",
            dataset_id="ds_xyz789",
            extraction_id="ext_m_abc123",  # Valid
            hyperparameters=hp,
        )

        assert training.extraction_id == "ext_m_abc123"


class TestTrainingHyperparametersDefaults:
    """Test default values for TrainingHyperparameters."""

    def test_default_warmup_steps(self):
        """Test default warmup_steps is 0."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        assert hp.warmup_steps == 0

    def test_default_weight_decay(self):
        """Test default weight_decay is 0.0."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        assert hp.weight_decay == 0.0

    def test_default_checkpoint_interval(self):
        """Test default checkpoint_interval is 1000."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        assert hp.checkpoint_interval == 1000

    def test_default_log_interval(self):
        """Test default log_interval is 100."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        assert hp.log_interval == 100

    def test_default_architecture_type(self):
        """Test default architecture_type is STANDARD."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        assert hp.architecture_type == SAEArchitectureType.STANDARD

    def test_default_resample_dead_neurons(self):
        """Test default resample_dead_neurons is True."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        assert hp.resample_dead_neurons is True

    def test_default_dead_neuron_threshold(self):
        """Test default dead_neuron_threshold is 1000."""
        hp = TrainingHyperparameters(
            hidden_dim=768,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0003,
            batch_size=4096,
            total_steps=100000,
        )

        assert hp.dead_neuron_threshold == 1000
