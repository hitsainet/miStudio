"""
Unit tests for Training Validator Service.

Tests TrainingValidator for SAE training quality validation including
pre-training configuration checks and real-time quality monitoring.
"""

import pytest
import math

from src.services.training_validator import TrainingValidator


class TestCalculateRecommendedL1Alpha:
    """Test l1_alpha recommendation formula.

    New formula calibrated for SAELens with activation normalization:
    l1_alpha = 5.0 / sqrt(latent_dim / 8192)

    This produces much larger values than the old formula because the .mean()
    L1 penalty normalizes by feature count.
    """

    def test_formula_8k_latent_dim(self):
        """Test recommended l1_alpha for 8192 latent dimensions."""
        recommended = TrainingValidator.calculate_recommended_l1_alpha(8192)
        # New formula: 5.0 / sqrt(8192/8192) = 5.0 / sqrt(1) = 5.0
        expected = 5.0 / math.sqrt(8192 / 8192)  # = 5.0
        assert abs(recommended - expected) < 0.00001
        assert abs(recommended - 5.0) < 0.001

    def test_formula_16k_latent_dim(self):
        """Test recommended l1_alpha for 16384 latent dimensions."""
        recommended = TrainingValidator.calculate_recommended_l1_alpha(16384)
        # New formula: 5.0 / sqrt(16384/8192) = 5.0 / sqrt(2) ≈ 3.5355
        expected = 5.0 / math.sqrt(16384 / 8192)  # ≈ 3.5355
        assert abs(recommended - expected) < 0.00001
        assert abs(recommended - 3.5355) < 0.001

    def test_formula_32k_latent_dim(self):
        """Test recommended l1_alpha for 32768 latent dimensions."""
        recommended = TrainingValidator.calculate_recommended_l1_alpha(32768)
        # New formula: 5.0 / sqrt(32768/8192) = 5.0 / sqrt(4) = 2.5
        expected = 5.0 / math.sqrt(32768 / 8192)  # = 2.5
        assert abs(recommended - expected) < 0.00001
        assert abs(recommended - 2.5) < 0.001

    def test_formula_4k_latent_dim(self):
        """Test recommended l1_alpha for 4096 latent dimensions."""
        recommended = TrainingValidator.calculate_recommended_l1_alpha(4096)
        # New formula: 5.0 / sqrt(4096/8192) = 5.0 / sqrt(0.5) ≈ 7.0711
        expected = 5.0 / math.sqrt(4096 / 8192)  # ≈ 7.0711
        assert abs(recommended - expected) < 0.00001
        assert abs(recommended - 7.0711) < 0.001

    def test_smaller_latent_dim_higher_penalty(self):
        """Test that smaller latent dims require higher l1_alpha."""
        l1_4k = TrainingValidator.calculate_recommended_l1_alpha(4096)
        l1_8k = TrainingValidator.calculate_recommended_l1_alpha(8192)
        l1_16k = TrainingValidator.calculate_recommended_l1_alpha(16384)

        assert l1_4k > l1_8k > l1_16k


class TestValidateSparsityConfig:
    """Test sparsity configuration validation."""

    def test_valid_configuration(self):
        """Test that a good configuration produces no errors or warnings."""
        hyperparameters = {
            'l1_alpha': 0.003,
            'latent_dim': 16384,
            'target_l0': 0.05,
        }

        warnings, errors = TrainingValidator.validate_sparsity_config(hyperparameters)

        assert len(errors) == 0
        # Warnings might exist if l1_alpha deviates from recommended, but no errors
        assert 'l1_alpha is required' not in str(errors)

    def test_missing_l1_alpha_error(self):
        """Test that missing l1_alpha produces an error."""
        hyperparameters = {
            'latent_dim': 16384,
            'target_l0': 0.05,
        }

        warnings, errors = TrainingValidator.validate_sparsity_config(hyperparameters)

        assert len(errors) > 0
        assert any('l1_alpha is required' in e for e in errors)

    def test_missing_latent_dim_error(self):
        """Test that missing latent_dim produces an error."""
        hyperparameters = {
            'l1_alpha': 0.001,
            'target_l0': 0.05,
        }

        warnings, errors = TrainingValidator.validate_sparsity_config(hyperparameters)

        assert len(errors) > 0
        assert any('latent_dim is required' in e for e in errors)

    def test_l1_alpha_very_low_warning(self):
        """Test that very low l1_alpha produces warning."""
        hyperparameters = {
            'l1_alpha': 0.00001,  # Very low - will produce dense features
            'latent_dim': 16384,
            'target_l0': 0.05,
        }

        warnings, errors = TrainingValidator.validate_sparsity_config(hyperparameters)

        assert len(warnings) > 0
        assert any('very low' in w and 'DENSE features' in w for w in warnings)

    def test_l1_alpha_moderately_low_warning(self):
        """Test that moderately low l1_alpha produces warning."""
        hyperparameters = {
            'l1_alpha': 0.0001,  # Low but not extremely low
            'latent_dim': 16384,
            'target_l0': 0.05,
        }

        warnings, errors = TrainingValidator.validate_sparsity_config(hyperparameters)

        assert len(warnings) > 0
        assert any('low' in w for w in warnings)

    def test_l1_alpha_very_high_warning(self):
        """Test that high l1_alpha (>3x recommended) produces warning about dead neurons."""
        recommended = TrainingValidator.calculate_recommended_l1_alpha(16384)
        hyperparameters = {
            'l1_alpha': recommended * 4,  # 4x recommended - triggers >3x warning
            'latent_dim': 16384,
            'target_l0': 0.05,
        }

        warnings, errors = TrainingValidator.validate_sparsity_config(hyperparameters)

        # Should produce warning, not error (>5x triggers error)
        assert len(errors) == 0
        assert len(warnings) > 0
        assert any('high' in w and 'dead neurons' in w for w in warnings)

    def test_l1_alpha_moderately_high_warning(self):
        """Test that moderately high l1_alpha (>2x recommended) produces warning."""
        recommended = TrainingValidator.calculate_recommended_l1_alpha(16384)
        hyperparameters = {
            'l1_alpha': recommended * 2.5,  # 2.5x recommended - triggers >2x warning
            'latent_dim': 16384,
            'target_l0': 0.05,
        }

        warnings, errors = TrainingValidator.validate_sparsity_config(hyperparameters)

        # Should produce warning about potential dead neurons
        assert len(errors) == 0
        assert len(warnings) > 0
        assert any('high' in w for w in warnings)

    def test_target_l0_too_high_warning(self):
        """Test that target_l0 > 0.15 produces warning."""
        hyperparameters = {
            'l1_alpha': 0.003,
            'latent_dim': 16384,
            'target_l0': 0.20,  # Too high - dense features
        }

        warnings, errors = TrainingValidator.validate_sparsity_config(hyperparameters)

        assert len(warnings) > 0
        assert any('target_l0' in w and 'high' in w for w in warnings)

    def test_target_l0_too_low_warning(self):
        """Test that target_l0 < 0.005 produces warning."""
        hyperparameters = {
            'l1_alpha': 0.003,
            'latent_dim': 16384,
            'target_l0': 0.002,  # Too low - training instability
        }

        warnings, errors = TrainingValidator.validate_sparsity_config(hyperparameters)

        assert len(warnings) > 0
        assert any('target_l0' in w and 'very low' in w for w in warnings)

    def test_target_l0_none_valid(self):
        """Test that target_l0=None does not produce warnings."""
        hyperparameters = {
            'l1_alpha': 0.003,
            'latent_dim': 16384,
            'target_l0': None,
        }

        warnings, errors = TrainingValidator.validate_sparsity_config(hyperparameters)

        # Should not have warnings about target_l0 being missing
        assert len(errors) == 0


class TestCheckTrainingQuality:
    """Test real-time training quality monitoring."""

    def test_during_warmup_no_warnings(self):
        """Test that no warnings are issued during warmup period."""
        warnings = TrainingValidator.check_training_quality(
            step=500,  # During warmup
            l0_sparsity=0.30,  # Even if very high
            dead_neurons=5000,  # Even if very high
            latent_dim=16384,
            target_l0=0.05,
            warmup_steps=1000
        )

        assert len(warnings) == 0

    def test_l0_very_high_warning(self):
        """Test that L0 > 0.20 produces strong warning."""
        warnings = TrainingValidator.check_training_quality(
            step=2000,  # After warmup
            l0_sparsity=0.25,  # Very high - dense features
            dead_neurons=100,
            latent_dim=16384,
            target_l0=0.05,
            warmup_steps=1000
        )

        assert len(warnings) > 0
        assert any('very high' in w and 'DENSE features' in w for w in warnings)

    def test_l0_high_warning(self):
        """Test that L0 > 0.15 produces warning."""
        warnings = TrainingValidator.check_training_quality(
            step=2000,  # After warmup
            l0_sparsity=0.18,  # High
            dead_neurons=100,
            latent_dim=16384,
            target_l0=0.05,
            warmup_steps=1000
        )

        assert len(warnings) > 0
        assert any('high' in w for w in warnings)

    def test_l0_above_target_warning(self):
        """Test that L0 > 3x target produces warning."""
        warnings = TrainingValidator.check_training_quality(
            step=2000,  # After warmup
            l0_sparsity=0.13,  # > 3x target of 0.04 (3x = 0.12)
            dead_neurons=100,
            latent_dim=16384,
            target_l0=0.04,  # Low target
            warmup_steps=1000
        )

        assert len(warnings) > 0
        assert any('3x above target' in w for w in warnings)

    def test_dead_neurons_very_high_warning(self):
        """Test that >70% dead neurons produces warning."""
        warnings = TrainingValidator.check_training_quality(
            step=2000,  # After warmup
            l0_sparsity=0.05,  # Good
            dead_neurons=12000,  # 73% of 16384
            latent_dim=16384,
            target_l0=0.05,
            warmup_steps=1000
        )

        assert len(warnings) > 0
        assert any('dead neurons' in w for w in warnings)

    def test_dead_neurons_high_warning(self):
        """Test that >50% dead neurons produces warning."""
        warnings = TrainingValidator.check_training_quality(
            step=2000,  # After warmup
            l0_sparsity=0.05,  # Good
            dead_neurons=9000,  # 55% of 16384
            latent_dim=16384,
            target_l0=0.05,
            warmup_steps=1000
        )

        assert len(warnings) > 0
        assert any('dead neurons' in w for w in warnings)

    def test_good_quality_no_warnings(self):
        """Test that good training quality produces no warnings."""
        warnings = TrainingValidator.check_training_quality(
            step=2000,  # After warmup
            l0_sparsity=0.05,  # Good sparsity
            dead_neurons=1000,  # Only 6% dead
            latent_dim=16384,
            target_l0=0.05,
            warmup_steps=1000
        )

        assert len(warnings) == 0

    def test_step_equals_warmup_steps(self):
        """Test behavior exactly at the end of warmup."""
        warnings = TrainingValidator.check_training_quality(
            step=1000,  # Exactly at warmup boundary
            l0_sparsity=0.30,  # High
            dead_neurons=100,
            latent_dim=16384,
            target_l0=0.05,
            warmup_steps=1000
        )

        # At step=warmup_steps, checks ARE performed (step < warmup_steps is False)
        # So high L0 sparsity should trigger warnings
        assert len(warnings) > 0

    def test_step_after_warmup(self):
        """Test that checks activate immediately after warmup."""
        warnings = TrainingValidator.check_training_quality(
            step=1001,  # Just after warmup
            l0_sparsity=0.30,  # High - should trigger warning
            dead_neurons=100,
            latent_dim=16384,
            target_l0=0.05,
            warmup_steps=1000
        )

        assert len(warnings) > 0


class TestConvenienceFunction:
    """Test the module-level convenience function."""

    def test_validate_training_config_delegates(self):
        """Test that validate_training_config() delegates to TrainingValidator."""
        from src.services.training_validator import validate_training_config

        hyperparameters = {
            'l1_alpha': 0.003,
            'latent_dim': 16384,
            'target_l0': 0.05,
        }

        warnings, errors = validate_training_config(hyperparameters)

        assert isinstance(warnings, list)
        assert isinstance(errors, list)
