"""
Training Configuration Validator Service

Validates training hyperparameters to ensure high-quality sparse feature learning.
Provides warnings and recommendations for sparsity configuration.
"""

import logging
import math
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class TrainingValidator:
    """Validator for SAE training hyperparameters and quality metrics."""

    @staticmethod
    def calculate_recommended_l1_alpha(latent_dim: int) -> float:
        """
        Calculate recommended l1_alpha based on latent dimension.

        Formula: l1_alpha = 10 / sqrt(latent_dim)

        This provides a reasonable starting point that scales with SAE width.
        Larger SAEs need proportionally smaller penalties to avoid killing all features.

        Args:
            latent_dim: SAE latent dimension (width)

        Returns:
            Recommended l1_alpha value

        Examples:
            latent_dim=8192  → l1_alpha ≈ 0.0035
            latent_dim=16384 → l1_alpha ≈ 0.0025
            latent_dim=32768 → l1_alpha ≈ 0.0017
        """
        return 10.0 / math.sqrt(latent_dim)

    @staticmethod
    def validate_sparsity_config(
        hyperparameters: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Validate sparsity configuration for training quality.

        Checks if l1_alpha is appropriate for the latent dimension and
        provides warnings/errors if the configuration is likely to produce
        poor quality features (too dense or too sparse).

        Args:
            hyperparameters: Training hyperparameters dictionary

        Returns:
            Tuple of (warnings, errors) as lists of strings
        """
        warnings = []
        errors = []

        l1_alpha = hyperparameters.get('l1_alpha')
        latent_dim = hyperparameters.get('latent_dim')
        target_l0 = hyperparameters.get('target_l0')

        if l1_alpha is None:
            errors.append(
                "l1_alpha is required. This is the sparsity penalty coefficient "
                "that enforces sparse feature learning."
            )
            return warnings, errors

        if latent_dim is None:
            errors.append("latent_dim is required for sparsity validation.")
            return warnings, errors

        # Calculate recommended l1_alpha
        recommended_l1_alpha = TrainingValidator.calculate_recommended_l1_alpha(latent_dim)

        # Check if l1_alpha is too low (will produce dense features)
        if l1_alpha < recommended_l1_alpha * 0.1:
            warnings.append(
                f"⚠️  l1_alpha ({l1_alpha:.6f}) is very low for latent_dim ({latent_dim}). "
                f"Recommended: {recommended_l1_alpha:.6f}. "
                f"This will likely produce DENSE features (L0 > 0.20) which are not interpretable. "
                f"Consider increasing l1_alpha to at least {recommended_l1_alpha * 0.5:.6f}."
            )
        elif l1_alpha < recommended_l1_alpha * 0.5:
            warnings.append(
                f"⚠️  l1_alpha ({l1_alpha:.6f}) is low for latent_dim ({latent_dim}). "
                f"Recommended: {recommended_l1_alpha:.6f}. "
                f"Features may be denser than ideal (L0 > 0.10). "
                f"Consider using {recommended_l1_alpha:.6f} for better sparsity."
            )

        # Check if l1_alpha is too high (will kill features)
        if l1_alpha > recommended_l1_alpha * 10:
            warnings.append(
                f"⚠️  l1_alpha ({l1_alpha:.6f}) is very high for latent_dim ({latent_dim}). "
                f"Recommended: {recommended_l1_alpha:.6f}. "
                f"This may cause too many dead neurons (>50%). "
                f"Consider decreasing l1_alpha to {recommended_l1_alpha * 2:.6f}."
            )
        elif l1_alpha > recommended_l1_alpha * 5:
            warnings.append(
                f"⚠️  l1_alpha ({l1_alpha:.6f}) is high for latent_dim ({latent_dim}). "
                f"Recommended: {recommended_l1_alpha:.6f}. "
                f"This may cause many dead neurons. Monitor dead_neurons metric during training."
            )

        # Validate target_l0 if provided
        if target_l0 is not None:
            if target_l0 > 0.15:
                warnings.append(
                    f"⚠️  target_l0 ({target_l0:.2f}) is high. "
                    f"For interpretable features, aim for L0 < 0.05 (5% activation rate). "
                    f"L0 > 0.15 typically indicates dense, polysemantic features."
                )
            elif target_l0 < 0.005:
                warnings.append(
                    f"⚠️  target_l0 ({target_l0:.3f}) is very low. "
                    f"This may be too sparse and cause training instability. "
                    f"Consider target_l0 between 0.01-0.05."
                )

        return warnings, errors

    @staticmethod
    def check_training_quality(
        step: int,
        l0_sparsity: float,
        dead_neurons: int,
        latent_dim: int,
        target_l0: float = 0.05,
        warmup_steps: int = 1000
    ) -> List[str]:
        """
        Check training quality metrics during training.

        Provides real-time warnings if L0 sparsity or dead neuron count
        indicate poor training quality.

        Args:
            step: Current training step
            l0_sparsity: Current L0 sparsity (fraction of active features)
            dead_neurons: Current count of dead neurons
            latent_dim: SAE latent dimension
            target_l0: Target L0 sparsity
            warmup_steps: Number of warmup steps (skip checks during warmup)

        Returns:
            List of warning messages
        """
        warnings = []

        # Skip quality checks during warmup period
        if step < warmup_steps:
            return warnings

        # Check L0 sparsity (too dense)
        if l0_sparsity > 0.20:
            warnings.append(
                f"⚠️  Step {step}: L0 sparsity ({l0_sparsity:.4f}) is very high (>20%). "
                f"Training is producing DENSE features which are not interpretable. "
                f"Consider stopping and retraining with higher l1_alpha."
            )
        elif l0_sparsity > 0.15:
            warnings.append(
                f"⚠️  Step {step}: L0 sparsity ({l0_sparsity:.4f}) is high (>15%). "
                f"Target: {target_l0:.2f} ({target_l0*100:.0f}% activation). "
                f"Features may be too dense for good interpretability."
            )
        elif l0_sparsity > target_l0 * 3 and target_l0 > 0:
            warnings.append(
                f"⚠️  Step {step}: L0 sparsity ({l0_sparsity:.4f}) is 3x above target ({target_l0:.2f}). "
                f"Consider increasing l1_alpha if L0 doesn't decrease."
            )

        # Check dead neurons (too sparse)
        dead_neuron_fraction = dead_neurons / latent_dim
        if dead_neuron_fraction > 0.7:
            warnings.append(
                f"⚠️  Step {step}: {dead_neuron_fraction*100:.1f}% dead neurons ({dead_neurons}/{latent_dim}). "
                f"Too many features are not learning. Consider decreasing l1_alpha."
            )
        elif dead_neuron_fraction > 0.5:
            warnings.append(
                f"⚠️  Step {step}: {dead_neuron_fraction*100:.1f}% dead neurons ({dead_neurons}/{latent_dim}). "
                f"Many features are not activating. Monitor if this improves with more training."
            )

        return warnings


def validate_training_config(hyperparameters: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Convenience function to validate training configuration.

    Args:
        hyperparameters: Training hyperparameters dictionary

    Returns:
        Tuple of (warnings, errors)
    """
    return TrainingValidator.validate_sparsity_config(hyperparameters)
