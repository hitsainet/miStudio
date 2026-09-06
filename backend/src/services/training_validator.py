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

        Formula: l1_alpha = 5e-4 * sqrt(16384 / latent_dim)

        Calibrated for the .sum(dim=-1).mean() L1 penalty formulation used by
        SparseAutoencoder.forward():
            l1_penalty = z.abs().sum(dim=-1).mean()
            loss = reconstruction_loss + l1_alpha * l1_penalty

        The sum accumulates across all latent features, so the penalty magnitude
        scales with the number of active features. Larger SAEs need less l1_alpha.

        Args:
            latent_dim: SAE latent dimension (width)

        Returns:
            Recommended l1_alpha value

        Examples:
            latent_dim=6144  → l1_alpha ≈ 8.2e-4 (GPT-2 / Pythia-160m)
            latent_dim=8192  → l1_alpha ≈ 7.1e-4
            latent_dim=16384 → l1_alpha = 5.0e-4 (baseline)
            latent_dim=32768 → l1_alpha ≈ 3.5e-4
            latent_dim=65536 → l1_alpha = 2.5e-4

        Reference:
            SAELens baseline: ~5e-4 for 16K features with .sum().mean() L1
        """
        BASE_LATENT_DIM = 16384
        BASE_L1_ALPHA = 5e-4
        return BASE_L1_ALPHA * math.sqrt(BASE_LATENT_DIM / latent_dim)

    @staticmethod
    def validate_sparsity_config(
        hyperparameters: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Validate sparsity configuration for training quality.

        Framework-specific validation:
        - standard_saelens / skip / transcoder: validates l1_alpha (SAELens scale ~5e-4)
        - standard_anthropic: validates l1_alpha (Anthropic scale ~5.0)
        - jumprelu: validates sparsity_coeff (L0 penalty)
        - topk: validates top_k (structural sparsity, no penalty)

        Args:
            hyperparameters: Training hyperparameters dictionary

        Returns:
            Tuple of (warnings, errors) as lists of strings
        """
        from ..core.framework_defaults import get_framework_defaults

        warnings = []
        errors = []

        l1_alpha = hyperparameters.get('l1_alpha')
        latent_dim = hyperparameters.get('latent_dim')
        target_l0 = hyperparameters.get('target_l0')
        architecture_type = hyperparameters.get('architecture_type', 'standard')

        # Normalize legacy 'standard'
        if architecture_type == 'standard':
            architecture_type = 'standard_saelens'

        fw = get_framework_defaults(architecture_type)
        sparsity_type = fw['sparsity_type']

        if latent_dim is None:
            errors.append("latent_dim is required for sparsity validation.")
            return warnings, errors

        if sparsity_type == 'topk':
            # TopK: validate top_k parameter
            top_k = hyperparameters.get('top_k')
            top_k_sparsity = hyperparameters.get('top_k_sparsity')
            if top_k is None and top_k_sparsity is None:
                errors.append("top_k is required for TopK architecture. Specify the number of active features per sample.")
                return warnings, errors
            effective_k = top_k if top_k is not None else max(1, int((top_k_sparsity / 100.0) * latent_dim))
            if effective_k > latent_dim * 0.1:
                warnings.append(
                    f"top_k ({effective_k}) is >10% of latent_dim ({latent_dim}). "
                    f"This produces dense features. Typical range: 32-256."
                )

        elif sparsity_type == 'l0':
            # JumpReLU: validate sparsity_coeff (count-based L0 per Gemma Scope paper)
            # L0 penalty = sparsity_coeff * avg_active_feature_count
            # Paper uses λ ≈ 6e-4. At L0=50: loss_l0 = 6e-4 * 50 = 0.03
            sparsity_coeff = hyperparameters.get('sparsity_coeff')
            if sparsity_coeff is not None:
                if sparsity_coeff > 0.01:
                    warnings.append(
                        f"sparsity_coeff ({sparsity_coeff}) is very high for count-based L0. "
                        f"At L0=50, loss_l0 = {sparsity_coeff * 50:.2f} which will dominate reconstruction. "
                        f"Typical range: 1e-4 to 5e-3 (default: 1e-3, Gemma Scope: 6e-4)."
                    )
                elif sparsity_coeff > 0.005:
                    warnings.append(
                        f"sparsity_coeff ({sparsity_coeff}) is high for count-based L0. "
                        f"Typical range: 1e-4 to 5e-3 (default: 1e-3, Gemma Scope: 6e-4). "
                        f"May cause excessive dead neurons."
                    )

        elif sparsity_type == 'l1':
            # L1-based: validate l1_alpha
            if l1_alpha is None:
                errors.append(
                    "l1_alpha is required for L1-based frameworks. "
                    "This is the sparsity penalty coefficient."
                )
                return warnings, errors

            if architecture_type == 'standard_anthropic':
                # Anthropic normalization: l1_alpha is typically ~5.0
                if l1_alpha < 0.5:
                    warnings.append(
                        f"l1_alpha ({l1_alpha}) is low for Standard (Anthropic). "
                        f"With Anthropic normalization (E[||x||²]=d_model), typical l1_alpha is ~5.0."
                    )
                elif l1_alpha > 20.0:
                    warnings.append(
                        f"l1_alpha ({l1_alpha}) is very high for Standard (Anthropic). "
                        f"Typical range: 1-10. Values >20 may cause race to zero."
                    )
            else:
                # SAELens / Skip / Transcoder: validate against recommended formula
                recommended_l1_alpha = TrainingValidator.calculate_recommended_l1_alpha(latent_dim)

                if l1_alpha < recommended_l1_alpha * 0.1:
                    warnings.append(
                        f"l1_alpha ({l1_alpha:.6f}) is very low for latent_dim ({latent_dim}). "
                        f"Recommended: {recommended_l1_alpha:.6f}. "
                        f"This will likely produce DENSE features (L0 > 0.20)."
                    )
                elif l1_alpha < recommended_l1_alpha * 0.5:
                    warnings.append(
                        f"l1_alpha ({l1_alpha:.6f}) is low for latent_dim ({latent_dim}). "
                        f"Recommended: {recommended_l1_alpha:.6f}."
                    )

                if l1_alpha > recommended_l1_alpha * 10:
                    errors.append(
                        f"l1_alpha ({l1_alpha:.6f}) is {l1_alpha/recommended_l1_alpha:.0f}x higher than recommended ({recommended_l1_alpha:.6f}). "
                        f"This WILL cause degenerate training. Use l1_alpha <= {recommended_l1_alpha * 3:.6f}."
                    )
                elif l1_alpha > recommended_l1_alpha * 5:
                    warnings.append(
                        f"l1_alpha ({l1_alpha:.6f}) is {l1_alpha/recommended_l1_alpha:.0f}x higher than recommended ({recommended_l1_alpha:.6f}). "
                        f"Will cause excessive dead neurons (>50%)."
                    )
                elif l1_alpha > recommended_l1_alpha * 3:
                    warnings.append(
                        f"l1_alpha ({l1_alpha:.6f}) is {l1_alpha/recommended_l1_alpha:.1f}x higher than recommended ({recommended_l1_alpha:.6f}). "
                        f"May cause many dead neurons (30-50%)."
                    )

        # Validate target_l0 if provided (relevant for all frameworks)
        if target_l0 is not None:
            if target_l0 > 0.15:
                warnings.append(
                    f"target_l0 ({target_l0:.2f}) is high. "
                    f"For interpretable features, aim for L0 < 0.05 (5% activation rate)."
                )
            elif target_l0 < 0.005:
                warnings.append(
                    f"target_l0 ({target_l0:.3f}) is very low. "
                    f"May cause training instability. Consider 0.01-0.05."
                )

        return warnings, errors

    # Class variable to track L0 history for race-to-zero detection
    _l0_history = {}

    @staticmethod
    def check_training_quality(
        step: int,
        l0_sparsity: float,
        dead_neurons: int,
        latent_dim: int,
        target_l0: float = 0.05,
        warmup_steps: int = 1000,
        training_id: str = None,
        sparsity_warmup_steps: int = 0
    ) -> List[str]:
        """
        Check training quality metrics during training.

        Provides real-time warnings if L0 sparsity or dead neuron count
        indicate poor training quality. Detects "race to zero" degenerate
        training where SAE learns to output all zeros.

        Args:
            step: Current training step
            l0_sparsity: Current L0 sparsity (fraction of active features)
            dead_neurons: Current count of dead neurons
            latent_dim: SAE latent dimension
            target_l0: Target L0 sparsity
            warmup_steps: Number of LR warmup steps (skip checks during warmup)
            training_id: Training ID for tracking L0 history
            sparsity_warmup_steps: Number of sparsity warmup steps

        Returns:
            List of warning messages
        """
        warnings = []

        # Guard against None values (hp.get() returns None when key exists with None value)
        if target_l0 is None:
            target_l0 = 0.05

        # Skip quality checks during warmup period (both LR and sparsity warmup)
        effective_warmup = max(warmup_steps, sparsity_warmup_steps)
        if step < effective_warmup:
            return warnings

        # Track L0 history for race-to-zero detection
        if training_id:
            if training_id not in TrainingValidator._l0_history:
                TrainingValidator._l0_history[training_id] = []

            history = TrainingValidator._l0_history[training_id]
            history.append((step, l0_sparsity))

            # Keep only last 50 checkpoints to detect trends
            if len(history) > 50:
                history.pop(0)

            # Detect "race to zero" - dramatic L0 collapse
            if len(history) >= 5:
                # Check if L0 dropped by >70% in recent steps
                recent_l0 = [l0 for _, l0 in history[-5:]]
                if len(history) >= 10:
                    earlier_l0 = [l0 for _, l0 in history[-10:-5]]
                    earlier_avg = sum(earlier_l0) / len(earlier_l0)
                    recent_avg = sum(recent_l0) / len(recent_l0)

                    if earlier_avg > 0.15 and recent_avg < 0.05:
                        drop_pct = ((earlier_avg - recent_avg) / earlier_avg) * 100
                        warnings.append(
                            f"🚨 RACE TO ZERO DETECTED at step {step}! "
                            f"L0 sparsity collapsed {drop_pct:.0f}% (from {earlier_avg:.2f} to {recent_avg:.2f}). "
                            f"This indicates degenerate training where SAE learns to output zeros. "
                            f"RECOMMENDATION: Stop training and retrain with LOWER l1_alpha."
                        )

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
