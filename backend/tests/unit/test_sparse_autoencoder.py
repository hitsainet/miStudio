"""
Unit tests for Sparse Autoencoder PyTorch models.

Tests SparseAutoencoder, SkipAutoencoder, and Transcoder architectures,
including forward pass, loss calculation, dead neuron detection, and ghost gradient penalty.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.ml.sparse_autoencoder import (
    SparseAutoencoder,
    SkipAutoencoder,
    Transcoder,
    create_sae,
)


class TestSparseAutoencoderForwardPass:
    """Test SparseAutoencoder forward pass and output shapes."""

    def test_forward_pass_output_shapes(self):
        """Test that forward pass returns correct output shapes."""
        hidden_dim = 768
        latent_dim = 8192
        batch_size = 32

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        # Create input
        x = torch.randn(batch_size, hidden_dim)

        # Forward pass
        x_reconstructed, z, losses = model(x, return_loss=True)

        # Verify shapes
        assert x_reconstructed.shape == (batch_size, hidden_dim), "Reconstructed output shape mismatch"
        assert z.shape == (batch_size, latent_dim), "Latent representation shape mismatch"

    def test_forward_pass_no_nan_values(self):
        """Test that forward pass produces no NaN values."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        x_reconstructed, z, losses = model(x, return_loss=True)

        # Check for NaN values
        assert not torch.isnan(x_reconstructed).any(), "Reconstructed output contains NaN"
        assert not torch.isnan(z).any(), "Latent representation contains NaN"
        assert not torch.isnan(losses['loss']), "Total loss is NaN"

    def test_forward_pass_no_inf_values(self):
        """Test that forward pass produces no Inf values."""
        hidden_dim = 256
        latent_dim = 2048
        batch_size = 8

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        x_reconstructed, z, losses = model(x, return_loss=True)

        # Check for Inf values
        assert not torch.isinf(x_reconstructed).any(), "Reconstructed output contains Inf"
        assert not torch.isinf(z).any(), "Latent representation contains Inf"
        assert not torch.isinf(losses['loss']), "Total loss is Inf"

    def test_latent_activations_are_non_negative(self):
        """Test that latent activations are non-negative (after ReLU)."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        _, z, _ = model(x, return_loss=True)

        # All latent activations should be >= 0 (ReLU output)
        assert (z >= 0).all(), "Latent activations contain negative values"

    def test_forward_pass_without_loss(self):
        """Test forward pass without computing loss."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        x_reconstructed, z, losses = model(x, return_loss=False)

        # Verify shapes
        assert x_reconstructed.shape == (batch_size, hidden_dim)
        assert z.shape == (batch_size, latent_dim)
        assert losses == {}, "Loss dict should be empty when return_loss=False"


class TestSparseAutoencoderLossCalculation:
    """Test SAE loss calculation components."""

    def test_loss_has_all_components(self):
        """Test that loss dict contains all expected components."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        _, _, losses = model(x, return_loss=True)

        # Check all loss components exist
        assert 'loss' in losses, "Total loss missing"
        assert 'loss_reconstruction' in losses, "Reconstruction loss missing"
        assert 'loss_zero' in losses, "Zero ablation loss missing"
        assert 'l1_penalty' in losses, "L1 penalty missing"
        assert 'l0_sparsity' in losses, "L0 sparsity missing"
        assert 'ghost_penalty' in losses, "Ghost penalty missing"

    def test_reconstruction_loss_is_mse(self):
        """Test that reconstruction loss is MSE between input and output."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.0,  # No L1 penalty for this test
        )

        x = torch.randn(batch_size, hidden_dim)
        x_reconstructed, _, losses = model(x, return_loss=True)

        # Manually compute MSE
        expected_mse = torch.nn.functional.mse_loss(x_reconstructed, x, reduction='mean')

        # Compare with model's reconstruction loss
        assert torch.allclose(losses['loss_reconstruction'], expected_mse, atol=1e-6), \
            "Reconstruction loss does not match MSE"

    def test_l1_penalty_calculation(self):
        """Test that L1 penalty is mean absolute value of latent activations."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        _, z, losses = model(x, return_loss=True)

        # Manually compute L1 penalty
        expected_l1 = z.abs().mean()

        # Compare with model's L1 penalty
        assert torch.allclose(losses['l1_penalty'], expected_l1, atol=1e-6), \
            "L1 penalty does not match mean absolute activation"

    def test_l0_sparsity_calculation(self):
        """Test that L0 sparsity is fraction of active (non-zero) features."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        _, z, losses = model(x, return_loss=True)

        # Manually compute L0 sparsity
        expected_l0 = (z > 0).float().mean()

        # Compare with model's L0 sparsity
        assert torch.allclose(losses['l0_sparsity'], expected_l0, atol=1e-6), \
            "L0 sparsity does not match fraction of active features"

    def test_total_loss_composition(self):
        """Test that total loss = reconstruction_loss + l1_alpha * l1_penalty + ghost_penalty."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16
        l1_alpha = 0.005
        ghost_penalty_coeff = 0.01

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=l1_alpha,
            ghost_gradient_penalty=ghost_penalty_coeff,
        )

        x = torch.randn(batch_size, hidden_dim)
        _, _, losses = model(x, return_loss=True)

        # Manually compute total loss
        expected_total = (
            losses['loss_reconstruction']
            + l1_alpha * losses['l1_penalty']
            + ghost_penalty_coeff * losses['ghost_penalty']
        )

        # Compare with model's total loss
        assert torch.allclose(losses['loss'], expected_total, atol=1e-5), \
            "Total loss does not match sum of components"

    def test_ghost_gradient_penalty_when_disabled(self):
        """Test that ghost penalty is zero when ghost_gradient_penalty=0."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
            ghost_gradient_penalty=0.0,
        )

        x = torch.randn(batch_size, hidden_dim)
        _, _, losses = model(x, return_loss=True)

        # Ghost penalty should be zero
        assert losses['ghost_penalty'].item() == 0.0, \
            "Ghost penalty should be zero when disabled"

    def test_ghost_gradient_penalty_when_enabled(self):
        """Test that ghost penalty is computed when ghost_gradient_penalty > 0."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
            ghost_gradient_penalty=0.01,
        )

        x = torch.randn(batch_size, hidden_dim)
        _, _, losses = model(x, return_loss=True)

        # Ghost penalty should be non-negative
        assert losses['ghost_penalty'].item() >= 0.0, \
            "Ghost penalty should be non-negative"


class TestSparseAutoencoderArchitectureVariants:
    """Test different SAE architecture variants."""

    def test_tied_weights_reduces_parameters(self):
        """Test that tied weights reduces parameter count."""
        hidden_dim = 512
        latent_dim = 4096

        model_untied = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            tied_weights=False,
        )

        model_tied = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            tied_weights=True,
        )

        # Count parameters
        params_untied = sum(p.numel() for p in model_untied.parameters())
        params_tied = sum(p.numel() for p in model_tied.parameters())

        # Tied weights should have fewer parameters (no separate decoder weights)
        assert params_tied < params_untied, "Tied weights should reduce parameter count"

    def test_skip_autoencoder_forward_pass(self):
        """Test SkipAutoencoder forward pass with residual connection."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SkipAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        x_reconstructed, z, losses = model(x, return_loss=True)

        # Verify shapes
        assert x_reconstructed.shape == (batch_size, hidden_dim)
        assert z.shape == (batch_size, latent_dim)

        # Check no NaN/Inf
        assert not torch.isnan(x_reconstructed).any()
        assert not torch.isinf(x_reconstructed).any()

    def test_transcoder_forward_pass(self):
        """Test Transcoder forward pass for layer-to-layer mapping."""
        input_dim = 512
        output_dim = 768
        latent_dim = 4096
        batch_size = 16

        model = Transcoder(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x_input = torch.randn(batch_size, input_dim)
        x_target = torch.randn(batch_size, output_dim)

        x_transcoded, z, losses = model(x_input, x_target, return_loss=True)

        # Verify shapes
        assert x_transcoded.shape == (batch_size, output_dim), "Transcoded output shape mismatch"
        assert z.shape == (batch_size, latent_dim), "Latent representation shape mismatch"

        # Check no NaN/Inf
        assert not torch.isnan(x_transcoded).any()
        assert not torch.isinf(x_transcoded).any()


class TestSparseAutoencoderDeadNeurons:
    """Test dead neuron detection and feature magnitudes."""

    def test_get_feature_magnitudes(self):
        """Test feature magnitude calculation."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        _, z, _ = model(x, return_loss=True)

        magnitudes = model.get_feature_magnitudes(z)

        # Verify shape
        assert magnitudes.shape == (latent_dim,), "Feature magnitudes shape mismatch"

        # Verify non-negative
        assert (magnitudes >= 0).all(), "Feature magnitudes should be non-negative"

    def test_get_dead_neurons(self):
        """Test dead neuron detection."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        _, z, _ = model(x, return_loss=True)

        dead_mask = model.get_dead_neurons(z, threshold=1e-6)

        # Verify shape
        assert dead_mask.shape == (latent_dim,), "Dead neuron mask shape mismatch"

        # Verify boolean
        assert dead_mask.dtype == torch.bool, "Dead neuron mask should be boolean"

    def test_dead_neurons_with_zero_activations(self):
        """Test that neurons with zero activations are marked as dead."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        # Create zero activations
        z = torch.zeros(batch_size, latent_dim)

        dead_mask = model.get_dead_neurons(z, threshold=1e-6)

        # All neurons should be dead
        assert dead_mask.all(), "All neurons should be marked as dead with zero activations"


class TestSparseAutoencoderFactory:
    """Test create_sae factory function."""

    def test_create_standard_sae(self):
        """Test creating standard SAE."""
        model = create_sae(
            architecture_type='standard',
            hidden_dim=512,
            latent_dim=4096,
            l1_alpha=0.001,
        )

        assert isinstance(model, SparseAutoencoder), "Should create SparseAutoencoder"
        assert not isinstance(model, SkipAutoencoder), "Should not be SkipAutoencoder"

    def test_create_skip_sae(self):
        """Test creating skip SAE."""
        model = create_sae(
            architecture_type='skip',
            hidden_dim=512,
            latent_dim=4096,
            l1_alpha=0.001,
        )

        assert isinstance(model, SkipAutoencoder), "Should create SkipAutoencoder"

    def test_create_transcoder(self):
        """Test creating transcoder."""
        model = create_sae(
            architecture_type='transcoder',
            hidden_dim=512,
            latent_dim=4096,
            l1_alpha=0.001,
            output_dim=768,
        )

        assert isinstance(model, Transcoder), "Should create Transcoder"

    def test_create_sae_invalid_architecture(self):
        """Test that invalid architecture type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown architecture_type"):
            create_sae(
                architecture_type='invalid',
                hidden_dim=512,
                latent_dim=4096,
                l1_alpha=0.001,
            )

    def test_create_sae_case_insensitive(self):
        """Test that architecture type is case-insensitive."""
        model1 = create_sae(
            architecture_type='STANDARD',
            hidden_dim=512,
            latent_dim=4096,
            l1_alpha=0.001,
        )

        model2 = create_sae(
            architecture_type='Standard',
            hidden_dim=512,
            latent_dim=4096,
            l1_alpha=0.001,
        )

        assert isinstance(model1, SparseAutoencoder)
        assert isinstance(model2, SparseAutoencoder)


class TestSparseAutoencoderGradientFlow:
    """Test gradient flow through SAE."""

    def test_gradients_flow_through_model(self):
        """Test that gradients flow through the model during backprop."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        _, _, losses = model(x, return_loss=True)

        # Backward pass
        loss = losses['loss']
        loss.backward()

        # Check that gradients exist and are not None
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient for {name} is None"

    def test_no_nan_gradients(self):
        """Test that gradients contain no NaN values."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        _, _, losses = model(x, return_loss=True)

        # Backward pass
        loss = losses['loss']
        loss.backward()

        # Check for NaN gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"

    def test_no_inf_gradients(self):
        """Test that gradients contain no Inf values."""
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.001,
        )

        x = torch.randn(batch_size, hidden_dim)
        _, _, losses = model(x, return_loss=True)

        # Backward pass
        loss = losses['loss']
        loss.backward()

        # Check for Inf gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf"
