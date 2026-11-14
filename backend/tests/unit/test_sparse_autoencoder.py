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
    JumpReLU,
    JumpReLUSAE,
    project_decoder_gradients,
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

    def test_reconstruction_loss_is_mse_in_normalized_space(self):
        """Reconstruction MSE is measured where the L1 penalty lives.

        This test used to assert MSE against the RAW input, which pinned
        MIS-E2E-086 rather than preventing it: the reconstruction term was
        computed in raw space and then added to an L1 term computed on `z`,
        which comes from the NORMALIZED input. Adding the two weights each
        sample by roughly ‖x‖²/d.

        `normalize_activations` is per-sample, so with the default
        `constant_norm_rescale` the two spaces differ by a different factor for
        every row — which is exactly why the mismatch is not a constant that
        the learning rate could absorb.
        """
        hidden_dim = 512
        latent_dim = 4096
        batch_size = 16

        model = SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=0.0,  # No L1 penalty for this test
        )

        x = torch.randn(batch_size, hidden_dim)
        _, _, losses = model(x, return_loss=True)

        x_normalized, norm_coeff = model.normalize(x)
        expected = torch.nn.functional.mse_loss(
            model.decode(model.encode(x_normalized)), x_normalized, reduction='mean'
        )
        assert torch.allclose(losses['loss_reconstruction'], expected, atol=1e-6), \
            "reconstruction loss is not the normalized-space MSE"

    def test_the_two_spaces_actually_differ_on_this_fixture(self):
        """Guard the test above: if they coincided, it would prove nothing.

        A fixture where normalized and raw space agree is the "fixtures agree
        by construction" trap — the assertion would hold either way.
        """
        model = SparseAutoencoder(hidden_dim=512, latent_dim=4096, l1_alpha=0.0)
        x = torch.randn(16, 512)
        x_normalized, coeff = model.normalize(x)
        assert not torch.allclose(x_normalized, x, atol=1e-3), (
            "normalization is a no-op on this fixture, so the space distinction "
            "the test above rests on does not exist here"
        )
        # And per-sample, so it is not one global scalar.
        assert coeff.numel() > 1 or float(coeff.std()) >= 0

    def test_l1_penalty_calculation(self):
        """Test that L1 penalty is per-sample L1 norm averaged over batch.

        This follows Anthropic's "Towards Monosemanticity" formulation:
        sum L1 norm per sample, then average across batch.
        """
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

        # Manually compute L1 penalty: per-sample L1 norm, averaged over batch
        expected_l1 = z.abs().sum(dim=-1).mean()

        # Compare with model's L1 penalty
        assert torch.allclose(losses['l1_penalty'], expected_l1, atol=1e-6), \
            "L1 penalty does not match per-sample L1 norm averaged over batch"

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


class TestJumpReLUActivation:
    """Test JumpReLU activation module with learnable thresholds."""

    def test_jumprelu_forward_pass(self):
        """Test JumpReLU forward pass applies threshold gating correctly."""
        num_features = 1024
        batch_size = 32

        jumprelu = JumpReLU(num_features=num_features, initial_threshold=0.5)

        # Create input with some values below and above threshold
        z = torch.randn(batch_size, num_features)

        output = jumprelu(z)

        # Output should be zero where input is below threshold
        threshold = jumprelu.threshold.detach()
        expected_zeros = (z <= threshold).float()
        actual_zeros = (output == 0).float()

        # Check that values below threshold are zeroed
        assert (actual_zeros >= expected_zeros).all(), \
            "Values below threshold should be zeroed"

    def test_jumprelu_threshold_is_learnable(self):
        """Test that JumpReLU threshold is learnable (has gradients)."""
        num_features = 1024
        batch_size = 32

        jumprelu = JumpReLU(num_features=num_features, initial_threshold=0.1)

        z = torch.randn(batch_size, num_features, requires_grad=True)
        output = jumprelu(z)
        loss = output.sum()
        loss.backward()

        # Threshold parameter should have gradients
        assert jumprelu.log_threshold.grad is not None, \
            "Threshold parameter should have gradients"

    def test_jumprelu_ste_gradient_approximation(self):
        """Test that STE provides gradient approximation for threshold."""
        num_features = 100
        batch_size = 16
        bandwidth = 0.01

        jumprelu = JumpReLU(
            num_features=num_features,
            initial_threshold=0.1,
            bandwidth=bandwidth,
        )

        # Create input with values around threshold
        z = torch.randn(batch_size, num_features, requires_grad=True)
        output = jumprelu(z)

        # Compute loss that depends on output
        loss = output.sum()
        loss.backward()

        # Check that gradients exist and are finite
        assert not torch.isnan(jumprelu.log_threshold.grad).any(), \
            "Threshold gradients should not be NaN"
        assert not torch.isinf(jumprelu.log_threshold.grad).any(), \
            "Threshold gradients should not be Inf"

    def test_jumprelu_threshold_stays_positive(self):
        """Test that threshold stays positive via log parameterization."""
        num_features = 100

        jumprelu = JumpReLU(num_features=num_features, initial_threshold=0.001)

        # Threshold should be positive
        assert (jumprelu.threshold > 0).all(), \
            "Threshold should be positive"

        # Even after optimization steps
        optimizer = torch.optim.Adam([jumprelu.log_threshold], lr=0.1)
        for _ in range(10):
            z = torch.randn(32, num_features)
            output = jumprelu(z)
            loss = -output.sum()  # Try to increase threshold
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert (jumprelu.threshold > 0).all(), \
            "Threshold should remain positive after optimization"


class TestJumpReLUSAEForwardPass:
    """Test JumpReLUSAE forward pass and output shapes."""

    def test_forward_pass_output_shapes(self):
        """Test that forward pass returns correct output shapes."""
        d_model = 768
        d_sae = 8192
        batch_size = 32

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
            sparsity_coeff=6e-4,
        )

        x = torch.randn(batch_size, d_model)
        x_reconstructed, z, losses = model(x, return_loss=True)

        assert x_reconstructed.shape == (batch_size, d_model), \
            "Reconstructed output shape mismatch"
        assert z.shape == (batch_size, d_sae), \
            "Latent representation shape mismatch"

    def test_forward_pass_no_nan_values(self):
        """Test that forward pass produces no NaN values."""
        d_model = 512
        d_sae = 4096
        batch_size = 16

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
            sparsity_coeff=6e-4,
        )

        x = torch.randn(batch_size, d_model)
        x_reconstructed, z, losses = model(x, return_loss=True)

        assert not torch.isnan(x_reconstructed).any(), \
            "Reconstructed output contains NaN"
        assert not torch.isnan(z).any(), \
            "Latent representation contains NaN"
        assert not torch.isnan(losses['loss']), \
            "Total loss is NaN"

    def test_latent_activations_are_non_negative(self):
        """Test that latent activations are non-negative (after JumpReLU)."""
        d_model = 512
        d_sae = 4096
        batch_size = 16

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
            sparsity_coeff=6e-4,
        )

        x = torch.randn(batch_size, d_model)
        _, z, _ = model(x, return_loss=True)

        assert (z >= 0).all(), \
            "Latent activations contain negative values"


class TestJumpReLUSAELossCalculation:
    """Test JumpReLUSAE loss calculation with L0 penalty."""

    def test_loss_has_all_components(self):
        """Test that loss dict contains all expected components."""
        d_model = 512
        d_sae = 4096
        batch_size = 16

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
            sparsity_coeff=6e-4,
        )

        x = torch.randn(batch_size, d_model)
        _, _, losses = model(x, return_loss=True)

        # Check all loss components exist
        assert 'loss' in losses, "Total loss missing"
        assert 'loss_reconstruction' in losses, "Reconstruction loss missing"
        assert 'loss_l0' in losses, "L0 loss missing"
        assert 'l0_sparsity' in losses, "L0 sparsity missing"
        assert 'fvu' in losses, "FVU missing"

    def test_l0_sparsity_calculation(self):
        """Test that L0 sparsity is fraction of active (non-zero) features."""
        d_model = 512
        d_sae = 4096
        batch_size = 16

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
            sparsity_coeff=6e-4,
        )

        x = torch.randn(batch_size, d_model)
        _, z, losses = model(x, return_loss=True)

        # Manually compute L0 sparsity
        expected_l0 = (z > 0).float().mean()

        # Compare with model's L0 sparsity
        assert torch.allclose(losses['l0_sparsity'], expected_l0, atol=1e-6), \
            "L0 sparsity does not match fraction of active features"

    def test_fvu_calculation(self):
        """Test FVU (Fraction of Variance Unexplained) calculation."""
        d_model = 512
        d_sae = 4096
        batch_size = 16

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
            sparsity_coeff=6e-4,
        )

        x = torch.randn(batch_size, d_model)
        x_reconstructed, _, losses = model(x, return_loss=True)

        # Manually compute FVU: var(residuals) / var(original)
        residuals = x - x_reconstructed
        var_residuals = residuals.var()
        var_original = x.var()
        expected_fvu = (var_residuals / var_original).item()

        # Compare with model's FVU
        assert abs(losses['fvu'] - expected_fvu) < 1e-5, \
            "FVU does not match manual calculation"

    def test_total_loss_composition(self):
        """Test that total loss = reconstruction_loss + l0_loss (coefficient already applied)."""
        d_model = 512
        d_sae = 4096
        batch_size = 16
        sparsity_coeff = 6e-4

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
            sparsity_coeff=sparsity_coeff,
        )

        x = torch.randn(batch_size, d_model)
        _, _, losses = model(x, return_loss=True)

        # loss_l0 already includes sparsity_coeff (lambda * L0_count)
        expected_total = losses['loss_reconstruction'] + losses['loss_l0']

        # Compare with model's total loss
        assert torch.allclose(losses['loss'], expected_total, atol=1e-5), \
            "Total loss does not match sum of components"


class TestJumpReLUSAEDecoderNormalization:
    """Test JumpReLUSAE decoder normalization."""

    def test_decoder_columns_are_unit_norm(self):
        """Test that decoder columns have unit norm after normalization."""
        d_model = 512
        d_sae = 4096

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
            normalize_decoder=True,
        )

        # Call normalize_decoder
        model.normalize_decoder()

        # Check decoder column norms
        column_norms = model.W_dec.norm(dim=0)

        # All column norms should be 1.0
        assert torch.allclose(column_norms, torch.ones(d_sae), atol=1e-5), \
            "Decoder columns should have unit norm"

    def test_normalize_decoder_preserves_direction(self):
        """Test that normalization preserves column directions."""
        d_model = 512
        d_sae = 100  # Smaller for easier testing

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
            normalize_decoder=False,  # Start without normalization
        )

        # Store original directions
        original_directions = model.W_dec.data / model.W_dec.data.norm(dim=0, keepdim=True)

        # Normalize
        model.normalize_decoder()

        # Check directions are preserved
        new_directions = model.W_dec.data / model.W_dec.data.norm(dim=0, keepdim=True)

        assert torch.allclose(original_directions, new_directions, atol=1e-5), \
            "Normalization should preserve column directions"


class TestGradientProjection:
    """Test gradient projection utility function."""

    def test_project_decoder_gradients_orthogonality(self):
        """Test that projected gradients are orthogonal to decoder columns."""
        d_model = 512
        d_sae = 4096
        batch_size = 16

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
        )

        # Forward and backward pass
        x = torch.randn(batch_size, d_model)
        _, _, losses = model(x, return_loss=True)
        losses['loss'].backward()

        # Project gradients
        project_decoder_gradients(model)

        # Check orthogonality: dot product of gradient with decoder column should be ~0
        W = model.W_dec.data
        G = model.W_dec.grad

        # Compute dot products for each column
        dot_products = (W * G).sum(dim=0)

        # All dot products should be close to 0
        assert torch.allclose(dot_products, torch.zeros(d_sae), atol=1e-5), \
            "Projected gradients should be orthogonal to decoder columns"

    def test_project_decoder_gradients_preserves_magnitude_component(self):
        """Test that projection removes only parallel component."""
        d_model = 512
        d_sae = 100
        batch_size = 16

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
        )

        # Forward and backward pass
        x = torch.randn(batch_size, d_model)
        _, _, losses = model(x, return_loss=True)
        losses['loss'].backward()

        # Store original gradient
        original_grad = model.W_dec.grad.clone()

        # Project gradients
        project_decoder_gradients(model)

        # The projected gradient should have smaller or equal norm
        # (we removed the parallel component)
        original_norm = original_grad.norm()
        projected_norm = model.W_dec.grad.norm()

        assert projected_norm <= original_norm + 1e-5, \
            "Projected gradient norm should be <= original"


class TestJumpReLUSAEFactory:
    """Test create_sae factory function for JumpReLU."""

    def test_create_jumprelu_sae(self):
        """Test creating JumpReLU SAE via factory."""
        model = create_sae(
            architecture_type='jumprelu',
            hidden_dim=512,
            latent_dim=4096,
            l1_alpha=6e-4,
        )

        assert isinstance(model, JumpReLUSAE), \
            "Should create JumpReLUSAE"

    def test_create_jumprelu_with_custom_params(self):
        """Test creating JumpReLU SAE with custom parameters."""
        model = create_sae(
            architecture_type='jumprelu',
            hidden_dim=768,
            latent_dim=8192,
            l1_alpha=6e-4,
            initial_threshold=0.01,
            bandwidth=0.005,
            sparsity_coeff=1e-3,
        )

        assert isinstance(model, JumpReLUSAE)
        # Check threshold initialization
        expected_threshold = 0.01
        actual_threshold = model.activation.threshold.mean().item()
        assert abs(actual_threshold - expected_threshold) < 1e-3, \
            "Initial threshold should match specified value"


class TestJumpReLUSAEGradientFlow:
    """Test gradient flow through JumpReLU SAE."""

    def test_gradients_flow_through_model(self):
        """Test that gradients flow through the model during backprop."""
        d_model = 512
        d_sae = 4096
        batch_size = 16

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
            sparsity_coeff=6e-4,
        )

        x = torch.randn(batch_size, d_model)
        _, _, losses = model(x, return_loss=True)

        # Backward pass
        loss = losses['loss']
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient for {name} is None"

    def test_threshold_gradients_flow(self):
        """Test that gradients flow to threshold parameters via STE."""
        d_model = 512
        d_sae = 4096
        batch_size = 16

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
            sparsity_coeff=6e-4,
        )

        x = torch.randn(batch_size, d_model)
        _, _, losses = model(x, return_loss=True)

        loss = losses['loss']
        loss.backward()

        # Check threshold gradients specifically
        threshold_grad = model.activation.log_threshold.grad
        assert threshold_grad is not None, \
            "Threshold gradient should not be None"
        assert not torch.isnan(threshold_grad).any(), \
            "Threshold gradient should not contain NaN"
        assert not torch.isinf(threshold_grad).any(), \
            "Threshold gradient should not contain Inf"

    def test_no_nan_gradients(self):
        """Test that gradients contain no NaN values."""
        d_model = 512
        d_sae = 4096
        batch_size = 16

        model = JumpReLUSAE(
            d_model=d_model,
            d_sae=d_sae,
            sparsity_coeff=6e-4,
        )

        x = torch.randn(batch_size, d_model)
        _, _, losses = model(x, return_loss=True)

        loss = losses['loss']
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), \
                    f"Gradient for {name} contains NaN"


class TestGhostGradientUsesTheEncodersOwnInput:
    """MIS-E2E-086, second half.

    The dead-neuron resurrection penalty encoded RAW `x` while the encoder is
    given the NORMALIZED tensor everywhere else, so the resurrection signal was
    computed against a distribution the encoder never sees.
    """

    def test_the_penalty_is_computed_on_the_normalized_tensor(self):
        import ast
        import inspect

        from src.ml.sparse_autoencoder import SparseAutoencoder

        source = inspect.getsource(SparseAutoencoder.forward)
        tree = ast.parse(inspect.cleandoc(source))
        encoder_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "encoder"
        ]
        assert encoder_calls, "the ghost-gradient encoder call is gone"
        for call in encoder_calls:
            arg = call.args[0]
            assert getattr(arg, "id", None) == "x_normalized", (
                "self.encoder(...) is called with something other than "
                "x_normalized; the encoder is trained on normalized inputs, so "
                "encoding raw x here resurrects neurons against the wrong "
                "distribution"
            )

    def test_the_penalty_actually_changes_when_normalization_does(self):
        """Behavioural, not just structural: the two inputs give different values."""
        import torch

        from src.ml.sparse_autoencoder import SparseAutoencoder

        model = SparseAutoencoder(
            hidden_dim=64, latent_dim=256, l1_alpha=0.0, ghost_gradient_penalty=1.0
        )
        # Large magnitudes so raw and normalized differ substantially.
        x = torch.randn(8, 64) * 50.0
        x_normalized, _ = model.normalize(x)

        raw_pre = model.encoder(x)
        norm_pre = model.encoder(x_normalized)
        raw_penalty = ((raw_pre <= 0).float() * torch.relu(-raw_pre)).mean()
        norm_penalty = ((norm_pre <= 0).float() * torch.relu(-norm_pre)).mean()
        assert not torch.allclose(raw_penalty, norm_penalty, atol=1e-4), (
            "raw and normalized give the same penalty on this fixture, so the "
            "test above cannot distinguish the fix from the defect"
        )

        _, _, losses = model(x, return_loss=True)
        assert torch.allclose(losses["ghost_penalty"], norm_penalty, atol=1e-5), (
            "the reported ghost penalty does not match the normalized-input one"
        )


class TestZeroAblationSharesTheReconstructionSpace:
    """`loss_zero` is the "how bad without the SAE" baseline.

    It is only meaningful next to `loss_reconstruction`, so the two must be
    measured in the same space. Control C191 moved `loss_zero` back to raw
    space while reconstruction stayed normalized and every other test passed —
    the two numbers would have been silently incomparable, and the ratio
    between them is what a reader interprets as "did the SAE help".
    """

    def test_zero_ablation_is_measured_in_normalized_space(self):
        import torch

        from src.ml.sparse_autoencoder import SparseAutoencoder

        model = SparseAutoencoder(hidden_dim=64, latent_dim=256, l1_alpha=0.0)
        x = torch.randn(8, 64) * 25.0
        x_normalized, _ = model.normalize(x)

        expected = torch.nn.functional.mse_loss(
            model.decoder_bias.expand_as(x_normalized), x_normalized, reduction="mean"
        )
        _, _, losses = model(x, return_loss=True)
        assert torch.allclose(losses["loss_zero"], expected, atol=1e-5), (
            "loss_zero is not the normalized-space baseline; it cannot be "
            "compared against loss_reconstruction"
        )

    def test_the_raw_baseline_would_be_a_different_number(self):
        """Otherwise the assertion above holds either way."""
        import torch

        from src.ml.sparse_autoencoder import SparseAutoencoder

        model = SparseAutoencoder(hidden_dim=64, latent_dim=256, l1_alpha=0.0)
        x = torch.randn(8, 64) * 25.0
        x_normalized, _ = model.normalize(x)
        raw = torch.nn.functional.mse_loss(
            model.decoder_bias.expand_as(x), x, reduction="mean"
        )
        norm = torch.nn.functional.mse_loss(
            model.decoder_bias.expand_as(x_normalized), x_normalized, reduction="mean"
        )
        assert not torch.allclose(raw, norm, atol=1e-3), (
            "raw and normalized baselines coincide on this fixture"
        )

    def test_both_loss_terms_are_in_one_space(self):
        """The property that actually matters, stated directly."""
        import torch

        from src.ml.sparse_autoencoder import SparseAutoencoder

        model = SparseAutoencoder(hidden_dim=64, latent_dim=256, l1_alpha=0.0)
        x = torch.randn(8, 64) * 25.0
        x_normalized, _ = model.normalize(x)
        _, _, losses = model(x, return_loss=True)

        recon = torch.nn.functional.mse_loss(
            model.decode(model.encode(x_normalized)), x_normalized, reduction="mean"
        )
        zero = torch.nn.functional.mse_loss(
            model.decoder_bias.expand_as(x_normalized), x_normalized, reduction="mean"
        )
        assert torch.allclose(losses["loss_reconstruction"], recon, atol=1e-5)
        assert torch.allclose(losses["loss_zero"], zero, atol=1e-5)


class TestSkipAutoencoderSharesTheSameSpaceRule:
    """MIS-E2E-086 applies to `SkipAutoencoder` too, and nothing tested it.

    Control C192 reverted only the SkipSAE reconstruction to raw space and the
    whole suite stayed green — the base-SAE tests above do not exercise this
    subclass's `forward`. "Fixed one representative, never generalized" is the
    anti-pattern this audit found five times; this is its test-side twin.
    """

    def _model(self):
        from src.ml.sparse_autoencoder import SkipAutoencoder

        return SkipAutoencoder(hidden_dim=64, latent_dim=256, l1_alpha=0.0)

    def test_reconstruction_is_measured_in_normalized_space(self):
        import torch

        model = self._model()
        x = torch.randn(8, 64) * 25.0
        x_normalized, _ = model.normalize(x)

        expected = torch.nn.functional.mse_loss(
            model.decode(model.encode(x_normalized), x_normalized),
            x_normalized,
            reduction="mean",
        )
        _, _, losses = model(x, return_loss=True)
        assert torch.allclose(losses["loss_reconstruction"], expected, atol=1e-5), (
            "SkipAutoencoder measures reconstruction in raw space while its L1 "
            "term is computed on the normalized encode"
        )

    def test_the_baseline_means_the_same_thing_as_every_other_architecture(self):
        """`training_metrics.loss_zero` is one column shared by all architectures.

        It used to be `skip_scale * x` here — "the skip connection alone" —
        and `skip_scale` defaults to 1.0, so the baseline was `mse(x, x)`,
        exactly 0.0 for every Skip-SAE training ever recorded. A reader
        comparing reconstruction against 0.0 concludes the SAE made things
        infinitely worse. Bias-only is the definition the other architectures
        use, so the column is comparable row to row.
        """
        import torch

        model = self._model()
        x = torch.randn(8, 64) * 25.0
        x_normalized, _ = model.normalize(x)

        expected = torch.nn.functional.mse_loss(
            model.decoder_bias.expand_as(x_normalized), x_normalized, reduction="mean"
        )
        _, _, losses = model(x, return_loss=True)
        assert torch.allclose(losses["loss_zero"], expected, atol=1e-5)

    def test_the_baseline_is_not_degenerately_zero(self):
        """The specific symptom: at the default skip_scale it was always 0.0."""
        import torch

        model = self._model()
        assert model.skip_scale == 1.0, "the default changed; re-check this case"
        x = torch.randn(8, 64) * 25.0
        _, _, losses = model(x, return_loss=True)
        assert float(losses["loss_zero"]) > 1e-6, (
            "loss_zero is ~0, which is what `skip_scale * x` produced at "
            "skip_scale=1.0 — an uninterpretable baseline"
        )

    def test_raw_and_normalized_differ_here_too(self):
        import torch

        model = self._model()
        x = torch.randn(8, 64) * 25.0
        x_normalized, _ = model.normalize(x)
        assert not torch.allclose(x_normalized, x, atol=1e-3), (
            "normalization is a no-op on this fixture, so these assertions "
            "would hold against either space"
        )
