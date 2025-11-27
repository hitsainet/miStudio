"""
Sparse Autoencoder (SAE) PyTorch implementations for mechanistic interpretability.

This module provides four SAE architectures:
1. SparseAutoencoder - Standard SAE with L1 sparsity penalty
2. SkipAutoencoder - SAE with residual/skip connections
3. Transcoder - Layer-to-layer SAE for transcoding activations
4. JumpReLUSAE - Gemma Scope-style SAE with JumpReLU activation and L0 penalty

All implementations support:
- L1/L0 sparsity penalty for feature learning
- Dead neuron tracking and optional resampling
- Flexible encoder/decoder initialization
- Comprehensive loss computation with multiple components

JumpReLU implementation based on:
- Gemma Scope: arXiv:2408.05147v2
- JumpReLU paper: arXiv:2407.14435 (Rajamanoharan et al. 2024)
"""

from typing import Tuple, Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class SparseAutoencoder(nn.Module):
    """
    Standard Sparse Autoencoder for learning interpretable features.

    Architecture:
        x → ReLU(W_enc @ x + b_enc) → z (latent)
        z → W_dec @ z + b_dec → x_reconstructed

    Loss:
        L_total = L_reconstruction + l1_alpha * L1(z) + L_zero_ablation

    Args:
        hidden_dim: Input/output dimension (e.g., 768 for transformer hidden states)
        latent_dim: Latent dimension (SAE width, typically 8-32x hidden_dim)
        l1_alpha: L1 sparsity penalty coefficient
        tied_weights: If True, W_dec = W_enc^T (reduces parameters)
        init_scale: Initialization scale for weights
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        l1_alpha: float = 0.001,
        tied_weights: bool = False,
        init_scale: float = 0.1,
        ghost_gradient_penalty: float = 0.0,
        normalize_activations: str = 'constant_norm_rescale',
        top_k_sparsity: Optional[float] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.l1_alpha = l1_alpha
        self.tied_weights = tied_weights
        self.ghost_gradient_penalty = ghost_gradient_penalty
        self.normalize_activations = normalize_activations
        self.top_k_sparsity = top_k_sparsity

        # Calculate k for Top-K if enabled (convert percentage to fraction)
        if top_k_sparsity is not None:
            fraction = top_k_sparsity / 100.0  # Convert percentage to fraction
            self.k = max(1, int(fraction * latent_dim))
        else:
            self.k = None

        # Encoder: x → z
        self.encoder = nn.Linear(hidden_dim, latent_dim, bias=True)

        # Decoder: z → x
        if tied_weights:
            self.decoder = None  # Will use encoder.weight.T
        else:
            self.decoder = nn.Linear(latent_dim, hidden_dim, bias=True)

        # Decoder bias (always separate, even with tied weights)
        self.decoder_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Initialize weights
        self._initialize_weights(init_scale)

    def _initialize_weights(self, scale: float) -> None:
        """Initialize weights with small random values."""
        nn.init.normal_(self.encoder.weight, mean=0.0, std=scale)
        nn.init.zeros_(self.encoder.bias)

        if not self.tied_weights:
            nn.init.normal_(self.decoder.weight, mean=0.0, std=scale)
            nn.init.zeros_(self.decoder.bias)

        nn.init.zeros_(self.decoder_bias)

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize activations according to specified method.

        Args:
            x: Input activations [batch, hidden_dim]

        Returns:
            x_normalized: Normalized activations
            norm_coeff: Normalization coefficients for denormalization
        """
        if self.normalize_activations == 'constant_norm_rescale':
            # SAELens standard: E(||x||) = sqrt(hidden_dim)
            import math
            x_norm = x.norm(dim=-1, keepdim=True)

            # Safety: Prevent division by zero
            x_norm = torch.clamp(x_norm, min=1e-6)

            norm_coeff = math.sqrt(self.hidden_dim) / x_norm
            x_normalized = x * norm_coeff
            return x_normalized, norm_coeff
        elif self.normalize_activations == 'none':
            # No normalization
            return x, torch.ones_like(x[:, :1])
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_activations}")

    def denormalize(self, x: torch.Tensor, norm_coeff: torch.Tensor) -> torch.Tensor:
        """
        Denormalize activations.

        Args:
            x: Normalized activations [batch, hidden_dim]
            norm_coeff: Normalization coefficients from normalize()

        Returns:
            x_denormalized: Original scale activations
        """
        if self.normalize_activations == 'constant_norm_rescale':
            return x / norm_coeff
        else:
            return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor [batch, hidden_dim]

        Returns:
            z: Latent activations [batch, latent_dim] (after ReLU, with optional Top-K)
        """
        z = F.relu(self.encoder(x))

        # Apply Top-K sparsity if enabled
        if self.k is not None:
            # Keep only top-K activations per sample
            # Get top-K values and indices
            topk_values, topk_indices = torch.topk(z, self.k, dim=-1)

            # Create sparse tensor with only top-K activations
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(-1, topk_indices, topk_values)

            return z_sparse

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            z: Latent activations [batch, latent_dim]

        Returns:
            x_reconstructed: Reconstructed input [batch, hidden_dim]
        """
        if self.tied_weights:
            # Use transposed encoder weights
            x_reconstructed = F.linear(z, self.encoder.weight.t())
        else:
            x_reconstructed = self.decoder(z)

        return x_reconstructed + self.decoder_bias

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through SAE.

        Args:
            x: Input activations [batch, hidden_dim]
            return_loss: Whether to compute and return loss components

        Returns:
            x_reconstructed: Reconstructed activations [batch, hidden_dim] (denormalized)
            z: Latent activations [batch, latent_dim]
            losses: Dictionary of loss components if return_loss=True, else empty dict
        """
        # Normalize inputs
        x_normalized, norm_coeff = self.normalize(x)

        # Encode
        z = self.encode(x_normalized)

        # Decode (still normalized)
        x_reconstructed_norm = self.decode(z)

        # Denormalize output
        x_reconstructed = self.denormalize(x_reconstructed_norm, norm_coeff)

        # Compute losses
        losses = {}
        if return_loss:
            # Reconstruction loss (MSE)
            loss_reconstruction = F.mse_loss(x_reconstructed, x, reduction='mean')

            # L1 sparsity penalty (per-sample L1 norm, then averaged over batch)
            # This is the correct formulation from Anthropic's "Towards Monosemanticity"
            # Sum L1 norm per sample, then average across batch
            # Shape: [batch, latent_dim] -> sum over latent_dim -> [batch] -> mean over batch -> scalar
            l1_penalty = z.abs().sum(dim=-1).mean()

            # L0 sparsity (fraction of active features)
            l0_sparsity = (z > 0).float().mean()

            # Zero ablation loss (how much worse is reconstruction without SAE features?)
            x_zero = self.decoder_bias.expand_as(x)
            loss_zero = F.mse_loss(x_zero, x, reduction='mean')

            # Ghost gradient penalty (encourages dead neurons to activate)
            ghost_penalty = torch.tensor(0.0, device=z.device)
            if self.ghost_gradient_penalty > 0:
                # Pre-activation values (before ReLU)
                pre_activation = self.encoder(x)
                # Dead neurons have pre_activation < 0 (would be zeroed by ReLU)
                dead_mask = (pre_activation <= 0).float()
                # Penalty for dead neurons (encourages positive pre-activations)
                ghost_penalty = (dead_mask * F.relu(-pre_activation)).mean()

            # Total loss
            loss_total = loss_reconstruction + self.l1_alpha * l1_penalty + self.ghost_gradient_penalty * ghost_penalty

            losses = {
                'loss': loss_total,
                'loss_reconstruction': loss_reconstruction,
                'loss_zero': loss_zero,
                'l1_penalty': l1_penalty,
                'l0_sparsity': l0_sparsity,
                'ghost_penalty': ghost_penalty,
            }

        return x_reconstructed, z, losses

    def get_feature_magnitudes(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get per-feature activation magnitudes.

        Args:
            z: Latent activations [batch, latent_dim]

        Returns:
            magnitudes: Per-feature average magnitude [latent_dim]
        """
        return z.mean(dim=0)

    def get_dead_neurons(
        self,
        z: torch.Tensor,
        threshold: float = 1e-6
    ) -> torch.Tensor:
        """
        Identify dead neurons (features that never activate).

        Args:
            z: Latent activations [batch, latent_dim]
            threshold: Activation threshold to consider a neuron alive

        Returns:
            dead_mask: Boolean mask [latent_dim] where True = dead neuron
        """
        magnitudes = self.get_feature_magnitudes(z)
        return magnitudes < threshold


class SkipAutoencoder(SparseAutoencoder):
    """
    Skip-connection Sparse Autoencoder with residual connections.

    Architecture:
        x → ReLU(W_enc @ x + b_enc) → z
        x_reconstructed = x + W_dec @ z + b_dec  (residual connection)

    The skip connection allows the SAE to learn only the "important"
    differences from the input, potentially improving reconstruction.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        l1_alpha: float = 0.001,
        tied_weights: bool = False,
        init_scale: float = 0.1,
        skip_scale: float = 1.0,
        normalize_activations: str = 'constant_norm_rescale',
        top_k_sparsity: Optional[float] = None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=l1_alpha,
            tied_weights=tied_weights,
            init_scale=init_scale,
            normalize_activations=normalize_activations,
            top_k_sparsity=top_k_sparsity,
        )
        self.skip_scale = skip_scale

    def decode(self, z: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        """
        Decode with skip connection.

        Args:
            z: Latent activations [batch, latent_dim]
            x_original: Original input [batch, hidden_dim] for skip connection

        Returns:
            x_reconstructed: Reconstructed input [batch, hidden_dim]
        """
        # Standard decode
        delta = super().decode(z)

        # Add skip connection
        return self.skip_scale * x_original + delta

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with skip connection.

        Args:
            x: Input activations [batch, hidden_dim]
            return_loss: Whether to compute and return loss components

        Returns:
            x_reconstructed: Reconstructed activations [batch, hidden_dim]
            z: Latent activations [batch, latent_dim]
            losses: Dictionary of loss components if return_loss=True
        """
        # Encode
        z = self.encode(x)

        # Decode with skip connection
        x_reconstructed = self.decode(z, x)

        # Compute losses (same as base SAE)
        losses = {}
        if return_loss:
            loss_reconstruction = F.mse_loss(x_reconstructed, x, reduction='mean')
            # L1 penalty: per-sample L1 norm, averaged over batch
            l1_penalty = z.abs().sum(dim=-1).mean()
            l0_sparsity = (z > 0).float().mean()

            # Zero ablation: just the skip connection
            x_zero = self.skip_scale * x
            loss_zero = F.mse_loss(x_zero, x, reduction='mean')

            loss_total = loss_reconstruction + self.l1_alpha * l1_penalty

            losses = {
                'loss': loss_total,
                'loss_reconstruction': loss_reconstruction,
                'loss_zero': loss_zero,
                'l1_penalty': l1_penalty,
                'l0_sparsity': l0_sparsity,
            }

        return x_reconstructed, z, losses


class Transcoder(nn.Module):
    """
    Transcoder SAE for layer-to-layer activation mapping.

    Architecture:
        x_layer_i → ReLU(W_enc @ x_i + b_enc) → z
        z → W_dec @ z + b_dec → x_layer_j

    Useful for understanding how information flows between transformer layers.

    Args:
        input_dim: Input dimension (layer i hidden size)
        output_dim: Output dimension (layer j hidden size)
        latent_dim: Latent dimension (SAE width)
        l1_alpha: L1 sparsity penalty coefficient
        init_scale: Initialization scale for weights
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        l1_alpha: float = 0.001,
        init_scale: float = 0.1,
        top_k_sparsity: Optional[float] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.l1_alpha = l1_alpha
        self.top_k_sparsity = top_k_sparsity

        # Calculate k for Top-K if enabled (convert percentage to fraction)
        if top_k_sparsity is not None:
            fraction = top_k_sparsity / 100.0  # Convert percentage to fraction
            self.k = max(1, int(fraction * latent_dim))
        else:
            self.k = None

        # Encoder: x_i → z
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)

        # Decoder: z → x_j
        self.decoder = nn.Linear(latent_dim, output_dim, bias=True)

        # Initialize weights
        self._initialize_weights(init_scale)

    def _initialize_weights(self, scale: float) -> None:
        """Initialize weights with small random values."""
        nn.init.normal_(self.encoder.weight, mean=0.0, std=scale)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=scale)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode layer i activations to latent."""
        z = F.relu(self.encoder(x))

        # Apply Top-K sparsity if enabled
        if self.k is not None:
            # Keep only top-K activations per sample
            topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(-1, topk_indices, topk_values)
            return z_sparse

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to layer j activations."""
        return self.decoder(z)

    def forward(
        self,
        x_input: torch.Tensor,
        x_target: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for transcoding.

        Args:
            x_input: Input activations from layer i [batch, input_dim]
            x_target: Target activations from layer j [batch, output_dim]
            return_loss: Whether to compute and return loss components

        Returns:
            x_transcoded: Transcoded activations [batch, output_dim]
            z: Latent activations [batch, latent_dim]
            losses: Dictionary of loss components if return_loss=True
        """
        # Encode
        z = self.encode(x_input)

        # Decode
        x_transcoded = self.decode(z)

        # Compute losses
        losses = {}
        if return_loss:
            # Reconstruction loss (how well do we predict layer j from layer i?)
            loss_reconstruction = F.mse_loss(x_transcoded, x_target, reduction='mean')

            # L1 sparsity penalty: per-sample L1 norm, averaged over batch
            l1_penalty = z.abs().sum(dim=-1).mean()

            # L0 sparsity
            l0_sparsity = (z > 0).float().mean()

            # Zero ablation (decoder bias only)
            x_zero = self.decoder.bias.expand_as(x_target)
            loss_zero = F.mse_loss(x_zero, x_target, reduction='mean')

            # Total loss
            loss_total = loss_reconstruction + self.l1_alpha * l1_penalty

            losses = {
                'loss': loss_total,
                'loss_reconstruction': loss_reconstruction,
                'loss_zero': loss_zero,
                'l1_penalty': l1_penalty,
                'l0_sparsity': l0_sparsity,
            }

        return x_transcoded, z, losses

    def get_feature_magnitudes(self, z: torch.Tensor) -> torch.Tensor:
        """Get per-feature activation magnitudes."""
        return z.mean(dim=0)

    def get_dead_neurons(
        self,
        z: torch.Tensor,
        threshold: float = 1e-6
    ) -> torch.Tensor:
        """Identify dead neurons."""
        magnitudes = self.get_feature_magnitudes(z)
        return magnitudes < threshold


# =============================================================================
# JumpReLU Implementation (Gemma Scope / arXiv:2407.14435)
# =============================================================================

class JumpReLUFunction(Function):
    """
    Custom autograd function for JumpReLU with Straight-Through Estimator (STE).

    Forward: JumpReLU_θ(z) = z ⊙ H(z - θ)
    Backward: Uses STE with kernel density estimation for threshold gradients.

    Reference: Rajamanoharan et al. 2024, "Jumping Ahead: Improving
    Reconstruction Fidelity with JumpReLU Sparse Autoencoders"
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, threshold: torch.Tensor, bandwidth: float = 0.001):
        """
        Forward pass for JumpReLU activation.

        Args:
            z: Pre-activations [batch, latent_dim]
            threshold: Per-feature thresholds [latent_dim]
            bandwidth: KDE bandwidth for STE gradient estimation (ε)

        Returns:
            Activated features with threshold gating
        """
        # Heaviside step function: H(z - θ) = 1 if z > θ, else 0
        gate = (z > threshold).float()

        # JumpReLU output: z ⊙ H(z - θ)
        output = z * gate

        # Save for backward pass
        ctx.save_for_backward(z, threshold, gate)
        ctx.bandwidth = bandwidth

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using Straight-Through Estimator.

        For z: Standard gradient through gate
        For θ: KDE approximation of step function derivative
        """
        z, threshold, gate = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # Gradient w.r.t. z: pass through where gate is active
        # grad_z = grad_output * gate
        grad_z = grad_output * gate

        # Gradient w.r.t. threshold using KDE (Gaussian kernel)
        # The derivative of Heaviside is approximated by a Gaussian kernel
        # centered at the threshold with bandwidth ε
        # d/dθ H(z - θ) ≈ -1/(ε√(2π)) * exp(-(z-θ)²/(2ε²))
        # This gives gradient that pushes threshold toward where z values are

        # Compute distance from threshold
        delta = z - threshold

        # Gaussian kernel density estimate
        # K(u) = 1/(ε√(2π)) * exp(-u²/(2ε²))
        kernel = torch.exp(-0.5 * (delta / bandwidth) ** 2) / (bandwidth * math.sqrt(2 * math.pi))

        # Gradient of threshold: negative because increasing θ decreases activation
        # Sum over batch dimension, keeping feature dimension
        # grad_θ = -sum_batch(grad_output * z * kernel)
        grad_threshold = -(grad_output * z * kernel).sum(dim=0)

        return grad_z, grad_threshold, None  # None for bandwidth (not a tensor)


class JumpReLU(nn.Module):
    """
    JumpReLU activation with learnable per-feature thresholds.

    JumpReLU_θ(z) = z ⊙ H(z - θ)

    Where:
        θ = learned threshold vector (positive values)
        H = Heaviside step function (1 if input > 0, else 0)
        ⊙ = element-wise multiplication

    The key innovation is that each feature has its own learnable threshold,
    allowing the SAE to better balance feature detection vs magnitude estimation.

    Args:
        num_features: Number of features (latent_dim)
        initial_threshold: Initial threshold value (default: 0.001)
        bandwidth: KDE bandwidth for STE gradient estimation (default: 0.001)
    """

    def __init__(
        self,
        num_features: int,
        initial_threshold: float = 0.001,
        bandwidth: float = 0.001,
    ):
        super().__init__()

        self.num_features = num_features
        self.bandwidth = bandwidth

        # Learnable thresholds initialized to small positive value
        # Using log-space to ensure thresholds stay positive
        self.log_threshold = nn.Parameter(
            torch.full((num_features,), math.log(initial_threshold), dtype=torch.float32)
        )

    @property
    def threshold(self) -> torch.Tensor:
        """Get positive thresholds from log-space parameters."""
        return torch.exp(self.log_threshold)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply JumpReLU activation.

        Args:
            z: Pre-activations [batch, latent_dim] or [batch, seq, latent_dim]

        Returns:
            Activated features with threshold gating
        """
        return JumpReLUFunction.apply(z, self.threshold, self.bandwidth)

    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, bandwidth={self.bandwidth}'


class JumpReLUSAE(nn.Module):
    """
    Sparse Autoencoder with JumpReLU activation (Gemma Scope architecture).

    This is the state-of-the-art SAE architecture from Google DeepMind's
    Gemma Scope project. Key differences from standard SAE:

    1. JumpReLU activation with learnable per-feature thresholds
    2. L0 sparsity penalty instead of L1
    3. Decoder columns constrained to unit norm
    4. Gradient projection to maintain unit norm

    Architecture:
        z = W_enc @ x + b_enc  (pre-activations)
        f = JumpReLU_θ(z)      (sparse features)
        x_hat = W_dec @ f + b_dec  (reconstruction)

    Loss:
        L = ||x - x_hat||² + λ||f||₀

    Args:
        d_model: Input/output dimension (model hidden size)
        d_sae: SAE latent dimension (number of features)
        sparsity_coeff: L0 sparsity penalty coefficient (λ)
        initial_threshold: Initial JumpReLU threshold value
        bandwidth: KDE bandwidth for threshold gradient estimation
        normalize_decoder: Whether to normalize decoder columns to unit norm
        tied_weights: Whether to tie encoder/decoder weights (not recommended)

    Reference:
        Lieberum et al. 2024, "Gemma Scope: Open Sparse Autoencoders
        Everywhere All At Once on Gemma 2" (arXiv:2408.05147v2)
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        sparsity_coeff: float = 6e-4,
        initial_threshold: float = 0.001,
        bandwidth: float = 0.001,
        normalize_decoder: bool = True,
        tied_weights: bool = False,
        normalize_activations: str = 'constant_norm_rescale',
    ):
        super().__init__()

        self.d_model = d_model
        self.d_sae = d_sae
        self.sparsity_coeff = sparsity_coeff
        self.normalize_decoder_flag = normalize_decoder
        self.tied_weights = tied_weights
        self.normalize_activations = normalize_activations

        # Aliases for compatibility with existing code
        self.hidden_dim = d_model
        self.latent_dim = d_sae
        self.l1_alpha = sparsity_coeff  # For compatibility, though we use L0

        # Encoder weights and bias
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder weights and bias
        if tied_weights:
            # W_dec will be computed as W_enc.T
            self.W_dec = None
        else:
            self.W_dec = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # JumpReLU activation with learnable thresholds
        self.activation = JumpReLU(
            num_features=d_sae,
            initial_threshold=initial_threshold,
            bandwidth=bandwidth,
        )

        # Initialize weights following Gemma Scope methodology
        self._init_weights()

        # Normalize decoder if requested
        if normalize_decoder:
            self.normalize_decoder()

    def _init_weights(self):
        """Initialize weights following Gemma Scope methodology."""
        # He-uniform initialization for decoder
        if not self.tied_weights:
            nn.init.kaiming_uniform_(self.W_dec, mode='fan_in', nonlinearity='relu')

        # Initialize encoder
        nn.init.kaiming_uniform_(self.W_enc, mode='fan_out', nonlinearity='relu')

        # Zero biases
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)

    @property
    def decoder_weight(self) -> torch.Tensor:
        """Get decoder weights (handles tied weights)."""
        if self.tied_weights:
            return self.W_enc.T
        return self.W_dec

    def normalize_decoder(self):
        """
        Project decoder columns to unit norm.

        Should be called after each optimizer step to maintain the
        unit norm constraint on decoder vectors.
        """
        if self.tied_weights:
            # For tied weights, normalize encoder rows instead
            with torch.no_grad():
                self.W_enc.data = F.normalize(self.W_enc.data, dim=1, p=2)
        else:
            with torch.no_grad():
                # Normalize each column (feature direction) to unit norm
                self.W_dec.data = F.normalize(self.W_dec.data, dim=0, p=2)

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize activations to have unit mean squared norm.

        This allows hyperparameter transfer across layers and sites.
        """
        if self.normalize_activations == 'constant_norm_rescale':
            x_norm = x.norm(dim=-1, keepdim=True)
            x_norm = torch.clamp(x_norm, min=1e-6)
            norm_coeff = math.sqrt(self.d_model) / x_norm
            x_normalized = x * norm_coeff
            return x_normalized, norm_coeff
        elif self.normalize_activations == 'none':
            return x, torch.ones_like(x[..., :1])
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_activations}")

    def denormalize(self, x: torch.Tensor, norm_coeff: torch.Tensor) -> torch.Tensor:
        """Denormalize activations."""
        if self.normalize_activations == 'constant_norm_rescale':
            return x / norm_coeff
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse feature representation.

        Args:
            x: Input activations [batch, d_model] or [batch, seq, d_model]

        Returns:
            f: Sparse features [batch, d_sae] or [batch, seq, d_sae]
        """
        # Pre-activations: z = W_enc @ x + b_enc
        z = F.linear(x, self.W_enc, self.b_enc)

        # Apply JumpReLU activation
        f = self.activation(z)

        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to input space.

        Args:
            f: Sparse features [batch, d_sae] or [batch, seq, d_sae]

        Returns:
            x_hat: Reconstructed activations [batch, d_model]
        """
        # x_hat = f @ W_dec.T + b_dec
        # F.linear expects weight of shape [out_features, in_features]
        # decoder_weight is [d_model, d_sae], which is [out_features, in_features]
        x_hat = F.linear(f, self.decoder_weight, self.b_dec)
        return x_hat

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass: encode then decode.

        Args:
            x: Input activations [batch, d_model]
            return_loss: Whether to compute and return loss components

        Returns:
            x_hat: Reconstructed activations [batch, d_model]
            f: Sparse feature activations [batch, d_sae]
            losses: Dictionary of loss components
        """
        # Normalize inputs
        x_normalized, norm_coeff = self.normalize(x)

        # Encode to sparse features
        f = self.encode(x_normalized)

        # Decode back to input space (still normalized)
        x_hat_norm = self.decode(f)

        # Denormalize output
        x_hat = self.denormalize(x_hat_norm, norm_coeff)

        # Compute losses
        losses = {}
        if return_loss:
            # Reconstruction loss (MSE)
            loss_reconstruction = F.mse_loss(x_hat, x, reduction='mean')

            # L0 sparsity: count of non-zero features per sample
            # This is the key difference from standard SAE
            l0_per_sample = (f != 0).float().sum(dim=-1)  # [batch] or [batch, seq]
            l0_mean = l0_per_sample.mean()  # Average active features

            # L0 sparsity as fraction (for logging compatibility)
            l0_sparsity = (f != 0).float().mean()

            # L0 penalty: λ * mean(||f||₀)
            loss_l0 = self.sparsity_coeff * l0_mean

            # Total loss
            loss_total = loss_reconstruction + loss_l0

            # Compute FVU (Fraction of Variance Unexplained)
            var_original = x.var()
            var_residuals = (x - x_hat).var()
            fvu = var_residuals / (var_original + 1e-8)

            # Zero ablation loss (decoder bias only)
            x_zero = self.b_dec.expand_as(x)
            loss_zero = F.mse_loss(x_zero, x, reduction='mean')

            # L1 penalty for compatibility with existing code
            l1_penalty = f.abs().sum(dim=-1).mean()

            losses = {
                'loss': loss_total,
                'loss_reconstruction': loss_reconstruction,
                'loss_l0': loss_l0,
                'loss_zero': loss_zero,
                'l0_sparsity': l0_sparsity,
                'l0_mean': l0_mean,  # Average number of active features
                'l1_penalty': l1_penalty,  # For compatibility
                'fvu': fvu,
                'threshold_mean': self.activation.threshold.mean(),
                'threshold_min': self.activation.threshold.min(),
                'threshold_max': self.activation.threshold.max(),
            }

        return x_hat, f, losses

    def get_l0(self, f: torch.Tensor) -> torch.Tensor:
        """
        Compute L0 norm (sparsity) of features.

        Args:
            f: Feature activations

        Returns:
            Mean number of active features per token
        """
        return (f != 0).float().sum(dim=-1).mean()

    def get_feature_magnitudes(self, z: torch.Tensor) -> torch.Tensor:
        """Get per-feature activation magnitudes."""
        return z.mean(dim=0)

    def get_dead_neurons(
        self,
        z: torch.Tensor,
        threshold: float = 1e-6
    ) -> torch.Tensor:
        """Identify dead neurons (features that never activate)."""
        magnitudes = self.get_feature_magnitudes(z)
        return magnitudes < threshold

    # Compatibility properties for existing code
    @property
    def encoder(self):
        """Compatibility: return a module-like object for encoder."""
        class EncoderWrapper:
            def __init__(wrapper_self, sae):
                wrapper_self.weight = sae.W_enc
                wrapper_self.bias = sae.b_enc
        return EncoderWrapper(self)

    @property
    def decoder(self):
        """Compatibility: return a module-like object for decoder."""
        class DecoderWrapper:
            def __init__(wrapper_self, sae):
                wrapper_self.weight = sae.decoder_weight.T if sae.W_dec is not None else sae.W_enc
                wrapper_self.bias = sae.b_dec
        return DecoderWrapper(self)

    @property
    def decoder_bias(self):
        """Compatibility: return decoder bias."""
        return self.b_dec


def project_decoder_gradients(model: nn.Module):
    """
    Project decoder gradients to be orthogonal to decoder columns.

    This maintains the unit norm constraint on decoder vectors during training.
    Should be called after loss.backward() and before optimizer.step().

    Args:
        model: A JumpReLUSAE model (or any model with W_dec parameter)
    """
    # Handle JumpReLUSAE
    if hasattr(model, 'W_dec') and model.W_dec is not None and model.W_dec.grad is not None:
        with torch.no_grad():
            W = model.W_dec.data  # [d_model, d_sae]
            G = model.W_dec.grad  # [d_model, d_sae]

            # Project out component parallel to W for each column
            # G_perp = G - W * (W^T @ G) for each column
            # Since columns are unit norm: parallel_component = (W * G).sum(dim=0)
            parallel_component = (W * G).sum(dim=0, keepdim=True)
            G_perp = G - W * parallel_component

            model.W_dec.grad = G_perp

    # Handle standard SAE with decoder.weight
    elif hasattr(model, 'decoder') and model.decoder is not None:
        if hasattr(model.decoder, 'weight') and model.decoder.weight.grad is not None:
            with torch.no_grad():
                W = model.decoder.weight.data  # [hidden_dim, latent_dim]
                G = model.decoder.weight.grad

                parallel_component = (W * G).sum(dim=0, keepdim=True)
                G_perp = G - W * parallel_component

                model.decoder.weight.grad = G_perp


def create_sae(
    architecture_type: str,
    hidden_dim: int,
    latent_dim: int,
    l1_alpha: float = 0.001,
    **kwargs
) -> nn.Module:
    """
    Factory function to create SAE models.

    Args:
        architecture_type: One of 'standard', 'skip', 'transcoder'
        hidden_dim: Hidden dimension (or input_dim for transcoder)
        latent_dim: Latent dimension
        l1_alpha: L1 sparsity penalty
        **kwargs: Additional architecture-specific parameters

    Returns:
        Initialized SAE model

    Raises:
        ValueError: If architecture_type is not recognized
    """
    architecture_type = architecture_type.lower()

    # JumpReLU-specific parameters to filter out for non-JumpReLU architectures
    jumprelu_params = {'initial_threshold', 'bandwidth', 'sparsity_coeff', 'normalize_decoder', 'tied_weights'}

    if architecture_type == 'standard':
        # Filter out JumpReLU-specific and unsupported parameters
        standard_kwargs = {k: v for k, v in kwargs.items() if k not in jumprelu_params}
        return SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=l1_alpha,
            **standard_kwargs
        )
    elif architecture_type == 'skip':
        # SkipAutoencoder doesn't support ghost_gradient_penalty or JumpReLU params
        skip_kwargs = {k: v for k, v in kwargs.items()
                       if k != 'ghost_gradient_penalty' and k not in jumprelu_params}
        return SkipAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=l1_alpha,
            **skip_kwargs
        )
    elif architecture_type == 'transcoder':
        # Transcoder doesn't support ghost_gradient_penalty or JumpReLU params
        output_dim = kwargs.pop('output_dim', hidden_dim)
        transcoder_kwargs = {k: v for k, v in kwargs.items()
                             if k != 'ghost_gradient_penalty' and k not in jumprelu_params}
        return Transcoder(
            input_dim=hidden_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            l1_alpha=l1_alpha,
            **transcoder_kwargs
        )
    elif architecture_type == 'jumprelu':
        # JumpReLU SAE with learnable thresholds (Gemma Scope architecture)
        # Extract JumpReLU-specific parameters with None handling
        # (hp.get() may pass explicit None values which we should treat as "use default")
        initial_threshold = kwargs.pop('initial_threshold', None)
        if initial_threshold is None:
            initial_threshold = 0.001
        bandwidth = kwargs.pop('bandwidth', None)
        if bandwidth is None:
            bandwidth = 0.001
        normalize_decoder = kwargs.pop('normalize_decoder', None)
        if normalize_decoder is None:
            normalize_decoder = True
        tied_weights = kwargs.pop('tied_weights', None)
        if tied_weights is None:
            tied_weights = False
        normalize_activations = kwargs.pop('normalize_activations', None)
        if normalize_activations is None:
            normalize_activations = 'constant_norm_rescale'
        # JumpReLU uses L0 loss with sparsity_coeff instead of L1
        # Map l1_alpha to sparsity_coeff for consistency
        sparsity_coeff = kwargs.pop('sparsity_coeff', None)
        if sparsity_coeff is None:
            sparsity_coeff = l1_alpha
        return JumpReLUSAE(
            d_model=hidden_dim,
            d_sae=latent_dim,
            sparsity_coeff=sparsity_coeff,
            initial_threshold=initial_threshold,
            bandwidth=bandwidth,
            normalize_decoder=normalize_decoder,
            tied_weights=tied_weights,
            normalize_activations=normalize_activations,
        )
    else:
        raise ValueError(
            f"Unknown architecture_type: {architecture_type}. "
            f"Must be one of: 'standard', 'skip', 'transcoder', 'jumprelu'"
        )
