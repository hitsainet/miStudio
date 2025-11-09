"""
Sparse Autoencoder (SAE) PyTorch implementations for mechanistic interpretability.

This module provides three SAE architectures:
1. SparseAutoencoder - Standard SAE with L1 sparsity penalty
2. SkipAutoencoder - SAE with residual/skip connections
3. Transcoder - Layer-to-layer SAE for transcoding activations

All implementations support:
- L1 sparsity penalty for feature learning
- Dead neuron tracking and optional resampling
- Flexible encoder/decoder initialization
- Comprehensive loss computation with multiple components
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            norm_coeff = math.sqrt(self.hidden_dim) / x.norm(dim=-1, keepdim=True)
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

            # L1 sparsity penalty (SAELens standard)
            # Mean across both features and batch
            # This normalizes penalty by number of features, making l1_alpha transferable
            l1_penalty = z.abs().mean()

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
            l1_penalty = z.abs().mean()
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

            # L1 sparsity penalty (SAELens standard)
            # Mean across both features and batch
            # This normalizes penalty by number of features, making l1_alpha transferable
            l1_penalty = z.abs().mean()

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

    if architecture_type == 'standard':
        return SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=l1_alpha,
            **kwargs
        )
    elif architecture_type == 'skip':
        # SkipAutoencoder doesn't support ghost_gradient_penalty
        skip_kwargs = {k: v for k, v in kwargs.items() if k != 'ghost_gradient_penalty'}
        return SkipAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=l1_alpha,
            **skip_kwargs
        )
    elif architecture_type == 'transcoder':
        # Transcoder doesn't support ghost_gradient_penalty
        output_dim = kwargs.pop('output_dim', hidden_dim)
        transcoder_kwargs = {k: v for k, v in kwargs.items() if k != 'ghost_gradient_penalty'}
        return Transcoder(
            input_dim=hidden_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            l1_alpha=l1_alpha,
            **transcoder_kwargs
        )
    else:
        raise ValueError(
            f"Unknown architecture_type: {architecture_type}. "
            f"Must be one of: 'standard', 'skip', 'transcoder'"
        )
