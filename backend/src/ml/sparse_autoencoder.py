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


def normalize_activations(
    x: torch.Tensor, mode: str, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shared activation normalization for all SAE architectures.

    Modes:
      - 'constant_norm_rescale': scale each sample so ‖x‖ = sqrt(dim)
                                 (SAELens standard)
      - 'anthropic_rescale':     **an alias for the above** — see below
      - 'none':                  no-op (coeff = ones)

    ── 'anthropic_rescale' IS NOT A SECOND METHOD (MIS-E2E-085) ──────────────

    This docstring used to claim it scales so ``E[‖x‖²] = dim`` (Templeton et
    al. 2024), and the two branches were written as if they differed. They do
    not: ``sqrt(dim / ‖x‖²)`` and ``sqrt(dim) / ‖x‖`` are the same number.
    Measured, not assumed — max |difference| **7.2e-7** on float32 inputs at
    two shapes, i.e. epsilon.

    The claimed property is a DATASET expectation; the code computes a
    PER-SAMPLE rescale. Those are genuinely different things — a dataset-wide
    scalar preserves the relative magnitude between tokens, a per-sample one
    discards it — so the paper's method is not implemented here at all.

    Collapsed to one code path rather than reimplemented, deliberately:

      * implementing it properly needs a calibration pass over the corpus to
        estimate E[‖x‖²], that scalar persisted in the checkpoint, and the
        checkpoint format changed to carry it; and
      * every SAE ever trained with ``normalize_activations='anthropic_rescale'``
        was in fact trained with these semantics. Changing what the string means
        would silently alter the meaning of existing artifacts rather than fix
        them.

    So the string keeps working and keeps its behaviour, and now says what it
    does. TRACKED DEBT: the real Templeton normalisation, and PPRD §2.1's "six
    paper-grounded frameworks" (it is five plus an alias).

    Uses ``x[..., :1]`` for the 'none' coefficient so it works for both
    [batch, dim] and [batch, seq, dim] inputs.

    Returns (x_normalized, norm_coeff) where norm_coeff feeds ``denormalize_activations``.
    """
    # One branch for both names — they computed the same thing, and two copies
    # of the same arithmetic can only ever drift into being differently wrong.
    if mode in ('constant_norm_rescale', 'anthropic_rescale'):
        x_norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-6)
        norm_coeff = math.sqrt(dim) / x_norm
        return x * norm_coeff, norm_coeff
    elif mode == 'none':
        return x, torch.ones_like(x[..., :1])
    else:
        raise ValueError(f"Unknown normalization method: {mode}")


def denormalize_activations(
    x: torch.Tensor, norm_coeff: torch.Tensor, mode: str
) -> torch.Tensor:
    """Inverse of ``normalize_activations`` (identity for mode='none')."""
    if mode in ('constant_norm_rescale', 'anthropic_rescale'):
        return x / norm_coeff
    return x


class SparseAutoencoder(nn.Module):
    """
    Standard Sparse Autoencoder for learning interpretable features.

    Architecture:
        x → ReLU(W_enc @ (x - b_dec) + b_enc) → z (latent)
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
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.l1_alpha = l1_alpha
        self.tied_weights = tied_weights
        self.ghost_gradient_penalty = ghost_gradient_penalty
        self.normalize_activations = normalize_activations

        # Encoder: x → z
        self.encoder = nn.Linear(hidden_dim, latent_dim, bias=True)

        # Decoder: z → x (bias=False — decoder_bias is the shared b_dec)
        if tied_weights:
            self.decoder = None  # Will use encoder.weight.T
        else:
            self.decoder = nn.Linear(latent_dim, hidden_dim, bias=False)

        # Shared decoder bias (b_dec): subtracted from input before encoding,
        # added to decoder output. Learns the data mean so encoder sees centered residuals.
        self.decoder_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Initialize weights
        self._initialize_weights(init_scale)

    def _initialize_weights(self, scale: float) -> None:
        """
        Initialize weights following SAELens methodology.

        Decoder columns are initialized with Kaiming uniform and normalized to unit norm.
        Encoder is initialized as the transpose of the decoder (W_enc = W_dec^T) so that
        encoder-decoder pairs start aligned, giving decent reconstruction from step 0.
        """
        if not self.tied_weights:
            # Decoder: Kaiming uniform, then normalize columns to unit norm
            nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='relu')
            with torch.no_grad():
                self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0, p=2)

            # Encoder: transpose of decoder (SAELens standard initialization)
            # This ensures encoder-decoder pairs start aligned
            self.encoder.weight.data = self.decoder.weight.data.T.contiguous().clone()
        else:
            nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')

        nn.init.zeros_(self.encoder.bias)
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
        return normalize_activations(x, self.normalize_activations, self.hidden_dim)

    def denormalize(self, x: torch.Tensor, norm_coeff: torch.Tensor) -> torch.Tensor:
        """
        Denormalize activations.

        Args:
            x: Normalized activations [batch, hidden_dim]
            norm_coeff: Normalization coefficients from normalize()

        Returns:
            x_denormalized: Original scale activations
        """
        return denormalize_activations(x, norm_coeff, self.normalize_activations)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Centers input by subtracting decoder bias (b_dec) before encoding,
        following the standard SAE formulation from Bricken et al. 2023:
            z = ReLU(W_enc @ (x - b_dec) + b_enc)

        Args:
            x: Input tensor [batch, hidden_dim]

        Returns:
            z: Latent activations [batch, latent_dim] (after ReLU)
        """
        # Center by decoder bias so encoder sees zero-mean residuals
        x_centered = x - self.decoder_bias
        z = F.relu(self.encoder(x_centered))
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
            # MIS-E2E-086: reconstruction is measured in NORMALIZED space,
            # the same space the L1 penalty lives in.
            #
            # It used to be `F.mse_loss(x_reconstructed, x)` — raw — while the
            # L1 term is `z.abs().sum()`, and `z` comes from encoding the
            # NORMALIZED input. Adding those two together weights each sample
            # by roughly ‖x‖²/d, so high-norm tokens dominate the objective and
            # the effective sparsity coefficient varies per sample. That defeats
            # the stated purpose of normalization: making one set of
            # hyperparameters transfer across layers whose activation scales
            # differ. `constant_norm_rescale` is the DEFAULT, so this was live
            # on every training that did not opt out.
            #
            # `loss_zero` moves with it, so the recon/zero ratio still compares
            # two quantities in one space. The reported FVU is unaffected — the
            # JumpReLU path computes it separately as a raw-space variance
            # ratio, which is self-consistent and is the right space to report
            # reconstruction quality in.
            loss_reconstruction = F.mse_loss(x_reconstructed_norm, x_normalized, reduction='mean')

            # L1 sparsity penalty (per-sample L1 norm, then averaged over batch)
            # This is the correct formulation from Anthropic's "Towards Monosemanticity"
            # Sum L1 norm per sample, then average across batch
            # Shape: [batch, latent_dim] -> sum over latent_dim -> [batch] -> mean over batch -> scalar
            l1_penalty = z.abs().sum(dim=-1).mean()

            # L0 sparsity (fraction of active features)
            l0_sparsity = (z > 0).float().mean()

            # Zero ablation loss (how much worse is reconstruction without SAE features?)
            x_zero = self.decoder_bias.expand_as(x_normalized)
            loss_zero = F.mse_loss(x_zero, x_normalized, reduction='mean')

            # Ghost gradient penalty (encourages dead neurons to activate)
            ghost_penalty = torch.tensor(0.0, device=z.device)
            if self.ghost_gradient_penalty > 0:
                # MIS-E2E-086: the NORMALIZED tensor, which is what `encode`
                # is given everywhere else. Encoding raw `x` here computed the
                # dead-neuron resurrection signal on an input the encoder never
                # sees, so the neurons were being revived against the wrong
                # distribution.
                pre_activation = self.encoder(x_normalized)
                # Dead neurons have pre_activation < 0 (would be zeroed by ReLU)
                dead_mask = (pre_activation <= 0).float()
                # Penalty for dead neurons (encourages positive pre-activations)
                ghost_penalty = (dead_mask * F.relu(-pre_activation)).mean()

            # Total loss: reconstruction + L1 sparsity + ghost gradient
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
        x → ReLU(W_enc @ (x - b_dec) + b_enc) → z
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
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=l1_alpha,
            tied_weights=tied_weights,
            init_scale=init_scale,
            normalize_activations=normalize_activations,
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
            x_reconstructed: Reconstructed activations [batch, hidden_dim] (denormalized)
            z: Latent activations [batch, latent_dim]
            losses: Dictionary of loss components if return_loss=True
        """
        # Normalize inputs (consistent with base SparseAutoencoder)
        x_normalized, norm_coeff = self.normalize(x)

        # Encode (on normalized inputs)
        z = self.encode(x_normalized)

        # Decode with skip connection (on normalized inputs)
        x_reconstructed_norm = self.decode(z, x_normalized)

        # Denormalize output
        x_reconstructed = self.denormalize(x_reconstructed_norm, norm_coeff)

        # Compute losses (same as base SAE)
        losses = {}
        if return_loss:
            # MIS-E2E-086: normalized space, matching the L1 term below.
            loss_reconstruction = F.mse_loss(x_reconstructed_norm, x_normalized, reduction='mean')
            # L1 penalty: per-sample L1 norm, averaged over batch
            l1_penalty = z.abs().sum(dim=-1).mean()
            l0_sparsity = (z > 0).float().mean()

            # Zero ablation: decoder bias only, the SAME definition every other
            # architecture in this module uses.
            #
            # This used to be `skip_scale * x_normalized`, "the skip connection
            # alone". That is architecturally defensible but produces a number
            # that cannot be read: `skip_scale` defaults to 1.0, so the baseline
            # was `mse(x, x)` — exactly 0.0 for every Skip-SAE training ever
            # run. `loss_zero` is persisted to `training_metrics.loss_zero`, a
            # column shared by all architectures, and a reader comparing
            # reconstruction against a 0.0 baseline concludes the SAE made
            # things infinitely worse.
            #
            # A shared column has to mean one thing in every row. Bias-only is
            # that thing: "what you get with no latent features at all".
            x_zero = self.decoder_bias.expand_as(x_normalized)
            loss_zero = F.mse_loss(x_zero, x_normalized, reduction='mean')

            # Total loss: reconstruction + L1 sparsity
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
        x_layer_i → ReLU(W_enc @ (x_i - b_enc_center) + b_enc) → z
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
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.l1_alpha = l1_alpha

        # Encoder centering bias (learns mean of input activations)
        self.b_enc_center = nn.Parameter(torch.zeros(input_dim))

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
        nn.init.zeros_(self.b_enc_center)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode layer i activations to latent."""
        # Center by encoder centering bias before encoding
        x_centered = x - self.b_enc_center
        z = F.relu(self.encoder(x_centered))
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

            # Total loss: reconstruction + L1 sparsity
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
# TopK SAE Implementation (Gao et al. 2024, OpenAI / arXiv:2406.04093)
# =============================================================================

class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder (Gao et al. 2024, OpenAI).

    Uses structural sparsity (TopK selection) instead of penalty-based sparsity.
    Exactly K features activate per sample — no L1 or L0 penalty needed.
    Includes auxiliary loss for dead feature recovery.

    Architecture:
        z = ReLU(W_enc @ (x - b_pre) + b_enc)     (pre-activations)
        f = TopK(z, k)                              (keep top-K, zero rest)
        x_hat = W_dec @ f + b_pre                   (reconstruction)

    Loss:
        L_main = ||x - x_hat||²
        L_aux = alpha * ||e - e_hat_dead||²         (dead feature aux loss)
        L_total = L_main + L_aux

    Args:
        hidden_dim: Input/output dimension (model hidden size)
        latent_dim: SAE latent dimension (number of features)
        k: Number of top features to keep active per sample
        aux_k: Number of dead features for aux loss (default: k * 2)
        aux_loss_alpha: Weight for auxiliary dead feature loss (default: 1/32)
        normalize_activations: Normalization method
        init_scale: Initialization scale for weights

    Reference:
        Gao et al. 2024, "Scaling and evaluating sparse autoencoders"
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        k: int,
        aux_k: Optional[int] = None,
        aux_loss_alpha: float = 1.0 / 32,
        normalize_activations: str = 'constant_norm_rescale',
        init_scale: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.k = k
        self.aux_k = aux_k if aux_k is not None else min(k * 2, latent_dim)
        self.aux_loss_alpha = aux_loss_alpha
        self.normalize_activations = normalize_activations

        # Pre-encoder bias (subtracted from input, added to reconstruction)
        self.b_pre = nn.Parameter(torch.zeros(hidden_dim))

        # Encoder: (x - b_pre) → z
        self.encoder = nn.Linear(hidden_dim, latent_dim, bias=True)

        # Decoder: z → x_hat
        self.decoder = nn.Linear(latent_dim, hidden_dim, bias=False)

        # Initialize weights
        self._initialize_weights(init_scale)

    def _initialize_weights(self, scale: float) -> None:
        """Initialize weights with small random values."""
        nn.init.normal_(self.encoder.weight, mean=0.0, std=scale)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=scale)

    @property
    def decoder_bias(self):
        """Compatibility: decoder bias is b_pre for TopK."""
        return self.b_pre

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize activations."""
        return normalize_activations(x, self.normalize_activations, self.hidden_dim)

    def denormalize(self, x: torch.Tensor, norm_coeff: torch.Tensor) -> torch.Tensor:
        """Denormalize activations."""
        return denormalize_activations(x, norm_coeff, self.normalize_activations)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode with TopK selection.

        Args:
            x: Input (already normalized, b_pre subtracted) [batch, hidden_dim]

        Returns:
            z_topk: Sparse activations with only top-K nonzero [batch, latent_dim]
        """
        z = F.relu(self.encoder(x))

        # TopK selection: keep only top-K activations per sample
        topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
        z_topk = torch.zeros_like(z)
        z_topk.scatter_(-1, topk_indices, topk_values)

        return z_topk

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse features to reconstruction."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with TopK sparsity and auxiliary dead feature loss.

        Args:
            x: Input activations [batch, hidden_dim]
            return_loss: Whether to compute losses

        Returns:
            x_reconstructed: Reconstructed activations [batch, hidden_dim]
            z: Sparse latent activations [batch, latent_dim]
            losses: Loss components dict
        """
        # Normalize inputs
        x_normalized, norm_coeff = self.normalize(x)

        # Subtract pre-encoder bias
        x_centered = x_normalized - self.b_pre

        # Encode with TopK
        z = self.encode(x_centered)

        # Decode and add back b_pre
        x_reconstructed_norm = self.decode(z) + self.b_pre

        # Denormalize output
        x_reconstructed = self.denormalize(x_reconstructed_norm, norm_coeff)

        losses = {}
        if return_loss:
            # Main reconstruction loss
            loss_reconstruction = F.mse_loss(x_reconstructed, x, reduction='mean')

            # L0 sparsity (exact: always k/latent_dim)
            l0_sparsity = (z > 0).float().mean()

            # L1 for logging compatibility
            l1_penalty = z.abs().sum(dim=-1).mean()

            # Zero ablation loss
            x_zero_norm = self.b_pre.expand_as(x_normalized)
            x_zero = self.denormalize(x_zero_norm, norm_coeff)
            loss_zero = F.mse_loss(x_zero, x, reduction='mean')

            # Auxiliary dead feature loss (Gao et al. 2024)
            # Encourages dead features to learn useful directions
            aux_loss = torch.tensor(0.0, device=z.device)
            dead_mask = (z == 0).all(dim=0)  # [latent_dim]
            num_dead = dead_mask.sum().item()

            if num_dead > 0 and num_dead < self.latent_dim:
                # Re-encode without TopK to get all pre-activations
                z_pre = F.relu(self.encoder(x_centered))

                # Select dead feature activations and apply TopK among them
                z_dead = z_pre[:, dead_mask]  # [batch, num_dead]
                k_dead = min(self.aux_k, num_dead)
                if k_dead > 0:
                    topk_vals, topk_idx = torch.topk(z_dead, k_dead, dim=-1)
                    z_dead_sparse = torch.zeros_like(z_dead)
                    z_dead_sparse.scatter_(-1, topk_idx, topk_vals)

                    # Reconstruct from dead features
                    dead_W = self.decoder.weight[:, dead_mask]  # [hidden_dim, num_dead]
                    x_dead_recon = z_dead_sparse @ dead_W.t()

                    # Residual = what alive features couldn't reconstruct
                    residual = x_centered - (self.decode(z)).detach()
                    aux_loss = F.mse_loss(x_dead_recon, residual, reduction='mean')

            loss_total = loss_reconstruction + self.aux_loss_alpha * aux_loss

            losses = {
                'loss': loss_total,
                'loss_reconstruction': loss_reconstruction,
                'loss_zero': loss_zero,
                'l1_penalty': l1_penalty,
                'l0_sparsity': l0_sparsity,
                'aux_loss': aux_loss,
                'ghost_penalty': torch.tensor(0.0, device=z.device),
            }

        return x_reconstructed, z, losses

    def get_feature_magnitudes(self, z: torch.Tensor) -> torch.Tensor:
        """Get per-feature activation magnitudes."""
        return z.mean(dim=0)

    def get_dead_neurons(self, z: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
        """Identify dead neurons."""
        return self.get_feature_magnitudes(z) < threshold


# =============================================================================
# JumpReLU Implementation (Gemma Scope / arXiv:2407.14435)
# =============================================================================

class StraightThroughL0(Function):
    """
    Custom autograd function for differentiable L0 counting with STE.

    Forward: Exact binary indicator H(z - θ) (no phantom contributions from dead features)
    Backward: Gaussian kernel STE provides smooth gradients near the threshold boundary.

    This replaces the sigmoid surrogate which inflated L0 for dead features:
    sigmoid((0 - 0.001)/0.001) = 0.269, causing phantom L0 from every dead neuron.

    With STE, dead features contribute exactly 0 in forward, but still receive
    gradient via the Gaussian kernel to push them toward activation if beneficial.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, threshold: torch.Tensor, bandwidth: float = 0.001):
        """
        Forward: exact Heaviside H(z - θ).

        Args:
            z: Pre-activations [batch, latent_dim]
            threshold: Per-feature thresholds [latent_dim]
            bandwidth: Gaussian kernel width for backward STE

        Returns:
            Binary indicators: 1 where z > threshold, 0 otherwise
        """
        indicators = (z > threshold).to(z.dtype)
        ctx.save_for_backward(z, threshold)
        ctx.bandwidth = bandwidth
        return indicators

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward: Gaussian kernel STE for smooth gradients.

        d/dz H(z-θ) ≈ K(z-θ) = (1/(ε√2π)) exp(-(z-θ)²/(2ε²))
        d/dθ H(z-θ) ≈ -K(z-θ)
        """
        z, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        delta = z - threshold
        # Gaussian kernel: peaks at z = θ, decays for distant features
        kernel = torch.exp(-0.5 * (delta / bandwidth) ** 2) / (bandwidth * math.sqrt(2 * math.pi))

        # Gradient w.r.t. z: pushes pre-activations toward/away from threshold
        grad_z = grad_output * kernel
        # Gradient w.r.t. threshold: negative (increasing θ reduces L0)
        grad_threshold = (-grad_output * kernel).sum(dim=0)

        return grad_z, grad_threshold, None


class JumpReLUFunction(Function):
    """
    Custom autograd function for JumpReLU with Straight-Through Estimator (STE).

    Forward: JumpReLU_θ(z) = z ⊙ H(z - θ)
    Backward: Uses STE with kernel density estimation for threshold gradients.

    Reference: Rajamanoharan et al. 2024, "Jumping Ahead: Improving
    Reconstruction Fidelity with JumpReLU Sparse Autoencoders"
    """

    @staticmethod
    def forward(
        ctx,
        z: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float = 0.001,
        ste_bandwidth: float = 0.5,
    ):
        """
        Forward pass for JumpReLU activation.

        Args:
            z: Pre-activations [batch, latent_dim]
            threshold: Per-feature thresholds [latent_dim]
            bandwidth: KDE bandwidth for threshold gradient estimation (ε)
            ste_bandwidth: Sigmoid STE bandwidth for z gradients — controls
                how much reconstruction gradient leaks through to inactive
                features. Larger values = more gradient to dead features.

        Returns:
            Activated features with threshold gating
        """
        # Heaviside step function: H(z - θ) = 1 if z > θ, else 0
        # Use z.dtype to preserve half precision (float16) when model is in half mode
        gate = (z > threshold).to(z.dtype)

        # JumpReLU output: z ⊙ H(z - θ)
        output = z * gate

        # Save for backward pass
        ctx.save_for_backward(z, threshold, gate)
        ctx.bandwidth = bandwidth
        ctx.ste_bandwidth = ste_bandwidth

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using Straight-Through Estimator.

        For z: Sigmoid STE — smooth approximation of the hard gate that allows
            reconstruction gradients to flow to inactive (dead) features.
            Without this, dead features get ZERO gradient from reconstruction
            loss and can never recover (the root cause of dead neuron collapse).
        For θ: KDE approximation of step function derivative (unchanged).
        """
        z, threshold, gate = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        ste_bandwidth = ctx.ste_bandwidth

        # Gradient w.r.t. z: sigmoid STE for smooth gradient flow
        # Forward uses hard Heaviside H(z-θ), backward uses σ((z-θ)/b)
        # Active features (z >> θ): sigmoid ≈ 1.0, same as hard gate
        # Near-threshold features: sigmoid ≈ 0.5, partial gradient
        # Inactive features (z << θ): sigmoid > 0, allowing recovery
        ste_gate = torch.sigmoid((z - threshold) / ste_bandwidth)
        grad_z = grad_output * ste_gate

        # Gradient w.r.t. threshold using KDE (Gaussian kernel)
        # d/dθ H(z - θ) ≈ -1/(ε√(2π)) * exp(-(z-θ)²/(2ε²))
        delta = z - threshold
        kernel = torch.exp(-0.5 * (delta / bandwidth) ** 2) / (bandwidth * math.sqrt(2 * math.pi))

        # grad_θ = -sum_batch(grad_output * z * kernel)
        grad_threshold = -(grad_output * z * kernel).sum(dim=0)

        return grad_z, grad_threshold, None, None  # None for bandwidth, ste_bandwidth


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
        initial_threshold: Initial threshold value (default: 0.5)
        bandwidth: KDE bandwidth for STE gradient estimation (default: 0.01)
    """

    def __init__(
        self,
        num_features: int,
        initial_threshold: float = 0.5,
        bandwidth: float = 0.01,
        ste_bandwidth: float = 0.5,
    ):
        super().__init__()

        self.num_features = num_features
        self.bandwidth = bandwidth
        self.ste_bandwidth = ste_bandwidth

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
        return JumpReLUFunction.apply(z, self.threshold, self.bandwidth, self.ste_bandwidth)

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
        sparsity_coeff: L0 sparsity penalty coefficient (λ). Applied to the
            COUNT of active features per sample (Σᵢ H(zᵢ - θᵢ), summed over
            features and averaged over the batch), per the Gemma Scope paper
            formulation L = ‖x - x̂‖² + λ · Σᵢ H(zᵢ - θᵢ). NOT the L0 fraction.
            Paper-scale default 1e-3 (Gemma Scope uses 6e-4). At L0=50 active
            features this gives loss_l0 = 1e-3 × 50 = 0.05. Typical range:
            1e-4 to 5e-3. (Do NOT confuse with the L1 `l1_alpha` scale.)
        initial_threshold: Initial JumpReLU threshold value (default: 0.5, should match
            pre-activation magnitude with constant_norm_rescale normalization)
        bandwidth: KDE bandwidth for STE gradient estimation (default: 0.01)
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
        sparsity_coeff: float = 1e-3,
        initial_threshold: float = 0.5,
        bandwidth: float = 0.01,
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
        # ste_bandwidth=0.5 gives sigmoid STE in backward pass so dead features
        # get reconstruction gradient and can recover (see JumpReLUFunction.backward)
        self.activation = JumpReLU(
            num_features=d_sae,
            initial_threshold=initial_threshold,
            bandwidth=bandwidth,
            ste_bandwidth=0.5,
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
        return normalize_activations(x, self.normalize_activations, self.d_model)

    def denormalize(self, x: torch.Tensor, norm_coeff: torch.Tensor) -> torch.Tensor:
        """Denormalize activations."""
        return denormalize_activations(x, norm_coeff, self.normalize_activations)

    def calibrate_thresholds(self, x: torch.Tensor, target_l0: float = 0.05) -> torch.Tensor:
        """
        Set per-feature thresholds from the actual pre-activation distribution.

        Instead of using a fixed initial_threshold (e.g. 0.5) that may leave
        most features dead from the start, this computes the (1 - target_l0)
        percentile of pre-activations per feature and sets thresholds there.

        Args:
            x: Calibration activations [batch, d_model] (raw, pre-normalization)
            target_l0: Target fraction of active features (default: 0.05 = 5%)

        Returns:
            Calibrated threshold tensor [d_sae]
        """
        with torch.no_grad():
            x_normalized, _ = self.normalize(x)
            # Center by decoder bias before encoding (must match encode())
            x_centered = x_normalized - self.b_dec
            z = F.linear(x_centered, self.W_enc, self.b_enc)

            # Set threshold at (1 - target_l0) percentile per feature
            # E.g., target_l0=0.05 → 95th percentile → ~5% of samples activate
            percentile = 1.0 - target_l0
            thresholds = torch.quantile(z, percentile, dim=0)

            # Clamp to positive (log-space parameterization requires θ > 0)
            thresholds = torch.clamp(thresholds, min=1e-4)

            self.activation.log_threshold.data = torch.log(thresholds)
        return thresholds

    def encode(self, x: torch.Tensor, return_pre_activations: bool = False):
        """
        Encode input to sparse feature representation.

        Centers input by subtracting decoder bias (b_dec) before encoding,
        following the standard SAE formulation:
            z = W_enc @ (x - b_dec) + b_enc

        Args:
            x: Input activations [batch, d_model] or [batch, seq, d_model]
            return_pre_activations: If True, also return pre-activations z

        Returns:
            f: Sparse features [batch, d_sae] or [batch, seq, d_sae]
            z: (optional) Pre-activations before JumpReLU [batch, d_sae]
        """
        # Center by decoder bias so encoder sees zero-mean residuals
        x_centered = x - self.b_dec
        # Pre-activations: z = W_enc @ (x - b_dec) + b_enc
        z = F.linear(x_centered, self.W_enc, self.b_enc)

        # Apply JumpReLU activation
        f = self.activation(z)

        if return_pre_activations:
            return f, z
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

        # Encode to sparse features (get pre-activations for L0 surrogate)
        f, z = self.encode(x_normalized, return_pre_activations=True)

        # Decode back to input space (still normalized)
        x_hat_norm = self.decode(f)

        # Denormalize output
        x_hat = self.denormalize(x_hat_norm, norm_coeff)

        # Compute losses
        losses = {}
        if return_loss:
            # Reconstruction loss (MSE)
            loss_reconstruction = F.mse_loss(x_hat, x, reduction='mean')

            # L0 sparsity (actual count, non-differentiable, for metrics only)
            with torch.no_grad():
                l0_per_sample = (f != 0).float().sum(dim=-1)  # [batch] or [batch, seq]
                l0_mean = l0_per_sample.mean()  # Average active features
                l0_sparsity = (f != 0).float().mean()  # Fraction for logging

            # Differentiable L0 using Straight-Through Estimator
            # Forward: exact Heaviside H(z - θ) — no phantom L0 from dead features
            # Backward: Gaussian kernel STE — smooth gradients near threshold boundary
            threshold = self.activation.threshold
            bandwidth = self.activation.bandwidth
            l0_differentiable = StraightThroughL0.apply(z, threshold, bandwidth)

            # L0 as count of active features, averaged over batch
            # Matches Gemma Scope paper: L = E_batch[ ||x - x̂||² + λ · Σᵢ H(zᵢ - θᵢ) ]
            # Sum over features (count per sample), mean over batch
            l0_count = l0_differentiable.sum(dim=-1).mean()  # [batch, d_sae] → [batch] → scalar

            # L0 penalty: λ * L0_count (differentiable via STE backward)
            # Paper-scale λ ≈ 6e-4 to 1e-3 (Rajamanoharan et al. 2024)
            loss_l0 = self.sparsity_coeff * l0_count

            # Total loss
            loss_total = loss_reconstruction + loss_l0

            # Compute FVU (Fraction of Variance Unexplained)
            var_original = x.var()
            var_residuals = (x - x_hat).var()
            fvu = var_residuals / (var_original + 1e-8)

            # Zero ablation loss (decoder bias only)
            x_zero = self.b_dec.expand_as(x)
            loss_zero = F.mse_loss(x_zero, x, reduction='mean')

            losses = {
                'loss': loss_total,
                'loss_reconstruction': loss_reconstruction,
                'loss_l0': loss_l0,
                'loss_zero': loss_zero,
                'l0_sparsity': l0_sparsity,
                'l0_mean': l0_mean,  # Average active feature count (for metrics)
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


def encode_with_training_normalization(sae, x: torch.Tensor) -> torch.Tensor:
    """Encode `x` applying the SAE's OWN training-time normalization first.

    MIS-E2E-083. `encode()` does NOT normalize — `forward()` does, at line
    ``x_normalized, norm_coeff = self.normalize(x)``, and then calls `encode`.
    So a caller that reaches for `encode()` directly hands the dictionary raw
    activations when it was trained on normalized ones.

    The extraction path knew this and normalized inline, with a comment saying
    why. Circuit capture, attribution, faithfulness and intervention all called
    `encode()` bare. Every circuit discovered from a capture was therefore mined
    from activations the dictionary was not trained to decode: the features
    fire, the numbers are plausible, and the basis is wrong — the same
    silent-wrong-basis shape as MIS-E2E-064, in the discovery plane rather than
    the steering plane, and everything downstream (co-activation statistics,
    attribution, edge validation) inherits it.

    One helper rather than the same three lines in six places, because five of
    the six had already forgotten them once.

    DIFFERENTIABLE. `normalize` is a scalar rescale, so the attribution path's
    gradient still flows — and flows through the same transform training used,
    which is the point.

    Handles the JumpReLU-style tuple return so callers do not each unpack it.
    """
    if getattr(sae, "normalize_activations", "none") != "none" and hasattr(sae, "normalize"):
        x, _coeff = sae.normalize(x)
    z = sae.encode(x)
    if isinstance(z, tuple):  # JumpReLU return_pre_activations styles
        z = z[0]
    return z


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
        architecture_type: One of 'standard', 'standard_saelens', 'standard_anthropic',
                          'skip', 'transcoder', 'jumprelu', 'topk'
        hidden_dim: Hidden dimension (or input_dim for transcoder)
        latent_dim: Latent dimension
        l1_alpha: L1 sparsity penalty (not used for topk/jumprelu)
        **kwargs: Additional architecture-specific parameters

    Returns:
        Initialized SAE model

    Raises:
        ValueError: If architecture_type is not recognized
    """
    architecture_type = architecture_type.lower()

    # Backward compatibility: map 'standard' to 'standard_saelens'
    if architecture_type == 'standard':
        architecture_type = 'standard_saelens'

    # Parameters specific to non-standard architectures (filter these out for standard/skip/transcoder)
    jumprelu_params = {'initial_threshold', 'bandwidth', 'sparsity_coeff', 'normalize_decoder', 'tied_weights'}
    topk_params = {'top_k', 'top_k_sparsity', 'aux_k', 'aux_loss_alpha', 'adam_epsilon'}
    non_standard_params = jumprelu_params | topk_params

    if architecture_type in ('standard_saelens', 'standard_anthropic'):
        # Standard SAE — SAELens or Anthropic normalization variant
        standard_kwargs = {k: v for k, v in kwargs.items() if k not in non_standard_params}

        # Anthropic variant uses anthropic_rescale normalization
        if architecture_type == 'standard_anthropic':
            standard_kwargs['normalize_activations'] = 'anthropic_rescale'

        return SparseAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=l1_alpha,
            **standard_kwargs
        )
    elif architecture_type == 'skip':
        skip_kwargs = {k: v for k, v in kwargs.items()
                       if k != 'ghost_gradient_penalty' and k not in non_standard_params}
        return SkipAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_alpha=l1_alpha,
            **skip_kwargs
        )
    elif architecture_type == 'transcoder':
        output_dim = kwargs.pop('output_dim', hidden_dim)
        transcoder_kwargs = {k: v for k, v in kwargs.items()
                             if k != 'ghost_gradient_penalty' and k not in non_standard_params}
        return Transcoder(
            input_dim=hidden_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            l1_alpha=l1_alpha,
            **transcoder_kwargs
        )
    elif architecture_type == 'topk':
        # TopK SAE (Gao et al. 2024, OpenAI)
        top_k = kwargs.pop('top_k', None)
        if top_k is None:
            # Backward compat: convert top_k_sparsity percentage to integer
            top_k_sparsity = kwargs.pop('top_k_sparsity', None)
            if top_k_sparsity is not None:
                top_k = max(1, int((top_k_sparsity / 100.0) * latent_dim))
            else:
                top_k = 64  # Reasonable default
        aux_k = kwargs.pop('aux_k', None)
        aux_loss_alpha = kwargs.pop('aux_loss_alpha', None)
        if aux_loss_alpha is None:
            aux_loss_alpha = 1.0 / 32
        normalize_activations = kwargs.pop('normalize_activations', None)
        if normalize_activations is None:
            normalize_activations = 'constant_norm_rescale'

        return TopKSAE(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            k=top_k,
            aux_k=aux_k,
            aux_loss_alpha=aux_loss_alpha,
            normalize_activations=normalize_activations,
        )
    elif architecture_type == 'jumprelu':
        # JumpReLU SAE (Rajamanoharan et al. 2024, Gemma Scope)
        initial_threshold = kwargs.pop('initial_threshold', None)
        if initial_threshold is None:
            initial_threshold = 0.5
        bandwidth = kwargs.pop('bandwidth', None)
        if bandwidth is None:
            bandwidth = 0.01
        normalize_decoder = kwargs.pop('normalize_decoder', None)
        if normalize_decoder is None:
            normalize_decoder = True
        tied_weights = kwargs.pop('tied_weights', None)
        if tied_weights is None:
            tied_weights = False
        normalize_activations = kwargs.pop('normalize_activations', None)
        if normalize_activations is None:
            normalize_activations = 'constant_norm_rescale'
        sparsity_coeff = kwargs.pop('sparsity_coeff', None)
        if sparsity_coeff is None:
            sparsity_coeff = 1e-3
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
            f"Must be one of: 'standard_saelens', 'standard_anthropic', "
            f"'skip', 'transcoder', 'topk', 'jumprelu'"
        )
