"""Held-out SAE evaluation: in bounded chunks, over a sample drawn from every source.

WHY THIS EXISTS (SAE training remediation item 2, 2026-09-15). The in-training
held-out check encoded every held-out token in ONE forward pass. At the 100,000
token cap and 16,384 latents that is ~6 GiB per latent-sized intermediate, and a
JumpReLU forward holds three of them — an OOM on a card that is already carrying
the training buffer. The failure was caught and logged, so the run went on with
no held-out number at all.

And the sample was not a sample. It took each extraction's held-out positions
LOWEST INDEX FIRST until the cap was reached, so a multi-extraction run evaluated
the first rows of the first extraction and (at 100k tokens against ~49 2,048-token
blocks) often nothing else. The held-out FVU described one corpus.

WHAT THIS DOES INSTEAD.

* ``holdout_quotas`` splits the evaluation budget across sources by
  ``dataset_weights`` — EQUAL shares when no weights are set, with a source that
  cannot fill its share handing the remainder to the others.
* ``holdout_row_order`` / ``select_holdout_positions`` take WHOLE held-out rows in
  a seeded random order (never lowest-first), so a source's share is spread over
  its documents and each read stays a handful of rows.
* ``evaluate_holdout`` runs the SAE over ``chunk_tokens`` tokens at a time and
  aggregates SUMS, so the result is the same number an unchunked pass gives. The
  peak memory of one chunk is bounded by ``holdout_eval_peak_bytes``, which the
  training task subtracts from its GPU budget.

TWO FVUs, NAMED — and computed by ``ml.sae_metrics.FvuSums``, the one definition.

* ``fvu`` — the LEGACY global-mean value, ``(x - x_hat).var() / (x.var() + 1e-8)``
  over every element of the evaluated tokens, as ``JumpReLUSAE.forward`` computes
  it. Stored historical values mean this, so this key keeps meaning it.
* ``fvu_centred`` — ``Σ‖x − x̂‖² / Σ‖x − μ‖²`` with μ the PER-DIMENSION mean over
  the evaluated tokens, in raw activation space (WS-EVAL's definition). The legacy
  value subtracts one scalar mean from every dimension, so a dimension with a
  large constant offset inflates its denominator and it reads low.

This module used to carry its own float64 sums for both (review R1-B, 2026-09-15):
the same arithmetic as ``FvuSums``, written twice, with the residual taken in
float32 before summing. The post-run evaluation already aggregates through
``FvuSums``, so both evaluations now share it.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..ml.sae_metrics import FvuSums
from . import dataset_mixture

logger = logging.getLogger(__name__)

#: Held-out tokens evaluated per (layer, hook), across all sources.
DEFAULT_HOLDOUT_EVAL_TOKENS = 100_000
#: Tokens per forward pass. At 16,384 latents and 2,048 dims one chunk is bounded
#: by ~0.75 GiB (see :func:`holdout_eval_peak_bytes`); the unchunked 100,000-token
#: pass it replaces held ~18 GiB of latent-sized tensors alone.
DEFAULT_HOLDOUT_EVAL_CHUNK_TOKENS = 2_048
#: Separates the held-out row order's random stream from the training cycles',
#: which are seeded ``[seed, source_index]``.
_HOLDOUT_STREAM = 0x484F4C44


def holdout_eval_peak_bytes(chunk_tokens: int, hidden_dim: int, latent_dim: int) -> int:
    """An upper bound on the memory one chunk of :func:`evaluate_holdout` allocates.

    Latent-sized terms: the pre-activations, the gate and the gated output of a
    JumpReLU forward (three float32 tensors), plus the active-latent boolean and
    headroom — 16 bytes per chunk token per latent. Width-sized terms: the input
    copy, the normalised input and its reconstruction, the residual and the
    float64 casts of the sums — 64 bytes per chunk token per dimension. Measured
    against CPU allocations in ``test_holdout_evaluation.py``.
    """
    return int(chunk_tokens) * (16 * int(latent_dim) + 64 * int(hidden_dim))


def holdout_quotas(
    available: Sequence[int], total: int, weights: Optional[Sequence[float]] = None
) -> List[int]:
    """Held-out tokens to evaluate from each source.

    Proportional to ``weights`` — the training's ``dataset_weights``, positional
    over the same sources — and EQUAL when there are none: the evaluation is a
    statement about every source, not a re-run of the training mixture's
    proportional-to-size default. ``available`` caps each source; what a source
    cannot supply is redistributed (``dataset_mixture.allocate_tokens``).
    """
    n = len(available)
    if n == 0:
        return []
    shares = list(weights) if weights is not None else [1.0] * n
    return dataset_mixture.allocate_tokens(available, total, shares)


def holdout_row_order(n_rows: int, seed: int, source_index: int) -> np.ndarray:
    """The order a source's held-out rows are taken in: seeded, never lowest-first."""
    rng = np.random.default_rng([int(seed), _HOLDOUT_STREAM, int(source_index)])
    return rng.permutation(int(n_rows))


def select_holdout_positions(
    held_flat: np.ndarray, seq_len: int, quota: int, *, seed: int, source_index: int
) -> np.ndarray:
    """``quota`` held-out token positions from one extraction, as WHOLE rows in a seeded order.

    Returns ascending flat ``(row * seq_len + pos)`` positions (the order
    ``activation_mask.gather_tokens`` reads). The last row taken is cut to its
    first tokens so the quota is met exactly.
    """
    held_flat = np.sort(np.asarray(held_flat, dtype=np.int64))
    quota = int(quota)
    if quota <= 0 or held_flat.size == 0:
        return np.empty(0, dtype=np.int64)
    _, starts, counts = np.unique(held_flat // int(seq_len), return_index=True, return_counts=True)
    taken = []
    got = 0
    for i in holdout_row_order(starts.size, seed, source_index):
        take = min(int(counts[i]), quota - got)
        taken.append(held_flat[starts[i]: starts[i] + take])
        got += take
        if got >= quota:
            break
    return np.sort(np.concatenate(taken))


def _centering_bias(model) -> Optional[torch.Tensor]:
    """The bias a zero-latent reconstruction returns, in the normalised space, if the SAE has one."""
    for name in ("b_dec", "b_pre", "decoder_bias"):
        bias = getattr(model, name, None)
        if isinstance(bias, torch.Tensor):
            return bias
    return None


class HoldoutAccumulator:
    """Sums over chunks, from which the unchunked metrics are recovered exactly.

    Both FVUs come from one ``FvuSums`` (float64, chunk-invariant). The firing
    counts and the normalised-space losses stay on the device the chunks arrive
    on and are not synchronised until :meth:`result`.
    """

    def __init__(self) -> None:
        self.n_tokens = 0
        self.hidden_dim: Optional[int] = None
        self._fvu = FvuSums()
        self._firing: Optional[torch.Tensor] = None      # [latent], tokens each latent fired on
        self._recon_sq: Optional[torch.Tensor] = None    # Σ (x̂_norm - x_norm)²
        self._zero_sq: Optional[torch.Tensor] = None     # Σ (bias - x_norm)²

    def update(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        latents: torch.Tensor,
        *,
        x_norm: Optional[torch.Tensor] = None,
        x_hat_norm: Optional[torch.Tensor] = None,
        zero_bias: Optional[torch.Tensor] = None,
    ) -> None:
        f64 = torch.float64
        if self._firing is None:
            self.hidden_dim = int(x.shape[1])
            self._firing = torch.zeros(latents.shape[1], dtype=torch.int64, device=x.device)
        self.n_tokens += int(x.shape[0])

        self._fvu.update(x, x_hat)

        active = latents != 0
        self._firing += active.sum(dim=0)
        del active

        if x_norm is not None and x_hat_norm is not None:
            diff = x_hat_norm - x_norm
            chunk = (diff * diff).sum(dtype=f64)
            self._recon_sq = chunk if self._recon_sq is None else self._recon_sq + chunk
            del diff
            if zero_bias is not None:
                diff = zero_bias.to(x_norm.dtype) - x_norm
                chunk = (diff * diff).sum(dtype=f64)
                self._zero_sq = chunk if self._zero_sq is None else self._zero_sq + chunk

    def result(self) -> Dict[str, object]:
        """The metrics of every token seen so far.

        ``fvu`` is the legacy global-mean ratio; ``fvu_centred`` the per-dimension
        one. Either is None when its denominator is not positive (no tokens, or
        an input with no variance).
        """
        n = self.n_tokens
        if n == 0 or self._firing is None:
            return {
                "n_tokens": 0, "fvu": None, "fvu_centred": None, "l0_mean": None,
                "l0_sparsity": None, "firing_counts": None,
                "loss_reconstruction": None, "loss_zero": None,
            }
        elements = n * self.hidden_dim
        firing = self._firing.detach().to("cpu")
        active_total = int(firing.sum().item())
        latent = int(firing.numel())
        return {
            "n_tokens": n,
            "fvu": self._fvu.legacy(),
            "fvu_centred": self._fvu.centred(),
            "l0_mean": active_total / n,
            "l0_sparsity": active_total / (n * latent) if latent else None,
            "firing_counts": firing,
            "loss_reconstruction": (
                float(self._recon_sq.item()) / elements if self._recon_sq is not None else None
            ),
            "loss_zero": float(self._zero_sq.item()) / elements if self._zero_sq is not None else None,
        }


def evaluate_holdout(
    model: torch.nn.Module,
    held: torch.Tensor,
    device: torch.device,
    chunk_tokens: int = DEFAULT_HOLDOUT_EVAL_CHUNK_TOKENS,
    *,
    transcoder: bool = False,
) -> Dict[str, object]:
    """Evaluate ``model`` on ``held`` (``[tokens, d]``, any device), ``chunk_tokens`` at a time.

    Runs in eval mode under ``torch.no_grad`` and restores the model's mode
    however it ends. Every quantity is a sum over chunks, and every SAE here
    encodes each token independently (normalisation is per token), so the result
    does not depend on the chunk size.

    Returns :meth:`HoldoutAccumulator.result` plus ``loss_l0`` — the SAE's
    current ``sparsity_coeff`` times ``l0_mean`` for an SAE that has one (JumpReLU).
    ``loss_reconstruction`` and ``loss_zero`` are mean squared errors in the SAE's
    normalised input space (raw for a transcoder, which has none).
    """
    chunk_tokens = max(1, int(chunk_tokens))
    accumulator = HoldoutAccumulator()
    zero_bias = None if transcoder else _centering_bias(model)
    normalise = None if transcoder else getattr(model, "normalize", None)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for lo in range(0, int(held.shape[0]), chunk_tokens):
                x = held[lo:lo + chunk_tokens]
                if x.device != device:
                    x = x.to(device)
                if x.dtype != torch.float32:
                    x = x.float()
                if transcoder:
                    x_hat, latents, _ = model(x, x, return_loss=False)
                else:
                    x_hat, latents, _ = model(x, return_loss=False)
                x_norm = x_hat_norm = None
                if callable(normalise):
                    # Per-token normalisation, so the chunk does not change the
                    # coefficient: x_hat_norm is exactly what the forward decoded.
                    x_norm, coeff = normalise(x)
                    x_hat_norm = x_hat * coeff
                    del coeff
                elif transcoder:
                    x_norm, x_hat_norm = x, x_hat
                accumulator.update(
                    x, x_hat, latents, x_norm=x_norm, x_hat_norm=x_hat_norm, zero_bias=zero_bias,
                )
                del x, x_hat, latents, x_norm, x_hat_norm
    finally:
        if was_training:
            model.train()
    result = accumulator.result()
    coeff = getattr(model, "sparsity_coeff", None)
    result["loss_l0"] = (
        float(coeff) * result["l0_mean"]
        if isinstance(coeff, (int, float)) and result["l0_mean"] is not None else None
    )
    return result
