"""Dead-latent detection and resampling for SAE training.

WHAT WAS WRONG (tracker item 1, 2026-09-15). The routine in the training loop:

* judged "dead" by an activity EMA and never read ``dead_neuron_threshold``;
* set ``W_enc[dead] = 0.1 * x`` from the RAW activation — the encoder sees the
  NORMALISED, centred input (``constant_norm_rescale`` rescales every token to
  norm sqrt(d)), so the new row's scale bore no relation to the others;
* gave the decoder column a random direction unrelated to the encoder row;
* never touched a JumpReLU threshold, so a latent killed by its threshold stayed
  behind that threshold and could not fire on the input it was made from;
* left Adam's moments in place, so the next update kept pushing the new weights
  the way the dead ones had been pushed;
* ran only inside the logging block, so it fired only at common multiples of
  ``log_interval`` and ``resample_interval``.

WHAT IT DOES NOW. The Anthropic recipe ("Towards Monosemanticity", neuron
resampling), in the space the encoder sees:

1. A latent is dead once it has not fired on any token for
   ``dead_neuron_threshold`` consecutive TRAINING STEPS (:class:`DeadLatentTracker`).
2. Inputs are drawn from the current batch with probability proportional to the
   SQUARE of their reconstruction loss (per-token squared error, measured in the
   normalised space the training loss uses), without replacement.
3. For a dead latent given input ``x``, with ``c = normalize(x) - b_dec`` (the
   encoder's own input) and ``u = c / |c|``:
     * encoder row = ``u * 0.2 * mean(|W_enc row|)`` over the latents still alive;
     * decoder column = ``u`` (unit norm, the matching direction);
     * encoder bias = 0;
     * JumpReLU threshold = half the new latent's pre-activation on ``x``, which is
       ``0.2 * mean alive norm * |c|``. It therefore fires on ``x``, and on inputs
       whose projection onto ``u`` is at least half of ``x``'s.
4. Adam's ``exp_avg`` and ``exp_avg_sq`` are zeroed for exactly the re-initialised
   slices — the rows, the columns, the bias entries and the threshold entries —
   and any accumulated gradient in those slices is cleared. Every other slice
   keeps its moments.

The work happens under ``torch.no_grad`` with autocast DISABLED, on the float32
parameters, after the step's ``scaler.update()``. A GradScaler keeps no
per-parameter state, so nothing in it needs resetting.

ARCHITECTURES. Standard (SAELens and "Anthropic"), Skip, Transcoder and JumpReLU
use this routine. TopK does not: it revives dead latents with its auxiliary loss
(Gao et al. 2024), and the training loop never resamples it. Tied weights are
refused because one matrix cannot take a separate encoder row and decoder column
(training never builds a tied SAE).

KNOWN COST OF STEP 4, kept because the recipe specifies it. Adam's bias
correction depends on the parameter's global step count, which cannot be reset
per slice. A zeroed second moment therefore makes the first updates to a
resampled slice larger than a steady-state update — about ``1/sqrt(1 - beta2)``
(~32x at beta2 = 0.999) on the first step, shrinking as the moment refills.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..ml.sparse_autoencoder import (
    JumpReLUSAE,
    SparseAutoencoder,
    TopKSAE,
    Transcoder,
)

logger = logging.getLogger(__name__)

#: The unit of ``dead_neuron_threshold``: consecutive training steps without a
#: single firing on any token of the step's batch.
DEAD_NEURON_THRESHOLD_UNIT = "consecutive training steps with no firing"

#: Encoder rows are re-initialised to this fraction of the mean alive row norm.
DEFAULT_ENCODER_SCALE = 0.2

#: A resampled JumpReLU threshold is this fraction of the latent's pre-activation
#: on the input it was made from.
DEFAULT_THRESHOLD_FRACTION = 0.5

#: The threshold is stored as a log, so it must stay positive.
THRESHOLD_FLOOR = 1e-6

_MOMENT_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


#: How many resample events a tracker remembers. A drift measurement compares a
#: revived latent's firing rate at the resample with its rate thousands of steps
#: later, and it reads those from two different checkpoints — so a tracker only
#: has to carry the events since the last save, not the whole run. 64 spans
#: 32,000 steps at the default interval of 500 and bounds what a checkpoint
#: carries: a fully dead 16,384-latent SAE is 128 KB per event.
RESAMPLE_HISTORY_EVENTS = 64


class DeadLatentTracker:
    """How long each latent has gone without firing, in training steps.

    ``update`` runs once per training step with that step's latent activations;
    a latent that fires on any token resets to 0. Its state is one integer per
    latent, saved in ``training_state.pt`` so a resumed run judges death exactly
    as an uninterrupted one would.

    It also remembers WHICH latents each resample revived. Nothing else can:
    ``update`` zeroes ``steps_since_fired`` for every latent that fired this
    step, and ``mark_revived`` then zeroes it for the revived ones, so after a
    step the two are indistinguishable. Without this record the only trace of a
    resample is its COUNT in a log line, and "did the latents we revived go on
    to fire?" cannot be answered from a checkpoint at all.
    """

    def __init__(self, latent_dim: int, device: Optional[torch.device] = None) -> None:
        self.steps_since_fired = torch.zeros(int(latent_dim), dtype=torch.long, device=device)
        #: (step, latent indices) per resample, oldest first. CPU tensors.
        self.resampled: list[tuple[int, torch.Tensor]] = []

    def update(self, z: torch.Tensor) -> None:
        with torch.no_grad():
            fired = (z.detach() != 0).reshape(-1, z.shape[-1]).any(dim=0)
            # The counter follows the activations to their device on first use, so
            # a tracker can be built (or restored from a checkpoint) on the CPU
            # before anything is placed on a GPU.
            if self.steps_since_fired.device != fired.device:
                self.steps_since_fired = self.steps_since_fired.to(fired.device)
            self.steps_since_fired = torch.where(
                fired,
                torch.zeros_like(self.steps_since_fired),
                self.steps_since_fired + 1,
            )

    def dead_mask(self, threshold_steps: int) -> torch.Tensor:
        return self.steps_since_fired >= int(threshold_steps)

    def mark_revived(self, latents: torch.Tensor) -> None:
        if latents.numel():
            self.steps_since_fired[latents.to(self.steps_since_fired.device)] = 0

    def record_resample(self, step: int, latents: torch.Tensor) -> None:
        """Remember which latents a resample revived at ``step``.

        Called AFTER :meth:`mark_revived`, which has already lost the
        distinction: it zeroes the same counter ``update`` zeroes for every
        latent that fired. Kept on the CPU and cloned, so nothing pins a GPU
        tensor or aliases the caller's.
        """
        if latents is None or latents.numel() == 0:
            return
        self.resampled.append((int(step), latents.detach().to("cpu", torch.long).clone()))
        if len(self.resampled) > RESAMPLE_HISTORY_EVENTS:
            del self.resampled[: -RESAMPLE_HISTORY_EVENTS]

    def state_dict(self) -> dict:
        return {
            "steps_since_fired": self.steps_since_fired.detach().cpu().clone(),
            "resampled": [(int(s), t.clone()) for s, t in self.resampled],
        }

    def load_state_dict(self, state: dict) -> None:
        restored = state["steps_since_fired"]
        if tuple(restored.shape) != tuple(self.steps_since_fired.shape):
            raise ValueError(
                f"dead-latent state has {tuple(restored.shape)} latents; "
                f"this SAE has {tuple(self.steps_since_fired.shape)}"
            )
        self.steps_since_fired = restored.to(
            device=self.steps_since_fired.device, dtype=torch.long
        ).clone()
        # ABSENT IN EVERY CHECKPOINT WRITTEN BEFORE THIS FIELD EXISTED. A resume
        # from one of those starts an empty history rather than refusing: the
        # run's own resamples are still recorded from here on.
        self.resampled = [
            (int(s), t.to("cpu", torch.long).clone()) for s, t in (state.get("resampled") or [])
        ]


def resample_due(
    step: int,
    *,
    interval: int,
    warmup_steps: int = 0,
    sparsity_warmup_steps: int = 0,
    total_steps: Optional[int] = None,
    lr_decay_steps: int = 0,
) -> bool:
    """Whether the training loop resamples at ``step``.

    Every ``interval`` steps, once both the LR warmup and the sparsity warmup are
    over — and never inside the LR decay window, where the learning rate heading
    to zero would leave a new latent no room to train.
    """
    interval = int(interval)
    if step <= 0 or interval <= 0 or step % interval != 0:
        return False
    if step < max(int(warmup_steps or 0), int(sparsity_warmup_steps or 0)):
        return False
    decay = int(lr_decay_steps or 0)
    if decay > 0 and total_steps is not None and step >= int(total_steps) - decay:
        return False
    return True


def resampling_unsupported_reason(model: torch.nn.Module) -> Optional[str]:
    """None when :func:`resample_dead_latents` can resample ``model``; else why not."""
    if isinstance(model, TopKSAE):
        return "TopK revives dead latents with its auxiliary loss (Gao et al. 2024)"
    if isinstance(model, JumpReLUSAE):
        if model.tied_weights or model.W_dec is None:
            return "tied encoder/decoder weights cannot take a separate row and column"
        return None
    if isinstance(model, Transcoder):
        if model.input_dim != model.output_dim:
            return "a transcoder whose output width differs from its input has no matching decoder direction"
        return None
    if isinstance(model, SparseAutoencoder):
        if model.tied_weights or model.decoder is None:
            return "tied encoder/decoder weights cannot take a separate row and column"
        return None
    return f"no resampling routine for {type(model).__name__}"


@dataclass
class ResampleResult:
    """What one resample did. Indices are CPU tensors."""

    latents: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    source_rows: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    dead_before: int = 0
    encoder_norm: float = 0.0
    skipped_reason: Optional[str] = None

    @property
    def count(self) -> int:
        return int(self.latents.numel())


@dataclass
class _Slices:
    enc_weight: torch.nn.Parameter  # [latent, d_in]; latents are ROWS
    enc_bias: torch.nn.Parameter  # [latent]
    dec_weight: torch.nn.Parameter  # [d_out, latent]; latents are COLUMNS
    center: torch.Tensor  # [d_in], subtracted from the input before the encoder
    log_threshold: Optional[torch.nn.Parameter]  # [latent], JumpReLU only


def _slices(model: torch.nn.Module) -> _Slices:
    if isinstance(model, JumpReLUSAE):
        return _Slices(model.W_enc, model.b_enc, model.W_dec, model.b_dec, model.activation.log_threshold)
    if isinstance(model, Transcoder):
        return _Slices(model.encoder.weight, model.encoder.bias, model.decoder.weight, model.b_enc_center, None)
    # SparseAutoencoder and its SkipAutoencoder subclass.
    return _Slices(model.encoder.weight, model.encoder.bias, model.decoder.weight, model.decoder_bias, None)


def _encoder_inputs_and_errors(model: torch.nn.Module, x: torch.Tensor):
    """``(what the encoder is given before centring, per-token squared error)``.

    The error is measured where the training loss is: in the normalised space.
    ``forward`` returns the reconstruction DENORMALISED (``x_hat_norm / coeff``),
    so ``(x - x_hat) * coeff`` is ``x_norm - x_hat_norm`` for every mode, and
    ``coeff`` is 1 for ``'none'``.
    """
    if isinstance(model, Transcoder):
        # The training loop trains transcoders to reproduce their input.
        x_hat, _, _ = model(x, x, return_loss=False)
        return x, (x - x_hat).pow(2).sum(dim=-1)
    x_norm, coeff = model.normalize(x)
    x_hat, _, _ = model(x, return_loss=False)
    return x_norm, ((x - x_hat) * coeff).pow(2).sum(dim=-1)


def _clear_slice(optimizer, param: torch.nn.Parameter, dim: int, index: torch.Tensor) -> None:
    """Zero ``param``'s Adam moments and accumulated gradient along ``index`` of ``dim``."""
    if param.grad is not None:
        param.grad.index_fill_(dim, index.to(param.grad.device), 0)
    if optimizer is None:
        return
    # `.get`, not `[]`: optimizer.state is a defaultdict, and indexing it would
    # create an empty entry for a parameter that has not been stepped yet.
    state = optimizer.state.get(param)
    if not state:
        return
    for name in _MOMENT_KEYS:
        buf = state.get(name)
        if torch.is_tensor(buf) and buf.shape == param.shape:
            buf.index_fill_(dim, index.to(buf.device), 0)


def resample_dead_latents(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    x: torch.Tensor,
    dead_mask: torch.Tensor,
    *,
    encoder_scale: float = DEFAULT_ENCODER_SCALE,
    threshold_fraction: float = DEFAULT_THRESHOLD_FRACTION,
    generator: Optional[torch.Generator] = None,
) -> ResampleResult:
    """Re-initialise the latents in ``dead_mask`` from high-loss rows of ``x``.

    ``x`` is a batch of RAW activations, as the SAE is trained on. At most one
    latent per row of ``x`` is resampled; any dead latents left over stay dead
    until the next resample. The input rows are drawn with the global CPU torch
    RNG unless ``generator`` is given, so a resumed run (which restores that RNG)
    draws the same rows.
    """
    dead_idx = torch.nonzero(dead_mask.detach().to("cpu"), as_tuple=False).flatten()
    reason = resampling_unsupported_reason(model)
    if reason is not None:
        return ResampleResult(dead_before=int(dead_idx.numel()), skipped_reason=reason)
    if dead_idx.numel() == 0 or x.shape[0] == 0:
        return ResampleResult(dead_before=int(dead_idx.numel()))

    slices = _slices(model)
    device = slices.enc_weight.device
    with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
        inputs, errors = _encoder_inputs_and_errors(model, x.detach().float())

        count = min(int(dead_idx.numel()), int(inputs.shape[0]))
        latents = dead_idx[:count]

        # Probability proportional to the SQUARE of each input's loss. A floor
        # keeps every row drawable, so drawing without replacement never asks
        # for more rows than have non-zero weight.
        weights = errors.detach().double().cpu().pow(2)
        if not bool(torch.isfinite(weights).all()) or float(weights.sum()) <= 0.0:
            weights = torch.ones_like(weights)
        weights = weights.clamp_min(float(weights.max()) * 1e-12 + 1e-300)
        rows = torch.multinomial(weights, count, replacement=False, generator=generator)

        centred = inputs[rows.to(inputs.device)].float() - slices.center.detach().float()
        norms = centred.norm(dim=-1, keepdim=True)
        directions = centred / norms.clamp_min(1e-8)

        latent_index = latents.to(device)
        alive = ~dead_mask.detach().to(device)
        row_norms = slices.enc_weight.detach().float().norm(dim=1)
        reference = row_norms[alive].mean() if bool(alive.any()) else row_norms.mean()
        if not bool(torch.isfinite(reference)) or float(reference) <= 0.0:
            reference = torch.tensor(1.0, device=device)
        scale = encoder_scale * reference

        dtype = slices.enc_weight.dtype
        slices.enc_weight[latent_index] = (scale * directions).to(device=device, dtype=dtype)
        slices.enc_bias[latent_index] = 0
        slices.dec_weight[:, latent_index] = directions.T.to(
            device=slices.dec_weight.device, dtype=slices.dec_weight.dtype
        )
        if slices.log_threshold is not None:
            # The new latent's pre-activation on its own input is
            # u . (scale * u) * |c| + 0 = scale * |c|.
            pre_activation = scale * norms.squeeze(-1).to(device)
            threshold = (threshold_fraction * pre_activation).clamp_min(THRESHOLD_FLOOR)
            slices.log_threshold[latent_index] = torch.log(threshold).to(slices.log_threshold.dtype)

    _clear_slice(optimizer, slices.enc_weight, 0, latent_index)
    _clear_slice(optimizer, slices.enc_bias, 0, latent_index)
    _clear_slice(optimizer, slices.dec_weight, 1, latent_index)
    if slices.log_threshold is not None:
        _clear_slice(optimizer, slices.log_threshold, 0, latent_index)

    return ResampleResult(
        latents=latents.cpu(),
        source_rows=rows.cpu(),
        dead_before=int(dead_idx.numel()),
        encoder_norm=float(scale),
    )
