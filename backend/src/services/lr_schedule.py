"""The learning-rate schedule SAE training runs under.

Linear warmup from 0, then constant, then — only when ``lr_decay_steps`` is set —
a linear decay to 0 over the final ``lr_decay_steps`` steps of the run.

Everything is expressed in TRAINING STEPS, the unit ``warmup_steps``,
``lr_decay_steps`` and ``total_steps`` are configured in. The scheduler itself
counts OPTIMIZER steps, and with gradient accumulation (batch sizes under 64)
there are several training steps per optimizer step. The old inline lambda read
its argument as training steps, so an accumulating run warmed up ``k`` times too
slowly; :func:`build_lr_scheduler` converts.

A function and a builder rather than a lambda inside the training task, so the
curve is tested at its boundary steps and the scheduler the task builds is the
one the tests drive.
"""

from __future__ import annotations

import torch


def lr_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    decay_steps: int = 0,
) -> float:
    """The factor applied to the configured learning rate at training ``step``.

    * ``step < warmup_steps``: ``step / warmup_steps`` (0 at step 0).
    * inside the final ``decay_steps`` steps: ``(total_steps - step) / decay_steps``,
      so the factor is 1 where the decay starts and reaches 0 at ``total_steps``
      (the last step that runs uses ``1 / decay_steps``).
    * otherwise 1.

    The schema refuses ``warmup_steps + decay_steps > total_steps``. A row that
    predates that check can still overlap the two windows; the smaller factor
    wins, so the curve never jumps.
    """
    step = max(0, int(step))
    total = int(total_steps)
    warmup = max(0, int(warmup_steps))
    decay = max(0, int(decay_steps))

    factor = 1.0
    if warmup > 0 and step < warmup:
        factor = step / warmup
    if decay > 0 and step >= total - decay:
        factor = min(factor, max(0.0, (total - step) / decay))
    return factor


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    decay_steps: int = 0,
    grad_accum_steps: int = 1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """A ``LambdaLR`` following :func:`lr_multiplier` in training steps.

    The training loop steps the scheduler once per OPTIMIZER step. After ``e``
    optimizer steps, ``e * grad_accum_steps`` training steps have run, which is
    the step the factor is read at.

    Its ``state_dict()`` carries the position (``last_epoch``), so a resumed run
    continues the curve where it stopped — including part-way through a decay.
    """
    accum = max(1, int(grad_accum_steps))

    def factor_at(optimizer_steps: int) -> float:
        return lr_multiplier(
            optimizer_steps * accum,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor_at)
