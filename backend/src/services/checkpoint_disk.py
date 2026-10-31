"""Will this training's checkpoints fit on the disk? (debt items R2D-7 / R3D-15)

WHY THIS IS THE WORST FAILURE MODE THIS SYSTEM HAS. A full disk raises OSError
inside the training loop, the task's outer handler marks the run FAILED — and a
FAILED run's checkpoints are exactly what a resume needs
(``TrainingService.resume_training`` accepts PAUSED **or** FAILED). So running
out of disk mid-run does not merely stop the run: it stops it in the state whose
recovery depends on files the same full disk may have left half-written. Pausing
is strictly better than failing, and it is what this module's callers do.

WHAT A CHECKPOINT STEP COSTS, measured on the 16,384-latent runs (R3-D):

    one SAE's weights          268,575,168 B   (safetensors)
    one SAE's training state   537,301,604 B   (Adam's two moments, ~2x weights)
    three SAEs, one step     2,417,630,316 B
    ... ending mid-accumulation
        (the state also saves gradients)
                             3,223,355,820 B

THE FORECAST IS PINNED TO THE WRITERS, NOT TO THOSE NUMBERS. A formula copied
out of a measurement drifts the moment an architecture changes its parameters —
and the six architectures here genuinely differ (JumpReLU carries a latent-wide
``log_threshold`` that the standard SAE does not, so its checkpoint is 65 KB per
SAE larger at 16,384 latents). :func:`sae_footprint` therefore builds the REAL
SAE through the REAL ``create_sae`` on the **meta device** — shapes and dtypes,
no memory — and reads the same ``state_dict()`` that
``CheckpointService.save_checkpoint`` hands to ``save_file``. Change the model
and the forecast changes with it.

``test_checkpoint_disk_forecast.py`` closes the loop from both ends: it pins the
forecast against the four measured figures above, AND it writes a real small
checkpoint through the real writers and requires the forecast to bound what
landed on disk.

WHAT RETENTION CAN AND CANNOT RESCUE. ``checkpoint_retention`` never prunes a
training that is PENDING, INITIALIZING, RUNNING or PAUSED
(``ACTIVE_TRAINING_STATUSES``), so it can free nothing from the run being
forecast — that run is active for its whole life. Pruning relief therefore comes
only from OTHER trainings in terminal states, and only when retention is both
enabled and out of dry-run. Assuming otherwise would let a run start on the
strength of a prune that is structurally incapable of happening.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..core.config import settings

logger = logging.getLogger(__name__)

GiB = 1024 ** 3

#: safetensors writes a JSON header before the tensor data. Measured at 448 B for
#: a five-tensor JumpReLU SAE; allowed generously because over-forecasting costs
#: a little headroom and under-forecasting costs the run.
SAFETENSORS_HEADER_ALLOWANCE = 4 * 1024

#: Per-SAE bookkeeping inside ``training_state.pt`` beyond Adam's two moments:
#: the optimizer's per-parameter step counts, the LR scheduler, the GradScaler
#: and the dead-latent tracker. The latent-wide ``activation_ema`` and
#: ``firing_rate`` are counted exactly, below, because they scale with the SAE.
#:
#: SET SO THAT EVERY COMPONENT IS INDIVIDUALLY >= WHAT WAS MEASURED, not merely
#: the total. At 8 KiB the per-SAE state came out 12,900 B UNDER the 537,301,604
#: recorded for one 16,384-latent SAE, and only the per-step overhead below
#: dragged the step total back over the line. A forecast whose parts are wrong in
#: opposite directions is one refactor away from being wrong overall.
_STATE_OVERHEAD_PER_SAE = 24 * 1024

#: Per-STEP bookkeeping in that one file: the RNG states (python, numpy, torch,
#: one per CUDA device), the activation source's position, the storage plan, the
#: resume history and the zip container itself.
_STATE_OVERHEAD_PER_STEP = 64 * 1024

#: Never let a training run the volume down to nothing, whatever the arithmetic
#: says. Mirrors ``circuit_capture_service.MIN_FREE_DISK_BYTES`` and
#: ``jlens_acquire_service.MIN_FREE_DISK_BYTES``, the two other floors here.
MIN_RESERVE_BYTES = 5 * GiB

#: The reserve is at least this many whole checkpoint steps, so a run whose single
#: step is larger than the floor still keeps room to write one and pause.
RESERVE_CHECKPOINT_STEPS = 2

#: Bytes Adam keeps per parameter: ``exp_avg`` and ``exp_avg_sq``, both the
#: parameter's own dtype. This is the multiplier, not a measured constant — it
#: follows from ``torch.optim.Adam``'s state, which ``build_training_state``
#: saves via ``optimizer.state_dict()``.
_ADAM_MOMENTS_PER_PARAM = 2

#: A training in any of these states will still write checkpoints, so its
#: remaining run competes for the same volume. Imported rather than redefined:
#: one list of "still going" statuses for the pruner and for this module.
from .checkpoint_retention import ACTIVE_TRAINING_STATUSES  # noqa: E402


@dataclass(frozen=True)
class SaeFootprint:
    """What ONE SAE of this configuration costs on disk, from the real model."""

    state_dict_bytes: int
    param_bytes: int
    latent_dim: int
    architecture_type: str

    @property
    def weights_file_bytes(self) -> int:
        """One ``checkpoint.safetensors``: every state_dict entry, plus its header."""
        return self.state_dict_bytes + SAFETENSORS_HEADER_ALLOWANCE

    def state_bytes(self, *, include_grads: bool = False) -> int:
        """This SAE's share of the step's ``training_state.pt``.

        Adam's two moments cover the PARAMETERS (buffers have no optimizer
        state), the two latent-wide trackers are counted exactly, and gradients
        appear only when the step ended inside a gradient-accumulation window —
        ``build_training_state(include_grads=...)``, which the loop sets from
        ``(step + 1) % grad_accum_steps != 0``.
        """
        total = _ADAM_MOMENTS_PER_PARAM * self.param_bytes
        # activation_ema and firing_rate, one float32 per latent each.
        total += 2 * self.latent_dim * 4
        total += _STATE_OVERHEAD_PER_SAE
        if include_grads:
            total += self.param_bytes
        return total


def sae_shape_kwargs(hp: Dict[str, Any]) -> Dict[str, Any]:
    """The ``create_sae`` arguments that determine an SAE's SHAPE, from hyperparameters.

    Deliberately a small pure function rather than a copy of the training task's
    call: only these arguments change what ``state_dict()`` contains, and this
    repo's recurring defect is a guard that agrees with the code it guards by
    construction. ``test_checkpoint_disk_forecast.py`` builds the task's full
    argument list and this one for every architecture and requires identical
    state_dict shapes, so a new shape-bearing parameter cannot be added to the
    model without the forecast noticing.
    """
    architecture_type = hp.get("architecture_type", "standard")
    architecture_type = getattr(architecture_type, "value", architecture_type)
    architecture_type = str(architecture_type).lower()
    if architecture_type == "standard":
        architecture_type = "standard_saelens"

    kwargs: Dict[str, Any] = {
        "architecture_type": architecture_type,
        "hidden_dim": int(hp["hidden_dim"]),
        "latent_dim": int(hp["latent_dim"]),
    }
    if hp.get("tied_weights"):
        kwargs["tied_weights"] = True
    if architecture_type == "transcoder" and hp.get("output_dim"):
        kwargs["output_dim"] = int(hp["output_dim"])
    return kwargs


def sae_footprint(hp: Dict[str, Any]) -> SaeFootprint:
    """Build the configured SAE on the META device and measure what would be saved.

    The meta device allocates nothing, so this costs microseconds and no memory
    even at 16,384 latents — which matters, because the create-time refusal runs
    inside an API request.
    """
    import torch

    from ..ml.sparse_autoencoder import create_sae

    kwargs = sae_shape_kwargs(hp)
    with torch.device("meta"):
        model = create_sae(**kwargs)

    state_dict = model.state_dict()
    state_dict_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return SaeFootprint(
        state_dict_bytes=state_dict_bytes,
        param_bytes=param_bytes,
        latent_dim=int(hp["latent_dim"]),
        architecture_type=kwargs["architecture_type"],
    )


def checkpoint_step_bytes(
    footprint: SaeFootprint, num_saes: int, *, include_grads: bool = False
) -> int:
    """Bytes ONE checkpoint step writes: every SAE's weights, plus one state file.

    ``save_multilayer_checkpoint`` writes one ``checkpoint.safetensors`` per
    (layer, hook); ``save_training_state`` then writes ONE ``training_state.pt``
    for the step, holding every SAE's entry.
    """
    num_saes = int(num_saes)
    weights = num_saes * footprint.weights_file_bytes
    state = num_saes * footprint.state_bytes(include_grads=include_grads)
    return weights + state + _STATE_OVERHEAD_PER_STEP


def export_bytes(footprint: SaeFootprint, num_saes: int) -> int:
    """The Community Standard export the run writes once, at the end, and keeps.

    ``save_sae_community_format`` writes ``sae_weights.safetensors`` per SAE from
    the same state_dict (converted, not resized) plus a small ``cfg.json``.
    """
    return int(num_saes) * footprint.weights_file_bytes


def checkpoint_steps_remaining(
    *, total_steps: int, checkpoint_interval: int, start_step: int = 0
) -> int:
    """How many periodic checkpoints a run still has to write.

    The loop's condition is ``step % checkpoint_interval == 0 and step > 0`` over
    ``range(start_step, total_steps)``, so this counts the multiples of the
    interval that are ``> max(start_step - 1, 0)`` and ``< total_steps``.
    """
    total_steps = int(total_steps)
    interval = max(1, int(checkpoint_interval))
    start_step = max(0, int(start_step))
    if total_steps <= 0:
        return 0
    # Multiples of `interval` in [1, total_steps - 1] ...
    upto = (total_steps - 1) // interval
    # ... minus those already written, i.e. at or below start_step - 1.
    already = max(0, start_step - 1) // interval
    return max(0, upto - already)


def num_saes_for(hp: Dict[str, Any]) -> int:
    """SAEs this training builds: layers x hook types, the loop's own product."""
    layers = hp.get("training_layers", [0])
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    hooks = hp.get("hook_types", hp.get("hook_type", ["residual"]))
    if isinstance(hooks, str):
        hooks = [hooks]
    if not hooks:
        hooks = ["residual"]
    return max(1, len(layers) * len(hooks))


@dataclass(frozen=True)
class RunForecast:
    """What one training will still write to the checkpoint volume."""

    per_step_bytes: int
    per_step_bytes_with_grads: int
    steps_remaining: int
    checkpoints_bytes: int
    export_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.checkpoints_bytes + self.export_bytes


def forecast_run(hp: Dict[str, Any], *, start_step: int = 0) -> RunForecast:
    """Everything ``hp`` will still write, from its current step to the end.

    The per-step figure used for the RUN assumes the worst of the two save
    shapes: a step that ends inside a gradient-accumulation window also saves
    gradients, which is a third again. Which steps those are depends on the
    accumulation factor and is not worth predicting precisely; over-forecasting
    by a third of the checkpoints is the safe direction, and the run-level
    decision is about hundreds of gigabytes either way.
    """
    footprint = sae_footprint(hp)
    num_saes = num_saes_for(hp)
    per_step = checkpoint_step_bytes(footprint, num_saes)
    per_step_grads = checkpoint_step_bytes(footprint, num_saes, include_grads=True)
    steps = checkpoint_steps_remaining(
        total_steps=hp.get("total_steps", 0),
        checkpoint_interval=hp.get("checkpoint_interval", 1000),
        start_step=start_step,
    )
    return RunForecast(
        per_step_bytes=per_step,
        per_step_bytes_with_grads=per_step_grads,
        steps_remaining=steps,
        checkpoints_bytes=steps * per_step_grads,
        export_bytes=export_bytes(footprint, num_saes),
    )


def free_bytes_for(path: "str | Path") -> int:
    """Bytes an UNPRIVILEGED writer can still use on the filesystem holding ``path``.

    ``f_bavail``, never ``f_bfree``: the difference is the root-reserved
    fraction, which the backend (running unprivileged) can never allocate. On
    this project's own workstation volume that gap is 101 GB — large enough to
    turn a correct refusal into a run that fills the disk and fails.

    Walks up to the nearest existing ancestor, because the checkpoint directory
    is created by the training task and does not exist at create time.
    """
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    stat = os.statvfs(probe)
    return int(stat.f_bavail) * int(stat.f_frsize)


def checkpoint_root_for(training_id: Optional[str] = None) -> Path:
    """The directory this training's checkpoints are written into.

    The same expression the training task uses
    (``settings.data_dir / "trainings" / id / "checkpoints"``), resolved through
    ``settings.resolve_data_path`` so a deployment whose data_dir differs from
    the recorded prefix probes the volume it will actually write to.
    """
    relative = Path("trainings")
    if training_id:
        relative = relative / str(training_id) / "checkpoints"
    return settings.resolve_data_path(relative)


def reserve_bytes_for(per_step_bytes: int) -> int:
    """Headroom kept free beyond the forecast: 5 GiB, or two checkpoint steps."""
    return max(MIN_RESERVE_BYTES, RESERVE_CHECKPOINT_STEPS * int(per_step_bytes))


@dataclass(frozen=True)
class DiskVerdict:
    """The decision, with every number that produced it."""

    fits: bool
    forecast_bytes: int
    other_runs_bytes: int
    reclaimable_bytes: int
    reserve_bytes: int
    free_bytes: int
    per_step_bytes: int
    steps_remaining: int
    path: str
    other_runs: int = 0
    shortfall_bytes: int = 0

    @property
    def required_bytes(self) -> int:
        return self.forecast_bytes + self.other_runs_bytes + self.reserve_bytes

    @property
    def available_bytes(self) -> int:
        return self.free_bytes + self.reclaimable_bytes

    def message(self) -> str:
        """One sentence naming the forecast, the free space and the reserve."""
        parts = [
            f"needs {_gb(self.forecast_bytes)} of checkpoints",
            f"{_gb(self.free_bytes)} free on {self.path}",
            f"{_gb(self.reserve_bytes)} reserve",
        ]
        if self.other_runs_bytes:
            parts.insert(1, f"{_gb(self.other_runs_bytes)} already promised to "
                            f"{self.other_runs} other running job(s)")
        if self.reclaimable_bytes:
            parts.insert(-1, f"{_gb(self.reclaimable_bytes)} reclaimable by the pruner")
        joined = "; ".join(parts)
        if self.fits:
            return f"Checkpoint disk OK: {joined}"
        return (
            f"Not enough disk for this training's checkpoints: {joined}. "
            f"Short by {_gb(self.shortfall_bytes)}. Free space, lower total_steps, "
            f"raise checkpoint_interval, or enable checkpoint pruning."
        )


def _gb(value: int) -> str:
    return f"{value / GiB:.1f} GiB"


def decide(
    *,
    forecast: RunForecast,
    free_bytes: int,
    path: str,
    other_runs_bytes: int = 0,
    other_runs: int = 0,
    reclaimable_bytes: int = 0,
) -> DiskVerdict:
    """The pure decision, given every input. No filesystem, no database.

    Kept session-free and IO-free so the rule can be tested directly and so the
    create path, the task's start and the loop's per-save check cannot drift into
    three slightly different rules.
    """
    reserve = reserve_bytes_for(forecast.per_step_bytes)
    required = forecast.total_bytes + int(other_runs_bytes) + reserve
    available = int(free_bytes) + int(reclaimable_bytes)
    fits = available >= required
    return DiskVerdict(
        fits=fits,
        forecast_bytes=forecast.total_bytes,
        other_runs_bytes=int(other_runs_bytes),
        reclaimable_bytes=int(reclaimable_bytes),
        reserve_bytes=reserve,
        free_bytes=int(free_bytes),
        per_step_bytes=forecast.per_step_bytes,
        steps_remaining=forecast.steps_remaining,
        path=str(path),
        other_runs=int(other_runs),
        shortfall_bytes=max(0, required - available),
    )


def other_active_runs_bytes(rows: Iterable[Any], *, exclude_id: Optional[str] = None) -> tuple:
    """``(bytes, count)`` the other still-running trainings will still write.

    A forecast that looks only at the run being started is wrong the moment two
    trainings share a volume, which is the normal case here — the queue runs them
    back to back and a PAUSED run still owns everything it has written and
    everything it will write when resumed.

    Rows are duck-typed (``status``, ``hyperparameters``, ``current_step``) so
    both the async create path and the sync worker can pass what they already
    hold. A row whose hyperparameters cannot be forecast is skipped with a
    warning rather than blocking the decision.
    """
    total = 0
    counted = 0
    for row in rows:
        row_id = getattr(row, "id", None)
        if exclude_id is not None and row_id == exclude_id:
            continue
        status = getattr(row, "status", None)
        status = getattr(status, "value", status)
        if status not in ACTIVE_TRAINING_STATUSES:
            continue
        hp = getattr(row, "hyperparameters", None) or {}
        try:
            forecast = forecast_run(hp, start_step=int(getattr(row, "current_step", 0) or 0))
        except Exception as exc:  # noqa: BLE001 - one bad row must not block the decision
            logger.warning(
                "Could not forecast checkpoint disk for active training %s (%s); "
                "it is not counted against this decision", row_id, exc,
            )
            continue
        total += forecast.total_bytes
        counted += 1
    return total, counted


def reclaimable_by_pruning(rows: Iterable[Any], policy: Any) -> int:
    """Bytes a prune WOULD free, counting only what it is actually allowed to touch.

    ZERO unless retention is enabled AND out of dry-run: a dry-run prune reports
    and deletes nothing, so counting its plan as headroom would start a run on
    space that is never going to appear.

    And zero for any training that is still active. ``ACTIVE_TRAINING_STATUSES``
    puts PENDING, INITIALIZING, RUNNING and PAUSED off limits, which includes the
    run being forecast for the whole of its life — so pruning can never rescue
    the run asking the question, only make room by clearing terminal ones.
    """
    if policy is None or not getattr(policy, "enabled", False):
        return 0
    if getattr(policy, "dry_run", True):
        return 0

    from .checkpoint_retention import plan_from_checkpoints

    total = 0
    for row in rows:
        status = getattr(row, "status", None)
        status = getattr(status, "value", status)
        if status in ACTIVE_TRAINING_STATUSES:
            continue
        checkpoints = list(getattr(row, "checkpoints", None) or [])
        if not checkpoints:
            continue
        try:
            plan = plan_from_checkpoints(
                training_id=getattr(row, "id", ""),
                training_status=status,
                checkpoints=checkpoints,
                policy=policy,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not plan a prune for %s (%s)", getattr(row, "id", "?"), exc)
            continue
        total += int(plan.estimated_bytes or 0)
    return total


def verdict_from_rows(
    hp: Dict[str, Any],
    *,
    training_id: Optional[str] = None,
    start_step: int = 0,
    rows: Sequence[Any] = (),
    policy: Any = None,
) -> DiskVerdict:
    """The whole decision from hyperparameters and the trainings that share the volume.

    ONE rule with two thin adapters: the create path holds an ``AsyncSession``
    and the worker a sync one, so each fetches its own rows and hands them here.
    Three independently-written versions of "is there room" is exactly how the
    create-time answer and the run-time answer drift apart.
    """
    forecast = forecast_run(hp, start_step=start_step)
    path = checkpoint_root_for(training_id)
    other_bytes, other_count = other_active_runs_bytes(rows, exclude_id=training_id)
    return decide(
        forecast=forecast,
        free_bytes=free_bytes_for(path),
        path=str(path),
        other_runs_bytes=other_bytes,
        other_runs=other_count,
        reclaimable_bytes=reclaimable_by_pruning(rows, policy),
    )


def verdict_from_sync_session(
    db: Any,
    hp: Dict[str, Any],
    *,
    training_id: Optional[str] = None,
    start_step: int = 0,
) -> DiskVerdict:
    """:func:`verdict_from_rows` for the Celery worker, which holds a sync session."""
    from ..models.training import Training
    from .checkpoint_retention import load_policy

    policy = load_policy(db)
    rows: List[Any] = list(db.query(Training).all() or [])
    return verdict_from_rows(
        hp, training_id=training_id, start_step=start_step, rows=rows, policy=policy
    )


def room_for_one_step(
    *, per_step_bytes: int, checkpoint_dir: "str | Path"
) -> tuple:
    """``(has_room, free_bytes, needed_bytes)`` for ONE more checkpoint step.

    The loop's per-save question, which is narrower than the run-level one: not
    "will the whole run fit" but "can this save complete and still leave the
    volume able to take one more". Called immediately before each write, because
    free space is shared and a co-tenant can consume it between two saves.
    """
    free = free_bytes_for(checkpoint_dir)
    # This save, plus a reserve that already includes two steps' worth: enough to
    # write the pause checkpoint that follows a refusal.
    needed = int(per_step_bytes) + reserve_bytes_for(per_step_bytes)
    return free >= needed, free, needed


def remove_partial_step(step_dir: "str | Path") -> int:
    """Delete a checkpoint step directory left half-written by a full disk.

    Returns bytes removed (best effort, 0 if nothing was there).

    WHY THIS IS NOT OPTIONAL. A step directory whose files are truncated is
    selectable by TWO different consumers, and neither reads the database to
    decide:

      * ``training_finalize_service.list_checkpoint_steps`` scans the filesystem
        for ``checkpoint_<n>`` directories, so Stop & Finalize would happily
        rebuild the run's exported SAEs from a torn step.
      * ``select_resume_checkpoint`` is row-driven and therefore safe *only*
        while the rows are written after the files — which is the current order,
        but the filesystem is what survives a crash, and a partial directory that
        stays is a trap for any future reader.

    Deleting it also returns the bytes, which is the difference between a pause
    the operator can resume from and one that cannot write its own checkpoint.
    """
    import shutil

    path = Path(step_dir)
    if not path.is_dir():
        return 0
    freed = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                freed += child.stat().st_size
        except OSError:
            continue
    try:
        shutil.rmtree(path)
        logger.warning(
            "Removed the partially written checkpoint directory %s (%s); a torn step "
            "must not be visible to a resume or a finalize", path, _gb(freed),
        )
    except OSError as exc:
        logger.error("Could not remove the partial checkpoint directory %s: %s", path, exc)
        return 0
    return freed


def is_out_of_space(error: BaseException) -> bool:
    """Is this the disk being full, rather than any other OSError?

    ``errno.ENOSPC`` is the kernel's answer; ``EDQUOT`` is the same condition
    under a quota, which a shared volume can raise instead. Matched on errno, not
    on the message, because the message is locale-dependent.
    """
    import errno

    code = getattr(error, "errno", None)
    if code in (errno.ENOSPC, getattr(errno, "EDQUOT", None)):
        return True
    # torch.save wraps the underlying failure on some paths; check the cause too.
    cause = getattr(error, "__cause__", None) or getattr(error, "__context__", None)
    if cause is not None and cause is not error:
        return is_out_of_space(cause)
    return False
