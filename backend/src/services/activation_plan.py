"""The activation storage plan a training ran with, and what a resume does with it.

WHY THIS EXISTS (SAE training remediation, review round 1, R1D-1 and R1D-2,
2026-09-15). A training's buffer size, and so the per-refill quotas its activation
source is built with, were planned from the free GPU memory and available RAM
measured when the process started. A resumed run is a new process: it measured
again, planned again, built its source with slightly different quotas, and the
source's ``load_state_dict`` refused the checkpoint's position ("the saved
buffer's quotas [40014, 40014] differ from this buffer's [39093, 39092]"). The
tolerance was ~27 KB of free memory for a 3-layer run at width 2,048, and
MemAvailable moves by more than that every second, so every ``*_rolling`` resume
and every on-the-fly resume failed on real hardware. The only resume test passed
because it pinned the planner to a constant.

WHAT HAPPENS NOW.

* A checkpoint saves the plan (``build_plan``) in ``training_state.pt`` beside the
  source's position.
* A resume that finds a plan REUSES it when it still fits what this process can
  hold (``plan_fits``). The capacities it is compared against are the ones the
  planner itself is given, computed with the same reserves and factors, so a
  reused plan is never larger than a fresh plan could have been. The source is
  built with the saved quotas, and the resume is exact.
* When the plan no longer fits, the buffer is RE-PLANNED from this process's
  memory and the source continues from the saved position under the new quotas
  (``restore_source_position``). No row repeats and the held-out split is
  unchanged, but the batches differ from an uninterrupted run's. That is logged
  loudly with the old and new sizes and quotas (``describe_replan``) and recorded
  in every later checkpoint (``resume_report``).
* A checkpoint written before plans were saved re-plans as before, with a
  warning; its source position loads only if the fresh plan happens to match.

THE FITS RULE, EXACTLY. A saved plan fits when its ``buffer_tokens`` (per layer) is
at most this process's capacity for its mode: the GPU capacity for ``gpu_*`` modes,
the RAM capacity for ``cpu_*`` modes. Those capacities are the numbers passed to
``plan_activation_storage``: free GPU memory less the SAEs' pending optimizer
state, a step's intermediates (at least 1 GiB), the held-out evaluation chunk (and
one model forward on the fly), times 0.9; and half of MemAvailable. So a card that
now reads even one allocator block less free memory than when the plan filled it
re-plans rather than eating into those margins. One correction is made first: the
optimizer state a resumed run has already restored onto the card is credited back
to the reading (``free_bytes_for_planning``), because the budget counts that state
as still to come and a fresh run measured before it existed.

WHAT A RESUME REUSES BESIDES THE SIZES. The storage MODE (a rolling buffer cannot
continue a fixed pool's position, nor the reverse) and, on the fly, the per-source
real-token ESTIMATES the quotas were allocated over: they come from a sample of
rows, and a resumed run allocates from the run's own numbers rather than estimating
again, so a change to the estimator between pause and resume moves nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PLAN_KIND = "activation_storage_plan"
PLAN_VERSION = 1
PATHS = ("cached", "on_the_fly")

#: No checkpoint to resume from: the plan is the planner's.
FRESH = "fresh"
#: The checkpoint's plan still fits: the source is built exactly as the run's was.
REUSED = "reused"
#: The checkpoint's plan no longer fits: planned again from this process's memory.
REPLANNED = "replanned"
#: The checkpoint's training state predates saved plans.
NOT_RECORDED = "not_recorded"
#: The checkpoint holds SAE weights and no training state: only the weights continued.
WEIGHTS_ONLY = "weights_only"


def build_plan(
    *,
    path: str,
    mode: str,
    buffer_tokens: int,
    quotas: Optional[Sequence[int]],
    source_tokens: Sequence[int],
    gpu_capacity_tokens: int,
    ram_capacity_tokens: int,
    min_useful_tokens: int,
) -> Dict[str, Any]:
    """The plan a checkpoint records: plain values only (loaded with ``weights_only=True``).

    ``buffer_tokens`` is what the storage holds per layer at once; ``quotas`` the
    per-refill real-token quota of each source (None when the pool is loaded whole
    with no allocation). ``source_tokens`` is each source's trainable token count
    when planned, which identifies the data the quotas were allocated over. The
    capacities are recorded for the log, not compared.
    """
    if path not in PATHS:
        raise ValueError(f"unknown training path {path!r}")
    return {
        "kind": PLAN_KIND,
        "version": PLAN_VERSION,
        "path": path,
        "mode": str(mode),
        "buffer_tokens": int(buffer_tokens),
        "quotas": None if quotas is None else [int(q) for q in quotas],
        "source_tokens": [int(n) for n in source_tokens],
        "gpu_capacity_tokens": int(gpu_capacity_tokens),
        "ram_capacity_tokens": int(ram_capacity_tokens),
        "min_useful_tokens": int(min_useful_tokens),
    }


def plan_fits(plan: Dict[str, Any], *, gpu_capacity_tokens: int, ram_capacity_tokens: int) -> bool:
    """Whether this process can hold the saved plan's buffer where the plan put it.

    The capacities must be the ones the planner is given (see the module
    docstring), so its safety margins apply to a reused plan too.
    """
    capacity = gpu_capacity_tokens if str(plan["mode"]).startswith("gpu") else ram_capacity_tokens
    return int(plan["buffer_tokens"]) <= max(0, int(capacity))


def _tensor_bytes_on(optimizers: Dict[Any, Any], models: Dict[Any, Any], device: Any) -> int:
    """Bytes of optimizer-state tensors and gradients held on ``device``, capped at 3x the weights there."""
    import torch

    device = torch.device(device)
    seen = set()
    state_bytes = 0
    for optimizer in optimizers.values():
        for entry in optimizer.state.values():
            for value in entry.values():
                if torch.is_tensor(value) and value.device == device and value.data_ptr() not in seen:
                    seen.add(value.data_ptr())
                    state_bytes += value.numel() * value.element_size()
    weight_bytes = 0
    for model in models.values():
        for param in model.parameters():
            if param.device != device:
                continue
            weight_bytes += param.numel() * param.element_size()
            if param.grad is not None and param.grad.data_ptr() not in seen:
                seen.add(param.grad.data_ptr())
                state_bytes += param.grad.numel() * param.grad.element_size()
    return min(state_bytes, 3 * weight_bytes)


def restored_state_bytes(optimizers: Dict[Any, Any], models: Dict[Any, Any], device: Any) -> int:
    """Optimizer moments and gradients a resumed run has already put on its card. 0 off a card."""
    import torch

    if device is None or torch.device(device).type != "cuda":
        return 0
    return _tensor_bytes_on(optimizers, models, device)


def free_bytes_for_planning(
    measured_free_bytes: int, *, resuming: bool, optimizers: Dict[Any, Any], models: Dict[Any, Any], device: Any
) -> int:
    """The free-memory reading a buffer budget should see, with a resume's restored state credited back.

    ``sae_buffer_budget`` subtracts the SAEs' Adam moments and gradients as PENDING —
    allocated lazily at the first step — from the free memory measured at set-up. A
    fresh run measures before they exist. A resumed run measures AFTER
    ``restore_training_state`` has loaded the moments (and any mid-accumulation
    gradients) onto the card, so the budget counted them twice and every resume on a
    card planned a buffer smaller than the run it continues by exactly that state,
    re-planning where the saved plan fits (review round 1, reviewer B). Crediting what
    was restored removes only the double count: the reserves, the step intermediates
    and the 0.9 factor apply as they did when the plan was made.

    LOGGED, both on a fresh run and on a resume (review round 2, R2-A): whether a saved
    plan is reused turns on these readings agreeing to within a few hundred tokens per
    layer, and what moves them between two processes on a card (allocator pages, the
    CUDA context, other processes) can only be measured there. Each reading is logged
    with its credit and, on a card, the caching allocator's reserved and allocated bytes.
    """
    import torch

    credit = restored_state_bytes(optimizers, models, device) if resuming else 0
    allocator = ""
    if device is not None and torch.device(device).type == "cuda":
        try:
            allocator = (
                f"; the caching allocator holds {torch.cuda.memory_reserved(device):,} bytes reserved and "
                f"{torch.cuda.memory_allocated(device):,} allocated"
            )
        except Exception:  # noqa: BLE001 - a log line never fails the plan
            allocator = ""
    logger.info(
        "Free memory reading for the activation storage plan on %s (%s): %s bytes free, %s bytes of "
        "restored optimizer state credited back%s",
        device, "resuming" if resuming else "fresh run", f"{int(measured_free_bytes):,}", f"{credit:,}", allocator,
    )
    return int(measured_free_bytes) + credit


@dataclass(frozen=True)
class StoragePlanChoice:
    """What the training builds its activation storage from."""

    mode: str
    buffer_tokens: int
    #: The saved per-refill quotas when the saved plan is reused; None means allocate.
    quotas: Optional[List[int]]
    outcome: str
    saved: Optional[Dict[str, Any]]
    fresh: Tuple[str, int]
    gpu_capacity_tokens: int
    ram_capacity_tokens: int


def choose_storage_plan(
    *,
    saved_plan: Optional[Dict[str, Any]],
    resuming: bool,
    path: str,
    source_tokens: Sequence[int],
    fresh: Tuple[str, int],
    gpu_capacity_tokens: int,
    ram_capacity_tokens: int,
) -> StoragePlanChoice:
    """Reuse the checkpoint's plan when it fits, else the planner's ``fresh`` plan.

    ``resuming`` is True when the checkpoint carries training state (a source
    position to continue). Refuses a saved plan for another path or for data
    whose trainable token counts changed: its quotas were allocated over other
    sources, and the saved position cannot mean anything in these.
    """
    fresh_mode, fresh_tokens = str(fresh[0]), int(fresh[1])

    def choice(mode, tokens, quotas, outcome):
        return StoragePlanChoice(
            mode=mode, buffer_tokens=int(tokens), quotas=quotas, outcome=outcome, saved=saved_plan,
            fresh=(fresh_mode, fresh_tokens), gpu_capacity_tokens=int(gpu_capacity_tokens),
            ram_capacity_tokens=int(ram_capacity_tokens),
        )

    if not resuming:
        return choice(fresh_mode, fresh_tokens, None, FRESH)
    if saved_plan is None:
        return choice(fresh_mode, fresh_tokens, None, NOT_RECORDED)
    if saved_plan.get("kind") != PLAN_KIND or int(saved_plan.get("version", 0)) != PLAN_VERSION:
        raise ValueError(f"not a {PLAN_KIND} v{PLAN_VERSION}: {saved_plan.get('kind')!r}")
    if saved_plan.get("path") != path:
        raise ValueError(
            f"the checkpoint's activation plan is for the {saved_plan.get('path')} path and this run "
            f"takes the {path} path; resume the training it was saved by"
        )
    saved_tokens = [int(n) for n in saved_plan["source_tokens"]]
    if saved_tokens != [int(n) for n in source_tokens]:
        raise ValueError(
            f"Cannot resume: the checkpoint's activation plan was made over {saved_tokens} trainable "
            f"tokens per source and the training's sources now hold {[int(n) for n in source_tokens]}. "
            "The extractions or datasets changed after the checkpoint, so its data position means "
            "nothing in them. Start a new training on the current data."
        )
    if plan_fits(saved_plan, gpu_capacity_tokens=gpu_capacity_tokens, ram_capacity_tokens=ram_capacity_tokens):
        quotas = saved_plan.get("quotas")
        return choice(
            str(saved_plan["mode"]), int(saved_plan["buffer_tokens"]),
            None if quotas is None else [int(q) for q in quotas], REUSED,
        )
    mode = fresh_mode
    if str(saved_plan["mode"]).endswith("_rolling") and mode.endswith("_all"):
        # KEEP CYCLING. The run was reading its pool without replacement; a pool
        # that now fits whole would be sampled WITH replacement by the fixed-pool
        # sampler, repeating rows the run has not finished its pass over. A
        # rolling buffer as large as the pool keeps the no-repeat guarantee.
        mode = mode[: -len("_all")] + "_rolling"
    return choice(mode, fresh_tokens, None, REPLANNED)


def _state_matches(source: Any, state: Dict[str, Any]) -> bool:
    """Whether ``source.load_state_dict(state)`` resumes exactly: same kind, same quotas or pool."""
    from .activation_buffer import FixedPoolSampler, RollingActivationBuffer
    from .model_activation_source import STATE_KIND as MODEL_STATE_KIND
    from .model_activation_source import ModelActivationSource

    kind = state.get("kind")
    if isinstance(source, RollingActivationBuffer):
        return kind == source.STATE_KIND and [int(q) for q in state.get("quotas", [])] == source.quotas
    if isinstance(source, ModelActivationSource):
        return kind == MODEL_STATE_KIND and [int(q) for q in state.get("quotas", [])] == source.quotas
    if isinstance(source, FixedPoolSampler):
        return kind == source.STATE_KIND and int(state.get("num_samples", -1)) == source.num_samples
    return False


def restore_source_position(source: Any, state: Dict[str, Any], choice: StoragePlanChoice) -> Dict[str, Any]:
    """Put ``source`` where the checkpoint's run stood. Returns what the resume report records.

    * The saved state matches the source (its plan was reused, or a fresh plan
      happened to equal it): exact ``load_state_dict``.
    * The plan was re-planned: the source continues the saved position under its
      new quotas (``load_state_dict_after_replan``), skipping the interrupted
      buffer's unserved tail; or, when the run drew from a fixed pool (with
      replacement, so there is no pass to continue), the rolling buffer starts
      its first pass.
    * Anything else goes through ``load_state_dict``, which refuses a mismatch
      exactly as a resume did before plans were saved.
    """
    from .activation_buffer import FixedPoolSampler, RollingActivationBuffer
    from .model_activation_source import STATE_KIND as MODEL_STATE_KIND
    from .model_activation_source import ModelActivationSource

    if _state_matches(source, state) or choice.outcome != REPLANNED:
        source.load_state_dict(state)
        return {"bit_identical": True, "tokens_skipped": 0, "source_restarted": False}
    kind = state.get("kind")
    if (isinstance(source, RollingActivationBuffer) and kind == source.STATE_KIND) or (
        isinstance(source, ModelActivationSource) and kind == MODEL_STATE_KIND
    ):
        skipped = source.load_state_dict_after_replan(state)
        return {"bit_identical": False, "tokens_skipped": int(skipped), "source_restarted": False}
    if isinstance(source, RollingActivationBuffer) and kind == FixedPoolSampler.STATE_KIND:
        return {"bit_identical": False, "tokens_skipped": 0, "source_restarted": True}
    raise ValueError(
        f"Cannot resume: the checkpoint's data position is a {kind!r} state and the re-planned run "
        f"reads through a {type(source).__name__}, which cannot continue it without repeating rows. "
        "Free memory on the training's card (or host) so the checkpoint's plan fits again, and resume."
    )


def _summary(plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if plan is None:
        return None
    return {"mode": plan["mode"], "buffer_tokens": int(plan["buffer_tokens"]), "quotas": plan.get("quotas")}


def selection_unchanged(saved: Optional[Dict[str, Any]], plan: Dict[str, Any]) -> bool:
    """Whether a re-planned source serves exactly the saved one's batches: the storage moved, the selection did not.

    The rule ``restore_source_position`` loads exactly by (``_state_matches``): a rolling
    buffer over the same per-refill quotas, or a whole pool of the same size over the same
    quotas. It is what happens when a pool loaded whole moves between the card and the host.
    """
    if not saved:
        return False
    rolling = str(saved["mode"]).endswith("_rolling")
    if rolling != str(plan["mode"]).endswith("_rolling"):
        return False
    if saved.get("quotas") != plan.get("quotas"):
        return False
    return rolling or int(saved["buffer_tokens"]) == int(plan["buffer_tokens"])


def describe_replan(choice: StoragePlanChoice, plan: Dict[str, Any]) -> str:
    """The log line a re-planned resume emits: why, and the old and new sizes and quotas.

    A re-plan over the same selection (``selection_unchanged``) continues exactly and says
    so. It used to warn that every batch differed while the resume report beside it
    recorded bit_identical (review round 2, R2-A).
    """
    saved = choice.saved or {}
    where = "GPU" if str(saved.get("mode", "")).startswith("gpu") else "RAM"
    capacity = choice.gpu_capacity_tokens if where == "GPU" else choice.ram_capacity_tokens
    if selection_unchanged(choice.saved, plan):
        return (
            f"RESUME MOVED THE ACTIVATION STORAGE: the checkpoint's activation plan was {saved.get('mode')} "
            f"with {int(saved.get('buffer_tokens', 0)):,} tokens per layer and quotas {saved.get('quotas')}, "
            f"and this process's {where} holds only {int(capacity):,} tokens per layer with the planner's "
            f"margins. Re-planned as {plan['mode']} over the same selection "
            f"({int(plan['buffer_tokens']):,} tokens per layer, quotas {plan.get('quotas')}), so the batches "
            "continue exactly."
        )
    return (
        f"RESUME IS NOT BIT-IDENTICAL: the checkpoint's activation plan was {saved.get('mode')} with a "
        f"buffer of {int(saved.get('buffer_tokens', 0)):,} tokens per layer and per-refill quotas "
        f"{saved.get('quotas')}, and this process's {where} holds only {int(capacity):,} tokens per layer "
        f"with the planner's margins. Re-planned as {plan['mode']} with a buffer of "
        f"{int(plan['buffer_tokens']):,} tokens per layer and quotas {plan.get('quotas')}. No row repeats "
        "and the held-out split is unchanged, but every batch after the checkpoint differs from an "
        "uninterrupted run's."
    )


def resume_report(
    *,
    checkpoint_step: int,
    checkpoint_id: Optional[str],
    choice: StoragePlanChoice,
    plan: Optional[Dict[str, Any]],
    restored: Dict[str, Any],
) -> Dict[str, Any]:
    """One resume, as every later checkpoint records it (``resume_history``)."""
    return {
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_id": None if checkpoint_id is None else str(checkpoint_id),
        "activation_plan": choice.outcome,
        "bit_identical": bool(restored["bit_identical"]),
        "tokens_skipped": int(restored.get("tokens_skipped", 0)),
        "source_restarted": bool(restored.get("source_restarted", False)),
        "saved_plan": _summary(choice.saved),
        "plan": _summary(plan),
    }


def weights_only_resume_report(
    *, checkpoint_step: int, checkpoint_id: Optional[str], plan: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """A resume from a checkpoint with no ``training_state.pt``, as every later checkpoint records it.

    Adam, the learning-rate warmup, the loss scale, the dead-latent statistics, the RNG and
    the data position all restarted; only the weights continued. Nothing recorded such a
    resume, so the run's later checkpoints claimed none (review round 2, R2-A).
    """
    return {
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_id": None if checkpoint_id is None else str(checkpoint_id),
        "activation_plan": WEIGHTS_ONLY,
        "bit_identical": False,
        "tokens_skipped": 0,
        "source_restarted": True,
        "saved_plan": None,
        "plan": _summary(plan),
    }
