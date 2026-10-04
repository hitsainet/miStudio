"""
Task-queue rows for J-space work, so long jobs are VISIBLE while they run.

WHY THIS EXISTS. A 45-minute fit burned the GPU with nothing anywhere in the
product saying so. The J-Lens panel's own fit card only knows about a fit THIS
browser tab started — its polling lives in component state, so a fit queued from
the API, from MCP, from another tab, or before a refresh was invisible. The
System Monitor's Active Operations panel reads `task_queue`, and J-space tasks
were never writing rows to it.

This repo has the same defect on record already: finalize and prune create no
task_queue row, so they do not appear in Active Operations either. The fix is
the same shape — write the row where the task is QUEUED, update it where the
task reports progress.

SYNC SESSIONS ON PURPOSE. `TaskQueueService` is async and Celery workers are
not; the workers that already write these rows (model_tasks, dataset_tasks) use
a sync session directly, and this follows them rather than introducing an event
loop into a worker.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: `task_type` values. Prefixed so a reader of Active Operations can tell J-space
#: work apart from training and extraction at a glance.
FIT = "jlens_fit"
BAND_REPORT = "jlens_band_report"
INTERVENTION = "jlens_intervention"
READOUT = "jlens_readout"
PROBE = "jlens_probe"
ACQUIRE = "jlens_acquire"
PUBLISH = "jlens_publish"
REVALIDATE = "jlens_revalidate"

#: Statuses after which a task will never report again. `completed_at` is
#: stamped on entering any of them, so a finished row carries a real duration
#: rather than an open-ended one.
TERMINAL_STATUSES = ("completed", "failed", "cancelled")


def open_row(
    task_type: str,
    entity_id: str,
    task_id: str,
    retry_params: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Record that a J-space task has been queued. Returns the row id.

    `retry_params` is what a re-run must reuse. J-space rows have no request
    columns, so the GPU request travels here as `{"gpu": ...}` — the value the
    endpoint RESOLVED, a UUID rather than the index the caller may have typed,
    because indices shift when a card is added and a retry must land on the
    card the job was asked to use.

    NEVER RAISES. A bookkeeping row failing to write must not fail the fit it
    describes — the work is the point and the row is the narration. Failures
    are logged so a missing row is diagnosable rather than mysterious.
    """
    from ..core.database import get_sync_db
    from ..models.task_queue import TaskQueue

    row_id = f"tq_{uuid.uuid4().hex[:12]}"
    try:
        with get_sync_db() as db:
            db.add(
                TaskQueue(
                    id=row_id,
                    task_id=task_id,
                    task_type=task_type,
                    entity_id=entity_id,
                    entity_type="model",
                    status="queued",
                    progress=0.0,
                    retry_params=dict(retry_params or {}),
                    retry_count=0,
                )
            )
            db.commit()
        return row_id
    except Exception as exc:  # noqa: BLE001 - narration must not break the work
        logger.warning("Could not open a task_queue row for %s: %s", task_id, exc)
        return None


def update_row(
    task_id: str,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    error_message: Optional[str] = None,
) -> bool:
    """Move a queued row along. Located by CELERY task id, not by row id.

    By task id because the worker knows that and would otherwise have to be
    handed the row id through the task signature — one more argument to forget,
    and forgetting it silently leaves a row stuck at "queued" forever.

    RETURNS WHETHER A ROW WAS FOUND. The endpoints open the row AFTER `.delay()`,
    so a task that fails in its first milliseconds can arrive here before the row
    exists — and a silent `return` then leaves it at "queued 0%" forever, which
    reads as a job that never started rather than one that failed instantly. A
    caller that can retry needs to know the difference; one that cannot is
    unaffected, since the value is simply ignored.

    NOW A SHIM OVER `core.cancellation.record_progress`. The terminal guard, the
    started_at/completed_at stamping and the clamp were all written here first
    and then independently rediscovered in `training_tasks`; the core module is
    where that rule now lives, and the `jlens_task` scope is the description of
    this table. Keeping the function is deliberate — it is the name eleven call
    sites use, and it carries the "located by celery id" contract above.
    """
    from ..core.cancellation import record_progress

    return record_progress(
        "jlens_task",
        task_id,
        status=status,
        progress=progress,
        error_message=error_message,
    )



def request_cancel(task_id: str, reason: str = "cancelled by operator") -> bool:
    """Ask a running J-space task to stop at its next checkpoint.

    A SHIM over `core.cancellation.request_cancel`. The solo-pool reasoning that
    used to be spelled out here now lives once in that module's docstring; this
    keeps the name and the bool return that eleven call sites and two endpoints
    already use.
    """
    from ..core.cancellation import request_cancel as _request_cancel

    return _request_cancel("jlens_task", task_id, reason=reason).requested


def cancel_checker(task_id: str):
    """A callable the work loop polls; True once cancellation is requested.

    A SHIM over `core.cancellation.cancel_checker`. The old `every=` count
    throttle is gone: a count is a guess about one loop's unit cost and travels
    to no other loop. See that module for why the budget is time.
    """
    from ..core.cancellation import cancel_checker as _cancel_checker

    return _cancel_checker("jlens_task", task_id)


#: PERMANENT ALIAS. `TaskCancelled` is caught by name in the J-space tasks and
#: asserted by name in their tests. Pointing it at `OperatorCancelled` keeps
#: both working AND upgrades it to a BaseException, so the bare
#: `except Exception` handlers on those paths can no longer turn an operator's
#: stop into a crash report.
def _task_cancelled_alias():
    from ..core.cancellation import OperatorCancelled

    return OperatorCancelled


TaskCancelled = _task_cancelled_alias()


def mark_running(task_id: str, progress: float = 1.0, attempts: int = 10) -> bool:
    """First transition to `running`, RETRIED past the row's own creation.

    THE ROW MAY NOT EXIST YET. Every J-space endpoint opens it AFTER `.delay()`,
    so a worker that picks the task up immediately can arrive before the insert
    commits — `update_row` then finds nothing, returns, and the row lands as
    "queued 0%" and never moves. That reads as a job that never started rather
    than one already running.

    FOR THE FIRST-AND-ONLY EARLY TRANSITION. Acquire and publish mark themselves
    running once and then do long work, so a missed write is permanent. The fit,
    band and intervention tasks write `running` from a REPEATING callback — one
    per prompt or per layer — so a first write that loses the race is corrected
    milliseconds later by the next, and putting a sleeping retry inside a hot
    callback would cost more than the race does. They deliberately keep
    `update_row`.
    """
    import time as _time

    for _attempt in range(max(1, attempts)):
        if update_row(task_id, status="running", progress=progress):
            return True
        _time.sleep(0.1)
    logger.warning(
        "No task_queue row for %s after %d attempts; it will not show progress",
        task_id,
        attempts,
    )
    return False


def record_gpu(
    task_id: Optional[str],
    gpu_uuid: Optional[str],
    gpu_uuids: Optional[list] = None,
    attempts: int = 10,
) -> bool:
    """Write the card(s) a J-space task runs on to its row. Returns whether it landed.

    `gpu_uuid` is the first card; `gpu_uuids` lists every card of a SPLIT and is
    None for one card — `Placement.gpu_columns()`. Both are written: a row that
    named only the first card of a split would tell Active Operations the other
    card was free while the job's weights filled it.

    CALLED BEFORE THE MODEL LOAD, so a task that dies loading still says which
    card it tried. Retried past the row's own creation for `mark_running`'s
    reason: the endpoint opens the row AFTER `.delay()`, and a worker that picks
    the task up at once can arrive first.

    A CPU placement writes nothing — NULL already means "no card". NEVER RAISES:
    the row is the narration, and the work is the point.
    """
    if not task_id or not gpu_uuid:
        return False
    import time as _time

    from ..core.cancellation import record_progress

    try:
        for _attempt in range(max(1, attempts)):
            if record_progress(
                "jlens_task", task_id, gpu_uuid=gpu_uuid, gpu_uuids=gpu_uuids
            ):
                return True
            _time.sleep(0.1)
    except Exception as exc:  # noqa: BLE001 - narration must not break the work
        logger.warning("Could not record GPU %s for %s: %s", gpu_uuid, task_id, exc)
        return False
    logger.warning(
        "No task_queue row for %s after %d attempts; its GPU %s is not recorded",
        task_id,
        attempts,
        gpu_uuid,
    )
    return False


def place_on_card(
    task_id: Optional[str],
    gpu_request: Optional[str],
    required_mb: Optional[float] = None,
    allow_shard: bool = False,
    headroom_mb: Optional[float] = None,
):
    """Choose the card(s) a J-space task runs on, record them, and make the first current.

    `gpu_request` is what the endpoint stored: "auto", "all", or a card's UUID.
    Auto takes the card with the most free memory NOW, when the task starts; a
    named card is used or refused, never swapped for another.

    `required_mb` is the model's weights (`jlens_model_registry.estimate_weights_mb`).
    Without it Auto cannot tell that a model fits no single card, so it can
    never split one. The placement is asked for the weights PLUS the activation
    headroom the loader's preflight keeps (`resource_config._ACTIVATION_HEADROOM_GB`),
    as circuits and steering place (review round 2): sized at the weights alone,
    Auto put a model on the one card whose free memory just held its weights,
    and the first forward pass ran out of memory there. `allow_shard` is passed
    only by a task whose model code runs split — see each task for why it does or
    does not.

    `headroom_mb` replaces that fixed headroom when the task knows it needs more:
    a FIT keeps its forward graph for a batched backward
    (`jlens_model_registry.estimate_fit_working_mb`, review round 3). Never less
    than the fixed headroom.

    A `GpuPlacementError` PROPAGATES. `owns_its_failure` writes its message to
    the row and Celery records the failure — no fallback to another card or to
    the CPU, and no retry loop.

    Returns the `Placement`. The model load takes the placement itself
    (`load_for_readout(record, placement=...)`), never `placement.device`, which
    is only a split's first card.
    """
    from ..services.gpu_placement import place_job
    from ..services.resource_config import _ACTIVATION_HEADROOM_GB

    # The weights and the headroom their forward passes need, as every other job
    # places. No size stays no size: Auto then behaves as it always has.
    headroom = max(_ACTIVATION_HEADROOM_GB * 1024, headroom_mb or 0.0)
    needed_mb = None if required_mb is None else required_mb + headroom
    placement = place_job(gpu_request, required_mb=needed_mb, allow_shard=allow_shard)
    logger.info(
        "J-space task %s placed on %s as %s (requested %r)",
        task_id,
        placement.describe(),
        placement.all_devices,
        gpu_request,
    )
    columns = placement.gpu_columns()
    record_gpu(task_id, columns["gpu_uuid"], columns["gpu_uuids"])
    return placement


def reuse_card(
    task_id: Optional[str],
    card: Any,
    device: Any,
    gpu_request: Optional[str],
):
    """Run a J-space task on the card(s) a model is ALREADY loaded on, and record them.

    `place_on_card`'s counterpart for a copy a previous task left loaded. No
    placement is made: Auto would judge the cards by the memory free NOW, and
    the card holding the copy has less of it, so Auto would pick the other card
    and load a second copy there.

    `card` and `device` are one card and its device, or — for a copy loaded
    SPLIT — every card of the split and their devices, in the same order. A
    split is returned as a split placement with no budgets: it names the cards
    the copy holds, and the registry refuses to LOAD from it.

    The cards are still written to the row, and the first device made current,
    exactly as a placement does. Returns a `Placement` so `release_card` works on it.
    """
    from ..services.gpu_placement import Placement, make_current

    cards = tuple(card) if isinstance(card, (tuple, list)) else (card,)
    devices = tuple(device) if isinstance(device, (tuple, list)) else (device,)
    if len(cards) != len(devices):
        raise ValueError(f"{len(cards)} card(s) for {len(devices)} device(s): {cards} / {devices}")
    if len(cards) > 1:
        placement = Placement(card=cards[0], device=devices[0], cards=cards, devices=devices)
    else:
        placement = Placement(card=cards[0], device=devices[0])

    # PHASE 3: the cards a resident copy is on are leased like a fresh
    # placement's, or the job waits for them — never run on a card another job holds.
    from ..services.gpu_job_claim import claim_exact_cards

    known = [held for held in cards if held is not None]
    if known:
        claim_exact_cards(known, gpu_request)
    make_current(devices[0])
    logger.info(
        "J-space task %s reused the model already loaded on %s as %s (requested %r)",
        task_id,
        placement.describe(),
        devices,
        gpu_request,
    )
    columns = placement.gpu_columns()
    record_gpu(task_id, columns["gpu_uuid"], columns["gpu_uuids"])
    return placement


def release_card(placement: Any) -> None:
    """Drop the resident model from the card a J-space task ran on.

    CALL IT AFTER THE TASK HAS DROPPED ITS OWN REFERENCES. `clear_cache` runs gc
    and then `empty_cache`, and anything still holding the model keeps every
    block allocated.

    ON A PROPAGATING EXCEPTION THE TRACEBACK HOLDS THE MODEL. It keeps every
    frame between the task and the raise alive, and a helper's frame holds
    `loaded`, or a `ReadoutService` built with `model=loaded.model`. Those
    frames are cleared first. `sys.exc_info()` alone only reads them, which the
    acquire task learned the hard way.

    A CPU placement — or none, when the task failed before placing — releases
    nothing. It holds no card, and the single-entry cache keeps a CPU copy
    across tasks on purpose. A split releases every card it holds: the cache
    entry names them all, and `clear_cache` empties each.
    """
    if placement is None or not placement.all_cards:
        return
    import sys
    import traceback as _traceback

    pending = sys.exc_info()[1]
    if pending is not None and pending.__traceback__ is not None:
        _traceback.clear_frames(pending.__traceback__)

    from ..services.jlens_model_registry import clear_cache

    clear_cache()
    logger.info("Released the J-space model from %s", placement.describe())


def fail_row(task_id: str, exc: BaseException) -> None:
    """Record a task's OWN failure, with its OWN reason.

    THE TASK OWNS ITS TERMINAL STATE. Leaving this to the orphan janitor costs
    three things: up to five minutes of "queued 0%" on an idle GPU while the
    sweep waits for its next beat; the real reason — "unknown primitive
    'aditive'", "swap partner is 2 tokens", the near-parallel refusal — replaced
    by the janitor's prose about the BOOKKEEPING defect, which tells the caller
    nothing about their request; and a blind spot for anything that fails AFTER
    its first progress report, which the sweep's `looks_abandoned` rule
    deliberately never closes because a terminal Celery state is not an orphan.

    The janitor remains the backstop for a worker that dies without running any
    Python at all — an eviction, an OOM kill, a pod roll.
    """
    update_row(task_id, status="failed", error_message=f"{type(exc).__name__}: {exc}")


#: Attribute stamped on a wrapped task so the guard can ASK rather than guess.
#:
#: The first version of that guard scraped the source with a regex allowing at
#: most one decorator between `@celery_app.task(...)` and `def`. A task carrying
#: any second decorator matched nothing at all, so it never entered the list the
#: assertion checked — the scan failed OPEN, and an undecorated task would have
#: shipped green. `functools.wraps` copies `__dict__`, so this marker survives
#: any number of further wrappers.
OWNERSHIP_MARKER = "__jlens_owns_its_failure__"


def owns_its_failure(fn):
    """Decorator: a J-space task records its own failure before re-raising.

    Applied at the task rather than duplicated in five bodies, so a task added
    later inherits it by construction instead of by anyone remembering. The
    exception still propagates — Celery must see the FAILURE, and swallowing it
    here would trade one silent state for another.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        from ..core.cancellation import OperatorCancelled
        from ..services.gpu_job_claim import DuplicateExecution, JobHandoff

        try:
            return fn(self, *args, **kwargs)
        except OperatorCancelled as cancelled:
            from ..core.cancellation import GPU_LEASE_LOST

            if cancelled.reason == GPU_LEASE_LOST:
                # NOT THE OPERATOR'S STOP (multi-GPU Phase 3). The job lost its
                # GPU lease; nothing made its row terminal, so it is recorded as
                # the job's failure, with why — and returned, which acks it.
                request_id = getattr(getattr(self, "request", None), "id", None)
                if request_id:
                    update_row(
                        request_id, status="failed",
                        error_message=f"Stopped: {cancelled.detail or 'its GPU lease was lost'}. Run it again.",
                    )
                return {
                    "status": "failed",
                    "scope": cancelled.scope,
                    "target_id": cancelled.target_id,
                    "reason": GPU_LEASE_LOST,
                    "detail": cancelled.detail,
                }
            # A CANCELLATION IS NOT A FAILURE, AND MUST NOT BE RECORDED AS ONE.
            #
            # This handler catches BaseException, so without this branch a
            # cancellation reached `fail_row` and relabelled a row the operator
            # had just CANCELLED as FAILED — last write wins, and the last write
            # was the crash report.
            #
            # A task's OWN `except TaskCancelled` sits inside the decorated
            # function, so it runs before this decorator ever sees anything;
            # reaching here means the task has no local handler. Returning the
            # canonical cancelled result is what ACKS the acks_late message —
            # re-raising a BaseException would let it escape celery unacked,
            # which is the 12-hour strand this design exists to avoid.
            logger.info(
                "%s cancelled with no task-local handler; returning the "
                "canonical result", cancelled,
            )
            return {
                "status": "cancelled",
                "scope": cancelled.scope,
                "target_id": cancelled.target_id,
                "detail": cancelled.detail,
            }
        except (JobHandoff, DuplicateExecution):
            # NOT A FAILURE EITHER (multi-GPU Phase 3): the GPU claim is handing
            # the job to another worker, which runs it under the same task id
            # and row. Recording it here would mark a queued job failed. A
            # DuplicateExecution is a redelivered copy of this task while the
            # original still runs (review round 1, R1-7): the row is the
            # ORIGINAL's, and recording the copy's drop would fail a live job.
            raise
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
            request_id = getattr(getattr(self, "request", None), "id", None)
            if request_id:
                fail_row(request_id, exc)
            raise

    setattr(wrapper, OWNERSHIP_MARKER, True)
    return wrapper
