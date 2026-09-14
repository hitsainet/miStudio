"""The claim wrapper every GPU task runs under (multi-GPU Phase 3).

``@gpu_job(kind)`` sits between ``@celery_app.task(...)`` and the task's other
decorators. In per-card mode (``settings.gpu_worker_mode``) it:

1. mints this EXECUTION's lease holder (``gpu_job_claim.make_holder``);
2. for a whole-GPU task (``handoff=True``), prechecks before the body does
   anything — a job naming another card, a worker whose card another job holds,
   or a general worker that took a GPU job hands it on at once;
3. runs the body under ``gpu_job_claim.claiming``, so ``place_job`` leases the
   cards it places on, renews them on a timer, and — on EVERY exit: success,
   failure, OOM, cooperative cancel, a deleted row — releases them;
4. turns a ``JobHandoff`` into a re-dispatch of the same task, same arguments,
   SAME TASK ID (rows, cancellation and J-lens task rows key on it), and acks
   this delivery with ``Ignore`` so no result is stored for it — a poller must
   never read the hand-off as the job's result.

A task whose GPU step comes after work that must not be repeated passes
``handoff=False`` and waits for its cards in place instead (model download,
local labeling, logit lens, steering).

ORDER MATTERS: ``@gpu_job`` must be OUTSIDE ``owns_its_failure`` and
``cooperative_cancel``. ``owns_its_failure`` records any BaseException as the
task's failure, and a hand-off is not one.

In single mode the wrapper is a pass-through.
"""

from __future__ import annotations

import functools
import gc
import inspect
import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..services.gpu_dispatch import GPU_JOB_MARKER, per_card_workers
from ..services.gpu_job_claim import (
    DEFAULT_IN_PLACE_TIMEOUT_S,
    HANDOFF_WAIT_S,
    STALE_HEARTBEAT_S,
    ClaimContext,
    DuplicateExecution,
    JobHandoff,
    claiming,
    make_holder,
)
from ..services.gpu_placement import AUTO, GpuPlacementError
from ..services.gpu_worker_queues import mark_waiting, park_job
from .gpu_supervisor import WORKER_GPU_ENV

logger = logging.getLogger(__name__)

#: Immediate hand-offs of one job in a row before it is parked instead. Two
#: workers reading free memory a moment apart can disagree about which card is
#: best; without a damper they could pass a job back and forth.
MAX_IMMEDIATE_HOPS = 4

#: How long a steering generation waits in place for a card. Its hard time limit
#: is 180–360 s, and the generation itself needs most of that.
STEERING_LEASE_WAIT_S = 60.0


@dataclass(frozen=True)
class GpuJobSpec:
    kind: str
    handoff: bool
    #: Dispatched to gpu.<uuid> / gpu.auto (and prechecked there), rather than
    #: claiming a GPU part-way through work on its own queue.
    gpu_queue: bool
    #: Whether the job can run split: True, False, or a callable over the bound
    #: arguments (the answer is on the job's row). Read by the precheck.
    can_split: Any = True


def _executing_task(args: tuple) -> Any:
    from celery import Task, current_task

    if args and isinstance(args[0], Task):
        return args[0]
    try:
        return current_task._get_current_object()
    except Exception:  # noqa: BLE001 - no task outside a worker
        return None


def _requested(signature: inspect.Signature, request_from, args: tuple, kwargs: dict) -> Any:
    try:
        arguments = signature.bind_partial(*args, **kwargs).arguments
    except TypeError:
        return AUTO
    if request_from is None:
        return arguments.get("gpu_request") or AUTO
    try:
        return request_from(arguments) or AUTO
    except Exception:  # noqa: BLE001 - a precheck guess; the placement reads the truth
        logger.warning("Could not read the GPU request for a precheck; assuming auto", exc_info=True)
        return AUTO


def _splits(signature: inspect.Signature, can_split: Any, args: tuple, kwargs: dict) -> bool:
    """The job's answer to "can you run split?", for the precheck. A callable reads the bound
    arguments; one that fails counts as True, which keeps the job waiting as it always did."""
    if not callable(can_split):
        return bool(can_split)
    try:
        return bool(can_split(signature.bind_partial(*args, **kwargs).arguments))
    except Exception:  # noqa: BLE001 - a precheck guess; the placement decides with the truth
        logger.warning("Could not tell whether this job can split; assuming it can", exc_info=True)
        return True


def _hops(request: Any) -> int:
    value = getattr(request, "gpu_hops", None)
    if value is None:
        value = (getattr(request, "headers", None) or {}).get("gpu_hops")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def clear_exception_frames(exc: BaseException) -> None:
    """Clear the locals of every frame a failure unwound: its own traceback and its chain's.

    ``traceback.clear_frames`` on the raised exception alone leaves the frames of the
    exception it wraps (``raise ... from exc``) or was raised while handling
    (``__context__``) — and the models and buffers those frames hold — alive until
    the exception is gone, which is after the claim releases the cards. Exception
    groups are walked too. Frames still executing are skipped by
    ``traceback.clear_frames`` itself. Line numbers and messages survive.
    """
    pending = [exc]
    seen: set = set()
    while pending:
        current = pending.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
        pending.extend((current.__cause__, current.__context__))
        pending.extend(getattr(current, "exceptions", None) or ())


def release_job_memory(task: Any, held: tuple) -> None:
    """Free a finished job's memory on the cards it leased, before the leases go.

    Run by the claim (``ClaimContext.release_memory``) on every exit that held a
    card. A task that keeps memory on ITSELF rather than in its body's frames
    (training's rolling activation buffer) frees it in ``release_job_memory()``;
    then garbage is collected, each leased card's cached blocks are returned to
    the driver, and PyTorch's pinned host cache is emptied — the host RAM guard
    of the next job reads MemAvailable. The task's ``after_return`` still runs
    later and finds nothing left to free.
    """
    # ONE ORDERED RELEASE (review round 1's R1-1 composed with 3dac5fb8), bounded by
    # the claim, before the leases go, on every exit:
    #   1. what the task keeps on itself — only a job that placed has any;
    #   2. the idle copies this process keeps (a J-lens readout's kept model, the
    #      logit-lens cache) — on EVERY exit, a hand-off included: they sit on a
    #      card no lease covers;
    #   3. garbage, then each leased card's cache, then pinned host memory — after
    #      1 and 2, so the blocks they dropped are returned.
    if held:
        hook = getattr(task, "release_job_memory", None)
        if callable(hook):
            hook()
    release_idle_copies_at_claim_end()
    gc.collect()
    from ..services.gpu_placement import empty_cache_on_cards

    if held:
        empty_cache_on_cards(held)
        _empty_pinned_host_cache_on(held)


def _empty_pinned_host_cache_on(held: tuple) -> None:
    """Empty PyTorch's pinned host cache from the job's own card, never from another.

    This runs on the claim's release thread (``ClaimContext._release_memory``),
    and a thread that never selected a card has device 0 current. The host release
    there CREATES a CUDA context on cuda:0: measured on the node 2026-09-14, a
    training on the 3090 left 253 MiB on the 3080 Ti, a card none of its leases
    covered. Only a job that held a card pinned anything, so one that held none
    (a hand-off) has nothing to release here.
    """
    import torch

    if not torch.cuda.is_available():
        return
    from ..services.activation_buffer import _empty_pinned_host_cache
    from ..services.gpu_placement import GpuPlacementError, torch_device

    try:
        device = torch_device(held[0])
    except GpuPlacementError:
        return
    _empty_pinned_host_cache(device=device)


def requeue(
    task: Any,
    request: Any,
    handed: JobHandoff,
    *,
    park: Optional[Callable] = None,
    now: Optional[Callable[[], float]] = None,
) -> None:
    """Publish the job again where ``handed`` says, with the same arguments and task id."""
    now = now or time.time
    hops = _hops(request)
    delay = handed.delay_s
    if delay <= 0 and hops >= MAX_IMMEDIATE_HOPS:
        logger.warning(
            "Job %s was handed between GPU workers %d times in a row; parking it for %.0f s",
            request.id, hops, HANDOFF_WAIT_S,
        )
        delay = HANDOFF_WAIT_S
    options = {
        "queue": handed.queue,
        "task_id": request.id,
        "headers": {"gpu_hops": 0 if delay > 0 else hops + 1},
    }
    limits = tuple(getattr(request, "timelimit", None) or ()) + (None, None)
    if limits[0]:
        options["time_limit"] = limits[0]
    if limits[1]:
        options["soft_time_limit"] = limits[1]
    for key in ("argsrepr", "kwargsrepr"):
        value = getattr(request, key, None)
        if value:
            options[key] = value
    args = list(getattr(request, "args", None) or ())
    kwargs = dict(getattr(request, "kwargs", None) or {})
    logger.info(
        "GPU job %s (%s) handed to %s%s: %s", request.id, task.name, handed.queue,
        f" in {delay:.0f} s" if delay > 0 else "", handed.reason,
    )
    # BEFORE it is republished: from here until it leases a card, a janitor that
    # finds its row looking started must know it is waiting, not dead.
    mark_waiting(request.id)
    if delay > 0:
        (park or park_job)(task.name, args, kwargs, options, due_at=now() + delay)
    else:
        task.apply_async(args=args, kwargs=kwargs, **options)


def waiting_for_gpu(task_id: Optional[str]) -> bool:
    """For a janitor about to reap a job: is the job only WAITING for a GPU?

    A job handed off at placement re-runs its pre-placement row writes each time
    it is republished, so its row can look started while its clock is frozen; a
    job waiting in place sends no heartbeat. Such a job is spared. A dead job has
    no waiting mark, no parked entry and no queued message, and is reaped as
    before. Inert (False) in single mode, where nothing is handed off.
    """
    if not task_id or not per_card_workers():
        return False
    from ..services.gpu_worker_queues import is_waiting_for_gpu

    return is_waiting_for_gpu(task_id)


def release_idle_copies_at_claim_end() -> None:
    """Free the GPU memory idle caches in this process hold, as a per-card job's claim ends.

    PHASE 3 REVIEW ROUND 1, ITEM 4. A J-lens readout may keep its model loaded for
    the next readout (``unload_after=False``), and the logit lens keeps the models
    it loaded. Such a copy sits on a card with NO lease once its job's claim ends,
    and is freed only by the next placement IN THIS WORKER
    (``gpu_placement.release_idle_gpu_memory``). Every other card's worker judges
    that card from its leases — idle — and from NVML's free memory, which the copy
    has taken: it refuses, or waits for, a job that would fit once the copy went,
    and nothing will ever free it from there. With a shared ``gpu.auto`` queue the
    next readout is as likely to reach another worker as this one, where it loads
    a second copy.

    Two ways out were weighed. Leasing a kept copy needs a lease with no running
    job: renewed from a thread of a solo worker, handed over to the next
    execution's holder, released after an idle timeout, and known to every janitor
    that frees leases by task id. Freeing the copy when the claim ends needs this
    one call, keeps "a card with no lease holds nothing of ours" true, and costs a
    reload only when two readouts in a row land on the same worker. The claim is
    still held here, so no other job can be placed on the card before the memory
    is back. Single mode never gets here (the wrapper is a pass-through there), so
    the kept copy — and the unload toggle — behave there as before.

    Never raises: ``release_idle_gpu_memory`` logs a failing release and goes on.

    COMPOSED WITH REVIEW ROUND 1's R1-1 (cherry-picked 2026-09-14): it runs as a step
    of :func:`release_job_memory` — the claim's ONE ordered, bounded release — on every
    exit, a hand-off included, and before the leases are released.
    """
    from ..services.gpu_placement import release_idle_gpu_memory

    release_idle_gpu_memory()


def gpu_job(
    kind: str,
    *,
    handoff: bool = True,
    precheck: Optional[bool] = None,
    request_from: Optional[Callable[[dict], Any]] = None,
    wait_timeout_s: float = DEFAULT_IN_PLACE_TIMEOUT_S,
    can_split: Any = True,
):
    """Run a Celery task under a GPU lease claim. See the module docstring.

    ``can_split`` — whether the job can run split across cards: True, False, or
    a callable taking the bound arguments (for a task whose answer is on its
    row). Read by the precheck only, which has no size: an ``"all"`` job that
    cannot split goes on to its placement, which refuses it with the placement's
    message, instead of being parked while a card is busy. A callable that raises
    counts as True — the job waits, as before.

    Args:
        kind: The job kind, for the lease holder and logs.
        handoff: True for a task that places before doing anything it must not
            repeat; False to wait for cards in place.
        precheck: Hand the job on at task start when this worker should not take
            it. Defaults to ``handoff``. Only for a task dispatched to the GPU
            queues: on the general worker a precheck would forward a task that
            belongs there.
        request_from: For a task whose request is on its ROW, not in its
            arguments: takes the bound arguments, returns the request. Used only
            by the precheck; the placement always reads the real one.
        wait_timeout_s: The bound on waiting in place (``handoff=False``).
    """
    if precheck is None:
        precheck = handoff

    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not per_card_workers():
                return fn(*args, **kwargs)
            task = _executing_task(args)
            request = getattr(task, "request", None)
            task_id = getattr(request, "id", None)
            ctx = ClaimContext(
                holder=make_holder(kind, task_id),
                task_id=task_id,
                worker_uuid=os.environ.get(WORKER_GPU_ENV) or None,
                handoff=handoff,
                wait_timeout_s=wait_timeout_s,
                release_memory=functools.partial(release_job_memory, task),
                redelivered=bool((getattr(request, "delivery_info", None) or {}).get("redelivered")),
            )
            try:
                with claiming(ctx):
                    try:
                        if precheck:
                            ctx.precheck(
                                _requested(signature, request_from, args, kwargs),
                                can_split=_splits(signature, can_split, args, kwargs),
                            )
                        return fn(*args, **kwargs)
                    except JobHandoff:
                        raise
                    except BaseException as exc:
                        # THE TRACEBACK HOLDS THE JOB'S MEMORY. Every frame the
                        # exception unwound — the body's models, optimizers,
                        # buffers — stays alive until the exception is handled,
                        # which is after `claiming` releases the cards. Cleared
                        # here so the release before the lease release frees
                        # something. Line numbers survive; locals do not. The
                        # WHOLE CHAIN (review round 2): a wrapped failure keeps
                        # the original, and every frame it unwound, on
                        # __cause__ / __context__.
                        if ctx.held:
                            clear_exception_frames(exc)
                        raise
                    # NO `finally` HERE: the idle copies (3dac5fb8) are freed by
                    # release_job_memory, the claim's one ordered, bounded release,
                    # which `claiming` runs on every exit before the leases go.
            except DuplicateExecution as duplicate:
                # A redelivered copy of a job another execution is still running
                # (review round 1, R1-7): ack it, run nothing, republish nothing.
                logger.error(
                    "GPU job %s (%s) was delivered again while %s still runs it; this copy is "
                    "dropped", task_id, kind, duplicate.holders,
                )
                from celery.exceptions import Ignore

                raise Ignore() from None
            except JobHandoff as handed:
                if task is None or not task_id:
                    raise GpuPlacementError(
                        f"{kind}: {handed.reason}, and this job has no Celery task to re-queue."
                    ) from None
                requeue(task, request, handed)
                from celery.exceptions import Ignore

                raise Ignore() from None

        setattr(wrapper, GPU_JOB_MARKER,
                GpuJobSpec(kind=kind, handoff=handoff, gpu_queue=precheck, can_split=can_split))
        return wrapper

    return decorate


# ── the requests of tasks that read them from their rows ─────────────────────


def _row_value(model_loader: Callable[[], Any], column: str, filters: dict) -> Any:
    from ..core.database import get_sync_db

    model = model_loader()
    with get_sync_db() as db:
        query = db.query(getattr(model, column))
        for name, value in filters.items():
            query = query.filter(getattr(model, name) == value)
        if hasattr(model, "created_at"):
            query = query.order_by(model.created_at.desc())
        row = query.first()
    return row[0] if row else None


def training_request(arguments: dict) -> Any:
    def model():
        from ..models.training import Training

        return Training

    return _row_value(model, "gpu_request", {"id": arguments.get("training_id")})


def training_can_split(arguments: dict) -> bool:
    """Whether a training can run split: only when it loads a base model (it extracts on the fly).

    ``extraction_ids_of`` is the one definition the endpoint and the placement
    already share. A row that is gone counts as True: the job's own placement
    deals with it, and the precheck only decides whether to wait.
    """
    from ..core.database import get_sync_db
    from ..models.training import Training
    from .training_tasks import extraction_ids_of

    with get_sync_db() as db:
        row = db.query(Training).filter(Training.id == arguments.get("training_id")).first()
        return True if row is None else extraction_ids_of(row) is None


def capture_request(arguments: dict) -> Any:
    def model():
        from ..models.circuit_runs import CircuitCaptureRun

        return CircuitCaptureRun

    return _row_value(model, "gpu_request", {"id": arguments.get("run_id")})


def record_request(arguments: dict) -> Any:
    def model():
        from ..models.steering_record_run import SteeringRecordRun

        return SteeringRecordRun

    return _row_value(model, "gpu_request", {"id": arguments.get("record_run_id")})


def sae_extraction_request(arguments: dict) -> Any:
    def model():
        from ..models.extraction_job import ExtractionJob

        return ExtractionJob

    # The same row the task reads: the SAE's newest extraction job.
    return _row_value(model, "gpu_request", {"external_sae_id": arguments.get("sae_id")})


# ── janitors ────────────────────────────────────────────────────────────────


def release_reaped_leases(task_id: Optional[str], *, session: Optional[Callable] = None) -> int:
    """For a janitor that has just reaped a job: free the cards its dead execution held.

    Only leases whose holder stopped renewing (``STALE_HEARTBEAT_S``); a live job
    a janitor reaped by mistake keeps its card. Inert in single mode, where no
    leases are taken. Never raises: the lease expires at its TTL regardless.
    """
    if not task_id or not per_card_workers():
        return 0
    from ..services import gpu_leases

    if session is None:
        from ..core.database import get_sync_db as session
    try:
        with session() as db:
            return gpu_leases.release_stale_for_task(db, task_id, stale_after_s=STALE_HEARTBEAT_S)
    except Exception:  # noqa: BLE001 - the TTL frees it anyway
        logger.exception("Could not release the GPU leases of reaped task %s", task_id)
        return 0
