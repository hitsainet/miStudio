"""Which queue a GPU job is sent to (multi-GPU Phase 3) — the one place that decides.

Phase 3 runs one solo worker per card (``workers/gpu_supervisor.py``). Each
consumes its card's own queue and one shared queue:

* a job that NAMES a card goes to that card's queue, ``gpu.<uuid>``;
* an Auto or ``"all"`` job goes to the shared ``gpu.auto``, so its card is chosen
  when a worker is free to run it (decision 5, 2026-09-14), against live free
  memory — see ``services/gpu_claim.decide_claim``.

Every GPU task is dispatched through :func:`dispatch_gpu_task` (or its
``.delay``-shaped form :func:`gpu_delay`); ``tests/unit/test_gpu_job_wiring.py``
walks the AST of ``src/`` for any ``.delay``/``.apply_async`` on a GPU task that
bypasses them. A GPU task that did would land on its static route — a queue no
card worker consumes in per-card mode, or the general worker, whose claim
wrapper then forwards it (one wasted hop, never a job run without a lease).

WHICH TASKS: those whose ``@gpu_job`` spec says they run on the GPU queues
(``GpuJobSpec.gpu_queue``). A task that claims a GPU part-way through its work
on another queue (a model download, a local labeling judge, a steering
generation) keeps its own queue, so the helper leaves its routing alone.

TWO DEPLOYMENT MODES, chosen by ``settings.gpu_worker_mode``:

* ``"single"`` (the default: docker-compose, a development box, every test that
  does not set it) — one worker consumes every GPU queue, exactly as before
  Phase 3. The task goes out on its static route with the very call the site
  made before, so behaviour, and every existing test double, is unchanged.
* ``"per_card"`` (the k8s manifest) — the queue is chosen here.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from .gpu_claim import AUTO_QUEUE, queue_for
from .gpu_placement import GpuRequest

#: Attribute ``@gpu_job`` stamps on a task function (a ``GpuJobSpec``).
#: ``functools.wraps`` copies ``__dict__``, so it survives further decorators.
GPU_JOB_MARKER = "__gpu_job__"

#: A card as a job row stores it once resolved at submit: NVML's ``GPU-<hex…>``.
_CARD_UUID = re.compile(r"^gpu-[0-9a-f][0-9a-f-]*$", re.IGNORECASE)

_FROM_KWARGS = object()


def per_card_workers() -> bool:
    """Whether this deployment runs one GPU worker per card (``gpu_worker_mode``)."""
    from ..core.config import settings

    return getattr(settings, "gpu_worker_mode", "single") == "per_card"


def gpu_job_spec(task: Any) -> Any:
    """The ``GpuJobSpec`` of a Celery task (or task function), or None."""
    return getattr(getattr(task, "run", task), GPU_JOB_MARKER, None)


def runs_on_gpu_queues(task: Any) -> bool:
    spec = gpu_job_spec(task)
    return bool(spec is not None and getattr(spec, "gpu_queue", False))


def is_named_card(requested: GpuRequest) -> bool:
    """True for a card UUID. Auto, ``"all"``, an index or junk are not a card's queue."""
    return isinstance(requested, str) and bool(_CARD_UUID.match(requested.strip()))


def gpu_queue_for(requested: GpuRequest) -> str:
    """The queue a job with this request waits on in per-card mode.

    A named card's own queue, and ``gpu.auto`` for everything else. ANYTHING
    THAT IS NOT A CARD UUID GOES TO ``gpu.auto``, not to a queue built from it:
    an unresolved index ("1") would otherwise name ``gpu.gpu-1``, a queue no
    worker consumes, and the job would wait there silently for ever. On
    ``gpu.auto`` the worker that takes it resolves the request and, if it names
    nothing, fails the job with the placement's own message.
    """
    if is_named_card(requested):
        return queue_for(requested)
    return AUTO_QUEUE


def dispatch_gpu_task(
    task: Any,
    *,
    gpu_request: GpuRequest,
    args: Optional[Sequence[Any]] = None,
    kwargs: Optional[dict] = None,
    **options: Any,
) -> Any:
    """Queue a GPU task where the worker that should run it will take it.

    Args:
        task: The Celery task.
        gpu_request: The job's request as stored at submit — ``"auto"``, ``"all"``
            or a card UUID. None is Auto.
        args / kwargs: The task's arguments, passed on as given.
        **options: ``apply_async`` options (``task_id``, time limits, ``kwargsrepr``…).

    Returns:
        The ``AsyncResult``.
    """
    call = dict(options)
    if args is not None:
        call["args"] = args
    if kwargs is not None:
        call["kwargs"] = kwargs
    if per_card_workers() and runs_on_gpu_queues(task):
        call["queue"] = gpu_queue_for(gpu_request)
        return task.apply_async(**call)
    if not options:
        # The `.delay()` the site always made.
        return task.delay(*(args or ()), **(kwargs or {}))
    return task.apply_async(**call)


def gpu_delay(task: Any, gpu_request: Any = _FROM_KWARGS):
    """``task.delay`` routed by :func:`dispatch_gpu_task`: ``gpu_delay(task)(*args, **kwargs)``.

    The request is the call's own ``gpu_request`` keyword unless given here — pass
    it for a task that reads its request from its row.
    """

    def delay(*args: Any, **kwargs: Any) -> Any:
        requested = kwargs.get("gpu_request") if gpu_request is _FROM_KWARGS else gpu_request
        return dispatch_gpu_task(task, gpu_request=requested, args=args or None, kwargs=kwargs or None)

    return delay
