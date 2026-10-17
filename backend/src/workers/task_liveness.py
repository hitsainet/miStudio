"""Process liveness owned by the task WRAPPER, not by stage code (False_Reap fix #1).

THE DEFECT THIS CLOSES (`0xcc/tasks/DEBT_TASKS|False_Reap_Of_Live_Jobs_2026-10-01.md`).
`pmr_54563518010f` was marked failed while its worker trained probe 24 of 24 at 217% CPU
holding 22.6 GB of GPU. Its `training` stage never called a heartbeat, so the row's
`updated_at` froze — and the janitor's backstop fell back to that same clock. One stage
forgetting one heartbeat defeated both layers together. Fix #2 (`task_is_actively_running`,
an `inspect()` broadcast) added a second signal, but a solo worker busy inside a GPU task
often does not answer a broadcast at all, so on exactly the jobs that matter it returns
None and the janitor falls back to the row clock again.

THE FIX: liveness is stamped by the job's OUTERMOST wrapper — Celery's `task_prerun` /
`task_postrun` signals, which fire around every task this worker runs — on a background
thread, for the task's whole lifetime, independent of what any stage does. So a stage
that forgets to heartbeat can no longer cause a false reap. Per-stage heartbeats stay, for
PROGRESS (which the UI needs), not for liveness.

Why the signal and not ``@gpu_job`` or ``@cooperative_cancel``: ``@gpu_job`` is a
pass-through in single mode (docker-compose) and wraps GPU tasks only, while the job that
first exposed the janitor problem was a CPU tokenization; ``@cooperative_cancel`` decorates
two tasks. The signal is the one wrapper every task already runs under, so a NEW task is
covered with no registration to forget — the failure mode this estate keeps repeating.

THE STAMP IS KEYED BY CELERY TASK ID with a TTL, in the broker's Redis — the same shape as
the GPU waiting mark (`services/gpu_worker_queues.mark_waiting`) that `task_looks_alive`
already consults. A killed worker stops stamping and the key expires within
``LIVENESS_TTL_S``, so a genuinely dead job is still reaped, by the old rules, a few
minutes later. It says "the process running this task is alive", not "the task is making
progress": a deadlocked task keeps stamping. That is the deliberate trade — a hung job is
visible to an operator; a live job reaped as dead destroys hours of GPU work.

Never raises: liveness is narration for the janitors, and must not break the work.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Redis key prefix; one key per running task id.
LIVENESS_PREFIX = "mistudio:task:alive:"

#: How often the background thread re-stamps.
LIVENESS_INTERVAL_S = 60.0

#: How long a stamp outlives its last write. Three intervals: one missed write (a GC
#: pause, a broker reconnect) must not read as death, and a dead worker's stamp must be
#: gone well before `task_heartbeat.STALE_AFTER_SECONDS` (600 s) matters.
LIVENESS_TTL_S = 180


def _client() -> Any:
    import redis

    from ..core.config import settings

    return redis.Redis.from_url(
        str(settings.celery_broker_url), socket_connect_timeout=1.0, socket_timeout=2.0
    )


def stamp(task_id: Optional[str], *, client: Any = None, ttl_s: int = LIVENESS_TTL_S) -> bool:
    """Write (or refresh) the liveness stamp. Returns whether it was written."""
    if not task_id:
        return False
    try:
        (client if client is not None else _client()).set(
            LIVENESS_PREFIX + str(task_id), "1", ex=int(ttl_s)
        )
        return True
    except Exception:  # noqa: BLE001 - narration must not break the work
        logger.warning("Could not stamp liveness for task %s", task_id, exc_info=True)
        return False


def clear(task_id: Optional[str], *, client: Any = None) -> None:
    """The task has finished: drop its stamp. Never raises (the stamp expires anyway)."""
    if not task_id:
        return
    try:
        (client if client is not None else _client()).delete(LIVENESS_PREFIX + str(task_id))
    except Exception:  # noqa: BLE001
        logger.warning("Could not clear the liveness stamp of task %s", task_id, exc_info=True)


def has_live_stamp(task_id: Optional[str], *, client: Any = None) -> Optional[bool]:
    """Is the process running this task id alive right now?

    True: a fresh stamp exists. False: the broker answered and there is none.
    None: the broker could not be read — no evidence, which a caller must NOT read
    as death (every janitor here spares a job on a broker hiccup).
    """
    if not task_id:
        return None
    try:
        return bool((client if client is not None else _client()).exists(
            LIVENESS_PREFIX + str(task_id)
        ))
    except Exception:  # noqa: BLE001 - an unreadable broker is not a verdict
        return None


class LivenessStamper:
    """Stamp a task id now and every ``interval_s`` on a daemon thread until stopped."""

    def __init__(self, task_id: str, *, interval_s: float = LIVENESS_INTERVAL_S,
                 client: Any = None) -> None:
        self.task_id = task_id
        self.interval_s = interval_s
        self._client = client
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "LivenessStamper":
        # The first stamp is written SYNCHRONOUSLY: a task must be visibly alive from its
        # first instruction, not from the thread's first scheduling slice.
        stamp(self.task_id, client=self._client)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"liveness-{self.task_id}"
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            stamp(self.task_id, client=self._client)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        clear(self.task_id, client=self._client)


#: Running stampers, by task id. A solo worker runs one task at a time; the dict keeps
#: the bookkeeping correct under any pool.
_ACTIVE: Dict[str, LivenessStamper] = {}
_LOCK = threading.Lock()


def _is_eager(task: Any) -> bool:
    request = getattr(task, "request", None)
    return bool(getattr(request, "is_eager", False))


def on_task_prerun(sender: Any = None, task_id: Optional[str] = None, task: Any = None,
                   **_: Any) -> None:
    """`task_prerun` handler: start stamping for the task about to run.

    An EAGER call (tests, or a task applied in-process) has no worker and no janitor
    watching it, so it is skipped rather than writing to a broker it may not have.
    """
    task = task if task is not None else sender
    if not task_id or _is_eager(task):
        return
    try:
        stamper = LivenessStamper(str(task_id)).start()
    except Exception:  # noqa: BLE001
        logger.warning("Could not start the liveness stamper for %s", task_id, exc_info=True)
        return
    with _LOCK:
        previous = _ACTIVE.pop(str(task_id), None)
        _ACTIVE[str(task_id)] = stamper
    if previous is not None:
        previous._stop.set()


def on_task_postrun(sender: Any = None, task_id: Optional[str] = None, **_: Any) -> None:
    """`task_postrun` handler: stop stamping and drop the stamp — the task has returned."""
    if not task_id:
        return
    with _LOCK:
        stamper = _ACTIVE.pop(str(task_id), None)
    if stamper is not None:
        try:
            stamper.stop()
        except Exception:  # noqa: BLE001
            logger.warning("Could not stop the liveness stamper for %s", task_id, exc_info=True)


def connect_signals() -> None:
    """Wire the stamper to every task this process runs. Called from `core.celery_app`.

    `weak=False`: Celery signals hold receivers weakly by default, and a module-level
    function is safe either way, but the wiring must not depend on that.
    `dispatch_uid` makes a second call a no-op rather than a second stamper.
    """
    from celery.signals import task_postrun, task_prerun

    task_prerun.connect(on_task_prerun, weak=False, dispatch_uid="mistudio-task-liveness-prerun")
    task_postrun.connect(on_task_postrun, weak=False, dispatch_uid="mistudio-task-liveness-postrun")
