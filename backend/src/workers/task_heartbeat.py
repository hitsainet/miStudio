"""
Liveness stamping for long-running tasks.

THE PROBLEM THIS SOLVES, observed rather than imagined. Celery's result backend
holds whatever a task last reported. If the worker dies — a pod roll, an OOM
kill, a node drain — nothing writes a terminal state, so `AsyncResult.state`
returns PROGRESS forever. A band report that was killed by a deploy kept
reading as "profiling" for forty minutes, and three separate status checks
reported it as still working.

A STALE HEARTBEAT IS THE ONLY HONEST SIGNAL AVAILABLE. Celery offers no
"my worker vanished" event a poller can see: `acks_late` re-queues the task
(which would silently re-run a forty-minute GPU job), and `inspect().active()`
is a broadcast RPC too heavy to put behind a status endpoint the UI polls every
few seconds. So the task stamps a timestamp with every progress report, and a
reader compares it to the clock.

THE THRESHOLD MUST EXCEED THE SLOWEST GAP BETWEEN BEATS, or a genuinely slow
task is declared dead. That is why the tasks beat inside their loops rather
than only at stage boundaries: a stage boundary beat on a 45-minute stage would
force a threshold so generous the check stops being useful.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

#: How long a task may go without reporting before a reader treats it as dead.
#:
#: Generous on purpose. A false "dead" on a working task is worse than a slow
#: truth: it would send someone to re-run a job that was going to finish.
STALE_AFTER_SECONDS = 600


def beat(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Progress meta carrying a liveness timestamp.

    Every `update_state` on a long task should go through this. A progress
    report without a timestamp is indistinguishable from one made an hour ago.
    """
    meta: Dict[str, Any] = dict(extra or {})
    meta["heartbeat"] = time.time()
    return meta


def seconds_since_beat(info: Any, now: Optional[float] = None) -> Optional[float]:
    """Age of a task's last heartbeat, or None when it never sent one.

    None is NOT "stale". Tasks predating this, and tasks that have not reached
    their first progress report, legitimately have no heartbeat — reporting
    those as dead would condemn every short task that never beats at all.
    """
    if not isinstance(info, dict):
        return None
    stamp = info.get("heartbeat")
    if not isinstance(stamp, (int, float)):
        return None
    return max(0.0, (now if now is not None else time.time()) - float(stamp))


def looks_orphaned(state: str, info: Any, now: Optional[float] = None) -> bool:
    """Whether a task reporting progress has actually stopped reporting.

    Only ever true for a task claiming to be in progress. A SUCCESS or FAILURE
    is terminal and its age says nothing — a report finished last week is not
    orphaned, it is done.

    THIS SEES ONLY HALF THE PROBLEM. It needs the worker's last PROGRESS report
    to still be in the result backend. When that entry has expired — or was
    never persisted, because the pod died between reports — Celery answers
    PENDING with no info, and this returns False forever. Use `looks_abandoned`
    anywhere a task_queue row is being judged; it covers both.
    """
    if state not in ("PROGRESS", "STARTED"):
        return False
    age = seconds_since_beat(info, now=now)
    return age is not None and age > STALE_AFTER_SECONDS


def looks_abandoned(
    state: str,
    info: Any,
    seconds_since_row_update: Optional[float] = None,
    now: Optional[float] = None,
) -> bool:
    """Whether a row claiming to be RUNNING belongs to a task that is gone.

    A task disappears in two ways, and the heartbeat rule alone sees only one:

      1. The worker died and the result backend still holds its last PROGRESS.
         The heartbeat is stale — `looks_orphaned` catches it.

      2. The worker died AND the result entry expired or was never written.
         Celery reports PENDING with `info=None`, which is indistinguishable
         from a task id that was never dispatched. `looks_orphaned` returns
         False, so the row sits at "running 21.5%" forever with the GPU idle.
         OBSERVED ON HARDWARE: a fit killed by a pod roll at 16:19 was still
         claiming to run hours later, through a janitor written specifically to
         clear it.

    Case 2 has no heartbeat to consult, so it falls back to the ROW's own
    clock. That is only safe because a row reaches "running" when the task
    ITSELF reports progress: work still QUEUED behind a long job is status
    "queued" and never reaches this check. Passing a row-update age for a
    queued row would condemn everything waiting on a single-GPU queue.

    A terminal state (SUCCESS/FAILURE/REVOKED) can never reach a True return
    here: neither branch below admits it. That is asserted by test rather than
    restated as a guard — an `if state in TERMINAL_STATES: return False` line
    looked protective, but no mutation of it could change any answer, which
    makes it decoration that a future reader would trust.
    """
    if looks_orphaned(state, info, now=now):
        return True
    if state == "PENDING" and seconds_since_row_update is not None:
        return seconds_since_row_update > STALE_AFTER_SECONDS
    return False


def seconds_since_row_update(row: Any, now: Optional[float] = None) -> Optional[float]:
    """Age of a task_queue row's last write, in seconds.

    Handles naive and tz-aware timestamps, because the column has carried both:
    a naive value compared against an aware one raises, and a janitor that
    raises on one row stops sweeping the rest.
    """
    from datetime import datetime, timezone

    stamp = getattr(row, "updated_at", None) or getattr(row, "created_at", None)
    if stamp is None:
        return None
    try:
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        current = (
            datetime.fromtimestamp(now, tz=timezone.utc)
            if now is not None
            else datetime.now(timezone.utc)
        )
        return max(0.0, (current - stamp).total_seconds())
    except Exception:  # noqa: BLE001 - an unreadable timestamp is not a verdict
        return None


def task_looks_alive(celery_task_id: Optional[str], row: Any, *, started: bool) -> bool:
    """Whether the Celery task behind `row` is plausibly still running.

    MIS-E2E-092. Four janitors asked this question as
    `state in ("PENDING", "STARTED", "RETRY")`, which can NEVER be false for a
    row that has a `celery_task_id`: Celery reports PENDING for any task id it
    holds no result for, and that covers both a queued task and one whose worker
    died. `task_track_started` is unset and the long tasks never call
    `update_state`, so a live task and a dead one are indistinguishable by state
    alone.

    The consequence per subsystem: a drained training is never reclaimed;
    extractions reclaim only the no-task rows — the case the janitor was NOT
    written for; a dead activation extraction never gets `error_type=TIMEOUT`
    and never emits `extraction:failed`, so the UI spinner never resolves; and
    enhanced labeling's own error text ("the worker was restarted or the task
    was lost") names precisely the state Celery reports as PENDING and the
    janitor read as healthy.

    `cleanup_stuck_circuit_runs` was written for exactly this trap, solved it
    with `looks_abandoned`, and documented the solution as general — then it was
    applied to one of five. This is that fix, extracted so there is one
    implementation rather than five copies.

    Args:
        celery_task_id: the row's task id, or None if none was ever recorded.
        row: the ORM row, read for its `updated_at` clock.
        started: True when the row's status means the task HAS run. Only then is
            the row's own age admissible — a QUEUED row waiting behind a long
            job on a single-GPU queue would otherwise be condemned for waiting.

    Returns:
        True when the task should be left alone.
    """
    from src.core.celery_app import celery_app

    if not celery_task_id:
        # No task was ever recorded. The caller has already applied its
        # staleness filter, so nothing can be waiting on this row.
        return False

    try:
        result = celery_app.AsyncResult(celery_task_id, app=celery_app)
        state, info = result.state, result.info
    except Exception:  # broker hiccup — treat as alive, never false-kill
        return True

    age = seconds_since_row_update(row) if started else None
    return not looks_abandoned(state, info, age)
