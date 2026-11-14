"""A GPU worker gives its host memory back after a job by recycling itself (ENH-006 G14).

The GPU workers run ``--pool=solo``, and Celery silently ignores
``--max-tasks-per-child`` on the solo pool, so a worker outlives every job it
runs. Python and the allocators keep what a job touched: after one 13-minute
probe evaluation an idle worker held 19.5 GiB, and the next probe run's host
memory forecast was refused against it. Freeing that took an operator
restarting the worker by hand inside the pod.

Now, after any task, a worker started by the GPU supervisor checks its own
resident memory. Above :data:`DEFAULT_RECYCLE_RSS_GIB` (``MIWORKBENCH_GPU_RECYCLE_RSS_GIB``)
it writes a recycle marker named by its PID and asks the broker to shut it
down. The request travels as a control message, and a solo worker reads control
messages only while idle, so the shutdown can never land inside the next task;
tasks ack late, so the finished one is acked first. The supervisor reads the
marker when the worker exits and restarts it at once, without counting the exit
toward the crash-loop limit (:func:`consume_marker`). A worker the supervisor did
not start has no recycle directory and never recycles.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..core import legacy_names

logger = logging.getLogger(__name__)

#: Directory for recycle markers. Set by the GPU supervisor on the workers it starts (042 FR-042.8: the
#: new name; the old MISTUDIO_GPU_RECYCLE_DIR is retired since ENH-009 P7).
RECYCLE_DIR_ENV = "MIWORKBENCH_GPU_RECYCLE_DIR"
#: Resident memory, in GiB, above which a worker recycles after a task (MIWORKBENCH_GPU_RECYCLE_RSS_GIB; the old MISTUDIO_ name is retired since ENH-009 P7).
RECYCLE_RSS_ENV = "MIWORKBENCH_GPU_RECYCLE_RSS_GIB"
#: A fresh worker with torch imported sits near 1-2 GiB; a job that kept 4 GiB left something behind.
DEFAULT_RECYCLE_RSS_GIB = 4.0
#: A marker older than this is not trusted to explain an exit (a lost shutdown request).
MARKER_MAX_AGE_S = 600.0

GIB = 1024**3


def marker_path(directory: str | os.PathLike, pid: int) -> Path:
    return Path(directory) / f"{pid}.recycle"


def resident_bytes(status_path: str = "/proc/self/status") -> Optional[int]:
    """This process's anonymous resident memory (``RssAnon``); None where /proc does not have it.

    Anonymous, not the whole RSS: memory-mapped token files are page cache the kernel drops on its
    own, and a worker must not recycle over memory it is not keeping.
    """
    try:
        with open(status_path, encoding="ascii") as handle:
            for line in handle:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def threshold_bytes(environ: Optional[dict] = None) -> int:
    """The recycle threshold. A missing, empty or invalid value falls back to the default."""
    raw = legacy_names.env_from(environ, RECYCLE_RSS_ENV, "") or ""
    try:
        gib = float(raw) if raw.strip() else DEFAULT_RECYCLE_RSS_GIB
    except ValueError:
        gib = DEFAULT_RECYCLE_RSS_GIB
    if not gib > 0:
        gib = DEFAULT_RECYCLE_RSS_GIB
    return int(gib * GIB)


def should_recycle(rss: Optional[int], threshold: int) -> bool:
    """Recycle at or above the threshold. An unreadable RSS never recycles."""
    return rss is not None and rss >= threshold


def on_task_postrun(
    sender: Any = None,
    *,
    environ: Optional[dict] = None,
    read_rss: Callable[[], Optional[int]] = resident_bytes,
    request_shutdown: Optional[Callable[[str], None]] = None,
    pid: Optional[int] = None,
    clock: Callable[[], float] = time.time,
    **_: Any,
) -> bool:
    """`task_postrun` handler. Returns True when it asked for a recycle (for tests and logs)."""
    env = environ if environ is not None else os.environ
    directory = legacy_names.env_from(env, RECYCLE_DIR_ENV, "") or ""
    if not directory:
        return False
    rss = read_rss()
    limit = threshold_bytes(env)
    if not should_recycle(rss, limit):
        return False
    hostname = getattr(getattr(sender, "request", None), "hostname", None)
    if not hostname:
        logger.warning(
            "GPU worker holds %.1f GiB after a task but its hostname is unknown; not recycling",
            (rss or 0) / GIB,
        )
        return False
    pid = pid if pid is not None else os.getpid()
    marker = marker_path(directory, pid)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "pid": pid,
                    "hostname": hostname,
                    "rss_bytes": rss,
                    "threshold_bytes": limit,
                    "task": getattr(sender, "name", None),
                    "at": clock(),
                }
            )
        )
    except OSError:
        logger.exception("Could not write the recycle marker %s; not recycling", marker)
        return False
    logger.info(
        "GPU worker %s holds %.1f GiB after %s (limit %.1f GiB): recycling to give it back",
        hostname,
        rss / GIB,
        getattr(sender, "name", "a task"),
        limit / GIB,
    )
    if request_shutdown is None:
        from ..core.celery_app import celery_app

        def request_shutdown(name: str) -> None:
            celery_app.control.shutdown(destination=[name])

    try:
        request_shutdown(hostname)
    except Exception:  # noqa: BLE001 - a failed request leaves the worker running, as before
        logger.exception("Recycle request for %s failed; the worker keeps running", hostname)
        try:
            marker.unlink()
        except OSError:
            pass
        return False
    return True


def consume_marker(
    directory: Optional[str], pid: Optional[int], *, now: Optional[float] = None
) -> bool:
    """Supervisor side: True when ``pid`` exited because it asked to recycle. Removes the marker.

    A marker older than :data:`MARKER_MAX_AGE_S` is removed and does not count: a worker
    whose shutdown request was lost and that crashed much later is a crash.
    """
    if not directory or pid is None:
        return False
    marker = marker_path(directory, pid)
    try:
        payload = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    try:
        marker.unlink()
    except OSError:
        pass
    at = payload.get("at") if isinstance(payload, dict) else None
    if not isinstance(at, (int, float)):
        return False
    current = time.time() if now is None else now
    return 0 <= current - at <= MARKER_MAX_AGE_S


def connect_signals() -> None:
    """Wire the recycle check to every task this process runs. Called from `core.celery_app`."""
    from celery.signals import task_postrun

    task_postrun.connect(_postrun_receiver, weak=False, dispatch_uid="mistudio-gpu-worker-recycle")


def _postrun_receiver(sender: Any = None, **kwargs: Any) -> None:
    try:
        on_task_postrun(sender)
    except Exception:  # noqa: BLE001 - never fail a task over the recycle check
        logger.exception("GPU worker recycle check failed")
