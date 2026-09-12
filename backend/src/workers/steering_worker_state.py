"""
Shared state for the dedicated steering worker's lifecycle.

Three parties coordinate through this module and a marker file:

- task_prerun/task_postrun (steering_tasks.py) mark the worker busy/idle,
  both in-process (``busy_task_id``) and on disk (the busy-marker file the
  backend API reads — separate processes, shared /data/run volume).
- The SIGTERM handler (steering_service.py) consults ``busy_task_id``:
  terminating mid-task is what crashed the solo pool with celery's
  "cannot unpack non-iterable ExceptionInfo" — so when busy it only sets
  ``shutdown_deferred`` and returns; task_postrun completes the shutdown.
- The API's _ensure_steering_worker_running reads the marker file to avoid
  SIGKILLing a worker that is mid-generation (which stranded the in-flight
  acks_late message for the 12h visibility timeout).

A marker older than BUSY_MARKER_STALE_SECONDS is ignored: the hard task
time_limit is 180s, so anything older is a hung/dead worker and killing it
is correct.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Past the 180s celery hard time_limit plus margin — a marker this old
# belongs to a hung or dead worker, not a live generation.
BUSY_MARKER_STALE_SECONDS = 240

# In-process state (the steering worker itself).
busy_task_id: Optional[str] = None
shutdown_deferred: bool = False


def _marker_path() -> Path:
    from ..core.config import settings

    return settings.run_dir / "steering-worker-busy.json"


def write_busy_marker(task_id: str) -> None:
    global busy_task_id
    busy_task_id = task_id
    try:
        _marker_path().write_text(
            json.dumps({"pid": os.getpid(), "task_id": task_id, "ts": time.time()})
        )
    except Exception:
        logger.exception("Could not write steering busy marker")


def clear_busy_marker() -> None:
    global busy_task_id
    busy_task_id = None
    try:
        _marker_path().unlink(missing_ok=True)
    except Exception:
        logger.exception("Could not clear steering busy marker")


def read_busy_marker() -> Optional[dict]:
    """Return the marker {pid, task_id, ts} if present AND fresh, else None."""
    try:
        raw = _marker_path().read_text()
        marker = json.loads(raw)
        if time.time() - float(marker.get("ts", 0)) > BUSY_MARKER_STALE_SECONDS:
            return None
        return marker
    except FileNotFoundError:
        return None
    except Exception:
        # Unreadable marker must not make a live generation killable — fail
        # toward "busy" — but a corrupt file must not deadlock ensure/spawn
        # forever either, so staleness falls back to the file's mtime.
        logger.exception("Unreadable steering busy marker; treating as busy")
        try:
            if time.time() - _marker_path().stat().st_mtime > BUSY_MARKER_STALE_SECONDS:
                return None
        except OSError:
            return None
        return {"pid": -1, "task_id": "unknown", "ts": time.time()}
