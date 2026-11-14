"""Old-name evidence in Celery workers and beat (042 FR-042.9; FTID §3.4; CF-5 M7; TPD-29).

A worker reads its configuration once, so a first-use-only record would age out of the 24-hour `in_use`
window while the old name is still set. This module keeps every long-lived Celery process reporting:

* `worker_ready` / `beat_init` set the database sink (`legacy_names_store.db_sink`, bounded) under the
  reporter name `worker:<node name>` or `beat`, and start the heartbeat;
* `task_prerun` calls `legacy_names.refresh()` (throttled to once a minute per process) and makes sure the
  heartbeat runs;
* the heartbeat is ONE daemon thread per process that calls `refresh()` every `HEARTBEAT_S`, so an idle
  worker and the beat process still move their `legacy_name_reporters.last_report` (an empty report is a
  heartbeat). Celery's own `heartbeat_sent` is not used: it fires only with task events on (TPD-29).

`connect_signals()` is called from `core/celery_app.py`; removing that line fails
`test_legacy_names_refresh.py` against the live signal registry.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from src.core import legacy_names

logger = logging.getLogger(__name__)

HEARTBEAT_S = 60.0
DISPATCH_PRERUN = "miworkbench-legacy-names-prerun"
DISPATCH_WORKER_READY = "miworkbench-legacy-names-worker-ready"
DISPATCH_BEAT_INIT = "miworkbench-legacy-names-beat-init"

_lock = threading.Lock()
_stop = threading.Event()
_thread: Optional[threading.Thread] = None


def _set_db_sink(reporter: str) -> None:
    from src.services import legacy_names_store

    if legacy_names.reporter() is None:
        legacy_names.set_sink(legacy_names_store.db_sink, reporter)


def _heartbeat_loop(interval: float, refresh: Callable[[], Any]) -> None:
    while not _stop.wait(interval):
        try:
            refresh()
        except Exception as exc:  # noqa: BLE001 — the heartbeat never dies
            logger.warning("legacy_names heartbeat failed (%s)", type(exc).__name__)


def _ensure_heartbeat(interval: Optional[float] = None) -> bool:
    """Start the one heartbeat thread of this process; True when this call started it."""
    global _thread
    interval = HEARTBEAT_S if interval is None else interval
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        _stop.clear()
        _thread = threading.Thread(target=_heartbeat_loop, args=(interval, legacy_names.refresh),
                                   name="legacy-names-heartbeat", daemon=True)
        _thread.start()
        return True


def stop_heartbeat() -> None:
    """Tests only: stop the heartbeat thread and forget it."""
    global _thread
    with _lock:
        _stop.set()
        thread, _thread = _thread, None
    if thread is not None:
        thread.join(timeout=5)


def on_task_prerun(**_: Any) -> None:
    try:
        if legacy_names.reporter() is None:
            import socket

            _set_db_sink(f"worker:{socket.gethostname()}"[:63])
        legacy_names.refresh()
    except Exception as exc:  # noqa: BLE001 — a task never fails because of evidence
        logger.warning("legacy_names refresh on task_prerun failed (%s)", type(exc).__name__)
    _ensure_heartbeat()


def on_worker_ready(sender: Any = None, **_: Any) -> None:
    hostname = getattr(sender, "hostname", None) or "worker"
    _set_db_sink(f"worker:{hostname}"[:63])
    _ensure_heartbeat()


def on_beat_init(sender: Any = None, **_: Any) -> None:
    _set_db_sink("beat")
    _ensure_heartbeat()


def connect_signals() -> None:
    """Wire the three receivers. `weak=False` so the wiring never depends on a held reference;
    `dispatch_uid` makes a second call a no-op (the `core/runtime_settings.py` pattern)."""
    from celery.signals import beat_init, task_prerun, worker_ready

    task_prerun.connect(on_task_prerun, weak=False, dispatch_uid=DISPATCH_PRERUN)
    worker_ready.connect(on_worker_ready, weak=False, dispatch_uid=DISPATCH_WORKER_READY)
    beat_init.connect(on_beat_init, weak=False, dispatch_uid=DISPATCH_BEAT_INIT)
