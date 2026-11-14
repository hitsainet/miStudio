"""The MCP pod's old-name evidence: kept in memory, pushed to the backend (042 FR-042.9; FTID §3.5; CF-5 M6).

The MCP pod has no database. `legacy_names.set_sink(state.sink, "mcp")` (wired in `build_server`) makes every
report land in `PushState`; the `/health` route schedules `push()` when one is due (every `PUSH_INTERVAL_S`)
WHETHER OR NOT there is anything to report — an empty push is the heartbeat that moves
`legacy_names_meta.reporters.mcp` on the backend, so a pod with nothing to say and a pod whose push fails
look different. A failed push keeps every use pending and the next due push re-sends them; a success
clears what it sent. The push goes through the backend client the tools already use
(`MiWorkbenchClient`, bounded by `PUSH_TIMEOUT_S`). The route never awaits it.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from src.core import legacy_names

logger = logging.getLogger(__name__)

PUSH_INTERVAL_S = 600.0
PUSH_TIMEOUT_S = 10.0
PATH = "/system/legacy-names"


def _clock() -> float:
    """Monotonic seconds; a module function so tests move time without touching the event loop's clock."""
    return time.monotonic()


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PushState:
    def __init__(self, mono: Optional[Callable[[], float]] = None) -> None:
        self._lock = threading.Lock()
        self._mono = mono or (lambda: _clock())
        self.pending: dict[tuple[str, str], dict[str, Any]] = {}
        self.last_attempt: Optional[float] = None
        self.last_ok: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.task: Optional[asyncio.Task] = None  # the strong reference to the running push

    def sink(self, report: dict[str, Any]) -> None:
        """`legacy_names`' sink in this pod: merge the report's uses into `pending` (never blocks)."""
        with self._lock:
            for use in report.get("uses", ()):
                key = (use["kind"], use["old"])
                held = self.pending.get(key)
                if held is None or use["last_seen"] >= held["last_seen"]:
                    self.pending[key] = dict(use)

    def push_due(self) -> bool:
        with self._lock:
            return self.last_attempt is None or self._mono() - self.last_attempt >= PUSH_INTERVAL_S

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"last_ok": _iso(self.last_ok) if self.last_ok else None,
                    "last_error": self.last_error, "pending": len(self.pending)}

    def schedule(self, client: Any) -> Optional[asyncio.Task]:
        """Start one background push if due and none is running; keeps a strong reference."""
        if not self.push_due() or (self.task is not None and not self.task.done()):
            return None
        with self._lock:
            self.last_attempt = self._mono()
        self.task = asyncio.create_task(self.push(client), name="legacy-names-push")
        return self.task

    async def push(self, client: Any) -> bool:
        """Send this process's uses plus everything pending; True on success."""
        own = legacy_names.report()
        self.sink(own)
        with self._lock:
            sent = {k: dict(v) for k, v in self.pending.items()}
        body = {"reporter": legacy_names.HTTP_REPORTER, "uses": [sent[k] for k in sorted(sent)],
                "also_set": own["also_set"], "conflicts": own["conflicts"]}
        try:
            await asyncio.wait_for(client.post(PATH, body), PUSH_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 — kept pending, retried at the next due time
            with self._lock:
                self.last_error = type(exc).__name__
            logger.warning("legacy_names push failed (%s); %d use(s) kept pending", type(exc).__name__, len(sent))
            return False
        with self._lock:
            for key, use in sent.items():
                held = self.pending.get(key)
                if held is not None and held["last_seen"] <= use["last_seen"]:
                    del self.pending[key]
            self.last_ok = datetime.now(timezone.utc)
            self.last_error = None
        return True
