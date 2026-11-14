"""The database half of the old-name evidence (042 FR-042.9; FTID §3.3; TPD-2, TPD-29; CF-4, CF-5, CF-18).

Two writers reach the two tables of `models/legacy_name.py`:

* **Celery workers and beat** call `write()` synchronously from their sink (`db_sink`), at most once a
  minute per process (`core/legacy_names` throttles); a worker has no event loop to block.
* **The API** must never block its event loop: `core/legacy_names.record()` runs inside the host middleware
  on the request path, so the API's sink (`ApiQueue.sink`) only MERGES the report into memory. The lifespan
  task `flush_loop()` writes it every 60 s through `asyncio.to_thread`, under `FLUSH_TIMEOUT_S`. A flush
  that fails, times out or is cancelled PUTS ITS USES BACK (merged with anything newer), so a drained use is
  never lost (miLLM P0 review finding: a cancelled flush lost its hits).

Every database call here goes through a dedicated small engine with a connect timeout and a statement/lock
timeout (`BOUND_S`), so a hung database answers with an error in seconds instead of holding a worker, the
flush or `/api/health` (miForge P0 review finding). This module is the only one that touches the tables;
the MCP pod never imports it (`test_mcp_tools_stay_on_http.py`).
"""
from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from src.core import legacy_names
from src.models.legacy_name import LegacyNameReporter, LegacyNameUse

logger = logging.getLogger(__name__)

#: Connect, statement and lock timeout for every store call, in seconds.
BOUND_S = 2.0
#: How long one API flush may take before its uses are put back.
FLUSH_TIMEOUT_S = 10.0
#: The API's flush cadence (also its refresh cadence, FTID §3.3).
FLUSH_INTERVAL_S = 60.0
#: The last flush at shutdown.
SHUTDOWN_FLUSH_TIMEOUT_S = 5.0

_engine_lock = threading.Lock()
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def _url() -> str:
    from src.core.database import sync_engine

    return sync_engine.url.render_as_string(hide_password=False)


def session_factory() -> sessionmaker:
    """An UNPOOLED engine bounded by `BOUND_S` on connect, statement and lock waits.

    No pool (review rev-mistudio-p0 F2, the miForge P0 black-hole finding): a pooled connection that goes
    silent mid-call (a failover, a dropped NAT entry) is never answered, and `statement_timeout` is enforced
    by a server that no longer hears it, so two such calls held both slots of the old `pool_size=1,
    max_overflow=1` pool and every later call failed while the database was healthy. Each call now opens a
    fresh, `connect_timeout`-bounded connection (at most a few a minute per process), and TCP keepalives
    plus `tcp_user_timeout` end a silent socket in seconds. `test_legacy_names_store_bounds.py` pins it."""
    global _engine, _session_factory
    with _engine_lock:
        if _session_factory is None:
            ms = int(BOUND_S * 1000)
            _engine = create_engine(
                _url(),
                poolclass=NullPool,
                connect_args={
                    "connect_timeout": max(1, int(BOUND_S)),
                    "options": f"-c statement_timeout={ms} -c lock_timeout={ms}",
                    "application_name": "mistudio_legacy_names",
                    "keepalives": 1,
                    "keepalives_idle": max(1, int(BOUND_S)),
                    "keepalives_interval": 1,
                    "keepalives_count": 2,
                    "tcp_user_timeout": ms,
                },
            )
            _session_factory = sessionmaker(bind=_engine, class_=Session, expire_on_commit=False)
        return _session_factory


def dispose() -> None:
    """Drop the engine (tests that switch databases; process shutdown)."""
    global _engine, _session_factory
    with _engine_lock:
        if _engine is not None:
            _engine.dispose()
        _engine, _session_factory = None, None


def _parse(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def write(report: dict[str, Any], now: Optional[datetime] = None) -> None:
    """Upsert every use of `report` and its reporter row, in one transaction. RAISES on failure (the
    caller keeps the uses). `last_seen` only moves forward (GREATEST) and never past `now`."""
    now = now or datetime.now(timezone.utc)
    reporter = report["reporter"]
    if not reporter or not legacy_names.REPORTER_PATTERN.match(reporter):
        raise ValueError(f"bad reporter name {reporter!r}")
    with session_factory()() as db:
        for use in report.get("uses", ()):
            if not legacy_names.known_use(use["kind"], use["old"], use["new"]):
                logger.warning("legacy_names_store: dropping an unknown use (%s, %s)", use["kind"], use["old"])
                continue
            seen = min(_parse(use["last_seen"]), now)
            db.execute(
                text(
                    "INSERT INTO legacy_name_uses (kind, old, new, first_seen, last_seen) "
                    "VALUES (:kind, :old, :new, :seen, :seen) "
                    "ON CONFLICT (kind, old) DO UPDATE SET "
                    "last_seen = GREATEST(legacy_name_uses.last_seen, excluded.last_seen), new = excluded.new"
                ),
                {"kind": use["kind"], "old": use["old"], "new": use["new"], "seen": seen},
            )
        db.execute(
            text(
                "INSERT INTO legacy_name_reporters (reporter, first_report, last_report) "
                "VALUES (:r, :now, :now) ON CONFLICT (reporter) DO UPDATE SET "
                "last_report = GREATEST(legacy_name_reporters.last_report, excluded.last_report)"
            ),
            {"r": reporter, "now": now},
        )
        db.commit()


def read() -> tuple[list[dict[str, Any]], dict[str, datetime], Optional[str]]:
    """`(uses, reporters, error)`; never raises: a failed read gives `([], {}, type(exc).__name__)`."""
    try:
        with session_factory()() as db:
            uses = [
                {"kind": r.kind, "old": r.old, "new": r.new, "first_seen": r.first_seen,
                 "last_seen": r.last_seen}
                for r in db.execute(select(LegacyNameUse)).scalars()
            ]
            reporters = {
                r.reporter: (r.first_report, r.last_report)
                for r in db.execute(select(LegacyNameReporter)).scalars()
            }
        return uses, reporters, None
    except Exception as exc:  # noqa: BLE001 — health must answer; the error is reported, not raised
        logger.warning("legacy_names_store: read failed (%s)", type(exc).__name__)
        return [], {}, type(exc).__name__


def db_sink(report: dict[str, Any]) -> None:
    """The sink of Celery workers and beat: a synchronous bounded write. `legacy_names` catches and logs
    a failure; the next refresh (a minute later) re-sends the whole registry, so nothing is lost."""
    write(report)


# ── the API's queue ───────────────────────────────────────────────────────────────────────────────────
class ApiQueue:
    """Merges reports in memory (never blocks); `drain()` takes them, `restore()` puts them back."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Optional[dict[str, Any]] = None

    def sink(self, report: dict[str, Any]) -> None:
        with self._lock:
            self._pending = _merge(self._pending, report)

    def drain(self) -> Optional[dict[str, Any]]:
        with self._lock:
            out, self._pending = self._pending, None
            return out

    def restore(self, report: dict[str, Any]) -> None:
        """Put a drained report back; anything that arrived since keeps its newer values."""
        with self._lock:
            self._pending = _merge(report, self._pending) if self._pending else report

    def pending(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return None if self._pending is None else dict(self._pending)


def _merge(older: Optional[dict[str, Any]], newer: Optional[dict[str, Any]]) -> dict[str, Any]:
    if older is None:
        return dict(newer) if newer is not None else {}
    if newer is None:
        return dict(older)
    uses: dict[tuple[str, str], dict[str, Any]] = {}
    for u in [*older.get("uses", ()), *newer.get("uses", ())]:
        key = (u["kind"], u["old"])
        if key not in uses or _parse(u["last_seen"]) >= _parse(uses[key]["last_seen"]):
            uses[key] = dict(u)
    return {"reporter": newer.get("reporter") or older.get("reporter"),
            "uses": sorted(uses.values(), key=lambda u: (u["kind"], u["old"])),
            "also_set": newer.get("also_set", []), "conflicts": newer.get("conflicts", [])}


API_QUEUE = ApiQueue()


async def flush_once(queue: ApiQueue = API_QUEUE, timeout: float = FLUSH_TIMEOUT_S) -> bool:
    """Write what the API queued, bounded by `timeout`. True when something was written. A failure,
    a timeout or a cancellation restores the drained report (CancelledError is re-raised)."""
    report = queue.drain()
    if not report:
        return False
    try:
        await asyncio.wait_for(asyncio.to_thread(write, report), timeout)
    except asyncio.CancelledError:
        queue.restore(report)
        raise
    except Exception as exc:  # noqa: BLE001 — TimeoutError and any database error
        queue.restore(report)
        logger.warning("legacy_names_store: API flush failed (%s); uses kept for the next flush",
                       type(exc).__name__)
        return False
    return True


async def flush_loop(queue: ApiQueue = API_QUEUE, interval: float = FLUSH_INTERVAL_S) -> None:
    """The API's lifespan task: every `interval` seconds re-stamp what is still in force (`refresh`, which
    also queues an empty report — the API's heartbeat) and flush the queue."""
    while True:
        try:
            legacy_names.refresh()
            await flush_once(queue)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — the loop must survive anything
            logger.warning("legacy_names_store: flush loop step failed (%s)", type(exc).__name__)
        await asyncio.sleep(interval)


async def shutdown_flush(queue: ApiQueue = API_QUEUE) -> None:
    """The last, bounded flush at API shutdown (uses of the final seconds may still be lost if it fails)."""
    try:
        await flush_once(queue, timeout=SHUTDOWN_FLUSH_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001
        logger.warning("legacy_names_store: shutdown flush failed (%s)", type(exc).__name__)
