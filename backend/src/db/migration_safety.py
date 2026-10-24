"""Migrations that fail fast, say why, and never leave a waiter behind.

WHY THIS MODULE EXISTS (incident 2026-10-21, 05:05-05:16 ET)
--------------------------------------------------------------
Deploying 73f2c603, the backend container's entrypoint ran ``alembic upgrade``. Migration
``c4e7a1b2d3f5`` began with ``ALTER TABLE probe_monitor_datasets ADD COLUMN``, which needs an
ACCESS EXCLUSIVE lock. A hung, two-day-old ``COPY`` (a pg_dump-style session) held ACCESS SHARE
on the table, so the ALTER waited — with no ``lock_timeout``, forever. The startup probe killed
the container twice (exit 137), and each killed attempt's ALTER STAYED QUEUED in Postgres: the
server does not notice a dead client while a backend sleeps on a lock, so every orphan kept its
place in the lock queue, and every later reader of the table queued behind it.

What this module does for the migration session (``alembic/env.py``, online mode):

* ``lock_timeout`` (``MIGRATION_LOCK_TIMEOUT``, default 15s): a lock wait errors out on its own.
* ``statement_timeout`` (``MIGRATION_STATEMENT_TIMEOUT``, default 150s): a single statement that
  runs longer than the startup window is cancelled and REPORTED, instead of the container being
  SIGKILLed mid-statement with nothing in the log. A migration that legitimately needs longer runs
  out of band with ``MIGRATION_STATEMENT_TIMEOUT=0``.
* ``idle_in_transaction_session_timeout`` (``MIGRATION_IDLE_TIMEOUT``, default 5min): a migration
  process that stops talking mid-transaction cannot hold its locks indefinitely.
* ``client_connection_check_interval`` (``MIGRATION_CLIENT_CHECK_INTERVAL``, default 2s, Postgres
  14+): a killed migration's backend notices the closed socket and aborts, even while it is
  sleeping on a lock — so a SIGKILLed attempt does not leave its ALTER in the queue.
* :class:`BlockerWatch`: while the migration waits on a lock, a second connection reads
  ``pg_blocking_pids`` and ``pg_stat_activity`` and logs exactly who blocks it (pid, application,
  user, client, state, transaction age, query). On a lock timeout the last reading is in the error.
* A deadline (``MIGRATION_DEADLINE_AT``, set by ``python -m src.db.migrate``): past it, the watch
  cancels the migration's backend, so a run is bounded as a whole, not just per statement.

``python -m src.db.migrate upgrade`` retries a lock timeout a bounded number of times with backoff
inside ``MIGRATION_DEADLINE`` (default 180s), which ``k8s/base/backend.yaml``'s startup probe window
(300s) comfortably exceeds — pinned by ``tests/unit/test_migration_lock_safety.py``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field

# A child of the `alembic` logger, so alembic.ini's INFO-level `alembic` logger prints it and
# env.py's fileConfig (disable_existing_loggers) keeps it enabled.
logger = logging.getLogger("alembic.mistudio.safety")

LOCK_NOT_AVAILABLE = "55P03"  # lock_timeout
QUERY_CANCELED = "57014"  # statement_timeout or pg_cancel_backend

DEFAULTS = {
    "MIGRATION_LOCK_TIMEOUT": "15s",
    "MIGRATION_STATEMENT_TIMEOUT": "150s",
    "MIGRATION_IDLE_TIMEOUT": "5min",
    "MIGRATION_CLIENT_CHECK_INTERVAL": "2s",
    "MIGRATION_ATTEMPTS": "4",
    "MIGRATION_RETRY_BACKOFF": "5s",
    "MIGRATION_DEADLINE": "180s",
}

#: Backoff never grows past this between two attempts.
MAX_BACKOFF_S = 30.0

_UNITS = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(us|ms|s|min|h|d)?\s*$")


def parse_duration(value: str) -> float:
    """Seconds in a Postgres duration (``15s``, ``4min``, ``500ms``; a bare number is MILLISECONDS,
    as Postgres reads it for these settings). ``0`` means "disabled"."""
    match = _DURATION.match(str(value))
    if not match:
        raise ValueError(f"not a duration: {value!r} (use e.g. 15s, 500ms, 4min)")
    number, unit = match.groups()
    return float(number) * _UNITS[unit or "ms"]


def _env(name: str) -> str:
    return os.environ.get(name, "").strip() or DEFAULTS[name]


@dataclass(frozen=True)
class MigrationSettings:
    lock_timeout: str
    statement_timeout: str
    idle_timeout: str
    client_check_interval: str
    attempts: int
    retry_backoff_s: float
    deadline_s: float

    @property
    def lock_timeout_s(self) -> float:
        return parse_duration(self.lock_timeout)

    @property
    def statement_timeout_s(self) -> float:
        return parse_duration(self.statement_timeout)

    def backoffs(self) -> list[float]:
        """The sleep before attempt 2, 3, … — doubling, capped at MAX_BACKOFF_S."""
        return [min(self.retry_backoff_s * 2**i, MAX_BACKOFF_S) for i in range(max(self.attempts - 1, 0))]

    def lock_retry_budget_s(self) -> float:
        """Worst case for a run whose every attempt dies on its first lock wait."""
        return self.attempts * self.lock_timeout_s + sum(self.backoffs())


def settings_from_env() -> MigrationSettings:
    values = {name: _env(name) for name in DEFAULTS}
    for name in ("MIGRATION_LOCK_TIMEOUT", "MIGRATION_STATEMENT_TIMEOUT", "MIGRATION_IDLE_TIMEOUT",
                 "MIGRATION_CLIENT_CHECK_INTERVAL", "MIGRATION_RETRY_BACKOFF", "MIGRATION_DEADLINE"):
        parse_duration(values[name])  # refuse a malformed value here, with its name, not in SQL
    attempts = int(values["MIGRATION_ATTEMPTS"])
    if attempts < 1:
        raise ValueError(f"MIGRATION_ATTEMPTS must be >= 1, got {attempts}")
    return MigrationSettings(
        lock_timeout=values["MIGRATION_LOCK_TIMEOUT"],
        statement_timeout=values["MIGRATION_STATEMENT_TIMEOUT"],
        idle_timeout=values["MIGRATION_IDLE_TIMEOUT"],
        client_check_interval=values["MIGRATION_CLIENT_CHECK_INTERVAL"],
        attempts=attempts,
        retry_backoff_s=parse_duration(values["MIGRATION_RETRY_BACKOFF"]),
        deadline_s=parse_duration(values["MIGRATION_DEADLINE"]),
    )


# ----------------------------------------------------------------------------- session guards


@dataclass(frozen=True)
class SessionGuards:
    backend_pid: int
    applied: dict[str, str]


def apply_session_guards(connection, settings: MigrationSettings | None = None) -> SessionGuards:
    """Set the migration session's timeouts (session-level, so they outlive this commit and hold
    for every transaction alembic opens, including autocommit blocks). ``connection`` is a
    SQLAlchemy (sync-facade) Connection; the transaction this opens is committed here."""
    from sqlalchemy import text

    settings = settings or settings_from_env()
    wanted = {
        "application_name": "mistudio-migrate",
        "lock_timeout": settings.lock_timeout,
        "statement_timeout": settings.statement_timeout,
        "idle_in_transaction_session_timeout": settings.idle_timeout,
    }
    version = int(connection.exec_driver_sql("SHOW server_version_num").scalar())
    if version >= 140000:
        wanted["client_connection_check_interval"] = settings.client_check_interval
    else:
        logger.warning("Postgres %s has no client_connection_check_interval; a killed migration's "
                       "backend is only reaped by lock_timeout", version)
    applied = {}
    for name, value in wanted.items():
        # set_config, not an interpolated SET: the values come from the environment.
        applied[name] = connection.execute(
            text("SELECT set_config(:name, :value, false)"), {"name": name, "value": str(value)}
        ).scalar()
    pid = int(connection.exec_driver_sql("SELECT pg_backend_pid()").scalar())
    connection.commit()
    logger.info("migration session pid=%s: lock_timeout=%s statement_timeout=%s "
                "idle_in_transaction_session_timeout=%s client_connection_check_interval=%s",
                pid, applied["lock_timeout"], applied["statement_timeout"],
                applied["idle_in_transaction_session_timeout"],
                applied.get("client_connection_check_interval", "n/a"))
    return SessionGuards(backend_pid=pid, applied=applied)


# ----------------------------------------------------------------------------- the blocker watch

_BLOCKERS_SQL = """
SELECT b.pid, b.application_name, b.usename, b.client_addr::text, b.state, b.backend_type,
       EXTRACT(EPOCH FROM now() - b.xact_start)::float, EXTRACT(EPOCH FROM now() - b.query_start)::float,
       left(regexp_replace(coalesce(b.query, ''), '\\s+', ' ', 'g'), 300),
       b.wait_event_type
  FROM pg_stat_activity w
 CROSS JOIN LATERAL unnest(pg_blocking_pids(w.pid)) AS blocker(pid)
  JOIN pg_stat_activity b ON b.pid = blocker.pid
 WHERE w.pid = %(pid)s
 ORDER BY b.xact_start NULLS LAST
"""


@dataclass(frozen=True)
class Blocker:
    pid: int
    application_name: str | None
    usename: str | None
    client_addr: str | None
    state: str | None
    backend_type: str | None
    xact_age_s: float | None
    query_age_s: float | None
    query: str
    waiting_on: str | None

    def describe(self) -> str:
        def age(seconds):
            if seconds is None:
                return "n/a"
            seconds = int(seconds)
            return f"{seconds // 86400}d{seconds % 86400 // 3600:02d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"

        return (f"pid={self.pid} application_name={self.application_name!r} user={self.usename!r} "
                f"client={self.client_addr or 'local'} state={self.state!r} "
                f"xact_age={age(self.xact_age_s)} query_age={age(self.query_age_s)}"
                + (f" (itself waiting on {self.waiting_on})" if self.waiting_on else "")
                + f" query={self.query!r}")


def _sync_connect_kwargs(url: str) -> dict:
    from sqlalchemy.engine import make_url

    u = make_url(url)
    kwargs = {"host": u.host or "localhost", "port": u.port or 5432, "user": u.username,
              "password": u.password, "dbname": u.database, "connect_timeout": 5}
    return {k: v for k, v in kwargs.items() if v is not None}


@dataclass
class BlockerWatch:
    """Polls who blocks the migration's backend; cancels it past an optional deadline.

    Used as a context manager around ``context.run_migrations()``. A watch that cannot connect
    logs it and steps aside: it is diagnostics, never a reason for a migration to fail.
    """

    url: str
    backend_pid: int
    deadline_at: float | None = None
    poll_s: float = 0.5
    blockers: list[Blocker] = field(default_factory=list)
    cancelled_for_deadline: bool = False
    available: bool = False

    def __post_init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_logged: tuple[int, ...] = ()

    def __enter__(self):
        try:
            import psycopg2

            self._conn = psycopg2.connect(application_name="mistudio-migrate-watch",
                                          options="-c statement_timeout=5000",
                                          **_sync_connect_kwargs(self.url))
            self._conn.autocommit = True
            self.available = True
        except Exception as exc:  # noqa: BLE001 — diagnostics must not fail a migration
            logger.warning("blocker watch unavailable (%s); a lock timeout will not name its blocker", exc)
            return self
        self._thread = threading.Thread(target=self._run, name="migration-blocker-watch", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self.available:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
        return False

    def read_blockers(self) -> list[Blocker]:
        with self._conn.cursor() as cur:
            cur.execute(_BLOCKERS_SQL, {"pid": self.backend_pid})
            return [Blocker(*row) for row in cur.fetchall()]

    def _run(self):
        while not self._stop.wait(self.poll_s):
            try:
                if self.deadline_at is not None and time.time() >= self.deadline_at and not self.cancelled_for_deadline:
                    self.cancelled_for_deadline = True
                    logger.error("migration deadline reached; cancelling the migration's backend pid=%s",
                                 self.backend_pid)
                    with self._conn.cursor() as cur:
                        cur.execute("SELECT pg_cancel_backend(%(pid)s)", {"pid": self.backend_pid})
                blockers = self.read_blockers()
            except Exception as exc:  # noqa: BLE001
                logger.warning("blocker watch stopped: %s", exc)
                return
            if blockers:
                self.blockers = blockers
                pids = tuple(b.pid for b in blockers)
                if pids != self._last_logged:
                    self._last_logged = pids
                    logger.warning("migration (pid=%s) is WAITING FOR A LOCK held or queued by:\n  %s",
                                   self.backend_pid, "\n  ".join(b.describe() for b in blockers))


# ----------------------------------------------------------------------------- errors


class MigrationSafetyError(RuntimeError):
    """A migration stopped by one of this module's guards. ``retryable`` says whether a later
    attempt can succeed without anyone intervening."""

    retryable = False


class MigrationLockTimeout(MigrationSafetyError):
    retryable = True

    def __init__(self, guards: SessionGuards, blockers: list[Blocker]):
        self.blockers = blockers
        if blockers:
            who = "\n  ".join(b.describe() for b in blockers)
            terminate = ", ".join(str(b.pid) for b in blockers)
            detail = (f"Blocked by:\n  {who}\nIf a blocker is stale, end it: "
                      f"SELECT pg_terminate_backend(pid) FROM unnest(ARRAY[{terminate}]) AS pid;")
        else:
            detail = "The blocker could not be identified (the watch saw none before the timeout)."
        super().__init__(
            f"migration (pid={guards.backend_pid}) gave up waiting for a lock after "
            f"lock_timeout={guards.applied.get('lock_timeout')}. Nothing was applied: the transaction "
            f"rolled back and no statement is left waiting. {detail}")


class MigrationStatementTimeout(MigrationSafetyError):
    def __init__(self, guards: SessionGuards):
        super().__init__(
            f"a migration statement ran longer than statement_timeout="
            f"{guards.applied.get('statement_timeout')} and was cancelled; nothing was applied. If it "
            f"legitimately needs longer, run it out of band: MIGRATION_STATEMENT_TIMEOUT=0 alembic upgrade heads")


class MigrationDeadlineExceeded(MigrationSafetyError):
    def __init__(self, guards: SessionGuards, blockers: list[Blocker]):
        tail = ("\n  last seen blocked by: " + "\n  ".join(b.describe() for b in blockers)) if blockers else ""
        super().__init__(
            f"migration (pid={guards.backend_pid}) passed its deadline (MIGRATION_DEADLINE) and was "
            f"cancelled; nothing was applied.{tail}")


def sqlstate_of(exc: BaseException) -> str | None:
    """The SQLSTATE anywhere in an exception's chain (SQLAlchemy wraps asyncpg/psycopg2 errors)."""
    seen = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        for attr in ("sqlstate", "pgcode"):
            code = getattr(current, attr, None)
            if isinstance(code, str) and code:
                return code
        stack.extend([getattr(current, "orig", None), current.__cause__, current.__context__])
    return None


def explain(exc: BaseException, guards: SessionGuards, watch: BlockerWatch) -> MigrationSafetyError | None:
    """The guard error a database error stands for, or None when it is something else."""
    code = sqlstate_of(exc)
    if code == LOCK_NOT_AVAILABLE:
        error: MigrationSafetyError = MigrationLockTimeout(guards, watch.blockers)
    elif code == QUERY_CANCELED and watch.cancelled_for_deadline:
        error = MigrationDeadlineExceeded(guards, watch.blockers)
    elif code == QUERY_CANCELED:
        error = MigrationStatementTimeout(guards)
    else:
        return None
    logger.error("%s", error)
    return error


def deadline_from_env() -> float | None:
    value = os.environ.get("MIGRATION_DEADLINE_AT", "").strip()
    return float(value) if value else None
