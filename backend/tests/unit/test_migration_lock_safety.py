"""A migration blocked on a lock fails fast, names its blocker, and leaves nothing queued.

INCIDENT 2026-10-21 05:05-05:16 ET. Deploying 73f2c603, the entrypoint's ``alembic upgrade``
reached ``c4e7a1b2d3f5`` (``ALTER TABLE probe_monitor_datasets ADD COLUMN …``) while a hung,
two-day-old COPY held ACCESS SHARE on that table. With no ``lock_timeout`` the ALTER waited
forever; the startup probe SIGKILLed the container twice, and each killed attempt's ALTER stayed
queued in Postgres behind the COPY — and in front of every later reader of the table.

These tests replay that against a REAL Postgres (the suite's own server; a scratch database per
test, dropped WITH (FORCE) afterwards), through the real ``alembic/env.py`` and the real
``python -m src.db.migrate`` the entrypoint runs. The blocker is a second session holding
ACCESS SHARE in an open transaction, exactly the COPY's lock.

MUTATION CONTROLS (2026-10-21; each alone, restored from saved bytes, sha256-checked) — all red:
  M1 lock_timeout set to "0" (disabled) in apply_session_guards
       -> test_a_blocked_migration_gives_up_names_the_blocker_and_then_succeeds (TimeoutExpired at 60 s)
  M2 client_connection_check_interval removed
       -> test_a_killed_migration_leaves_no_waiter_behind (the SIGKILLed ALTER stays queued — the incident)
  M3 env.py reads its pid and commits but installs no guards
       -> test_a_blocked_migration_gives_up_names_the_blocker_and_then_succeeds
  M4 the runner stops retrying a lock timeout (``retryable = False``)
       -> test_a_blocked_migration_gives_up_names_the_blocker_and_then_succeeds (rc 4, 1 attempt)
  M5 the watch never records blockers
       -> test_a_blocked_migration_gives_up_names_the_blocker_and_then_succeeds (pid not named)
  M6 the deadline cancel removed from the watch
       -> test_the_deadline_cancels_a_run_server_side
  M7 startupProbe failureThreshold lowered to 20 (200 s)
       -> test_the_startup_probe_outlasts_the_whole_migration_budget
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psycopg2
import pytest
import yaml
from sqlalchemy.engine import make_url

from src.db import migration_safety as ms

BACKEND = Path(__file__).resolve().parents[2]
MANIFEST = BACKEND.parent / "k8s" / "base" / "backend.yaml"
#: The revision before the incident's migration, and the incident's migration.
BEFORE_INCIDENT = "b36e0a7c1d42"
INCIDENT = "c4e7a1b2d3f5"
INCIDENT_TABLE = "probe_monitor_datasets"


def _server_url():
    raw = os.environ.get("DATABASE_URL_SYNC") or os.environ["DATABASE_URL"]
    return make_url(raw)


def _connect(dbname: str):
    u = _server_url()
    conn = psycopg2.connect(host=u.host or "localhost", port=u.port or 5432, user=u.username,
                            password=u.password, dbname=dbname, connect_timeout=5)
    conn.autocommit = True
    return conn


@pytest.fixture
def scratch_db():
    name = f"mig_lock_{uuid.uuid4().hex[:12]}"
    admin = _connect(_server_url().database)
    with admin.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{name}"')
    try:
        yield name
    finally:
        with admin.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


def _env(dbname: str, **extra) -> dict:
    u = _server_url().set(drivername="postgresql+asyncpg", database=dbname)
    env = dict(os.environ)
    env.update({
        "DATABASE_URL": u.render_as_string(hide_password=False),
        "DATABASE_URL_SYNC": u.set(drivername="postgresql+psycopg2").render_as_string(hide_password=False),
        "PYTHONUNBUFFERED": "1",
    })
    env.pop("MIGRATION_DEADLINE_AT", None)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _alembic(dbname: str, *args, timeout=240):
    return subprocess.run([sys.executable, "-m", "alembic", *args], cwd=BACKEND, env=_env(dbname),
                          capture_output=True, text=True, timeout=timeout)


def _migrate(dbname: str, timeout: float, **extra):
    started = time.monotonic()
    result = subprocess.run([sys.executable, "-m", "src.db.migrate", "upgrade"], cwd=BACKEND,
                            env=_env(dbname, **extra), capture_output=True, text=True, timeout=timeout)
    return result, time.monotonic() - started


def _version(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT version_num FROM alembic_version")
        return {row[0] for row in cur.fetchall()}


def _migration_sessions(conn, dbname: str) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute("SELECT pid, state, wait_event_type, query FROM pg_stat_activity "
                    "WHERE datname = %s AND application_name = 'mistudio-migrate'", (dbname,))
        return cur.fetchall()


@pytest.fixture
def at_the_incident(scratch_db):
    """A scratch database one revision before the incident's ALTER, and a session holding ACCESS
    SHARE on the table that ALTER needs, as the hung COPY did."""
    result = _alembic(scratch_db, "upgrade", BEFORE_INCIDENT)
    assert result.returncode == 0, result.stdout + result.stderr
    blocker = _connect(scratch_db)
    blocker.autocommit = False
    with blocker.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {INCIDENT_TABLE}")
        cur.execute("SELECT pg_backend_pid()")
        pid = cur.fetchone()[0]
    observer = _connect(scratch_db)
    try:
        yield scratch_db, blocker, pid, observer
    finally:
        try:
            blocker.rollback()
            blocker.close()
        finally:
            observer.close()


class TestTheIncidentReplayed:
    def test_a_blocked_migration_gives_up_names_the_blocker_and_then_succeeds(self, at_the_incident):
        dbname, blocker, blocker_pid, observer = at_the_incident
        # Budget: 2 attempts x 2 s lock_timeout + 1 s backoff. Without lock_timeout this run
        # would wait until the 150 s statement_timeout; the 60 s subprocess timeout turns that
        # into a failure rather than a hung suite.
        result, elapsed = _migrate(dbname, timeout=60, MIGRATION_LOCK_TIMEOUT="2s",
                                   MIGRATION_ATTEMPTS=2, MIGRATION_RETRY_BACKOFF="1s")
        out = result.stdout + result.stderr
        assert result.returncode == 3, out
        assert elapsed < 30, f"took {elapsed:.1f}s for a ~5s budget"
        assert out.count("gave up waiting for a lock") >= 2, "the lock timeout was not retried"
        assert f"pid={blocker_pid} " in out, f"the blocker (pid {blocker_pid}) is not named:\n{out}"
        assert "pg_terminate_backend" in out and "GIVING UP" in out
        assert _version(observer) == {BEFORE_INCIDENT}, "a blocked run must apply nothing"
        assert _migration_sessions(observer, dbname) == [], "a migration statement was left queued"

        blocker.rollback()  # the COPY finishes (or the operator terminates it)
        result, _ = _migrate(dbname, timeout=120, MIGRATION_LOCK_TIMEOUT="2s")
        assert result.returncode == 0, result.stdout + result.stderr
        from alembic.script import ScriptDirectory

        from src.db.migrate import DEFAULT_INI, _config

        assert _version(observer) == set(ScriptDirectory.from_config(_config(DEFAULT_INI)).get_heads())
        with observer.cursor() as cur:
            cur.execute("SELECT 1 FROM information_schema.columns "
                        "WHERE table_name = %s AND column_name = 'archived_at'", (INCIDENT_TABLE,))
            assert cur.fetchone(), "the incident's column was not added once the lock was free"

    def test_a_killed_migration_leaves_no_waiter_behind(self, at_the_incident):
        """The startup probe's SIGKILL must not leave the ALTER in the lock queue (2026-10-21: it
        did, twice). client_connection_check_interval reaps it long before lock_timeout would."""
        dbname, _blocker, _pid, observer = at_the_incident
        proc = subprocess.Popen([sys.executable, "-m", "src.db.migrate", "upgrade"], cwd=BACKEND,
                                env=_env(dbname, MIGRATION_LOCK_TIMEOUT="120s", MIGRATION_ATTEMPTS=1),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 60
            while not any(row[2] == "Lock" for row in _migration_sessions(observer, dbname)):
                assert proc.poll() is None, "the migration exited instead of waiting on the lock"
                assert time.monotonic() < deadline, "the migration never queued on the lock"
                time.sleep(0.2)
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=10)
            killed = time.monotonic()
            while _migration_sessions(observer, dbname):
                assert time.monotonic() - killed < 15, (
                    f"the killed migration's statement is still queued: {_migration_sessions(observer, dbname)}")
                time.sleep(0.25)
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_the_deadline_cancels_a_run_server_side(self, at_the_incident):
        dbname, _blocker, _pid, observer = at_the_incident
        result, elapsed = _migrate(dbname, timeout=60, MIGRATION_LOCK_TIMEOUT="120s",
                                   MIGRATION_ATTEMPTS=1, MIGRATION_DEADLINE="3s")
        out = result.stdout + result.stderr
        assert result.returncode == 4, out
        assert "passed its deadline" in out and elapsed < 30, out
        assert _migration_sessions(observer, dbname) == []

    def test_a_long_statement_is_cancelled_and_reported(self, at_the_incident):
        dbname, _blocker, _pid, observer = at_the_incident
        result, _ = _migrate(dbname, timeout=60, MIGRATION_LOCK_TIMEOUT="120s",
                             MIGRATION_STATEMENT_TIMEOUT="1s", MIGRATION_ATTEMPTS=3)
        out = result.stdout + result.stderr
        assert result.returncode == 4, out
        assert "statement_timeout=1s" in out and "MIGRATION_STATEMENT_TIMEOUT=0" in out
        assert out.count("Running upgrade") == 1, "a statement timeout must not be retried"


class TestTheBudget:
    def _backend_container(self):
        (deployment,) = [doc for doc in yaml.safe_load_all(MANIFEST.read_text())
                         if doc and doc.get("kind") == "Deployment" and doc["metadata"]["name"] == "mistudio-backend"]
        (backend,) = [c for c in deployment["spec"]["template"]["spec"]["containers"]
                      if {e["name"]: e.get("value") for e in c.get("env", [])}.get("SERVICE_TYPE") == "api"]
        return backend

    def test_the_startup_probe_outlasts_the_whole_migration_budget(self, monkeypatch):
        """The migration must report and exit before the kubelet SIGKILLs it: the deadline (which
        bounds everything, retries included) plus room for imports and uvicorn's boot must fit in
        the startup probe's window. Read with the manifest's own MIGRATION_* values applied."""
        backend = self._backend_container()
        for name in ms.DEFAULTS:
            monkeypatch.delenv(name, raising=False)
        for env in backend.get("env", []):
            if env["name"].startswith("MIGRATION_") and "value" in env:
                monkeypatch.setenv(env["name"], env["value"])
        settings = ms.settings_from_env()
        probe = backend["startupProbe"]
        window = probe.get("periodSeconds", 10) * probe.get("failureThreshold", 3) + probe.get("initialDelaySeconds", 0)
        assert window >= settings.deadline_s + 90, (window, settings.deadline_s)
        assert settings.statement_timeout_s < settings.deadline_s
        assert settings.lock_retry_budget_s() <= settings.deadline_s, (
            "the retries cannot all run inside the deadline")
        assert settings.attempts >= 2, "a single attempt cannot ride out a short-lived lock"


class TestParsing:
    @pytest.mark.parametrize("value, seconds", [("15s", 15), ("500ms", 0.5), ("4min", 240), ("2000", 2), ("0", 0)])
    def test_postgres_durations(self, value, seconds):
        assert ms.parse_duration(value) == pytest.approx(seconds)

    def test_a_malformed_setting_is_refused_by_name(self, monkeypatch):
        monkeypatch.setenv("MIGRATION_LOCK_TIMEOUT", "15 seconds; DROP TABLE x")
        with pytest.raises(ValueError, match="not a duration"):
            ms.settings_from_env()

    def test_the_sqlstate_is_found_through_the_wrapping(self):
        class Inner(Exception):
            sqlstate = "55P03"

        class Wrapped(Exception):
            def __init__(self):
                super().__init__("wrapped")
                self.orig = Exception("adapter")
                self.orig.__cause__ = Inner()

        assert ms.sqlstate_of(Wrapped()) == ms.LOCK_NOT_AVAILABLE
        assert ms.sqlstate_of(ValueError("no sql")) is None
