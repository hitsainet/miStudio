"""Celery workers and beat wait for the schema; the API migrates through the guarded runner.

INCIDENT 2026-10-21. The pod's Celery worker and beat containers start at the same moment as the
API container (one pod, strategy Recreate). While the API's migration was blocked, beat and the
workers were already querying ``probe_monitor_runs.behaviour`` — a column that migration adds —
and failing with "column does not exist".

``docker-entrypoint.sh`` is EXECUTED here with a fake ``setpriv``/``su`` that record what they are
asked to run (as in test_gpu_workers_entrypoint.py), and ``python -m src.db.migrate wait-for-head``
runs for real against a scratch Postgres database.

MUTATION CONTROLS (2026-10-21; each alone, restored from saved bytes, sha256-checked) — all red:
  W1 the entrypoint's wait-for-head call removed
       -> test_a_worker_waits_for_the_schema_before_it_starts[*], test_a_worker_whose_wait_fails_never_starts[*]
  W2 the wait's failure ignored (`|| true`)
       -> test_a_worker_whose_wait_fails_never_starts[*]
  W3 the API migrates with a bare `alembic upgrade heads` again
       -> test_the_api_migrates_through_the_guarded_runner
  W4 wait_for_head treats "behind" as head
       -> test_a_database_behind_the_code_is_waited_for
  W5 the poll loop dropped (one look, then give up)
       -> test_the_wait_ends_when_the_migration_lands
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psycopg2
import pytest
import yaml
from sqlalchemy.engine import make_url

BACKEND = Path(__file__).resolve().parents[2]
ENTRYPOINT = BACKEND / "docker-entrypoint.sh"
MANIFEST = BACKEND.parent / "k8s" / "base" / "backend.yaml"
WORKERS = ["celery-worker", "celery-gpu-workers", "celery-beat"]


# ----------------------------------------------------------------------------- the entrypoint


def _fake(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


def _run_entrypoint(service_type: str, tmp_path: Path, wait_rc=0, migrate_rc=0, **extra):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    for name in ("setpriv", "su"):
        _fake(bin_dir, name, (
            f'echo "{name} $*" >> "{calls}"\n'
            f'case "$*" in *"src.db.migrate wait-for-head"*) exit {wait_rc};; '
            f'*"src.db.migrate upgrade"*|*"alembic upgrade"*) exit {migrate_rc};; esac\n'
            "exit 0\n"))
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "SERVICE_TYPE": service_type,
           "DATA_DIR": str(tmp_path / "data"), "HOME": str(tmp_path), **extra}
    result = subprocess.run(["bash", str(ENTRYPOINT)], env=env, capture_output=True, text=True, timeout=60)
    return result, (calls.read_text().splitlines() if calls.exists() else [])


def _service_launches(calls):
    return [c for c in calls if "src.db.migrate" not in c and "alembic" not in c]


class TestTheEntrypoint:
    @pytest.mark.parametrize("service_type", WORKERS)
    def test_a_worker_waits_for_the_schema_before_it_starts(self, service_type, tmp_path):
        result, calls = _run_entrypoint(service_type, tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        waits = [i for i, c in enumerate(calls) if "python -m src.db.migrate wait-for-head" in c]
        assert len(waits) == 1, calls
        assert calls[waits[0]].startswith("setpriv --reuid=mistudio --regid=mistudio"), calls[waits[0]]
        launches = _service_launches(calls)
        assert len(launches) == 1 and calls.index(launches[0]) > waits[0], calls
        assert not any("migrate upgrade" in c for c in calls), "a worker must not migrate by default"

    @pytest.mark.parametrize("service_type", WORKERS)
    def test_a_worker_whose_wait_fails_never_starts(self, service_type, tmp_path):
        result, calls = _run_entrypoint(service_type, tmp_path, wait_rc=1)
        assert result.returncode != 0
        assert _service_launches(calls) == [], f"{service_type} started on a stale schema: {calls}"
        assert "not starting" in result.stdout + result.stderr

    def test_the_wait_can_be_switched_off(self, tmp_path):
        result, calls = _run_entrypoint("celery-worker", tmp_path, wait_rc=1, WAIT_FOR_SCHEMA="false")
        assert result.returncode == 0 and not any("wait-for-head" in c for c in calls), calls

    def test_the_api_migrates_through_the_guarded_runner(self, tmp_path):
        result, calls = _run_entrypoint("api", tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        assert calls[0].startswith("setpriv --reuid=mistudio") and calls[0].endswith(
            "python -m src.db.migrate upgrade"), calls
        assert not any("alembic upgrade" in c for c in calls), calls
        assert not any("wait-for-head" in c for c in calls), "the API is the one that migrates"
        assert "uvicorn src.main:app" in calls[-1]

    def test_the_api_does_not_start_when_the_migration_fails(self, tmp_path):
        result, calls = _run_entrypoint("api", tmp_path, migrate_rc=3)
        assert result.returncode != 0
        assert not any("uvicorn" in c for c in calls), calls


class TestTheManifest:
    def test_no_worker_container_opts_out_of_the_wait(self):
        for doc in yaml.safe_load_all(MANIFEST.read_text()):
            if not doc or doc.get("kind") != "Deployment" or doc["metadata"]["name"] != "mistudio-backend":
                continue
            for container in doc["spec"]["template"]["spec"]["containers"]:
                env = {e["name"]: e.get("value") for e in container.get("env", [])}
                if env.get("SERVICE_TYPE") in WORKERS:
                    assert env.get("WAIT_FOR_SCHEMA", "true") != "false", container["name"]
                    assert env.get("RUN_MIGRATIONS") != "true", (
                        f"{container['name']} would race the API's migration")


# ----------------------------------------------------------------------------- wait-for-head, for real


def _server_url():
    return make_url(os.environ.get("DATABASE_URL_SYNC") or os.environ["DATABASE_URL"])


def _connect(dbname: str):
    u = _server_url()
    conn = psycopg2.connect(host=u.host or "localhost", port=u.port or 5432, user=u.username,
                            password=u.password, dbname=dbname, connect_timeout=5)
    conn.autocommit = True
    return conn


@pytest.fixture
def scratch_db():
    name = f"schema_wait_{uuid.uuid4().hex[:12]}"
    admin = _connect(_server_url().database)
    with admin.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{name}"')
    conn = _connect(name)
    try:
        yield name, conn
    finally:
        conn.close()
        with admin.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


@pytest.fixture(scope="module")
def head_and_previous():
    from alembic.script import ScriptDirectory

    from src.db.migrate import DEFAULT_INI, _config

    script = ScriptDirectory.from_config(_config(DEFAULT_INI))
    (head,) = script.get_heads()
    return head, script.get_revision(head).down_revision


def _stamp(conn, revision: str | None):
    with conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS alembic_version (version_num varchar(32) PRIMARY KEY)")
        cur.execute("DELETE FROM alembic_version")
        if revision:
            cur.execute("INSERT INTO alembic_version VALUES (%s)", (revision,))


def _wait_cmd(dbname: str, timeout: float):
    u = _server_url().set(drivername="postgresql+asyncpg", database=dbname)
    env = dict(os.environ, DATABASE_URL=u.render_as_string(hide_password=False),
               MIGRATION_WAIT_TIMEOUT=str(timeout), MIGRATION_WAIT_POLL="0.2", PYTHONUNBUFFERED="1")
    return [sys.executable, "-m", "src.db.migrate", "wait-for-head"], env


def _wait(dbname: str, timeout: float):
    cmd, env = _wait_cmd(dbname, timeout)
    started = time.monotonic()
    result = subprocess.run(cmd, cwd=BACKEND, env=env, capture_output=True, text=True, timeout=timeout + 60)
    return result, time.monotonic() - started


class TestWaitForHead:
    def test_a_database_at_head_starts_at_once(self, scratch_db, head_and_previous):
        dbname, conn = scratch_db
        _stamp(conn, head_and_previous[0])
        result, elapsed = _wait(dbname, timeout=30)
        assert result.returncode == 0 and "schema is at head" in result.stderr, result.stderr
        assert elapsed < 25

    def test_a_database_behind_the_code_is_waited_for(self, scratch_db, head_and_previous):
        dbname, conn = scratch_db
        _stamp(conn, head_and_previous[1])
        result, elapsed = _wait(dbname, timeout=2)
        assert result.returncode == 1, result.stderr
        assert elapsed >= 2 and "GIVING UP" in result.stderr and "Not starting" in result.stderr

    def test_a_database_with_no_alembic_version_is_waited_for(self, scratch_db):
        dbname, _conn = scratch_db
        result, _ = _wait(dbname, timeout=1)
        assert result.returncode == 1, result.stderr

    def test_the_wait_ends_when_the_migration_lands(self, scratch_db, head_and_previous):
        dbname, conn = scratch_db
        head, previous = head_and_previous
        _stamp(conn, previous)
        cmd, env = _wait_cmd(dbname, timeout=60)
        proc = subprocess.Popen(cmd, cwd=BACKEND, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            time.sleep(3)
            assert proc.poll() is None, f"stopped waiting on a stale schema: {proc.communicate()[0]}"
            _stamp(conn, head)
            landed = time.monotonic()
            out, _ = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proc.returncode == 0, out
        assert "waiting for the schema" in out and time.monotonic() - landed < 25

    def test_a_database_ahead_of_the_code_starts_with_a_warning(self, scratch_db):
        dbname, conn = scratch_db
        _stamp(conn, "ffffffffffff")
        result, _ = _wait(dbname, timeout=30)
        assert result.returncode == 0 and "AHEAD" in result.stderr, result.stderr
