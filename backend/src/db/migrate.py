"""What the container entrypoint runs instead of a bare ``alembic upgrade heads``.

``python -m src.db.migrate upgrade``
    Runs the migrations with the session guards of :mod:`src.db.migration_safety`. A lock timeout
    is retried up to ``MIGRATION_ATTEMPTS`` times with doubling backoff (``MIGRATION_RETRY_BACKOFF``),
    all inside ``MIGRATION_DEADLINE`` — after which the run is cancelled server-side. Exit codes:
    0 applied (or nothing to do), 3 still blocked after every attempt (the blockers are in the log),
    4 deadline or statement timeout, 1 any other failure. The total stays well inside the backend's
    startup-probe window, so the process reports and exits instead of being SIGKILLed mid-wait.

``python -m src.db.migrate wait-for-head``
    For the Celery worker/beat containers, which start with the API in the same pod (strategy
    Recreate): wait until ``alembic_version`` is this code's head before starting, so a worker never
    queries a column the API's migration has not added yet (2026-10-21: beat and the workers hit
    ``probe_monitor_runs.behaviour does not exist`` while the migration was still blocked).
    Bounded by ``MIGRATION_WAIT_TIMEOUT`` (default 600s); exit 0 at head, 1 on timeout. A database
    AHEAD of this code (a rollback) proceeds with a warning: waiting could never end, and refusing
    would take every worker down on a rollback.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
DEFAULT_INI = BACKEND / "alembic.ini"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 3
EXIT_TIMED_OUT = 4


def _say(message: str) -> None:
    print(f"[migrate] {message}", file=sys.stderr, flush=True)


def _config(ini: Path):
    from alembic.config import Config

    cfg = Config(str(ini))
    location = cfg.get_main_option("script_location")
    if location and not os.path.isabs(location) and "%(here)s" not in location:
        cfg.set_main_option("script_location", str(ini.parent / location))
    return cfg


def upgrade(ini: Path = DEFAULT_INI, target: str = "heads") -> int:
    from alembic import command

    from src.db import migration_safety as ms

    try:
        settings = ms.settings_from_env()
    except ValueError as exc:
        _say(f"refusing to migrate: {exc}")
        return EXIT_FAILED
    started = time.time()
    deadline = started + settings.deadline_s
    os.environ["MIGRATION_DEADLINE_AT"] = repr(deadline)
    backoffs = settings.backoffs()
    _say(f"upgrading to {target}: up to {settings.attempts} attempt(s), lock_timeout={settings.lock_timeout}, "
         f"backoff {backoffs}, deadline {settings.deadline_s:.0f}s")
    last: Exception | None = None
    for attempt in range(1, settings.attempts + 1):
        try:
            command.upgrade(_config(ini), target)
        except ms.MigrationSafetyError as exc:
            last = exc
            if not exc.retryable:
                _say(f"attempt {attempt} FAILED and will not be retried: {exc}")
                return EXIT_TIMED_OUT
            if attempt == settings.attempts:
                break
            pause = backoffs[attempt - 1]
            if time.time() + pause + settings.lock_timeout_s > deadline:
                _say(f"attempt {attempt} blocked; no time left for another before the "
                     f"{settings.deadline_s:.0f}s deadline")
                break
            _say(f"attempt {attempt}/{settings.attempts} blocked on a lock; retrying in {pause:.0f}s")
            time.sleep(pause)
            continue
        except Exception as exc:  # noqa: BLE001 — report every failure the same way, then exit
            _say(f"migration FAILED (not a lock timeout; not retried): {type(exc).__name__}: {exc}")
            return EXIT_FAILED
        _say(f"migrations applied in {time.time() - started:.1f}s (attempt {attempt})")
        return EXIT_OK
    _say(f"GIVING UP after {time.time() - started:.0f}s: the migration could not get its locks. "
         f"Nothing was applied and no statement is left waiting in Postgres. {last}")
    return EXIT_BLOCKED


def _sync_url() -> str:
    from src.core.config import settings

    return str(settings.database_url)


def schema_state(ini: Path, url: str) -> tuple[str, set[str], set[str]]:
    """(``"head"`` | ``"behind"`` | ``"ahead"``, database revisions, code heads)."""
    import psycopg2

    from alembic.script import ScriptDirectory
    from src.db.migration_safety import _sync_connect_kwargs

    script = ScriptDirectory.from_config(_config(ini))
    heads = set(script.get_heads())
    conn = psycopg2.connect(application_name="mistudio-schema-wait", **_sync_connect_kwargs(url))
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.alembic_version') IS NOT NULL")
            if not cur.fetchone()[0]:
                return "behind", set(), heads
            cur.execute("SELECT version_num FROM alembic_version")
            current = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()
    if current == heads:
        return "head", current, heads
    known = {rev.revision for rev in script.walk_revisions()}
    if current - known:
        return "ahead", current, heads
    return "behind", current, heads


def wait_for_head(ini: Path = DEFAULT_INI, timeout_s: float | None = None, poll_s: float | None = None) -> int:
    timeout_s = float(os.environ.get("MIGRATION_WAIT_TIMEOUT", "600")) if timeout_s is None else timeout_s
    poll_s = float(os.environ.get("MIGRATION_WAIT_POLL", "5")) if poll_s is None else poll_s
    url = _sync_url()
    started = time.monotonic()
    last_report = None
    while True:
        try:
            state, current, heads = schema_state(ini, url)
            detail = f"database at {sorted(current) or 'no alembic_version'}, code head {sorted(heads)}"
        except Exception as exc:  # noqa: BLE001 — the database may still be starting
            state, detail = "unreachable", f"{type(exc).__name__}: {exc}".strip()
        if state == "head":
            _say(f"schema is at head ({detail}); starting")
            return EXIT_OK
        if state == "ahead":
            _say(f"WARNING: the database is AHEAD of this code ({detail}) — a rollback? Starting anyway: "
                 f"waiting cannot end, and the newer schema is expected to be additive")
            return EXIT_OK
        waited = time.monotonic() - started
        if waited >= timeout_s:
            _say(f"GIVING UP after {waited:.0f}s: schema is not at this code's head ({detail}). The API "
                 f"container runs the migration; check its log for the blocker. Not starting.")
            return EXIT_FAILED
        if last_report is None or time.monotonic() - last_report >= 15:
            last_report = time.monotonic()
            _say(f"waiting for the schema ({state}: {detail}); {waited:.0f}s of {timeout_s:.0f}s")
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.db.migrate")
    parser.add_argument("action", choices=["upgrade", "wait-for-head"])
    parser.add_argument("--config", type=Path, default=DEFAULT_INI, help="alembic.ini (default: the backend's)")
    args = parser.parse_args(argv)
    if args.action == "upgrade":
        return upgrade(args.config)
    return wait_for_head(args.config)


if __name__ == "__main__":
    sys.exit(main())
