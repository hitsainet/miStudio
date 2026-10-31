"""The judge-prompt library's revision (d5f8a2c41e7b): seeds, the legacy-run link, round trip.

Replayed by its OWN functions on a scratch database (named after this run's DATABASE_URL_SYNC, so
parallel agents never share it), over stand-in tables carrying exactly the columns it touches. The
shipped schema against the ORM is `test_schema_guards`' job.

What must hold, because each is a claim made to the operator:
* a pre-library run whose recorded hash IS the stakes prompt's is linked to the seeded
  ``stakes-rating/v1``, which is then LOCKED (it has been used);
* a run whose recorded hash is anything else is NOT linked — it measured with other text;
* no probe or run gains a behaviour (nothing recorded one; a guess would promote probes).
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

BACKEND = pathlib.Path(__file__).resolve().parents[2]
FILE = BACKEND / "alembic" / "versions" / "d5f8a2c41e7b_probe_judge_prompt_library.py"
TABLES = ("probe_monitor_judge_runs", "probe_monitors", "probe_monitor_runs", "probe_judge_prompts")


def _load():
    spec = importlib.util.spec_from_file_location("rev_d5f8a2c41e7b", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine():
    url = os.environ.get("DATABASE_URL_SYNC") or "postgresql://postgres:devpassword@localhost:5432/mistudio_test"
    url = url.replace("+psycopg2", "")
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_judge_lib_migration"
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
                conn.execute(text('CREATE DATABASE "' + name + '"'))
    finally:
        admin.dispose()
    return create_engine(server + "/" + name)


def _drop(c):
    for t in TABLES:
        c.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
    c.execute(text("DROP FUNCTION IF EXISTS probe_judge_prompts_immutable() CASCADE"))


def _run(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


@pytest.fixture
def engine():
    engine = _engine()
    with engine.begin() as c:
        _drop(c)
        c.execute(text("CREATE TABLE probe_monitor_runs (id VARCHAR(36) PRIMARY KEY)"))
        c.execute(text("CREATE TABLE probe_monitors (id VARCHAR(36) PRIMARY KEY)"))
        c.execute(text(
            "CREATE TABLE probe_monitor_judge_runs (id VARCHAR(36) PRIMARY KEY, "
            "prompt_version VARCHAR(64) NOT NULL, metrics JSONB NOT NULL DEFAULT '{}')"
        ))
        c.execute(text("INSERT INTO probe_monitor_runs VALUES ('pmr_1')"))
        c.execute(text("INSERT INTO probe_monitors VALUES ('pm_1')"))
    try:
        yield engine
    finally:
        with engine.begin() as c:
            _drop(c)
        engine.dispose()


def _legacy_run(engine, run_id, prompt_hash):
    metrics = {"prompt_version": "stakes-rating/v1"}
    if prompt_hash is not None:
        metrics["prompt_hash"] = prompt_hash
    with engine.begin() as c:
        c.execute(
            text("INSERT INTO probe_monitor_judge_runs (id, prompt_version, metrics) "
                 "VALUES (:i, 'stakes-rating/v1', CAST(:m AS JSONB))"),
            {"i": run_id, "m": json.dumps(metrics)},
        )


def test_a_matching_legacy_run_is_linked_and_the_seed_LOCKED(engine):
    migration = _load()
    _legacy_run(engine, "pmj_match", migration.STAKES_HASH)
    _legacy_run(engine, "pmj_other", "f" * 64)
    _legacy_run(engine, "pmj_nohash", None)
    with engine.begin() as c:
        _run(c, migration.upgrade)
    with engine.connect() as c:
        rows = {r.id: (r.prompt_id, r.prompt_hash) for r in c.execute(
            text("SELECT id, prompt_id, prompt_hash FROM probe_monitor_judge_runs"))}
        assert rows["pmj_match"] == ("jp_stakes_rating_v1", migration.STAKES_HASH)
        assert rows["pmj_other"] == (None, "f" * 64), "a run with other text must not be linked"
        assert rows["pmj_nohash"] == (None, None)
        seeds = {r.id: r.locked for r in c.execute(
            text("SELECT id, locked_at IS NOT NULL AS locked FROM probe_judge_prompts"))}
        assert seeds == {"jp_stakes_rating_v1": True, "jp_pii_presence_v1": False}
        for table in ("probe_monitors", "probe_monitor_runs"):
            assert c.execute(text(f"SELECT behaviour FROM {table}")).scalar() is None


def test_with_no_legacy_runs_nothing_is_locked(engine):
    migration = _load()
    with engine.begin() as c:
        _run(c, migration.upgrade)
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM probe_judge_prompts WHERE locked_at IS NOT NULL")).scalar() == 0


def test_the_trigger_is_live_after_upgrade_and_the_revision_round_trips(engine):
    migration = _load()
    _legacy_run(engine, "pmj_match", migration.STAKES_HASH)
    with engine.begin() as c:
        _run(c, migration.upgrade)
    with engine.connect() as c, pytest.raises(Exception, match="cannot be changed"):
        c.execute(text("UPDATE probe_judge_prompts SET system_prompt = 'x' WHERE id = 'jp_stakes_rating_v1'"))
    with engine.begin() as c:
        _run(c, migration.downgrade)
        assert c.execute(text("SELECT to_regclass('probe_judge_prompts')")).scalar() is None
        columns = {r[0] for r in c.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'probe_monitor_judge_runs'"))}
        assert "prompt_id" not in columns and "prompt_hash" not in columns
    with engine.begin() as c:
        _run(c, migration.upgrade)
