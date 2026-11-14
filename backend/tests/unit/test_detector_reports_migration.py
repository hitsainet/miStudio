"""Revision effadddf2061 (Feature 040 S2, FTASKS 6.1): round trip, append-only tables, the widened CHECK
and the downgrade that refuses while a `blend_fit` link exists (TI-21).

Replayed by the revision's own functions on a scratch database over stand-in tables carrying the columns it
touches; the shipped schema against the ORM is `test_orm_matches_migrated_schema`'s job.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from src.models import detector_report as models
from tests.support.test_databases import sync_test_url

FILE = pathlib.Path(__file__).resolve().parents[2] / "alembic" / "versions" / "effadddf2061_detector_reports.py"
NEW_TABLES = ("detector_success_reports", "probe_monitor_dataset_column_changes")


def _load():
    spec = importlib.util.spec_from_file_location("rev_effadddf2061", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine():
    url = sync_test_url().replace("+psycopg2", "")
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_040_reports_migration"
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
                conn.execute(text('CREATE DATABASE "' + name + '"'))
    finally:
        admin.dispose()
    return create_engine(server + "/" + name)


def _drop(c):
    for t in (*NEW_TABLES, "probe_behaviour_view_links", "probe_monitors", "probe_monitor_runs", "probe_monitor_datasets"):
        c.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
    for t in NEW_TABLES:
        c.execute(text(f"DROP FUNCTION IF EXISTS {t}_append_only() CASCADE"))


def _run(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


@pytest.fixture
def engine():
    engine = _engine()
    with engine.begin() as c:
        _drop(c)
        c.execute(text("CREATE TABLE probe_monitor_datasets (id VARCHAR(36) PRIMARY KEY)"))
        c.execute(text("CREATE TABLE probe_monitor_runs (id VARCHAR(36) PRIMARY KEY)"))
        c.execute(text("CREATE TABLE probe_monitors (id VARCHAR(36) PRIMARY KEY)"))
        c.execute(text("CREATE TABLE probe_behaviour_view_links (id SERIAL PRIMARY KEY, use VARCHAR(16) NOT NULL, "
                       "CONSTRAINT ck_probe_behaviour_view_links_use CHECK (use IN ('train', 'eval', 'calibration')))"))
        c.execute(text("INSERT INTO probe_monitor_datasets VALUES ('pmd_old')"))
        c.execute(text("INSERT INTO probe_monitors VALUES ('pm_old')"))
    try:
        yield engine
    finally:
        with engine.begin() as c:
            _drop(c)
        engine.dispose()


def _shape(engine):
    found = inspect(engine)
    return {
        "tables": sorted(found.get_table_names()),
        **{t: sorted(c["name"] for c in found.get_columns(t))
           for t in ("probe_monitor_datasets", "probe_monitor_runs", "probe_monitors")},
        "checks": [c["sqltext"] for c in found.get_check_constraints("probe_behaviour_view_links")],
    }


def test_the_trigger_ddl_is_the_models():
    migration = _load()
    assert migration.REPORTS_DDL == models.REPORTS_DDL
    assert migration.COLUMN_CHANGES_DDL == models.COLUMN_CHANGES_DDL


def test_round_trip(engine):
    migration = _load()
    before = _shape(engine)
    with engine.begin() as c:
        _run(c, migration.upgrade)
    after = _shape(engine)
    assert set(NEW_TABLES) <= set(after["tables"])
    assert {"group_columns", "sub_question", "behaviour_version", "column_snapshots"} <= set(after["probe_monitor_datasets"])
    assert {"calibration_positions_path", "calibration_n_negatives", "sub_question"} <= set(after["probe_monitors"])
    assert "blend_fit" in after["checks"][0]
    with engine.begin() as c:
        _run(c, migration.downgrade)
    assert _shape(engine) == before
    with engine.begin() as c:
        _run(c, migration.upgrade)
    assert _shape(engine) == after


def test_existing_rows_read_null(engine):
    with engine.begin() as c:
        _run(c, _load().upgrade)
        row = c.execute(text("SELECT calibration_positions_path, calibration_n_negatives, sub_question, "
                             "behaviour_version FROM probe_monitors WHERE id = 'pm_old'")).one()
    assert tuple(row) == (None, None, None, None)


def test_the_downgrade_refuses_while_a_blend_fit_link_exists(engine):
    migration = _load()
    with engine.begin() as c:
        _run(c, migration.upgrade)
        c.execute(text("INSERT INTO probe_behaviour_view_links (use) VALUES ('blend_fit')"))
    with pytest.raises(RuntimeError, match="1 behaviour view link"):
        with engine.begin() as c:
            _run(c, migration.downgrade)
    with engine.begin() as c:
        assert c.execute(text("SELECT count(*) FROM detector_success_reports")).scalar() == 0  # still upgraded


@pytest.mark.parametrize("table,insert", [
    ("detector_success_reports",
     "INSERT INTO detector_success_reports VALUES ('dsr_1', repeat('a', 64), 'k-a', 1, 'm', '{}', 1, '{}', 'b', "
     "repeat('c', 64), 'met', 'operator', now())"),
    ("probe_monitor_dataset_column_changes",
     "INSERT INTO probe_monitor_dataset_column_changes (view_id, field, actor, created_at) "
     "VALUES ('pmd_x', 'group_columns', 'operator', now())"),
])
def test_the_new_tables_are_append_only(engine, table, insert):
    with engine.begin() as c:
        _run(c, _load().upgrade)
        c.execute(text(insert))
    for statement in (f"UPDATE {table} SET created_at = now()", f"DELETE FROM {table}"):
        with pytest.raises(Exception, match="append-only"):
            with engine.begin() as c:
                c.execute(text(statement))


def test_one_report_per_identity(engine):
    with engine.begin() as c:
        _run(c, _load().upgrade)
        row = ("'{id}', repeat('a', 64), 'k-a', 1, 'm', '{{}}', 1, '{{}}', 'b', repeat('c', 64), 'met', 'o', now()")
        c.execute(text("INSERT INTO detector_success_reports VALUES (" + row.format(id="dsr_1") + ")"))
    with pytest.raises(Exception, match="identity_sha256"):
        with engine.begin() as c:
            c.execute(text("INSERT INTO detector_success_reports VALUES (" + row.format(id="dsr_2") + ")"))
