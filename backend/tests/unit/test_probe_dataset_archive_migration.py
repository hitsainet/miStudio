"""Revision c4e7a1b2d3f5 (probe dataset archive + purge tombstones) round-trips with rows present,
and its trigger keeps the tombstone append-only for raw SQL.

Replayed by the revision's OWN functions on a scratch database named after this run's
``DATABASE_URL_SYNC`` (parallel agents never share it), over a stand-in ``probe_monitor_datasets``
holding a row — an ADD COLUMN that only works on an empty table is the defect this catches.
``test_orm_matches_migrated_schema`` covers the shipped schema against the ORM.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

BACKEND = pathlib.Path(__file__).resolve().parents[2]
FILE = BACKEND / "alembic" / "versions" / "c4e7a1b2d3f5_probe_dataset_archive_and_purge.py"


def _load():
    spec = importlib.util.spec_from_file_location("rev_c4e7a1b2d3f5", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine():
    url = os.environ.get("DATABASE_URL_SYNC") or "postgresql://postgres:devpassword@localhost:5432/mistudio_test"
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_pmd_archive_migration"
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
                conn.execute(text('CREATE DATABASE "' + name + '"'))
    finally:
        admin.dispose()
    return create_engine(server + "/" + name)


def _drop(c):
    c.execute(text("DROP TABLE IF EXISTS probe_monitor_dataset_purges CASCADE"))
    c.execute(text("DROP TABLE IF EXISTS probe_monitor_datasets CASCADE"))
    c.execute(text("DROP FUNCTION IF EXISTS probe_monitor_dataset_purges_append_only() CASCADE"))


def _run(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


@pytest.fixture
def engine():
    engine = _engine()
    with engine.begin() as c:
        _drop(c)
        c.execute(text("CREATE TABLE probe_monitor_datasets (id VARCHAR(36) PRIMARY KEY, name VARCHAR(255))"))
        c.execute(text("INSERT INTO probe_monitor_datasets VALUES ('pmd_old', 'an existing view')"))
    try:
        yield engine
    finally:
        with engine.begin() as c:
            _drop(c)
        engine.dispose()


def test_identity():
    mod = _load()
    assert mod.revision == "c4e7a1b2d3f5" and mod.down_revision == "b36e0a7c1d42"


def test_upgrade_keeps_existing_views_unarchived_and_downgrade_round_trips(engine):
    mod = _load()
    with engine.begin() as c:
        _run(c, mod.upgrade)
    with engine.begin() as c:
        row = c.execute(text(
            "SELECT id, archived_at, archived_by, archive_reason FROM probe_monitor_datasets"
        )).one()
        assert tuple(row) == ("pmd_old", None, None, None), "an existing view must come up visible"
        cols = {col["name"] for col in sa.inspect(c).get_columns("probe_monitor_dataset_purges")}
        assert cols == {"id", "view_id", "name", "purged_by", "reason", "rows_removed", "removed",
                        "kept", "purged_at"}
    with engine.begin() as c:
        _run(c, mod.downgrade)
    with engine.begin() as c:
        insp = sa.inspect(c)
        assert not insp.has_table("probe_monitor_dataset_purges")
        assert {col["name"] for col in insp.get_columns("probe_monitor_datasets")} == {"id", "name"}
        assert c.execute(text("SELECT count(*) FROM probe_monitor_datasets")).scalar() == 1
    with engine.begin() as c:  # and up again
        _run(c, mod.upgrade)


@pytest.mark.parametrize("statement", [
    "UPDATE probe_monitor_dataset_purges SET reason = 'rewritten'",
    "DELETE FROM probe_monitor_dataset_purges",
])
def test_the_trigger_refuses_update_and_delete(engine, statement):
    mod = _load()
    with engine.begin() as c:
        _run(c, mod.upgrade)
        c.execute(text(
            "INSERT INTO probe_monitor_dataset_purges (id, view_id, name, purged_by, rows_removed, "
            "removed, kept, purged_at) VALUES ('pmp_1', 'pmd_x', 'x', 'operator', '{}', '{}', '[]', now())"
        ))
    with pytest.raises(sa.exc.DBAPIError, match="append-only"):
        with engine.begin() as c:
            c.execute(text(statement))
    with engine.begin() as c:
        assert c.execute(text("SELECT reason FROM probe_monitor_dataset_purges")).scalar() is None
