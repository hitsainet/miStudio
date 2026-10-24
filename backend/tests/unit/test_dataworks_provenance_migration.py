"""Feature 035's one Alembic revision round-trips, and its constraint means what it says (task 1.1).

Replayed by its OWN functions on a scratch database (named after this run's `DATABASE_URL_SYNC`,
so parallel agents never share it), over stand-in tables that carry an EXISTING row — so "no
backfill" is asserted on data, not assumed (FR-5). `test_orm_matches_migrated_schema` and the
pipeline test (which builds its database with `alembic upgrade head`) cover the shipped schema.
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
from sqlalchemy.exc import IntegrityError

BACKEND = pathlib.Path(__file__).resolve().parents[2]
FILE = BACKEND / "alembic" / "versions" / "a35d0e1f2c3b_dataworks_provenance_routes.py"
EVAL_COLUMNS = ("positions_path", "n_scored_path", "labels_path")


def _load():
    spec = importlib.util.spec_from_file_location("rev_a35d0e1f2c3b", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine():
    url = os.environ.get("DATABASE_URL_SYNC") or "postgresql://postgres:devpassword@localhost:5432/mistudio_test"
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_dw035_migration"
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
                conn.execute(text('CREATE DATABASE "' + name + '"'))
    finally:
        admin.dispose()
    return create_engine(server + "/" + name)


def _run(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


def _cols(conn, table):
    return {c["name"]: c for c in sa.inspect(conn).get_columns(table)}


def _indexes(conn, table):
    return {i["name"] for i in sa.inspect(conn).get_indexes(table)}


@pytest.fixture
def conn():
    engine = _engine()
    with engine.begin() as c:
        for t in ("datasets", "probe_monitor_evaluations", "probe_monitor_runs"):
            c.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
        c.execute(text("CREATE TABLE datasets (id VARCHAR(36) PRIMARY KEY)"))
        c.execute(text("CREATE TABLE probe_monitor_evaluations (id VARCHAR(36) PRIMARY KEY)"))
        c.execute(text("CREATE TABLE probe_monitor_runs (id VARCHAR(36) PRIMARY KEY, train_dataset_id VARCHAR(36))"))
        c.execute(text("INSERT INTO datasets (id) VALUES ('legacy')"))
        c.execute(text("INSERT INTO probe_monitor_evaluations (id) VALUES ('pme_legacy')"))
    try:
        yield engine
    finally:
        with engine.begin() as c:
            for t in ("datasets", "probe_monitor_evaluations", "probe_monitor_runs"):
                c.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
        engine.dispose()


def test_identity():
    module = _load()
    assert module.revision == "a35d0e1f2c3b" and module.down_revision == "c34b5e2a9d02"


def test_up_down_up(conn):
    module = _load()
    with conn.begin() as c:
        _run(c, module.upgrade)
        assert _cols(c, "datasets")["revision"]["nullable"] is True
        assert str(_cols(c, "datasets")["revision"]["type"]) == "VARCHAR(40)"
        for name in EVAL_COLUMNS:
            assert str(_cols(c, "probe_monitor_evaluations")[name]["type"]) == "VARCHAR(1000)"
        assert "ix_probe_monitor_runs_train_dataset_id" in _indexes(c, "probe_monitor_runs")
        _run(c, module.downgrade)
        assert "revision" not in _cols(c, "datasets")
        assert not set(EVAL_COLUMNS) & set(_cols(c, "probe_monitor_evaluations"))
        assert "ix_probe_monitor_runs_train_dataset_id" not in _indexes(c, "probe_monitor_runs")
        _run(c, module.upgrade)
        assert "revision" in _cols(c, "datasets")


def test_existing_rows_read_NULL_after_upgrade(conn):
    """FR-5: no backfill — "revision not recorded" is the only truthful value for old rows."""
    module = _load()
    with conn.begin() as c:
        _run(c, module.upgrade)
        assert c.execute(text("SELECT revision FROM datasets WHERE id='legacy'")).scalar() is None
        row = c.execute(text(
            "SELECT positions_path, n_scored_path, labels_path FROM probe_monitor_evaluations"
        )).one()
        assert tuple(row) == (None, None, None)


@pytest.mark.parametrize("bad", ["main", "v1.0", "a" * 39, "A" * 40, "a" * 39 + "g"])
def test_the_constraint_refuses_anything_but_a_commit(conn, bad):
    module = _load()
    with conn.begin() as c:
        _run(c, module.upgrade)
    with pytest.raises(IntegrityError):
        with conn.begin() as c:
            c.execute(text("UPDATE datasets SET revision=:r WHERE id='legacy'"), {"r": bad})


def test_the_constraint_accepts_a_commit_and_null(conn):
    module = _load()
    with conn.begin() as c:
        _run(c, module.upgrade)
        c.execute(text("UPDATE datasets SET revision=:r WHERE id='legacy'"), {"r": "0123456789abcdef" * 2 + "01234567"})
        c.execute(text("UPDATE datasets SET revision=NULL WHERE id='legacy'"))


def test_the_orm_declares_the_same_constraint():
    import src.models  # noqa: F401
    from src.core.database import Base

    table = Base.metadata.tables["datasets"]
    checks = {c.name: str(c.sqltext) for c in table.constraints if isinstance(c, sa.CheckConstraint)}
    assert checks.get("ck_datasets_revision_is_commit") == "revision IS NULL OR revision ~ '^[0-9a-f]{40}$'"
    assert "ix_probe_monitor_runs_train_dataset_id" in {
        i.name for i in Base.metadata.tables["probe_monitor_runs"].indexes
    }
