"""Revision 5ed6d79015d2 (Feature 040 S3, FTASKS 20.1): round trip, the partial unique index, the body-immutable
trigger, and the frozen DDL copy.

Replayed by the revision's own functions on a scratch database over stand-in tables carrying the columns it
touches; the shipped schema against the ORM is `test_orm_matches_migrated_schema`'s job (against a database
migrated to head).
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from src.models import rules_evaluation as models
from tests.support.test_databases import sync_test_url

FILE = pathlib.Path(__file__).resolve().parents[2] / "alembic" / "versions" / "5ed6d79015d2_rule_signal_evaluations.py"
NEW_TABLES = ("probe_rules_evaluations", "rule_verdict_cache")


def _load():
    spec = importlib.util.spec_from_file_location("rev_5ed6d79015d2", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine():
    url = sync_test_url().replace("+psycopg2", "")
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_040_rules_migration"
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
                conn.execute(text('CREATE DATABASE "' + name + '"'))
    finally:
        admin.dispose()
    return create_engine(server + "/" + name)


def _drop(c):
    for t in (*NEW_TABLES, "probe_monitor_datasets"):
        c.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
    c.execute(text("DROP FUNCTION IF EXISTS probe_rules_evaluations_body_immutable() CASCADE"))


def _run(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


@pytest.fixture
def engine():
    engine = _engine()
    with engine.begin() as c:
        _drop(c)
        c.execute(text("CREATE TABLE probe_monitor_datasets (id VARCHAR(36) PRIMARY KEY)"))
        c.execute(text("INSERT INTO probe_monitor_datasets VALUES ('pmd_old')"))
    try:
        yield engine
    finally:
        with engine.begin() as c:
            _drop(c)
        engine.dispose()


def _shape(engine):
    found = inspect(engine)
    return {"tables": sorted(found.get_table_names()),
            "view_columns": sorted(c["name"] for c in found.get_columns("probe_monitor_datasets"))}


def _insert(c, identity, status, body=None):
    c.execute(text(
        "INSERT INTO probe_rules_evaluations (id, identity_sha256, probe_id, view_ids, behaviour_key, behaviour_version, "
        "engine_digest, calibration, status, inputs, created_by, created_at, body) VALUES (:id, :i, 'pm', '[]', 'k', 1, "
        "'e', false, :s, '{}', 'op', now(), :b)"), {"id": f"pre_{identity[:6]}_{status[:3]}", "i": identity, "s": status,
                                                     "b": body})


def test_round_trip(engine):
    rev = _load()
    before = _shape(engine)
    with engine.begin() as c:
        _run(c, rev.upgrade)
    after = _shape(engine)
    assert set(NEW_TABLES) <= set(after["tables"])
    assert {"text_column", "rule_columns", "rule_set_listing"} <= set(after["view_columns"])
    pk = inspect(engine).get_pk_constraint("rule_verdict_cache")["constrained_columns"]
    assert pk == ["view_id", "rule_set_digest", "engine_digest", "reading_digest"]
    with engine.begin() as c:
        _run(c, rev.downgrade)
    assert _shape(engine) == before
    with engine.begin() as c:
        _run(c, rev.upgrade)
    assert _shape(engine) == after


def test_the_partial_unique_index_covers_live_rows_only(engine):
    rev = _load()
    with engine.begin() as c:
        _run(c, rev.upgrade)
        _insert(c, "a" * 64, "completed", "{}")
        _insert(c, "a" * 64, "failed")       # a failed or cancelled row never blocks
        _insert(c, "a" * 64, "cancelled")
    with pytest.raises(Exception, match="uq_probe_rules_evaluations_identity_live"):
        with engine.begin() as c:
            _insert(c, "a" * 64, "pending")


def test_a_completed_body_is_never_rewritten_or_deleted(engine):
    rev = _load()
    with engine.begin() as c:
        _run(c, rev.upgrade)
        _insert(c, "b" * 64, "running")
        c.execute(text("UPDATE probe_rules_evaluations SET status = 'completed', body = '{}' WHERE identity_sha256 = :i"),
                  {"i": "b" * 64})
        c.execute(text("UPDATE probe_rules_evaluations SET progress = '{}' WHERE identity_sha256 = :i"), {"i": "b" * 64})
    with pytest.raises(Exception, match="never rewritten"):
        with engine.begin() as c:
            c.execute(text("UPDATE probe_rules_evaluations SET body = 'changed' WHERE identity_sha256 = :i"),
                      {"i": "b" * 64})
    with pytest.raises(Exception, match="never deleted"):
        with engine.begin() as c:
            c.execute(text("DELETE FROM probe_rules_evaluations WHERE identity_sha256 = :i"), {"i": "b" * 64})


def test_the_frozen_ddl_copy_is_the_models():
    assert _load().BODY_IMMUTABLE_DDL == models.BODY_IMMUTABLE_DDL


def test_the_revision_follows_the_head_it_was_written_on():
    rev = _load()
    assert (rev.revision, rev.down_revision) == ("5ed6d79015d2", "b7e41c2d9a05")
