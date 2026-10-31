"""Revision cf0910e18fd8 (Feature 040 S1, FTASKS 2.1 – 2.2): round trip, pre-feature rows, append-only `rule_sets`.

Replayed by the revision's OWN functions on a scratch database named after this run's test database, over
stand-in tables carrying the 037 columns of `probe_behaviours` / `probe_behaviour_versions`. The shipped
schema against the ORM is `test_orm_matches_migrated_schema`'s job (against `SCHEMA_CHECK_DATABASE_URL`).
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from src.models import rule_set as rule_set_model
from tests.support.test_databases import sync_test_url

BACKEND = pathlib.Path(__file__).resolve().parents[2]
FILE = BACKEND / "alembic" / "versions" / "cf0910e18fd8_behaviour_detector_fields.py"
SIX = ("sub_questions", "success_criteria", "held_out_columns", "models", "calibration_floor", "rule_set_digest")
HEX = "a" * 64


def _load():
    spec = importlib.util.spec_from_file_location("rev_cf0910e18fd8", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine():
    url = sync_test_url().replace("+psycopg2", "")
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_040_detector_migration"
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
                conn.execute(text('CREATE DATABASE "' + name + '"'))
    finally:
        admin.dispose()
    return create_engine(server + "/" + name)


def _drop(c):
    c.execute(text("DROP TABLE IF EXISTS probe_behaviour_versions CASCADE"))
    c.execute(text("DROP TABLE IF EXISTS probe_behaviours CASCADE"))
    c.execute(text("DROP TABLE IF EXISTS rule_sets CASCADE"))
    c.execute(text("DROP FUNCTION IF EXISTS rule_sets_append_only() CASCADE"))


def _run(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


@pytest.fixture
def engine():
    engine = _engine()
    with engine.begin() as c:
        _drop(c)
        c.execute(text("CREATE TABLE probe_behaviours (key VARCHAR(64) PRIMARY KEY, current_version INTEGER NOT NULL)"))
        c.execute(text(
            "CREATE TABLE probe_behaviour_versions (id SERIAL PRIMARY KEY, "
            "key VARCHAR(64) NOT NULL REFERENCES probe_behaviours(key) ON DELETE CASCADE, "
            "version INTEGER NOT NULL, name VARCHAR(255) NOT NULL, positive_meaning TEXT NOT NULL, "
            "target_fpr DOUBLE PRECISION)"
        ))
        c.execute(text("INSERT INTO probe_behaviours VALUES ('k-before', 1)"))
        c.execute(text(
            "INSERT INTO probe_behaviour_versions (key, version, name, positive_meaning, target_fpr) "
            "VALUES ('k-before', 1, 'n', 'm', 0.01)"
        ))
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
        "columns": sorted(c["name"] for c in found.get_columns("probe_behaviour_versions")),
        "checks": sorted(c["name"] for c in found.get_check_constraints("probe_behaviour_versions")),
        "fks": sorted(fk["name"] for fk in found.get_foreign_keys("probe_behaviour_versions")),
    }


def test_the_trigger_ddl_is_the_models():
    assert _load().RULE_SETS_DDL == rule_set_model.RULE_SETS_DDL


def test_round_trip_upgrade_downgrade_upgrade(engine):
    migration = _load()
    before = _shape(engine)
    with engine.begin() as c:
        _run(c, migration.upgrade)
    after = _shape(engine)
    assert "rule_sets" in after["tables"]
    assert set(SIX) <= set(after["columns"])
    with engine.begin() as c:
        _run(c, migration.downgrade)
    assert _shape(engine) == before
    with engine.begin() as c:
        _run(c, migration.upgrade)
    assert _shape(engine) == after


def test_a_pre_feature_version_reads_every_field_null(engine):
    with engine.begin() as c:
        _run(c, _load().upgrade)
        row = c.execute(text(
            f"SELECT {', '.join(SIX)}, name, target_fpr FROM probe_behaviour_versions WHERE key = 'k-before'"
        )).one()
    assert tuple(row[:6]) == (None,) * 6
    assert row[6:] == ("n", 0.01)


def _insert_set(c, digest=HEX, key="k-before"):
    c.execute(text(
        "INSERT INTO rule_sets (digest, bytes, schema_version, behaviour_key, source, digest_scheme, "
        "created_by, created_at) VALUES (:d, :b, 'midataworks.rule-set/v1', :k, 'inline', "
        "'midataworks_rule_set_identity', 'operator', now())"
    ), {"d": digest, "b": b"{}", "k": key})


@pytest.mark.parametrize("statement", [
    "UPDATE rule_sets SET behaviour_key = 'other'",
    "DELETE FROM rule_sets",
])
def test_the_database_refuses_changing_a_rule_set(engine, statement):
    with engine.begin() as c:
        _run(c, _load().upgrade)
        _insert_set(c)
    with pytest.raises(Exception, match="append-only"):
        with engine.begin() as c:
            c.execute(text(statement))
    with engine.begin() as c:
        assert c.execute(text("SELECT behaviour_key FROM rule_sets")).scalar() == "k-before"


@pytest.mark.parametrize("statement,match", [
    ("INSERT INTO rule_sets (digest, bytes, schema_version, behaviour_key, source, digest_scheme, created_by, "
     "created_at) VALUES ('NOTHEX', '', 'f', 'k', 'inline', 's', 'o', now())", "ck_rule_sets_digest_hex"),
    (f"INSERT INTO rule_sets (digest, bytes, schema_version, behaviour_key, source, digest_scheme, created_by, "
     f"created_at) VALUES ('{'b' * 64}', '', 'f', 'k', 'pasted', 's', 'o', now())", "ck_rule_sets_source"),
    ("UPDATE probe_behaviour_versions SET calibration_floor = 0", "calibration_floor"),
    (f"UPDATE probe_behaviour_versions SET rule_set_digest = '{'c' * 64}'", "rule_set_digest_fkey"),
])
def test_constraints(engine, statement, match):
    with engine.begin() as c:
        _run(c, _load().upgrade)
    with pytest.raises(Exception, match=match):
        with engine.begin() as c:
            c.execute(text(statement))


def test_a_named_set_cannot_go_even_past_the_trigger(engine):
    """RESTRICT is the second line: with the trigger off, the FK still keeps a named set."""
    with engine.begin() as c:
        _run(c, _load().upgrade)
        _insert_set(c)
        c.execute(text(f"UPDATE probe_behaviour_versions SET rule_set_digest = '{HEX}'"))
    with pytest.raises(Exception, match="rule_set_digest_fkey"):
        with engine.begin() as c:
            c.execute(text("ALTER TABLE rule_sets DISABLE TRIGGER trg_rule_sets_append_only"))
            c.execute(text("DELETE FROM rule_sets"))
