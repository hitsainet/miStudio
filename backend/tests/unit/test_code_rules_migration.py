"""Revision c0de5a11e004 (Feature 040 S4, FTASKS 25.3; FTDD Appendix C.5, C.6, C.8): round trip, the `origin` CHECK,
the append-only triggers, and the frozen DDL copy.

Replayed by the revision's own functions on a scratch database over a stand-in `agent_approval_requests` (the one
table it references); the shipped schema against the ORM is `test_orm_matches_migrated_schema`'s job (against a
database migrated to head).

Controls (040 controls record, S4): drop the `origin = 'operator'` CHECK → `test_the_database_refuses_an_agent_decision`
red; drop a trigger → the append-only test red.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DBAPIError

from src.models import code_rule as models
from tests.support.test_databases import sync_test_url

FILE = pathlib.Path(__file__).resolve().parents[2] / "alembic" / "versions" / "c0de5a11e004_code_rules.py"
NEW_TABLES = ("code_rule_approvals", "code_rules")
SHA = "a" * 64


def _load():
    spec = importlib.util.spec_from_file_location("rev_c0de5a11e004", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine():
    url = sync_test_url().replace("+psycopg2", "")
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_040_code_rules_migration"
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
                conn.execute(text('CREATE DATABASE "' + name + '"'))
    finally:
        admin.dispose()
    return create_engine(server + "/" + name)


def _drop(c):
    for t in (*NEW_TABLES, "agent_approval_requests"):
        c.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
    c.execute(text("DROP FUNCTION IF EXISTS code_rules_append_only() CASCADE"))


def _run(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


@pytest.fixture
def engine():
    engine = _engine()
    with engine.begin() as c:
        _drop(c)
        c.execute(text("CREATE TABLE agent_approval_requests (id VARCHAR(36) PRIMARY KEY)"))
        c.execute(text("INSERT INTO agent_approval_requests VALUES ('req-1')"))
    try:
        yield engine
    finally:
        with engine.begin() as c:
            _drop(c)
        engine.dispose()


def _upgrade(engine):
    rev = _load()
    with engine.begin() as c:
        _run(c, rev.upgrade)
    return rev


def _document(c):
    c.execute(text("INSERT INTO code_rules (code_sha256, canonical, runner_contract, functions, source, created_by, "
                   "created_at) VALUES (:s, '\\x7b7d', 'midataworks.runner/v1', '[]', 'inline', 'operator', now())"),
              {"s": SHA})


def _decision(c, origin="operator", action="approve", reason="ok", changed_by="operator"):
    c.execute(text("INSERT INTO code_rule_approvals (code_sha256, action, reason, changed_by, origin, "
                   "approval_request_id, created_at) VALUES (:s, :a, :r, :b, :o, 'req-1', now())"),
              {"s": SHA, "a": action, "r": reason, "b": changed_by, "o": origin})


def test_it_revises_the_s3_head():
    rev = _load()
    assert (rev.revision, rev.down_revision) == ("c0de5a11e004", "5ed6d79015d2")


def test_round_trip(engine):
    rev = _upgrade(engine)
    assert set(NEW_TABLES) <= set(inspect(engine).get_table_names())
    with engine.begin() as c:
        _run(c, rev.downgrade)
    assert not set(NEW_TABLES) & set(inspect(engine).get_table_names())
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM pg_proc WHERE proname = 'code_rules_append_only'")).scalar() == 0
    with engine.begin() as c:
        _run(c, rev.upgrade)
    assert set(NEW_TABLES) <= set(inspect(engine).get_table_names())


def test_the_database_refuses_an_agent_decision(engine):
    """C.6: `origin` is CHECKed = 'operator'. Whatever a caller sends, a decision row is the operator's."""
    _upgrade(engine)
    with engine.begin() as c:
        _document(c)
        _decision(c)
    for bad in ({"origin": "agent"}, {"origin": "mcp"}, {"reason": "  "}, {"changed_by": ""}, {"action": "grant"}):
        with pytest.raises(DBAPIError), engine.begin() as c:
            _decision(c, **bad)
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM code_rule_approvals")).scalar() == 1


def test_both_tables_are_append_only(engine):
    _upgrade(engine)
    with engine.begin() as c:
        _document(c)
        _decision(c)
    for statement in ("UPDATE code_rules SET created_by = 'x'", "DELETE FROM code_rules",
                      "UPDATE code_rule_approvals SET action = 'revoke'", "DELETE FROM code_rule_approvals"):
        with pytest.raises(DBAPIError, match="append-only"), engine.begin() as c:
            c.execute(text(statement))


def test_the_frozen_ddl_equals_the_models():
    assert tuple(_load().APPEND_ONLY_DDL) == tuple(models.APPEND_ONLY_DDL)
