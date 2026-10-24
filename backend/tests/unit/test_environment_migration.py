"""Feature 036's Alembic revision round-trips, folds `ollama_url`, and keeps history append-only (FTASKS 1.3).

Replayed by its OWN functions on a scratch database named after this run's ``DATABASE_URL_SYNC`` (so
parallel agents never share it), over a stand-in ``app_settings`` table carrying the row D4 moves.
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
FILE = BACKEND / "alembic" / "versions" / "b36e0a7c1d42_runtime_environment_settings.py"
TABLES = ("environment_setting_changes", "environment_overrides", "app_settings")


def _load():
    spec = importlib.util.spec_from_file_location("rev_b36e0a7c1d42", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine():
    url = os.environ.get("DATABASE_URL_SYNC") or "postgresql://postgres:devpassword@localhost:5432/mistudio_test"
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_env036_migration"
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
    c.execute(text("DROP FUNCTION IF EXISTS environment_setting_changes_append_only() CASCADE"))


def _run(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


@pytest.fixture
def engine():
    engine = _engine()
    with engine.begin() as c:
        _drop(c)
        c.execute(text(
            "CREATE TABLE app_settings (id UUID PRIMARY KEY, key VARCHAR(255) UNIQUE NOT NULL, "
            "value TEXT NOT NULL, is_sensitive BOOLEAN NOT NULL DEFAULT false, "
            "category VARCHAR(50) NOT NULL DEFAULT 'general')"
        ))
    try:
        yield engine
    finally:
        with engine.begin() as c:
            _drop(c)
        engine.dispose()


def _seed(engine, value):
    with engine.begin() as c:
        c.execute(text(
            "INSERT INTO app_settings (id, key, value, is_sensitive, category) VALUES "
            "(gen_random_uuid(), 'ollama_url', :v, false, 'endpoints'), "
            "(gen_random_uuid(), 'openai_compatible_model', 'm', false, 'endpoints')"
        ), {"v": value})


def test_identity():
    mod = _load()
    assert mod.revision == "b36e0a7c1d42" and mod.down_revision == "a35d0e1f2c3b"


def test_upgrade_folds_ollama_url_and_downgrade_restores_it(engine):
    mod = _load()
    _seed(engine, "http://llm.example:8001")
    with engine.begin() as c:
        _run(c, mod.upgrade)
    with engine.begin() as c:
        assert c.execute(text("SELECT key, value, updated_by, reason FROM environment_overrides")).all() == [
            ("ollama_url", "http://llm.example:8001", "migration", mod.FOLD_REASON)
        ]
        history = c.execute(text(
            "SELECT key, action, previous_value, previous_source, new_value, new_source, actor "
            "FROM environment_setting_changes"
        )).all()
        assert history == [("ollama_url", "set", None, "env", "http://llm.example:8001", "override", "migration")]
        keys = {r[0] for r in c.execute(text("SELECT key FROM app_settings"))}
        assert keys == {"openai_compatible_model"}, "the row left app_settings; the others stayed"

    with engine.begin() as c:
        _run(c, mod.downgrade)
    with engine.begin() as c:
        insp = sa.inspect(c)
        assert not insp.has_table("environment_overrides") and not insp.has_table("environment_setting_changes")
        assert c.execute(text("SELECT value FROM app_settings WHERE key = 'ollama_url'")).scalar() == (
            "http://llm.example:8001"
        )
    with engine.begin() as c:  # and up again
        _run(c, mod.upgrade)


def test_an_empty_row_is_dropped_not_carried(engine):
    mod = _load()
    _seed(engine, "   ")
    with engine.begin() as c:
        _run(c, mod.upgrade)
    with engine.begin() as c:
        assert c.execute(text("SELECT count(*) FROM environment_overrides")).scalar() == 0
        assert c.execute(text("SELECT count(*) FROM environment_setting_changes")).scalar() == 0
        assert c.execute(text("SELECT count(*) FROM app_settings WHERE key = 'ollama_url'")).scalar() == 0


def test_a_fresh_database_gets_no_override(engine):
    mod = _load()
    with engine.begin() as c:
        _run(c, mod.upgrade)
    with engine.begin() as c:
        assert c.execute(text("SELECT count(*) FROM environment_overrides")).scalar() == 0


@pytest.mark.parametrize("statement", [
    "UPDATE environment_setting_changes SET reason = 'rewritten'",
    "DELETE FROM environment_setting_changes",
])
def test_the_trigger_refuses_update_and_delete(engine, statement):
    mod = _load()
    _seed(engine, "http://llm.example:8001")
    with engine.begin() as c:
        _run(c, mod.upgrade)
    with pytest.raises(sa.exc.DBAPIError, match="append-only"):
        with engine.begin() as c:
            c.execute(text(statement))
    with engine.begin() as c:
        assert c.execute(text("SELECT count(*) FROM environment_setting_changes")).scalar() == 1
