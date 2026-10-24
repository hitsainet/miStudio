"""The behaviour catalogue's revision (e7c1a4b9d2f6): round trip, frozen copies, and Part A untouched.

Replayed by its OWN functions on a scratch database named after this run's DATABASE_URL_SYNC (so
parallel agents never share it), over stand-in tables carrying exactly the columns it touches. The
shipped schema against the ORM is `test_orm_matches_migrated_schema`'s job.

What must hold:
* FR-3 / D25 — the revision never alters or rewrites Part A's three ``behaviour`` columns. Read from the
  file's ``op.*`` calls with ``ast``, not a regex (memory: source-scrape-guards-fail-open): a call that
  names one of those tables with ``alter_column``/``execute``/``drop_column`` fails here, and the guard
  is proven to see calls by a positive control on ``create_index``.
* the slug regex and both trigger bodies written into the migration equal the model module's constants;
* upgrade then downgrade leaves the stand-in tables exactly as they were.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import pathlib

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from src.models import probe_behaviour as models
from src.services.judge_prompt_forms import SLUG_PATTERN

BACKEND = pathlib.Path(__file__).resolve().parents[2]
FILE = BACKEND / "alembic" / "versions" / "e7c1a4b9d2f6_behaviour_catalogue.py"
PART_A_TABLES = ("probe_judge_prompts", "probe_monitor_runs", "probe_monitors")
NEW_TABLES = (
    "probe_behaviour_history",
    "probe_behaviour_view_links",
    "probe_behaviour_probe_roles",
    "probe_behaviour_versions",
    "probe_behaviours",
)
STAND_INS = ("probe_monitor_datasets", "probe_monitors", "probe_monitor_runs")


def _load():
    spec = importlib.util.spec_from_file_location("rev_e7c1a4b9d2f6", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _op_calls(source: str):
    """Every ``op.<name>(…)`` call in the file: (name, first positional string argument or None)."""
    calls = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "op"
        ):
            first = node.args[0] if node.args else None
            calls.append((node.func.attr, first.value if isinstance(first, ast.Constant) else None, node))
    return calls


class TestPartAColumnsAreNeverTouched:
    def test_the_guard_sees_the_files_op_calls(self):
        """Positive control: the walk finds the index this revision DOES create on probe_monitors."""
        calls = _op_calls(FILE.read_text())
        assert ("create_index", "ix_probe_monitors_behaviour") in {(n, a) for n, a, _ in calls}

    def test_no_op_alters_executes_on_or_drops_from_part_a_tables(self):
        source = FILE.read_text()
        offending = []
        for name, first, node in _op_calls(source):
            if name in {"alter_column", "drop_column", "add_column"} and first in PART_A_TABLES:
                offending.append(ast.unparse(node))
        # `op.execute(statement)` takes a variable here, so read every STRING in the file as SQL: none
        # may write to a Part A table.
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                sql = " ".join(node.value.lower().split())
                for table in PART_A_TABLES:
                    if f"update {table} " in sql or f"alter table {table} " in sql:
                        offending.append(sql[:120])
        assert not offending, (
            "the catalogue revision changes Part A's columns, which D25 forbids — adoption is a join: "
            f"{offending}"
        )


class TestFrozenCopies:
    def test_the_slug_regex_is_the_librarys(self):
        assert _load().SLUG_REGEX == models.SLUG_REGEX == SLUG_PATTERN

    def test_the_trigger_bodies_are_the_models(self):
        migration = _load()
        assert migration.VERSIONS_DDL == models.VERSIONS_DDL
        assert migration.HISTORY_DDL == models.HISTORY_DDL


def _engine():
    url = os.environ.get("DATABASE_URL_SYNC") or "postgresql://postgres:devpassword@localhost:5432/mistudio_test"
    url = url.replace("+psycopg2", "")
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_behaviour_migration"
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
                conn.execute(text('CREATE DATABASE "' + name + '"'))
    finally:
        admin.dispose()
    return create_engine(server + "/" + name)


def _drop(c):
    for t in NEW_TABLES + STAND_INS:
        c.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
    c.execute(text("DROP FUNCTION IF EXISTS probe_behaviour_versions_append_only() CASCADE"))
    c.execute(text("DROP FUNCTION IF EXISTS probe_behaviour_history_append_only() CASCADE"))


def _run(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


@pytest.fixture
def engine():
    engine = _engine()
    with engine.begin() as c:
        _drop(c)
        c.execute(text("CREATE TABLE probe_monitor_runs (id VARCHAR(36) PRIMARY KEY, behaviour VARCHAR(64))"))
        c.execute(text("CREATE TABLE probe_monitors (id VARCHAR(36) PRIMARY KEY, behaviour VARCHAR(64))"))
        c.execute(text("CREATE TABLE probe_monitor_datasets (id VARCHAR(36) PRIMARY KEY)"))
        c.execute(text("INSERT INTO probe_monitors VALUES ('pm_1', 'pii')"))
    try:
        yield engine
    finally:
        with engine.begin() as c:
            _drop(c)
        engine.dispose()


def _shape(engine):
    found = inspect(engine)
    return {
        table: (
            sorted(c["name"] for c in found.get_columns(table)),
            sorted(i["name"] for i in found.get_indexes(table)),
        )
        for table in STAND_INS
    }


def test_round_trip_leaves_the_tables_as_they_were(engine):
    migration = _load()
    before = _shape(engine)
    with engine.begin() as c:
        _run(c, migration.upgrade)
    found = inspect(engine)
    assert set(NEW_TABLES) <= set(found.get_table_names())
    assert {"behaviour", "positive_meaning", "positive_labels_confirmed"} <= {
        c["name"] for c in found.get_columns("probe_monitor_datasets")
    }
    with engine.begin() as c:
        # Part A's value on the probe is untouched by the upgrade.
        assert c.execute(text("SELECT behaviour FROM probe_monitors WHERE id = 'pm_1'")).scalar() == "pii"
    with engine.begin() as c:
        _run(c, migration.downgrade)
    assert _shape(engine) == before
    assert not set(NEW_TABLES) & set(inspect(engine).get_table_names())


def test_the_migrated_triggers_refuse(engine):
    migration = _load()
    with engine.begin() as c:
        _run(c, migration.upgrade)
        c.execute(text(
            "INSERT INTO probe_behaviours (key, current_version, created_by, created_at, updated_at) "
            "VALUES ('pii', 1, 'operator', now(), now())"
        ))
        c.execute(text(
            "INSERT INTO probe_behaviour_versions (key, version, name, positive_meaning, created_by, created_at) "
            "VALUES ('pii', 1, 'PII', 'has personal data', 'operator', now())"
        ))
        c.execute(text(
            "INSERT INTO probe_behaviour_history (key, subject_kind, field, actor, created_at) "
            "VALUES ('pii', 'behaviour', 'created', 'operator', now())"
        ))
    for statement in (
        "UPDATE probe_behaviour_versions SET name = 'x'",
        "DELETE FROM probe_behaviour_versions",
        "UPDATE probe_behaviour_history SET actor = 'x'",
        "DELETE FROM probe_behaviour_history",
        "INSERT INTO probe_behaviours (key, current_version, created_by, created_at, updated_at) "
        "VALUES ('Bad Key', 1, 'operator', now(), now())",
    ):
        with pytest.raises(Exception):
            with engine.begin() as c:
                c.execute(text(statement))
    # Deleting the behaviour itself cascades to its versions: the trigger allows it once the parent is gone.
    with engine.begin() as c:
        c.execute(text("DELETE FROM probe_behaviours WHERE key = 'pii'"))
        assert c.execute(text("SELECT count(*) FROM probe_behaviour_versions")).scalar() == 0
        assert c.execute(text("SELECT count(*) FROM probe_behaviour_history")).scalar() == 1
