"""Feature 034's two Alembic revisions round-trip up/down/up (tasks 2.2, 3.4).

Each revision is replayed by its OWN functions on a scratch database (named after this run's
`DATABASE_URL_SYNC`, so parallel agents never share it), over a stand-in table without the
column. The column must appear (with the ORM's type and nullability), vanish and reappear. `test_orm_matches_migrated_schema` separately compares the ORM with a
database migrated to head (`SCHEMA_CHECK_DATABASE_URL`).

Two revisions, one per phase (FTID TI7), so each phase reverts alone.
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
VERSIONS = BACKEND / "alembic" / "versions"

REVISIONS = {
    # revision id: (file, down_revision, table, column)
    "c34a0d1e7f01": ("c34a0d1e7f01_approval_consumed_at.py", "e2e024a11e5b",
                     "agent_approval_requests", "consumed_at"),
    "c34b5e2a9d02": ("c34b5e2a9d02_probe_dataset_version.py", "c34a0d1e7f01",
                     "probe_monitor_datasets", "dataset_version"),
}


def _load(revision_id: str):
    filename, down, _table, _column = REVISIONS[revision_id]
    spec = importlib.util.spec_from_file_location(f"rev_{revision_id}", VERSIONS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.revision == revision_id and module.down_revision == down
    return module


def _scratch_engine(suffix: str):
    url = os.environ.get("DATABASE_URL_SYNC") or (
        "postgresql://postgres:devpassword@localhost:5432/mistudio_test"
    )
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_" + suffix
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"),
                                {"n": name}).scalar():
                conn.execute(text('CREATE DATABASE "' + name + '"'))
    finally:
        admin.dispose()
    return create_engine(server + "/" + name)


def _run(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


def _has(conn, table, column):
    return column in {c["name"] for c in sa.inspect(conn).get_columns(table)}


@pytest.mark.parametrize("revision_id", sorted(REVISIONS))
def test_the_revision_round_trips_up_down_up(revision_id):
    import src.models  # noqa: F401 — register every table on the metadata
    from src.core.database import Base

    module = _load(revision_id)
    _file, _down, table, column = REVISIONS[revision_id]
    engine = _scratch_engine("dw034_migrations")
    try:
        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
            # The table as the ORM declares it minus this revision's column and its foreign
            # keys (a scratch database holds only this table): the revision must add exactly
            # the column the ORM carries.
            orm_column = Base.metadata.tables[table].c[column]
            conn.execute(text(f'CREATE TABLE "{table}" (id VARCHAR(36) PRIMARY KEY)'))
            assert not _has(conn, table, column)
            _run(conn, module.upgrade)
            added = {c["name"]: c for c in sa.inspect(conn).get_columns(table)}[column]
            assert added["nullable"] is True and orm_column.nullable is True
            assert added["type"].compile(conn.dialect) == orm_column.type.compile(
                conn.dialect
            ), (added["type"], orm_column.type)
            assert _has(conn, table, column)
            _run(conn, module.downgrade)
            assert not _has(conn, table, column)
            _run(conn, module.upgrade)
            assert _has(conn, table, column)
            conn.execute(text(f'DROP TABLE "{table}" CASCADE'))
    finally:
        engine.dispose()


def test_the_revisions_form_ONE_line_to_head():
    """The newest 034 revision is the script head, and each revises the one before it."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "alembic"))
    script = ScriptDirectory.from_config(config)
    assert len(script.get_heads()) == 1, script.get_heads()
    newest = max(REVISIONS, key=lambda r: list(REVISIONS).index(r))
    assert script.get_current_head() == newest
