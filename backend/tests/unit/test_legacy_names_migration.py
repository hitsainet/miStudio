"""042 FTASKS 3.2: the legacy-name revision upgrades, downgrades and upgrades again (FTID §4).

Replayed by the revision's OWN functions on a scratch database named after this run's test database (so
parallel runs never share it). The URL comes from the one helper (`tests/support/test_databases.py`),
never from a literal (R16). `test_orm_matches_migrated_schema.py` compares the shipped schema with the ORM.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from tests.support.test_databases import sync_test_url

BACKEND = pathlib.Path(__file__).resolve().parents[2]
FILE = BACKEND / "alembic" / "versions" / "e011a7c3d9f2_legacy_name_uses.py"
TABLES = ("legacy_name_uses", "legacy_name_reporters")


def _load():
    spec = importlib.util.spec_from_file_location("rev_e011a7c3d9f2", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def engine():
    url = sync_test_url().replace("+psycopg2", "")
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_legacy_names_migration"
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
                conn.execute(text('CREATE DATABASE "' + name + '"'))
    finally:
        admin.dispose()
    eng = create_engine(server + "/" + name)
    with eng.begin() as c:
        for t in TABLES:
            c.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
    try:
        yield eng
    finally:
        with eng.begin() as c:
            for t in TABLES:
                c.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
        eng.dispose()


def _run(engine, step):
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            step()


def _tables(engine) -> set[str]:
    return set(inspect(engine).get_table_names()) & set(TABLES)


def test_upgrade_downgrade_upgrade(engine):
    rev = _load()
    assert rev.down_revision == "e010d84a7c31"
    _run(engine, rev.upgrade)
    assert _tables(engine) == set(TABLES)
    insp = inspect(engine)
    uses = {c["name"]: c for c in insp.get_columns("legacy_name_uses")}
    assert set(uses) == {"kind", "old", "new", "first_seen", "last_seen"}
    assert insp.get_pk_constraint("legacy_name_uses")["constrained_columns"] == ["kind", "old"]
    assert isinstance(uses["last_seen"]["type"], sa.DateTime) and uses["last_seen"]["type"].timezone
    assert insp.get_pk_constraint("legacy_name_reporters")["constrained_columns"] == ["reporter"]
    # The kind check constraint refuses anything but env / host / category.
    with pytest.raises(sa.exc.IntegrityError):
        with engine.begin() as c:
            c.execute(text("INSERT INTO legacy_name_uses VALUES ('header','X','Y',now(),now())"))
    _run(engine, rev.downgrade)
    assert _tables(engine) == set()
    _run(engine, rev.upgrade)
    assert _tables(engine) == set(TABLES)
