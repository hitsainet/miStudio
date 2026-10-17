"""The ORM and the migrated database must describe the same schema.

WHY THIS EXISTS

`Base.metadata.create_all()` builds the unit-test schema from the ORM. The
migrations build production. Nothing compared them, and they diverged — in both
directions, on a single table:

    features — ORM declared 4 foreign keys; the database had 2
             — the database had uq_features_extraction_neuron; the ORM had none

So the suite ran against a schema that had constraints production lacked, and
lacked one production had. Every consequence was structurally unobservable:

  * `_cache_analysis` blind-INSERTed against a unique constraint the test schema
    did not have, so it passed in CI and 500'd in production once the 7-day
    cache expiry made the row invisible to the read but still present in the
    table (MIS-E2E-030 — predicted from source, then hit by a user).
  * Flipping every CASCADE on `features` to RESTRICT left 211 tests green
    (MIS-E2E-053), because nothing exercises a cascade.

Fixing the individual constraints does not fix that. This does: it fails when
the two schemas disagree, whichever way they drift.

SCOPE: constraints only — unique, foreign key, and primary key — because those
are what diverged and what the ORM can be authoritative about. Column presence
is already covered by `check_migrations.py`; column TYPE comparison is
deliberately out of scope (SQLAlchemy's type objects do not round-trip through
reflection cleanly enough to assert on without false positives).

NEGATIVE CONTROL: delete any `__table_args__` UniqueConstraint, or drop a
foreign key from a model, and `test_no_constraint_is_declared_but_missing` must
fail. Verified 2026-08-23.
"""

import os

import pytest
import sqlalchemy as sa

from src.core.database import Base
from src.models import *  # noqa: F401,F403 — registers every table on Base.metadata


# The audited tables. Deliberately not "every table": this guard is about the
# constraint divergence that was found and the tables it touched, and a
# whole-schema assertion would fail on pre-existing drift unrelated to it.
# Widen this list as tables are brought into conformance — do not weaken the
# assertions to accommodate a new one.
AUDITED_TABLES = ("features", "extraction_jobs", "feature_analysis_cache")


def _sync_url() -> str:
    """The MIGRATED database to reflect.

    `SCHEMA_CHECK_DATABASE_URL` takes precedence so CI can point this at a
    database that `conftest.async_engine` does not manage. That fixture is
    FUNCTION-scoped and runs `Base.metadata.drop_all` on teardown, so any
    database it touches has no tables by the time this module reflects it.

    Locally the two are already distinct — `DATABASE_URL` is `.../mistudio`, so
    conftest appends `_test` and works on `mistudio_test` while this reads
    `mistudio`. In CI both pointed at `mistudio_test`, conftest dropped the
    migrated schema out from under this module, and the guard failed with
    NoSuchTableError while passing locally. `test_reflects_a_database_conftest_
    does_not_manage` below makes that collision a loud failure rather than a
    confusing one.
    """
    return os.environ.get(
        "SCHEMA_CHECK_DATABASE_URL",
        os.environ.get(
            "DATABASE_URL_SYNC",
            "postgresql://postgres:devpassword@localhost:5432/mistudio",
        ),
    )


def _conftest_managed_db() -> str:
    """The database name `conftest.async_engine` drops and recreates."""
    url = os.environ.get("DATABASE_URL", "postgresql://localhost/mistudio")
    name = url.rsplit("/", 1)[-1].split("?")[0]
    return name if "test" in name else "mistudio_test"


def test_reflects_a_database_conftest_does_not_manage():
    """This module must not read the database the unit fixtures wipe.

    Not a skip — a FAILURE. If the two collide, every assertion in this file
    becomes order-dependent: it passes when it happens to run before the first
    fixture teardown and errors afterwards. A guard whose result depends on
    test ordering is worse than no guard, because the green runs look like
    evidence.
    """
    reflected = _sync_url().rsplit("/", 1)[-1].split("?")[0]
    managed = _conftest_managed_db()
    assert reflected != managed, (
        f"this module reflects {reflected!r}, which conftest.async_engine "
        f"drops after every test. Point SCHEMA_CHECK_DATABASE_URL at a "
        f"separate migrated database."
    )


@pytest.fixture(scope="module")
def inspector():
    """Reflect the MIGRATED database — not the create_all() one.

    This must not use the `async_session` fixture: that builds its schema from
    the ORM, which is the very thing under comparison. Reading it would compare
    the ORM against itself and pass unconditionally — the shape of guard this
    codebase has been bitten by four times.
    """
    engine = sa.create_engine(_sync_url())
    try:
        with engine.connect() as conn:
            yield sa.inspect(conn)
    finally:
        engine.dispose()


def _orm_fks(table: str) -> set:
    tbl = Base.metadata.tables[table]
    return {
        (
            list(c.columns)[0].name,
            c.elements[0].target_fullname,
            (c.ondelete or "NO ACTION").upper(),
        )
        for c in tbl.foreign_key_constraints
    }


def _db_fks(inspector, table: str) -> set:
    out = set()
    for fk in inspector.get_foreign_keys(table):
        cols = fk.get("constrained_columns") or []
        ref = fk.get("referred_columns") or []
        if not cols or not ref:
            continue
        ondelete = ((fk.get("options") or {}).get("ondelete") or "NO ACTION").upper()
        out.add((cols[0], f"{fk['referred_table']}.{ref[0]}", ondelete))
    return out


def _orm_uniques(table: str) -> set:
    tbl = Base.metadata.tables[table]
    return {
        tuple(sorted(c.name for c in con.columns))
        for con in tbl.constraints
        if isinstance(con, sa.UniqueConstraint)
    }


def _db_uniques(inspector, table: str) -> set:
    return {
        tuple(sorted(uc["column_names"]))
        for uc in inspector.get_unique_constraints(table)
    }


@pytest.mark.parametrize("table", AUDITED_TABLES)
def test_no_constraint_is_declared_but_missing(inspector, table):
    """ORM says it exists; the database must have it.

    This is the direction that produced MIS-E2E-033: three foreign keys declared
    on models and absent from the database, so `ondelete=` did nothing in
    production while the tests saw it working.
    """
    missing_fks = _orm_fks(table) - _db_fks(inspector, table)
    assert not missing_fks, (
        f"{table}: the ORM declares foreign keys the migrated database lacks: "
        f"{sorted(missing_fks)}. `ondelete=` is DDL-only — with no constraint in "
        f"the database it does nothing, while create_all() gives the tests one."
    )

    missing_uq = _orm_uniques(table) - _db_uniques(inspector, table)
    assert not missing_uq, (
        f"{table}: the ORM declares unique constraints the database lacks: "
        f"{sorted(missing_uq)}"
    )


@pytest.mark.parametrize("table", AUDITED_TABLES)
def test_no_constraint_exists_but_undeclared(inspector, table):
    """The database has it; the ORM must declare it.

    This is the direction that produced MIS-E2E-030/031: a unique constraint
    that existed only in a migration, so `create_all()` built a test schema
    without it and a blind INSERT could never fail in CI.
    """
    undeclared_uq = _db_uniques(inspector, table) - _orm_uniques(table)
    assert not undeclared_uq, (
        f"{table}: the database has unique constraints the ORM does not declare: "
        f"{sorted(undeclared_uq)}. create_all() will build a test schema without "
        f"them, so no test can observe a violation that production enforces."
    )

    undeclared_fk = _db_fks(inspector, table) - _orm_fks(table)
    assert not undeclared_fk, (
        f"{table}: the database has foreign keys the ORM does not declare: "
        f"{sorted(undeclared_fk)}"
    )
