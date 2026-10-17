"""The migrated schema and the ORM models must describe the same database.

WHY THIS EXISTS

Production is built by the migrations; the unit suite builds its schema from the
models with ``create_all``. On 2026-09-13 a brand-new production database (117
migrations replayed from empty) was compared with the models and ``alembic check``
reported 382 differences — index names, comments, server defaults, types, a
nullable column, the partition tables, and ``task_queue``, whose model env.py never
registered. Every test in the suite ran against a schema production did not have.

TWO GUARDS, because each is blind where the other sees:

* G1 ``test_alembic_drift_matches_the_ratchet`` — Alembic's own comparison (what
  ``alembic check`` sees) against ``alembic_drift_ratchet.json``. The ratchet may only
  SHRINK: a difference not listed is new drift and fails; a listed difference that no
  longer occurs also fails until its line is deleted. When the ratchet is empty it is
  deleted and this becomes "no differences at all".
* G2 ``test_the_orm_builds_the_same_schema_as_the_migrations`` — a pg_catalog snapshot
  of a ``create_all`` database against the migrated one. It sees what Alembic cannot:
  partial-index predicates, GIN, enum labels, CHECK constraints, triggers, partitions.
  It is a strict xfail until the reconciliation lands, so it cannot start passing
  without someone noticing.

WIRING: env.py and these tests share ``src.db.alembic_support``; the AST test pins
env.py's use of it and the subprocess test pins that alembic's own interpreter
registers every model the application does.

The migrated database is ``SCHEMA_CHECK_DATABASE_URL`` (CI migrates
``mistudio_schema_check`` before the suite). It must be at head; a stale local
database fails loudly rather than producing a confusing ratchet diff.
"""

import ast
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa

from alembic.config import Config
from alembic.script import ScriptDirectory
from src.db.schema_parity import alembic_drift, create_orm_schema, diff, snapshot

BACKEND = Path(__file__).resolve().parents[2]
RATCHET = Path(__file__).with_name("alembic_drift_ratchet.json")
PROBE_TABLE = "schema_guard_probe"


def _migrated_url() -> str:
    return os.environ.get(
        "SCHEMA_CHECK_DATABASE_URL",
        os.environ.get(
            "DATABASE_URL_SYNC", "postgresql://postgres:devpassword@localhost:5432/mistudio"
        ),
    )


def _script_head() -> str:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "alembic"))
    return ScriptDirectory.from_config(config).get_current_head()


def _assert_at_head(conn: sa.engine.Connection) -> None:
    current = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalars().all()
    head = _script_head()
    assert current == [head], (
        f"{sa.engine.make_url(_migrated_url()).database} is at {current}, not head {head}. "
        "Rebuild it: drop, create, `alembic upgrade head` — a stale database makes every "
        "comparison below meaningless."
    )


@pytest.fixture(scope="module")
def migrated_engine():
    engine = sa.create_engine(_migrated_url())
    try:
        yield engine
    finally:
        engine.dispose()


def test_alembic_drift_matches_the_ratchet(migrated_engine):
    with migrated_engine.connect() as conn:
        _assert_at_head(conn)
        current = set(alembic_drift(conn))
    recorded = set(json.loads(RATCHET.read_text()))

    new = sorted(current - recorded)
    assert not new, (
        f"{len(new)} NEW difference(s) between the migrated schema and the models. Fix the "
        f"model or add the migration in the same commit — do not add them to "
        f"{RATCHET.name}. First 25:\n" + "\n".join(new[:25])
    )
    fixed = sorted(recorded - current)
    assert not fixed, (
        f"{len(fixed)} difference(s) in {RATCHET.name} no longer occur. Delete those lines so "
        "the ratchet only shrinks. First 25:\n" + "\n".join(fixed[:25])
    )


def test_the_drift_comparison_can_see_a_difference(migrated_engine):
    """NEGATIVE CONTROL: a comparison that cannot report a difference passes forever."""
    probe = sa.MetaData()
    sa.Table(PROBE_TABLE, probe, sa.Column("id", sa.Integer, primary_key=True))
    with migrated_engine.connect() as conn:
        signatures = alembic_drift(conn, probe)
    assert f"add_table:{PROBE_TABLE}" in signatures
    assert "remove_table:features" in signatures


def test_the_ratchet_is_sorted_and_unique():
    entries = json.loads(RATCHET.read_text())
    assert entries == sorted(set(entries)), f"keep {RATCHET.name} sorted and free of duplicates"


def test_alembic_registers_every_model_the_app_registers():
    """alembic runs env.py in a fresh interpreter, where nothing else imported the models.

    Registration that only works because the test process imported ``src.main`` first
    is the exact failure that hid ``task_queue`` from every comparison.
    """
    import src.main  # noqa: F401  the application's full model registration
    from src.core.database import Base

    code = (
        "import json; from src.db.alembic_support import target_metadata; "
        "print(json.dumps(sorted(target_metadata().tables)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    alembic_tables = set(json.loads(result.stdout.strip().splitlines()[-1]))
    missing = sorted(set(Base.metadata.tables) - alembic_tables)
    assert not missing, f"alembic's metadata lacks tables the application registers: {missing}"


def test_env_py_uses_the_shared_metadata_and_compare_options():
    tree = ast.parse((BACKEND / "alembic" / "env.py").read_text())

    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "target_metadata" for t in node.targets)
    ]
    assert len(assigns) == 1, "env.py must assign target_metadata exactly once"
    value = assigns[0].value
    assert isinstance(value, ast.Call) and getattr(value.func, "id", None) == (
        "load_target_metadata"
    ), "env.py's target_metadata must come from src.db.alembic_support.target_metadata()"

    configures = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "configure"
    ]
    assert len(configures) == 2, "expected the offline and online context.configure calls"
    for call in configures:
        assert any(
            kw.arg is None and isinstance(kw.value, ast.Name) and kw.value.id == "COMPARE_OPTIONS"
            for kw in call.keywords
        ), f"context.configure at line {call.lineno} must pass **COMPARE_OPTIONS"
        assert any(kw.arg == "target_metadata" for kw in call.keywords)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "P1a-P1c of the Alembic plan reconcile the models with the migrations; strict so "
        "it cannot quietly start passing"
    ),
)
def test_the_orm_builds_the_same_schema_as_the_migrations(migrated_engine):
    url = sa.engine.make_url(_migrated_url())
    scratch = f"orm_parity_{uuid.uuid4().hex[:10]}"
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{scratch}"'))
    orm_engine = sa.create_engine(url.set(database=scratch))
    try:
        with orm_engine.begin() as conn:
            create_orm_schema(conn)
        with orm_engine.connect() as conn:
            orm = snapshot(conn)
        with migrated_engine.connect() as conn:
            _assert_at_head(conn)
            migrated = snapshot(conn)
    finally:
        orm_engine.dispose()
        with admin.connect() as conn:
            conn.execute(
                sa.text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :name AND pid <> pg_backend_pid()"
                ),
                {"name": scratch},
            )
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{scratch}"'))
        admin.dispose()

    differences = diff(migrated, orm, "migrated", "orm")
    assert not differences, f"{len(differences)} difference(s); first 40:\n" + "\n".join(
        differences[:40]
    )
