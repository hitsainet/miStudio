"""The metric table's unique key includes the hook type, on REAL Postgres (review R1-A, A5).

A training trains one SAE per (layer, hook type) and logs one metric row per SAE per log
step. The key was (training_id, step, layer_idx), so a training over two hook types wrote
the same key twice at step 0 and failed. Migration d5a1f3c7e9b2 adds hook_type and the
unique expression index (training_id, step, layer_idx, COALESCE(hook_type, '')).

WHAT IS REAL. Postgres, the task's own log_metric, and the table as the MIGRATIONS built
it: columns, unique constraints and every index are read from the migrated database
(SCHEMA_CHECK_DATABASE_URL, at head) and replayed into a scratch database named after this
run's DATABASE_URL_SYNC, less the foreign key. The same matrix runs against the table the
ORM builds, and the two index definitions are compared as Postgres renders them, because
Alembic's autogenerate skips expression indexes: the drift ratchet cannot see this key.
The migration test runs the revision's own upgrade() and downgrade() over historical rows.

These fail, never skip, when Postgres is unreachable.

MUTATION CONTROLS (one line broken at a time, this file run, bytes restored, sha256 and the
working tree verified clean; the full table is in the A5 record):
  K1 the ORM index without COALESCE(hook_type, '')       -> the ORM parity test and the
                                                             [orm] matrix
  K2 the migration's index without the hook expression   -> the migration test
  K3 log_metric does not pass hook_type to the row        -> both matrices
  K4 the downgrade's dedup removed                        -> the migration test
  K5 the downgrade's IS NOT DISTINCT FROM back to =       -> the migration test

REVIEW ROUND 2 (R2-B; same procedure, record review_sae_remediation_R2_B_2026-09-15.md):
  K5  only the keep rule's FIRST disjunct back to =        -> SURVIVED: no 'residual' row sat beside
        a NULL-hook row, so the one rule that disjunct decides was never exercised. The
        migration test now adds one; re-run: red
  K5b every IS NOT DISTINCT FROM back to =                  -> red
  K1  re-run                                               -> red, including the new
        test_the_unit_suite_schema_builds_the_migrated_key (conftest's create_all)
"""

import contextlib
import importlib.util
import logging
import os
import re
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateIndex, CreateTable

from src.models.training_metric import TrainingMetric
from src.workers import base_task, training_tasks

BACKEND = Path(__file__).resolve().parents[2]
REVISION_FILE = BACKEND / "alembic" / "versions" / "d5a1f3c7e9b2_training_metrics_hook_type.py"
NEW_KEY = "uq_training_metrics_tid_step_layer_hook"
OLD_KEY = "uq_training_metrics_tid_step_layer"
TABLE = "training_metrics"


# ── databases ────────────────────────────────────────────────────────────────


def _migrated_url() -> str:
    return (
        os.environ.get("SCHEMA_CHECK_DATABASE_URL")
        or os.environ.get("DATABASE_URL_SYNC")
        or "postgresql://postgres:devpassword@localhost:5432/mistudio"
    )


def scratch_engine(suffix: str):
    """An engine on a scratch database named after this run's DATABASE_URL_SYNC."""
    url = os.environ.get("DATABASE_URL_SYNC") or "postgresql://postgres:devpassword@localhost:5432/mistudio"
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_" + suffix
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
                conn.execute(text('CREATE DATABASE "' + name + '"'))
    finally:
        admin.dispose()
    return create_engine(server + "/" + name)


def _script_head() -> str:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "alembic"))
    return ScriptDirectory.from_config(config).get_current_head()


def _indexdefs(conn) -> dict:
    rows = conn.execute(
        text("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = :t"),
        {"t": TABLE},
    ).all()
    return {name: definition for name, definition in rows}


def _unique_constraints(conn) -> dict:
    return {uc["name"]: uc["column_names"] for uc in sa.inspect(conn).get_unique_constraints(TABLE)}


def _migrated_shape():
    """Columns, unique constraints and indexes of training_metrics as the migrations built it."""
    engine = create_engine(_migrated_url())
    try:
        with engine.connect() as conn:
            current = conn.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
            head = _script_head()
            assert current == [head], (
                str(engine.url.database) + " is at " + str(current) + ", not head " + head
                + ": migrate it (SCHEMA_CHECK_DATABASE_URL) before trusting any comparison here"
            )
            inspector = sa.inspect(conn)
            columns = inspector.get_columns(TABLE)
            primary = set(inspector.get_pk_constraint(TABLE)["constrained_columns"])
            return columns, primary, _unique_constraints(conn), _indexdefs(conn)
    finally:
        engine.dispose()


def replay_migrated_table(conn) -> None:
    """Create training_metrics on conn as the migrations built it, less the foreign key."""
    columns, primary, uniques, indexdefs = _migrated_shape()
    table = sa.Table(
        TABLE,
        sa.MetaData(),
        *[
            sa.Column(
                c["name"],
                sa.BigInteger if c["name"] in primary else c["type"],
                primary_key=c["name"] in primary,
                autoincrement=c["name"] in primary,
                nullable=c["nullable"],
                # The migrated server defaults (timestamp's now()); the key's own
                # sequence default comes from BIGSERIAL instead.
                server_default=(
                    sa.text(c["default"]) if c.get("default") is not None and c["name"] not in primary else None
                ),
            )
            for c in columns
        ],
        *[sa.UniqueConstraint(*cols, name=name) for name, cols in uniques.items()],
    )
    conn.execute(text("DROP TABLE IF EXISTS " + TABLE))
    conn.execute(CreateTable(table))
    for name, definition in indexdefs.items():
        if name.endswith("_pkey"):
            continue
        conn.execute(text(definition))


def _replay_orm_table(conn) -> None:
    table = TrainingMetric.__table__
    ddl = str(CreateTable(table).compile(conn))
    ddl = re.sub(r",\s*FOREIGN KEY\s*\([^)]*\)\s*REFERENCES\s+\w+\s*\([^)]*\)(\s+ON DELETE \w+)?", "", ddl)
    assert "FOREIGN KEY" not in ddl, ddl
    conn.execute(text("DROP TABLE IF EXISTS " + TABLE))
    conn.execute(text(ddl))
    for index in table.indexes:
        conn.execute(CreateIndex(index))


def task_session_factory(engine):
    """A get_sync_db over engine with the production session semantics."""
    make = sessionmaker(bind=engine, expire_on_commit=False)

    @contextlib.contextmanager
    def session():
        db = make()
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    return session


@pytest.fixture(params=["migrated", "orm"])
def metrics_table(request, monkeypatch):
    engine = scratch_engine("hook_key_" + request.param)
    with engine.begin() as conn:
        (replay_migrated_table if request.param == "migrated" else _replay_orm_table)(conn)
    session = task_session_factory(engine)
    monkeypatch.setattr(base_task, "get_sync_db", session)
    try:
        yield session
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS " + TABLE))
        engine.dispose()


def _log(training_id, step, layer_idx, hook_type, loss=1.0):
    training_tasks.train_sae_task.log_metric(
        training_id=training_id, step=step, loss=loss, layer_idx=layer_idx, hook_type=hook_type,
    )


AGGREGATE = -1000000000  # how the reads below spell a NULL layer_idx


# ── the key ──────────────────────────────────────────────────────────────────


def test_the_migrated_key_includes_the_hook_type_and_the_old_key_is_gone():
    _, _, uniques, indexdefs = _migrated_shape()
    assert OLD_KEY not in uniques, uniques
    assert NEW_KEY in indexdefs, sorted(indexdefs)
    definition = indexdefs[NEW_KEY]
    assert definition.startswith("CREATE UNIQUE INDEX " + NEW_KEY + " ON public." + TABLE), definition
    assert "(training_id, step, layer_idx, COALESCE(hook_type," in definition, definition


def test_the_orm_declares_the_key_the_migrations_built():
    """Rendered by Postgres on both sides. Alembic's autogenerate skips expression
    indexes, so the drift ratchet in test_schema_guards cannot see this key disagree."""
    _, _, _, migrated = _migrated_shape()
    engine = scratch_engine("hook_key_parity")
    try:
        with engine.begin() as conn:
            _replay_orm_table(conn)
            orm = _indexdefs(conn)
            conn.execute(text("DROP TABLE IF EXISTS " + TABLE))
    finally:
        engine.dispose()
    assert orm.get(NEW_KEY) == migrated[NEW_KEY]


async def test_the_unit_suite_schema_builds_the_migrated_key(async_engine):
    """The unit suite does not replay DDL: conftest's `async_engine` runs
    `Base.metadata.create_all`. The key it builds there, read back from Postgres, is the
    migrated one (review R2-B; the parity test above renders the ORM index itself)."""
    _, _, uniques_migrated, migrated = _migrated_shape()
    async with async_engine.connect() as conn:
        built = await conn.run_sync(_indexdefs)
        uniques_built = await conn.run_sync(_unique_constraints)
    assert built.get(NEW_KEY) == migrated[NEW_KEY], (built.get(NEW_KEY), migrated[NEW_KEY])
    assert OLD_KEY not in uniques_built and OLD_KEY not in built, (uniques_built, sorted(built))


def test_two_hooks_on_a_layer_are_two_rows_and_every_old_guarantee_holds(metrics_table):
    """Through the task's own write path, on the migrated table and on the ORM's."""
    for hook in ("residual", "mlp"):
        _log("train_multi", 0, 3, hook)
        _log("train_multi", 0, -4, hook)  # held-out, per hook
    _log("train_multi", 0, None, None)
    _log("train_multi", 0, None, None)  # aggregated rows were never unique, and still are not

    with pytest.raises(IntegrityError, match=NEW_KEY):
        _log("train_multi", 0, 3, "mlp")
    with pytest.raises(IntegrityError, match=NEW_KEY):
        _log("train_multi", 0, -4, "residual")

    # A row with no hook keeps the old (training, step, layer) guarantee, so a call
    # site that forgets the hook on a two-hook run fails loudly instead of quietly.
    _log("train_legacy", 0, 3, None)
    with pytest.raises(IntegrityError, match=NEW_KEY):
        _log("train_legacy", 0, 3, None)

    with metrics_table() as db:
        rows = sorted(
            (m.training_id, m.step, AGGREGATE if m.layer_idx is None else m.layer_idx, m.hook_type or "")
            for m in db.query(TrainingMetric).all()
        )
    assert rows == sorted([
        ("train_multi", 0, 3, "residual"), ("train_multi", 0, 3, "mlp"),
        ("train_multi", 0, -4, "residual"), ("train_multi", 0, -4, "mlp"),
        ("train_multi", 0, AGGREGATE, ""), ("train_multi", 0, AGGREGATE, ""),
        ("train_legacy", 0, 3, ""),
    ])


# ── the migration ────────────────────────────────────────────────────────────


def _revision():
    spec = importlib.util.spec_from_file_location("a5_revision_d5a1f3c7e9b2", REVISION_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.revision == "d5a1f3c7e9b2" and module.down_revision == "c4e8b2d6f1a9"
    return module


def _run(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


def _insert(conn, training_id, step, layer_idx, loss, hook_type=None, with_hook=True):
    columns = "training_id, step, layer_idx, loss" + (", hook_type" if with_hook else "")
    values = ":t, :s, :l, :loss" + (", :h" if with_hook else "")
    conn.execute(
        text("INSERT INTO " + TABLE + " (" + columns + ") VALUES (" + values + ")"),
        {"t": training_id, "s": step, "l": layer_idx, "loss": loss, "h": hook_type},
    )


def _all(conn, with_hook=True):
    hook = "COALESCE(hook_type, '')" if with_hook else "''"
    sql = (
        "SELECT training_id, step, COALESCE(layer_idx, " + str(AGGREGATE) + "), "
        + hook + ", loss FROM " + TABLE
    )
    return sorted(tuple(row) for row in conn.execute(text(sql)).all())


def _has_column(conn, name):
    return name in {c["name"] for c in sa.inspect(conn).get_columns(TABLE)}


def test_the_revision_upgrades_historical_rows_and_downgrades_cleanly(caplog):
    """The revision's own functions over the table as it stood at c4e8b2d6f1a9.

    The starting table is the migrated head's, taken back one revision by this
    revision's downgrade, then checked to be exactly the old shape: the old key on
    (training_id, step, layer_idx) and no hook_type column.
    """
    revision = _revision()
    engine = scratch_engine("hook_key_migration")
    try:
        with engine.begin() as conn:
            replay_migrated_table(conn)
            _run(conn, revision.downgrade)
            assert not _has_column(conn, "hook_type")
            assert _unique_constraints(conn) == {OLD_KEY: ["training_id", "step", "layer_idx"]}
            assert NEW_KEY not in _indexdefs(conn)

            # HISTORICAL ROWS: a duplicated aggregate (a resume before the discard wrote
            # those), per-layer, held-out and a legacy spliced-CE row.
            history = [
                ("train_hist", 0, None, 1.0), ("train_hist", 0, None, 1.0),
                ("train_hist", 0, 3, 1.1), ("train_hist", 0, 4, 1.2),
                ("train_hist", 0, -4, 1.3), ("train_hist", 0, -5, 1.4),
                ("train_hist", 100, None, 0.9), ("train_hist", 100, 3, 0.8),
                ("train_hist", 100, -1003, 2.5),
            ]
            for row in history:
                _insert(conn, *row, with_hook=False)
            before = _all(conn, with_hook=False)

            _run(conn, revision.upgrade)
            assert _has_column(conn, "hook_type")
            assert OLD_KEY not in _unique_constraints(conn)
            assert "COALESCE(hook_type," in _indexdefs(conn)[NEW_KEY]
            assert _all(conn) == before, "the upgrade changed a historical row"

        def refused(*row):
            with engine.connect() as conn:
                with pytest.raises(IntegrityError, match=NEW_KEY):
                    _insert(conn, *row)

        def accepted(*row):
            with engine.begin() as conn:
                _insert(conn, *row)

        refused("train_hist", 0, 3, 9.9, None)   # a historical row keeps its old uniqueness
        refused("train_hist", 0, -4, 9.9, None)
        accepted("train_hist", 0, None, 9.9, None)  # aggregates were never unique
        accepted("train_hist", 0, 3, 5.0, "mlp")     # beside a NULL-hook row: removed on downgrade
        # A 'residual' row beside a NULL-hook row: the residual row is KEPT and the historical
        # one goes. Review R2-B: with no such pair here, a downgrade whose first rule compared
        # `hook_type = 'residual'` (NULL on the historical row, so neither rule is true) kept
        # BOTH rows and could not recreate the old key, and this test stayed green (K5).
        accepted("train_hist", 0, 4, 5.1, "residual")

        # A multi-hook run written after the upgrade. Losses name the rows.
        accepted("train_multi", 0, 3, 3.1, "mlp")        # lower id, not residual
        accepted("train_multi", 0, 3, 3.2, "residual")   # the one the downgrade keeps
        accepted("train_multi", 0, -4, 4.1, "mlp")
        accepted("train_multi", 0, -4, 4.2, "residual")
        accepted("train_multi", 0, 7, 7.1, "mlp")        # neither residual: the lowest id stays
        accepted("train_multi", 0, 7, 7.2, "attention")
        accepted("train_multi", 0, None, 0.5, None)
        accepted("train_multi", 0, None, 0.5, None)
        refused("train_multi", 0, 3, 9.9, "mlp")

        caplog.clear()
        with engine.begin() as conn, caplog.at_level(logging.WARNING, logger="alembic.runtime.migration"):
            _run(conn, revision.downgrade)
            assert not _has_column(conn, "hook_type")
            assert _unique_constraints(conn) == {OLD_KEY: ["training_id", "step", "layer_idx"]}
            assert NEW_KEY not in _indexdefs(conn)
            survivors = _all(conn, with_hook=False)
        logged = [r.getMessage() for r in caplog.records if "d5a1f3c7e9b2 downgrade" in r.getMessage()]
        assert logged == ["d5a1f3c7e9b2 downgrade: removed 5 per-hook metric rows the old key cannot hold"], logged
        expected = sorted([row for row in before if row != ("train_hist", 0, 4, "", 1.2)] + [
            ("train_hist", 0, 4, "", 5.1),
            ("train_hist", 0, AGGREGATE, "", 9.9),
            ("train_multi", 0, 3, "", 3.2), ("train_multi", 0, -4, "", 4.2),
            ("train_multi", 0, 7, "", 7.1),
            ("train_multi", 0, AGGREGATE, "", 0.5), ("train_multi", 0, AGGREGATE, "", 0.5),
        ])
        assert survivors == expected

        # And the downgrade left a table the revision upgrades again.
        with engine.begin() as conn:
            _run(conn, revision.upgrade)
            assert _all(conn) == expected
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS " + TABLE))
        engine.dispose()
