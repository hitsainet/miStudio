"""ENH-008's two revisions: the history columns, and the drain of `millm_*` requests at switch-over (S5).

(ENH-006 D81 adds a third, `e008c3d5f7a9`: the `batch_id` column and index, round-tripped at the end.)

* `e008a1f2b3c4` adds `requested_by` and `decided_by` (String(64), nullable) to `agent_approval_requests`, with the
  ORM's type and nullability, and round-trips up/down/up.
* `e008b7c9d1e3` (`e008_millm_approvals_moved.py`) EXPIRES every row of the four tools that used to file in
  miStudio's queue while it is `pending` or `approved`, with the reason "moved: …" and `decided_by` "system", and
  leaves every other row — other tools in any state, and the four tools' already-decided rows — exactly as it was.

Each runs its OWN functions on a scratch database (named after this run's `DATABASE_URL_SYNC`, so parallel agents
never share it), over the table as the ORM declares it.

Mutation controls (0xcc/reviews/enh008_controls_2026-10-28.md): drop `'approved'` from the drain's status list →
red; widen the tool list to every tool → red.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import uuid

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

BACKEND = pathlib.Path(__file__).resolve().parents[2]
VERSIONS = BACKEND / "alembic" / "versions"
TABLE = "agent_approval_requests"


def _load(filename: str, revision: str, down: str):
    spec = importlib.util.spec_from_file_location(f"rev_{revision}", VERSIONS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.revision == revision and module.down_revision == down
    return module


def _scratch_engine(suffix: str):
    url = os.environ.get("DATABASE_URL_SYNC") or "postgresql://postgres:devpassword@localhost:5432/mistudio_test"
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


def _run(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


def _create_table(conn):
    """The table as the ORM declares it now (both new columns included)."""
    import src.models  # noqa: F401 — register every table on the metadata
    from src.core.database import Base

    conn.execute(text(f'DROP TABLE IF EXISTS "{TABLE}" CASCADE'))
    Base.metadata.tables[TABLE].create(conn)


def test_the_history_columns_round_trip_with_the_orms_type():
    import src.models  # noqa: F401
    from src.core.database import Base

    module = _load("e008a1f2b3c4_approval_history_fields.py", "e008a1f2b3c4", "c0de5a11e004")
    engine = _scratch_engine("enh008_migrations")
    try:
        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{TABLE}" CASCADE'))
            conn.execute(text(f'CREATE TABLE "{TABLE}" (id VARCHAR(36) PRIMARY KEY)'))
            _run(conn, module.upgrade)
            cols = {c["name"]: c for c in sa.inspect(conn).get_columns(TABLE)}
            for name in ("requested_by", "decided_by"):
                orm = Base.metadata.tables[TABLE].c[name]
                assert cols[name]["nullable"] is True and orm.nullable is True
                assert cols[name]["type"].compile(conn.dialect) == orm.type.compile(conn.dialect)
            _run(conn, module.downgrade)
            cols = {c["name"] for c in sa.inspect(conn).get_columns(TABLE)}
            assert not {"requested_by", "decided_by"} & cols
            _run(conn, module.upgrade)
            conn.execute(text(f'DROP TABLE "{TABLE}" CASCADE'))
    finally:
        engine.dispose()


#: (tool, status, should be drained)
ROWS = [
    ("millm_load_model", "pending", True),
    ("millm_load_model", "approved", True),
    ("millm_request_probe_limits", "pending", True),
    ("millm_measure_probe_capacity", "approved", True),
    ("millm_import_rule_set", "pending", True),
    # the four tools' already-decided rows are history; untouched
    ("millm_load_model", "consumed", False),
    ("millm_load_model", "denied", False),
    ("millm_import_rule_set", "expired", False),
    # every other tool, in the states the drain targets, untouched
    ("start_probe_training_run", "pending", False),
    ("start_probe_training_run", "approved", False),
    ("steer_compare", "pending", False),
    ("code_rule_approve", "approved", False),
]


def test_the_drain_expires_pending_and_approved_millm_rows_and_nothing_else():
    module = _load("e008_millm_approvals_moved.py", "e008b7c9d1e3", "e008a1f2b3c4")
    engine = _scratch_engine("enh008_migrations")
    try:
        with engine.begin() as conn:
            _create_table(conn)
            ids = {}
            for tool, status, _ in ROWS:
                rid = str(uuid.uuid4())
                ids[rid] = (tool, status)
                conn.execute(text(
                    f'INSERT INTO "{TABLE}" (id, tool_name, payload, status, reason, created_at, resolved_at) '
                    "VALUES (:id, :t, CAST(:p AS JSONB), :s, :r, now() - interval '1 hour', "
                    "CASE WHEN :s = 'pending' THEN NULL ELSE now() - interval '30 minutes' END)"),
                    {"id": rid, "t": tool, "p": json.dumps({"model_id": 1}), "s": status, "r": "before"})
            before = {r.id: dict(r._mapping) for r in conn.execute(text(f'SELECT * FROM "{TABLE}"'))}

            _run(conn, module.upgrade)

            after = {r.id: dict(r._mapping) for r in conn.execute(text(f'SELECT * FROM "{TABLE}"'))}
            for rid, (tool, status) in ids.items():
                drained = next(d for t, s, d in ROWS if (t, s) == (tool, status))
                row = after[rid]
                if drained:
                    assert row["status"] == "expired", (tool, status)
                    assert row["reason"] == "moved: miLLM now decides its own requests; ask again (ENH-008)"
                    assert row["decided_by"] == "system"
                    assert row["resolved_at"] is not None
                    if before[rid]["resolved_at"] is not None:
                        assert row["resolved_at"] == before[rid]["resolved_at"], "an approval time is kept"
                else:
                    assert row == before[rid], f"{tool}/{status} was changed by the drain"

            # The rows stay (History shows them) and the downgrade re-opens nothing.
            assert len(after) == len(ROWS)
            _run(conn, module.downgrade)
            assert {r.id: dict(r._mapping) for r in conn.execute(text(f'SELECT * FROM "{TABLE}"'))} == after
            conn.execute(text(f'DROP TABLE "{TABLE}" CASCADE'))
    finally:
        engine.dispose()


def test_the_drain_names_exactly_the_four_tools_that_moved():
    """Frozen in the migration, and the same four the endpoint no longer accepts."""
    from src.api.v1.endpoints import mcp_approvals
    from src.mcp_server.tools.millm_approvals import RELAYED_ACTIONS

    module = _load("e008_millm_approvals_moved.py", "e008b7c9d1e3", "e008a1f2b3c4")
    assert set(module.MOVED_TOOLS) == {"millm_load_model", "millm_request_probe_limits",
                                       "millm_measure_probe_capacity", "millm_import_rule_set"}
    assert set(module.MOVED_TOOLS) <= set(RELAYED_ACTIONS)
    assert not set(module.MOVED_TOOLS) & set(mcp_approvals.APPROVABLE_TOOLS)


def test_the_revisions_lie_on_the_one_line_to_head():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "alembic"))
    script = ScriptDirectory.from_config(config)
    assert len(script.get_heads()) == 1, script.get_heads()
    line = {rev.revision for rev in script.walk_revisions("base", script.get_current_head())}
    assert {"e008a1f2b3c4", "e008b7c9d1e3"} <= line


def test_the_batch_id_column_and_index_round_trip_with_the_orms_type():
    """ENH-006 D81: `e008c3d5f7a9` adds `batch_id` (String(36), nullable) and its index, matching the ORM."""
    import src.models  # noqa: F401
    from src.core.database import Base

    module = _load("e008c3d5f7a9_approval_batch_id.py", "e008c3d5f7a9", "e008b7c9d1e3")
    engine = _scratch_engine("d81_batch_migration")
    try:
        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{TABLE}" CASCADE'))
            conn.execute(text(f'CREATE TABLE "{TABLE}" (id VARCHAR(36) PRIMARY KEY)'))
            _run(conn, module.upgrade)
            col = {c["name"]: c for c in sa.inspect(conn).get_columns(TABLE)}["batch_id"]
            orm = Base.metadata.tables[TABLE].c["batch_id"]
            assert col["nullable"] is True and orm.nullable is True
            assert col["type"].compile(conn.dialect) == orm.type.compile(conn.dialect)
            orm_index = {i.name for i in Base.metadata.tables[TABLE].indexes if "batch_id" in i.columns}
            db_index = {i["name"] for i in sa.inspect(conn).get_indexes(TABLE) if i["column_names"] == ["batch_id"]}
            assert db_index == orm_index == {"ix_agent_approval_requests_batch_id"}
            _run(conn, module.downgrade)
            assert "batch_id" not in {c["name"] for c in sa.inspect(conn).get_columns(TABLE)}
            assert sa.inspect(conn).get_indexes(TABLE) == []
            _run(conn, module.upgrade)
            conn.execute(text(f'DROP TABLE "{TABLE}" CASCADE'))
    finally:
        engine.dispose()
