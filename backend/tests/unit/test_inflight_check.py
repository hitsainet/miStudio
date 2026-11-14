"""T-39 (042 FTASKS 13.9, FR-042.38; FTID §3.16; TPD-13): the read-only drain check before the P3 window.

`scripts/rebrand/inflight.py` exits 1 listing anything mid-flight (a non-terminal row in any status column, a
pending approval, a waiting queue message) and 0 when nothing is. The database half runs on the real Postgres
test database (the `async_engine` fixture builds the ORM's tables), so the SQL it sends is exercised; Redis is a
small fake with the four calls the script makes.

`test_every_status_value_classified` keeps a new status from silently reading as finished (TI-12): the status
columns are discovered from `Base.metadata`, and every value one of them can take must be classified — the
labels of a database enum, the values of the module's status enum the column defaults to, the defaults, and
every string literal `src/` assigns to, compares with or filters on a column of that name (or binds to a
`*<COLUMN>*` constant).
"""
from __future__ import annotations

import ast
import enum
import io
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import text

from scripts.rebrand import inflight
from src.core.database import Base
import src.models  # noqa: F401 — every mapped class

SRC = Path(__file__).resolve().parents[2] / "src"


class FakeRedis:
    """Only the read calls the script may make; a write would raise AttributeError."""

    def __init__(self, data: dict | None = None):
        self.data = data or {}  # key -> ("list"|"zset", length)

    def scan_iter(self, match: str):
        rx = re.compile("^" + re.escape(match).replace(r"\*", ".*") + "$", re.S)
        return [k.encode() for k in self.data if rx.match(k)]

    def type(self, key):
        return (self.data.get(key, ("none", 0))[0]).encode()

    def llen(self, key):
        kind, n = self.data.get(key, ("none", 0))
        assert kind == "list", key
        return n

    def zcard(self, key):
        kind, n = self.data.get(key, ("none", 0))
        assert kind == "zset", key
        return n


class ReadOnly:
    """The connection the script gets: every statement must be a SELECT (the script is read-only)."""

    def __init__(self, conn):
        self.conn, self.statements = conn, []

    def execute(self, statement, *args, **kwargs):
        sql = str(statement).strip()
        assert sql.upper().startswith("SELECT"), sql
        self.statements.append(sql)
        return self.conn.execute(statement, *args, **kwargs)


def _run(conn, redis: FakeRedis) -> tuple[int, str]:
    out = io.StringIO()
    guarded = ReadOnly(conn)

    @contextmanager
    def connect():
        yield guarded

    code = inflight.main([], connect=connect, redis=lambda: redis, out=out)
    assert len(guarded.statements) == len(inflight.status_columns(Base.metadata)), guarded.statements
    return code, out.getvalue()


async def _in_db(async_engine, body):
    async with async_engine.connect() as aconn:
        trans = await aconn.begin()
        try:
            return await aconn.run_sync(body)
        finally:
            await trans.rollback()


@pytest.mark.asyncio
async def test_exit_0_on_empty_world(async_engine):
    code, out = await _in_db(async_engine, lambda c: _run(c, FakeRedis()))
    assert code == 0, out
    assert "clear" in out and f"{len(inflight.status_columns(Base.metadata))} status columns" in out


@pytest.mark.asyncio
async def test_exit_1_on_unfinished_row(async_engine):
    def body(c):
        c.execute(text("INSERT INTO task_queue (id, task_type, entity_id, entity_type, status, retry_count) "
                       "VALUES ('tq_1', 'training', 'tr_1', 'training', 'running', 0)"))
        c.execute(text("INSERT INTO task_queue (id, task_type, entity_id, entity_type, status, retry_count) "
                       "VALUES ('tq_2', 'training', 'tr_2', 'training', 'completed', 0)"))
        return _run(c, FakeRedis())
    code, out = await _in_db(async_engine, body)
    assert code == 1, out
    assert "IN FLIGHT: task_queue.status = 'running': 1 row(s)" in out
    assert "completed" not in out.split("inflight:")[0], out


@pytest.mark.asyncio
async def test_exit_1_on_pending_approval(async_engine):
    def body(c):
        c.execute(text("INSERT INTO agent_approval_requests (id, tool_name, payload, status) "
                       "VALUES ('ap_1', 'steer_sweep', '{}'::jsonb, 'pending'), "
                       "('ap_2', 'steer_sweep', '{}'::jsonb, 'approved')"))
        return _run(c, FakeRedis())
    code, out = await _in_db(async_engine, body)
    assert code == 1, out
    found = [ln for ln in out.splitlines() if ln.startswith("IN FLIGHT")]
    assert found == ["IN FLIGHT: agent_approval_requests.status = 'pending': 1 row(s)"], out


@pytest.mark.asyncio
async def test_exit_1_on_a_waiting_queue_message(async_engine):
    queue = inflight.queue_names()[0]
    redis = FakeRedis({queue: ("list", 2), "gpu.gpu-247aa582\x06\x163": ("list", 1), "unrelated": ("list", 9),
                       "mistudio:gpu:waiting": ("zset", 1)})
    code, out = await _in_db(async_engine, lambda c: _run(c, redis))
    assert code == 1, out
    found = sorted(ln for ln in out.splitlines() if ln.startswith("IN FLIGHT"))
    assert found == sorted([f"IN FLIGHT: queue {queue}: 2 message(s)",
                            "IN FLIGHT: queue gpu.gpu-247aa582\x06\x163: 1 message(s)",
                            "IN FLIGHT: queue mistudio:gpu:waiting: 1 message(s)"]), out


def test_queue_names_come_from_the_routes():
    from src.core.celery_app import celery_app

    routed = {r["queue"] for r in celery_app.conf.task_routes.values() if isinstance(r, dict) and r.get("queue")}
    assert routed and routed <= set(inflight.queue_names()), (routed, inflight.queue_names())
    assert "celery" in inflight.queue_names()


# ── every value a status column can take is classified ─────────────────────────────────────────────


def _strs(node) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [v.value for v in node.elts if isinstance(v, ast.Constant) and isinstance(v.value, str)]
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("frozenset", "set", "tuple") and node.args:
        return _strs(node.args[0])
    return []


def _source_vocabulary(columns: set[str]) -> dict[str, set[str]]:
    vocab: dict[str, set[str]] = defaultdict(set)
    constant = {c: re.compile(rf"(^|_){c.upper()}(_|S?$)") for c in columns}
    for path in SRC.rglob("*.py"):
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Attribute) and t.attr in columns:
                        vocab[t.attr].update(_strs(n.value))
                    if isinstance(t, ast.Name):
                        for c, rx in constant.items():
                            if rx.search(t.id):
                                vocab[c].update(_strs(n.value))
            elif isinstance(n, ast.Compare) and isinstance(n.left, ast.Attribute) and n.left.attr in columns:
                for c in n.comparators:
                    vocab[n.left.attr].update(_strs(c))
            elif isinstance(n, ast.Call):
                f = n.func
                if (isinstance(f, ast.Attribute) and f.attr in ("in_", "notin_")
                        and isinstance(f.value, ast.Attribute) and f.value.attr in columns):
                    for a in n.args:
                        vocab[f.value.attr].update(_strs(a))
                for kw in n.keywords:
                    if kw.arg in columns:
                        vocab[kw.arg].update(_strs(kw.value))
    return vocab


def _column_values(table, column, vocab) -> set[str]:
    col = Base.metadata.tables[table].columns[column]
    values = set(vocab.get(column, ()))
    values.update(getattr(col.type, "enums", None) or ())
    default = getattr(col.default, "arg", None)
    server = getattr(col.server_default, "arg", None)
    for d in (default, server):
        if isinstance(d, enum.Enum):
            values.update(str(m.value) for m in type(d))
            values.add(d.name)
        elif isinstance(d, str):
            values.add(d)
    # A String column that defaults to `SomeStatus.X.value`: the enum in the model's module holds the vocabulary.
    mapper = next(m for m in Base.registry.mappers if m.local_table is Base.metadata.tables[table])
    module = sys.modules[mapper.class_.__module__]
    for obj in vars(module).values():
        if isinstance(obj, type) and issubclass(obj, enum.Enum) and not issubclass(obj, enum.Flag):
            members = {str(m.value) for m in obj}
            if (isinstance(default, str) and default in members) or (isinstance(server, str) and server in members):
                values |= members
    return values


def test_every_status_value_classified():
    discovered = inflight.status_columns(Base.metadata)
    assert len(discovered) >= 24, discovered  # FTDD §13: 24 models with a `status` column at 7ce9fe382
    assert set(inflight.TERMINAL) == set(discovered), (
        f"unclassified columns: {sorted(set(discovered) - set(inflight.TERMINAL))}; "
        f"stale entries: {sorted(set(inflight.TERMINAL) - set(discovered))}")
    vocab = _source_vocabulary({c for _, c in discovered})
    assert {"pending", "running", "completed"} <= vocab["status"], "the source scan found nothing: it is blind"
    problems = []
    for key in discovered:
        values = _column_values(*key, vocab)
        unclassified = sorted(values - inflight.TERMINAL[key] - inflight.ACTIVE)
        if unclassified:
            problems.append(f"{key[0]}.{key[1]}: {unclassified}")
    assert not problems, "classify each value as terminal or ACTIVE in scripts/rebrand/inflight.py:\n" + "\n".join(
        problems)
