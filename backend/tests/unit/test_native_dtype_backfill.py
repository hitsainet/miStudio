"""`models.native_dtype` is filled from each checkpoint's config.json, and can never fail a migration."""

from __future__ import annotations

import json

import sqlalchemy as sa

from src.db.native_dtype_backfill import backfill_native_dtype


def _engine():
    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE models (id TEXT PRIMARY KEY, file_path TEXT, native_dtype TEXT)"))
    return engine


def _snapshot(root, config):
    snap = root / "models--org--m" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(config if isinstance(config, str) else json.dumps(config))
    return str(root)


def test_reads_each_checkpoints_own_dtype_and_skips_what_it_cannot_read(tmp_path):
    engine = _engine()
    rows = {
        "m_bf16": _snapshot(tmp_path / "a", {"torch_dtype": "bfloat16"}),
        "m_fp16": _snapshot(tmp_path / "b", {"text_config": {"dtype": "float16"}}),
        "m_none": _snapshot(tmp_path / "c", {"model_type": "llama"}),
        "m_bad": _snapshot(tmp_path / "d", "{not json"),
        "m_gone": str(tmp_path / "missing"),
    }
    with engine.begin() as conn:
        for model_id, path in rows.items():
            conn.execute(sa.text("INSERT INTO models VALUES (:i, :p, NULL)"), {"i": model_id, "p": path})
        conn.execute(sa.text("INSERT INTO models VALUES ('m_set', :p, 'float32')"), {"p": rows["m_bf16"]})
        assert backfill_native_dtype(conn) == 2
        got = dict(conn.execute(sa.text("SELECT id, native_dtype FROM models")).fetchall())
    assert got == {"m_bf16": "bfloat16", "m_fp16": "float16", "m_none": None, "m_bad": None,
                   "m_gone": None, "m_set": "float32"}


def test_a_row_that_raises_does_not_fail_the_migration(tmp_path, monkeypatch):
    """⚠ The API refuses to serve on a failed migration; one bad row must cost one row."""
    from src.db import native_dtype_backfill

    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sa.text("INSERT INTO models VALUES ('m_x', :p, NULL)"),
                     {"p": _snapshot(tmp_path / "x", {"torch_dtype": "bfloat16"})})
        monkeypatch.setattr(native_dtype_backfill, "config_dir", lambda p: (_ for _ in ()).throw(OSError("boom")))
        assert backfill_native_dtype(conn) == 0


def test_the_migration_calls_it():
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "alembic" / "versions" / "e7a2c5d913f4_record_model_dtype.py").read_text()
    upgrade = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "upgrade")
    calls = [n for n in ast.walk(upgrade) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "backfill_native_dtype"]
    assert len(calls) == 1
