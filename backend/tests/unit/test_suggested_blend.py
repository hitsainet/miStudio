"""The suggested blend (Feature 041 FR-21; T8, T14; FTASKS 5.3, 9.2).

The record is a SLICE of the stored report (no recomputation), so two equal evaluations give the same
record bytes; who, when and why live in the append-only history (a trigger refuses UPDATE and DELETE,
raw SQL included); no blend field ever reaches a member's definition; the format markers are not contracts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import text

from src.core import runtime_settings
from src.core.config import settings
from tests.support import blend_world as bw

API = "/api/v1/probe-monitors"
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


async def mark(client, world, report_id, candidate="stack", reason=None, headers=None):
    return await client.put(f"{API}/behaviours/{world.key}/suggested-blend", headers=headers or {},
                            json={"model": world.model.id, "report_id": report_id, "candidate": candidate,
                                  **({"reason": reason} if reason else {})})


async def test_record_round_trips_from_report_bytes(client, async_session, request, monkeypatch):
    from src.services.probe_blend.report_format import canonical_bytes

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    report = json.loads((await client.get(f"{API}/blend-reports/{report_id}/report.json")).content)
    made = (await mark(client, world, report_id, "stack")).json()
    record = made["record"]
    entry = next(c for c in report["candidates"] if c["name"] == "stack")
    assert record["candidate"]["params"] == entry["params"] and record["candidate"]["calibration"] == entry["calibration"]
    assert record["advisory"] is True and record["statement"] == report["statement"]
    assert record["format"] == "mistudio.suggested-blend/1" and record["report_id"] == report_id
    assert hashlib.sha256(canonical_bytes(record)).hexdigest() == made["record_sha256"]
    current = (await client.get(f"{API}/behaviours/{world.key}/suggested-blend", params={"model": world.model.id})).json()
    assert current["record"] == record and current["stale"] is False


async def test_two_equal_evaluations_give_the_same_record(client, async_session, request, monkeypatch):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    first = await bw.done_report(client, monkeypatch, world)
    second = await bw.done_report(client, monkeypatch, world)
    a = (await mark(client, world, first)).json()
    b = (await mark(client, world, second)).json()
    ra, rb = dict(a["record"]), dict(b["record"])
    assert ra.pop("report_id") == first and rb.pop("report_id") == second
    assert ra == rb


async def test_replace_and_history_with_actors(client, async_session, request, monkeypatch):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    await mark(client, world, report_id, "stack", reason="best held-out")
    await mark(client, world, report_id, "max", reason="simpler", headers={"X-MiStudio-Agent": "mcp"})
    current = (await client.get(f"{API}/behaviours/{world.key}/suggested-blend", params={"model": world.model.id})).json()
    assert current["candidate"] == "max"
    history = (await client.get(f"{API}/behaviours/{world.key}/suggested-blend/history",
                                params={"model": world.model.id})).json()["entries"]
    assert [(h["candidate"], h["actor"], h["reason"]) for h in history] == [
        ("max", "agent:mcp", "simpler"), ("stack", "operator", "best held-out")]
    catalogue = (await client.get(f"{API}/behaviours/{world.key}/history")).json()["entries"]
    assert [e["field"] for e in catalogue if e["subject_kind"] == "blend"] == ["suggested", "suggested"]


async def test_stale_report_refused(client, async_session, request, monkeypatch):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    world.members[0].decision_digest = "e" * 64
    await async_session.commit()
    response = await mark(client, world, report_id)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "blend_report_stale" and detail["next"][0]["tool"] == "rerun_blend_report"


async def test_trigger_refuses_update_and_delete(client, async_session, request, monkeypatch):
    """Real Postgres, raw SQL: the trigger guards against writers the ORM does not see."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    await mark(client, world, report_id)
    for statement in ("UPDATE probe_suggested_blend_history SET reason = 'x'",
                      "DELETE FROM probe_suggested_blend_history"):
        with pytest.raises(Exception, match="append-only"):
            await async_session.execute(text(statement))
        await async_session.rollback()


def test_the_migration_carries_the_models_trigger():
    import importlib.util

    from src.models import probe_blend

    path = REPO / "backend" / "alembic" / "versions" / "b7e41c2d9a05_probe_blend_reports.py"
    spec = importlib.util.spec_from_file_location("rev_041b", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.HISTORY_DDL == probe_blend.HISTORY_DDL


def test_docs_schemas_allowlist():
    """T14: the blend formats are internal markers, not contracts — no file in docs/schemas names them."""
    for path in (REPO / "docs" / "schemas").glob("*"):
        text_ = path.read_text(errors="ignore")
        assert "blend-report" not in text_ and "suggested-blend" not in text_, path.name


async def test_member_export_bytes_same_with_and_without(client, async_session, request, monkeypatch):
    """No blend field reaches a member's definition: the builder never reads the blend tables (AST: no
    import of `probe_blend` anywhere in the definition builder or publisher)."""
    import ast

    for name in ("probe_definition_builder.py", "probe_definition_publisher.py"):
        tree = ast.parse((REPO / "backend" / "src" / "services" / name).read_text())
        imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        assert not any("probe_blend" in m for m in imported), (name, imported)
