"""Feature 041 works for ANY behaviour (FR-29; D20, D25; FTASKS 14.3).

* 040's guard covers 041's modules: a behaviour token planted in a copy of a 041 module is found, and the
  real 041 modules name none.
* The MCP-only sequence — family → blend report → export — runs through the REAL tool functions against
  the REAL app (an in-process transport) with the SAME tool names and call counts for three shapes of
  behaviour: a pilot-shaped one (declared sub-questions, specialists), a "high-stakes"-shaped one
  (multi-layer family, no sub-questions, no rule set: the report says `no_rule_set`) and a plain random
  key. No behaviour is named; nothing in `src/` differs between the three.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from src.core import runtime_settings
from src.core.config import settings
from tests.support import blend_world as bw
from tests.support.behaviour_tokens import BACKEND, Matcher, derive_tokens
from tests.unit.test_no_behaviour_names_in_source import scan_python

FILES_041 = [BACKEND / "src" / p for p in (
    "services/probe_family.py", "services/probe_family_scoring.py", "services/probe_family_export.py",
    "services/probe_blend/__init__.py", "services/probe_blend/candidates.py", "services/probe_blend/engine.py",
    "services/probe_blend/service.py", "services/probe_blend/report_format.py", "services/probe_blend/assemble.py",
    "api/v1/endpoints/probe_blends.py", "api/v1/endpoints/_probe_ids.py", "mcp_server/tools/probe_families.py",
    "workers/probe_blend_tasks.py", "workers/cleanup_stuck_blend_reports.py", "models/probe_blend.py",
    "schemas/probe_blend.py")]


@pytest.fixture(scope="module")
def matcher():
    return Matcher(derive_tokens())


def test_the_041_modules_name_no_behaviour(matcher):
    assert all(p.is_file() for p in FILES_041), [p for p in FILES_041 if not p.is_file()]
    assert scan_python(FILES_041, matcher) == []


def test_a_canary_planted_in_a_041_module_is_found(matcher, tmp_path):
    token = matcher.tokens[0]
    copy = tmp_path / "probe_family.py"
    copy.write_text(FILES_041[0].read_text() + f"\n# special case for {token}\n", encoding="utf-8")
    hits = scan_python([copy], matcher, base=tmp_path)
    assert [h.where for h in hits] == ["probe_family.py"], hits


# ── the MCP-only sequence, three shapes ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    runtime_settings.apply({"blend_max_concurrent": 8})
    yield tmp_path
    runtime_settings.reset()


def tools_on_the_app():
    from mcp.server.mcpserver import MCPServer

    from src.main import app
    from src.mcp_server.client import AGENT_HEADER, AGENT_NAME, MiStudioClient
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import probe_families

    client = MiStudioClient("http://test")
    client._http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test/api/v1",
                                     headers={AGENT_HEADER: AGENT_NAME})
    mcp = MCPServer("t")
    probe_families.register(mcp, client, MCPSettings(auth_token="x" * 32))
    return {name: tool.fn for name, tool in mcp._tool_manager._tools.items()}  # noqa: SLF001


async def sequence(world, monkeypatch, data_dir):
    from src.workers import probe_blend_tasks

    for probe in world.members:                         # built definitions on disk, so the export can run
        path = data_dir / "defs" / f"{probe.id}.probe.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": f"d-{probe.id}", "evidence": {"rung": 2}}))
        probe.definition_path = str(path)
    tools = tools_on_the_app()
    queued = []
    monkeypatch.setattr(probe_blend_tasks.run_blend_evaluation, "delay",
                        lambda rid: queued.append(rid) or type("T", (), {"id": "t"})())
    calls = []

    async def call(name, **kwargs):
        calls.append(name)
        return await tools[name](**kwargs)

    family = await call("get_probe_family", key=world.key, model=world.model.id)
    started = await call("evaluate_probe_blends", behaviour_key=world.key, model=world.model.id,
                         dataset_version="dv-blend", calibration_sample_rows=world.sample_rows)
    await asyncio.to_thread(probe_blend_tasks.run_blend_evaluation, started["report_id"])
    report = await call("get_blend_report", report_id=started["report_id"])
    exported = await call("export_probe_family", key=world.key, model=world.model.id)
    return calls, queued, family, report, exported


@pytest.mark.parametrize("shape", ["pilot", "high_stakes", "random"])
async def test_the_same_tool_sequence_for_every_shape(client, async_session, request, monkeypatch, data_dir, shape):
    world = await bw.build(client, async_session, seed=request.node.nodeid, signals=(2.0, 1.0, 0.5, 0.0))
    if shape == "pilot":
        subs = ["sq-" + world.key.replace("-", "")[:6], "sq-" + world.key.replace("-", "")[-6:]]
        made = await client.post(f"/api/v1/probe-monitors/behaviours/{world.key}/versions", json={
            "base_version": 2, "sub_questions": [{"key": s, "question": "q", "positive_meaning": "p"} for s in subs]})
        assert made.status_code == 201, made.text
    if shape == "high_stakes":
        for layer, probe in enumerate(world.members):
            probe.layer = layer * 4                     # a multi-layer family
    await async_session.commit()
    calls, queued, family, report, exported = await sequence(world, monkeypatch, data_dir)
    assert calls == ["get_probe_family", "evaluate_probe_blends", "get_blend_report", "export_probe_family"]
    assert len(queued) == 1
    assert report["status"] == "done" and report["report"]["has_rung"] is False
    assert "no_rule_set" in report["report"]["flags"]
    assert [m["probe_id"] for m in exported["members"]] == sorted(m.id for m in world.members)
    if shape == "pilot":
        assert [c["state"] for c in family["coverage"][:2]] == ["missing", "missing"]
