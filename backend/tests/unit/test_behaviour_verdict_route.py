"""The behaviour verdict export through the REAL routes on a real database (ENH-006 G19, AC 3, AC 8).

A 041 report runs for real (start route, task in-process) over 040's verdict cache seeded WITH vetoes, with the
veto scoped to one member (G19-3). The export route then slices the definition from the stored report:

* the scope reaches the report's candidate params and the definition's `veto_applies`;
* over the target the export is refused (`union_over_budget`, with the live `next` call), and an acknowledged
  export records the reason (G19-6);
* the definition validates against the published schema, names the report's sha256 and is byte-identical
  across two exports; a re-cut member makes the report stale and the export refused;
* the start route refuses a scope naming a non-member and a scope over a named rules ∪ <score kind>, and drops
  defaulted rules ∪ <score kind> candidates with a flag.

The route is in `app.openapi()["paths"]` and the MCP tool in the LIVE registry, posting exactly one request with
exactly the body the route takes (global CLAUDE.md: payload and call count).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import jsonschema
import pytest

from src.core import runtime_settings
from src.core.config import settings
from src.services import behaviour_verdict as bv
from tests.support import blend_world as bw
from tests.support import rule_set_fixtures as rf
from tests.support.next_calls import assert_next_calls_are_live

API = "/api/v1/probe-monitors"
REPO = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((REPO / "docs" / "schemas" / "behaviour-verdict-v1.json").read_text())


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "miserver_api_url", "", raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


async def with_identity(session, world):
    """Each member's built definition records a detector digest (039), as a real build does."""
    for j, probe in enumerate(world.members):
        probe.definition_build = {"resolved": {"detector_digest": hashlib.sha256(f"{probe.id}:{j}".encode()).hexdigest(),
                                               "decision_digest": probe.decision_digest, "decision_revision": 1}}
    await session.commit()


async def scoped_report(client, async_session, request, monkeypatch, **extra):
    seed = request.node.nodeid
    world = await bw.build(client, async_session, seed=seed, rule_set=lambda key: rf.masking_set(seed, key))
    await with_identity(async_session, world)
    await bw.seed_rule_verdicts(async_session, world, seed=seed, veto_share=0.3)
    report_id = await bw.done_report(client, monkeypatch, world, candidates=["any_of", "rules_or_any_of"],
                                     veto_scope=[world.members[0].id], **extra)
    return world, report_id


def export_body(world, report_id, **extra):
    return {"model": world.model.id, "report_id": report_id, "candidate": "rules_or_any_of", **extra}


async def test_a_scoped_report_exports_a_definition(client, async_session, request, monkeypatch):
    world, report_id = await scoped_report(client, async_session, request, monkeypatch)
    envelope = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    entry = next(c for c in envelope["report"]["candidates"] if c["name"] == "rules_or_any_of")
    assert entry["params"].get("veto_scope") == [world.members[0].id]
    url = f"{API}/behaviours/{world.key}/behaviour-verdict/export"

    # the world's members fire at 0.8 on N(0, 1) negatives: any_of's union is far over the 5% target
    assert entry["calibration"]["realised_fpr"] > envelope["report"]["behaviour"]["target_fpr"]
    refused = await client.post(url, json=export_body(world, report_id))
    assert refused.status_code == 409, refused.text
    detail = refused.json()["detail"]
    assert detail["code"] == "union_over_budget"
    assert detail["realised_union_fpr"] == entry["calibration"]["realised_fpr"]
    assert_next_calls_are_live(detail["next"])
    response = await client.post(url, json=export_body(world, report_id, acknowledge_over_budget="dry run"))
    assert response.status_code == 200, response.text
    body = response.json()
    definition = body["definition"]
    jsonschema.validate(definition, SCHEMA)
    assert body["definition_sha256"] == hashlib.sha256(bv.definition_bytes(definition)).hexdigest()
    assert definition["calibration"]["acknowledged"]["reason"] == "dry run"
    assert [m["veto_applies"] for m in definition["members"]] == [m.id == world.members[0].id
                                                                  for m in world.members]
    assert [m["probe_id"] for m in definition["members"]] == [m.id for m in world.members]
    assert definition["calibration"]["report_sha256"] == envelope["report_sha256"]
    assert definition["calibration"]["realised_union_fpr"] == entry["calibration"]["realised_fpr"]
    assert definition["rules"]["rule_set_digest"] == envelope["report"]["behaviour"]["rule_set"]["digest"]
    assert definition["rules"]["match"]["source"] in ("explicit", "inferred", "inferred_all_outputs")
    assert definition["rules"]["match"]["any_of"]
    again = await client.post(url, json=export_body(world, report_id, acknowledge_over_budget="dry run"))
    assert again.json()["definition_sha256"] == body["definition_sha256"]


async def test_the_route_forwards_the_callers_windows(client, async_session, request, monkeypatch):
    """Control A5 (suite agent, 2026-11-09): the route passing `windows=None` to the service survived every test, because
    every export asked for the default. A caller's windows must reach the definition, and a bad one must be refused."""
    world, report_id = await scoped_report(client, async_session, request, monkeypatch)
    url = f"{API}/behaviours/{world.key}/behaviour-verdict/export"
    body = export_body(world, report_id, acknowledge_over_budget="dry run", windows=["prompt", "last_user"])
    response = await client.post(url, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["definition"]["windows"] == ["prompt", "last_user"]
    refused = await client.post(url, json={**body, "windows": ["prompt", "everything"]})
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"]["code"] == "invalid_field"


async def test_a_member_without_a_detector_digest_is_refused(client, async_session, request, monkeypatch):
    seed = request.node.nodeid
    world = await bw.build(client, async_session, seed=seed, rule_set=lambda key: rf.masking_set(seed, key))
    await bw.seed_rule_verdicts(async_session, world, seed=seed, veto_share=0.3)
    report_id = await bw.done_report(client, monkeypatch, world, candidates=["any_of", "rules_or_any_of"])
    response = await client.post(f"{API}/behaviours/{world.key}/behaviour-verdict/export",
                                 json=export_body(world, report_id, acknowledge_over_budget="x"))
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "member_identity_missing" and detail["probe_ids"] == [m.id for m in world.members]
    assert_next_calls_are_live(detail["next"])


async def test_a_recut_member_makes_the_export_refused(client, async_session, request, monkeypatch):
    world, report_id = await scoped_report(client, async_session, request, monkeypatch)
    probe = world.members[1]
    probe.decision_digest = "9" * 64
    await async_session.commit()
    response = await client.post(f"{API}/behaviours/{world.key}/behaviour-verdict/export",
                                 json=export_body(world, report_id, acknowledge_over_budget="x"))
    assert response.status_code == 409 and response.json()["detail"]["code"] == "blend_report_stale"


async def test_the_start_route_refuses_a_scope_it_cannot_honour(client, async_session, request, monkeypatch):
    seed = request.node.nodeid
    world = await bw.build(client, async_session, seed=seed, rule_set=lambda key: rf.masking_set(seed, key))
    await bw.seed_rule_verdicts(async_session, world, seed=seed, veto_share=0.3)
    from src.workers import probe_blend_tasks

    monkeypatch.setattr(probe_blend_tasks.run_blend_evaluation, "delay", lambda rid: MagicMock(id="t"))
    stranger = await client.post(f"{API}/blend-reports", json=bw.start_body(
        world, candidates=["rules_or_any_of"], veto_scope=["pm_000000000000"]))
    assert stranger.status_code == 422 and stranger.json()["detail"]["code"] == "veto_scope_not_members"
    named = await client.post(f"{API}/blend-reports", json=bw.start_body(
        world, candidates=["rules_or_any_of", "rules_or_stack"], veto_scope=[world.members[0].id]))
    assert named.status_code == 422 and named.json()["detail"]["code"] == "veto_scope_needs_verdict_kind"
    defaulted = await client.post(f"{API}/blend-reports", json=bw.start_body(world, veto_scope=[world.members[0].id]))
    assert defaulted.status_code == 202, defaulted.text
    names = defaulted.json()["candidates"]
    assert "rules_or_any_of" in names and not any(n in names for n in ("rules_or_stack", "rules_or_max",
                                                                       "rules_or_best_single"))
    assert "veto_scope_partial_score_unions_dropped" in defaulted.json()["resolved"]["flags"]
    # naming every member is today's whole-blend veto, recorded as no scope
    every = await client.post(f"{API}/blend-reports", json=bw.start_body(
        world, candidates=["rules_or_stack"], veto_scope=[m.id for m in world.members]))
    assert every.status_code == 202, every.text


# ── wiring: the live registries ──────────────────────────────────────────────────────────────


def test_the_export_route_is_in_the_live_openapi():
    from src.main import app

    paths = app.openapi()["paths"]
    route = paths["/api/v1/probe-monitors/behaviours/{key}/behaviour-verdict/export"]
    assert "post" in route
    body = app.openapi()["components"]["schemas"]["BehaviourVerdictExport"]["properties"]
    assert set(body) == {"model", "report_id", "candidate", "windows", "acknowledge_over_budget"}
    assert "veto_scope" in app.openapi()["components"]["schemas"]["BlendStart"]["properties"]


def _live_tool(name: str):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mcp, client = build_server(MCPSettings(tool_categories="probe_monitors", allow_anonymous=True), stdio=True)
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    return mcp, client, tools.get(name)


def test_the_export_tool_is_live_and_posts_one_request_with_the_routes_body():
    mcp, client, tool = _live_tool("export_behaviour_verdict")
    assert tool is not None, "export_behaviour_verdict is absent from the live MCP registry"
    schema = getattr(tool, "input_schema", None) or tool.inputSchema
    assert set(schema["properties"]) == {"key", "model", "report_id", "candidate", "windows",
                                         "acknowledge_over_budget"}
    client.post = AsyncMock(return_value={"definition": {}})
    asyncio.run(mcp.call_tool("export_behaviour_verdict", {
        "key": "beh-x", "model": "m_1", "report_id": "pbr_1", "candidate": "rules_or_any_of",
        "acknowledge_over_budget": "why"}))
    client.post.assert_awaited_once_with(
        "/probe-monitors/behaviours/beh-x/behaviour-verdict/export",
        json_body={"model": "m_1", "report_id": "pbr_1", "candidate": "rules_or_any_of",
                   "acknowledge_over_budget": "why"})


def test_the_blend_tool_forwards_the_veto_scope():
    mcp, client, tool = _live_tool("evaluate_probe_blends")
    client.post = AsyncMock(return_value={})
    asyncio.run(mcp.call_tool("evaluate_probe_blends", {"behaviour_key": "beh-x", "veto_scope": ["pm_a"]}))
    client.post.assert_awaited_once_with("/probe-monitors/blend-reports",
                                         json_body={"behaviour_key": "beh-x", "seed": 0, "veto_scope": ["pm_a"]})
