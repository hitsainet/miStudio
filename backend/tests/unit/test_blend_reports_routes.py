"""The blend-report routes (Feature 041 FR-16, FR-19; FTASKS 9.1, 9.6).

Served routes are read from `app.openapi()` (miStudio memory `app-routes-is-not-a-route-list`). Paging is
stable; a report changes nothing a probe route serves; the report object has every key and no row text.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.core import runtime_settings
from src.core.config import settings
from tests.support import blend_world as bw

API = "/api/v1/probe-monitors"

ROUTES = {
    ("post", "/api/v1/probe-monitors/blend-reports"),
    ("get", "/api/v1/probe-monitors/blend-reports"),
    ("get", "/api/v1/probe-monitors/blend-reports/{report_id}"),
    ("get", "/api/v1/probe-monitors/blend-reports/{report_id}/report.json"),
    ("get", "/api/v1/probe-monitors/blend-reports/{report_id}/scores"),
    ("post", "/api/v1/probe-monitors/blend-reports/{report_id}/rerun"),
    ("post", "/api/v1/probe-monitors/blend-reports/{report_id}/cancel"),
    ("post", "/api/v1/probe-monitors/blend-reports/{report_id}/prune-rows"),
    ("get", "/api/v1/probe-monitors/behaviours/{key}/suggested-blend"),
    ("put", "/api/v1/probe-monitors/behaviours/{key}/suggested-blend"),
    ("get", "/api/v1/probe-monitors/behaviours/{key}/suggested-blend/history"),
}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


def test_routes_in_openapi():
    from src.main import app

    paths = app.openapi()["paths"]
    served = {(method, path) for path, ops in paths.items() for method in ops}
    assert ROUTES <= served, sorted(ROUTES - served)


async def test_pages_union_equals_unpaged(client, async_session, request, monkeypatch):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    runtime_settings.apply({"blend_max_concurrent": 8})
    from src.workers import probe_blend_tasks

    monkeypatch.setattr(probe_blend_tasks.run_blend_evaluation, "delay", lambda rid: type("T", (), {"id": "t"})())
    made = [(await client.post(f"{API}/blend-reports", json=bw.start_body(world, seed=i))).json()["report_id"]
            for i in range(5)]
    whole = (await client.get(f"{API}/blend-reports", params={"behaviour_key": world.key})).json()
    assert [r["report_id"] for r in whole["reports"]] == list(reversed(made))      # newest first
    paged, offset = [], 0
    while offset is not None:
        page = (await client.get(f"{API}/blend-reports", params={"behaviour_key": world.key, "offset": offset,
                                                                 "limit": 2})).json()
        paged += [r["report_id"] for r in page["reports"]]
        offset = page["next_offset"]
    assert paged == [r["report_id"] for r in whole["reports"]]


async def test_report_page_size_is_the_runtime_setting(client, async_session, request):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    runtime_settings.apply({"blend_report_page_size": 2})
    over = await client.get(f"{API}/blend-reports", params={"behaviour_key": world.key, "limit": 3})
    assert over.status_code == 422 and over.json()["detail"]["setting"] == "blend_report_page_size"
    assert (await client.get(f"{API}/blend-reports", params={"behaviour_key": world.key})).json()["limit"] == 2


async def test_probe_scores_route_byte_unchanged_by_a_report(client, async_session, request, monkeypatch):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    probe = world.members[0]
    evaluation = world.evaluations[probe.id][world.tests[0].id]
    url = f"{API}/probes/{probe.id}/evaluations/{evaluation.id}/scores"
    before = (await client.get(url)).content
    await bw.done_report(client, monkeypatch, world)
    assert (await client.get(url)).content == before


async def test_scores_page_and_limit(client, async_session, request, monkeypatch):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    view = world.tests[0].id
    whole = (await client.get(f"{API}/blend-reports/{report_id}/scores",
                              params={"candidate": "stack", "view_id": view})).json()
    assert whole["total"] == len(whole["rows"]) and whole["next_offset"] is None
    positions = [r["position"] for r in whole["rows"]]
    assert positions == sorted(positions)
    paged, offset = [], 0
    while offset is not None:
        page = (await client.get(f"{API}/blend-reports/{report_id}/scores",
                                 params={"candidate": "stack", "view_id": view, "offset": offset, "limit": 50})).json()
        paged += page["rows"]
        offset = page["next_offset"]
    assert paged == whole["rows"]
    verdict = (await client.get(f"{API}/blend-reports/{report_id}/scores",
                                params={"candidate": "k_of_n:2", "view_id": view})).json()
    assert all(r["verdict"] == (r["value"] >= 2) for r in verdict["rows"])
    runtime_settings.apply({"blend_scores_page_limit": 1000})
    over = await client.get(f"{API}/blend-reports/{report_id}/scores",
                            params={"candidate": "stack", "view_id": view, "limit": 1001})
    assert over.status_code == 422 and over.json()["detail"]["setting"] == "blend_scores_page_limit"


async def test_report_has_every_key_and_no_text(client, async_session, request, monkeypatch, data_dir):
    """FTDD §6.4: every top-level key present; no row text — a sentinel planted in the views' source text
    never reaches the bytes (the rows are never read; only ids, counts and positions are)."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    for view in (world.blend_fit, *world.tests):
        view.name = "SENTINEL-ROW-TEXT-" + view.id
    await async_session.commit()
    report_id = await bw.done_report(client, monkeypatch, world)
    raw = (await client.get(f"{API}/blend-reports/{report_id}/report.json")).content
    report = json.loads(raw)
    assert set(report) >= {"format", "engine_version", "report_id", "behaviour", "model_id", "dataset_version", "views",
                           "members", "alignment", "standardisation", "folds", "candidates", "members_own_figures",
                           "ranking", "flags", "statement", "has_rung"}
    assert report["has_rung"] is False and report["format"] == "mistudio.blend-report/1"
    assert b"SENTINEL" not in raw
    assert not any(key == "rung" for c in report["candidates"] for key in c)
    from src.services.probe_blend.report_format import canonical_bytes

    assert canonical_bytes(report) == raw


async def test_view_purge_plan_lists_no_041_blocker(client, async_session, request, monkeypatch):
    """T6 / 5.4: no FK from a report to a view — purging a view a report cites is neither blocked nor
    cascaded; the report stays readable (and reads stale only through what it records)."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    before = (await client.get(f"{API}/blend-reports/{report_id}/report.json")).content
    view = world.tests[1]
    assert (await client.post(f"{API}/datasets/{view.id}/archive", json={})).status_code == 200
    plan = (await client.post(f"{API}/datasets/purge/plan", json={"ids": [view.id]})).json()
    entry = plan["items"][0]
    # The view's own evaluations block it (035's rule); NOTHING of 041's does — a report is not a blocker.
    assert {b["code"] for b in entry["blockers"]} == {"evaluation_on_it"}
    assert not any("blend" in json.dumps(b) or "pbr_" in json.dumps(b) for b in entry["blockers"])
    # Remove those evaluations (as the operator would) and the purge goes through, the report untouched.
    from sqlalchemy import delete

    from src.models.probe_monitor import ProbeMonitorEvaluation

    await async_session.execute(delete(ProbeMonitorEvaluation).where(ProbeMonitorEvaluation.dataset_id == view.id))
    await async_session.commit()
    purged = await client.post(f"{API}/datasets/purge", json={"ids": [view.id], "reason": "privacy"})
    assert purged.status_code == 200 and purged.json()["results"][0]["outcome"] == "purged", purged.text
    assert (await client.get(f"{API}/blend-reports/{report_id}/report.json")).content == before


async def test_run_delete_keeps_report_rows(client, async_session, request, monkeypatch):
    from src.models.probe_blend import ProbeBlendReportMember

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    run_id = world.members[2].run_id
    assert (await client.delete(f"{API}/runs/{run_id}")).status_code == 204
    from sqlalchemy import select

    members = (await async_session.execute(select(ProbeBlendReportMember.probe_id).where(
        ProbeBlendReportMember.report_id == report_id))).scalars().all()
    assert len(members) == 3
    envelope = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    assert envelope["status"] == "done" and [r["code"] for r in envelope["stale_reasons"]] == ["member_gone"]
