"""Pruning a report's per-row values (Feature 041 FR-19; FTASKS 7.6).

The operator's alone (agents get 403 at the route's dependency); refused behind the current suggested
blend; the summary and `report.json` stay byte-identical; the scores route then answers `rows_pruned`.
"""
from __future__ import annotations

import pytest

from src.core import runtime_settings
from src.core.config import settings
from tests.support import blend_world as bw

API = "/api/v1/probe-monitors"
AGENT = {"X-MiStudio-Agent": "mcp"}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


async def test_agent_prune_403(client, async_session, request, monkeypatch):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    response = await client.post(f"{API}/blend-reports/{report_id}/prune-rows", json={}, headers=AGENT)
    assert response.status_code == 403
    assert (await client.get(f"{API}/blend-reports/{report_id}")).json()["rows_pruned_at"] is None


async def test_in_use_refused(client, async_session, request, monkeypatch):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    marked = await client.put(f"{API}/behaviours/{world.key}/suggested-blend",
                              json={"model": world.model.id, "report_id": report_id, "candidate": "max"})
    assert marked.status_code == 200, marked.text
    response = await client.post(f"{API}/blend-reports/{report_id}/prune-rows", json={})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "report_in_use"


async def test_summary_bytes_unchanged_after_prune(client, async_session, request, monkeypatch, data_dir):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    before = (await client.get(f"{API}/blend-reports/{report_id}/report.json")).content
    directory = data_dir / "probe_monitors" / "blends" / report_id
    assert list(directory.glob("*.npy"))
    response = await client.post(f"{API}/blend-reports/{report_id}/prune-rows", json={"reason": "space"})
    assert response.status_code == 200 and response.json()["rows_pruned_by"] == "operator"
    assert not list(directory.glob("*.npy")) and (directory / "report.json").is_file()
    assert (await client.get(f"{API}/blend-reports/{report_id}/report.json")).content == before


async def test_scores_after_prune_409_rows_pruned(client, async_session, request, monkeypatch):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    await client.post(f"{API}/blend-reports/{report_id}/prune-rows", json={})
    response = await client.get(f"{API}/blend-reports/{report_id}/scores",
                                params={"candidate": "stack", "view_id": world.tests[0].id})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "rows_pruned"
