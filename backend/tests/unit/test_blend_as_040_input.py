"""040's reports take a blend as input (Feature 041 FR-20; FTASKS 9.5).

* The report's per-subgroup figures ARE 040's: recomputed from the report's own files through
  `probe_breakdown.groups_of`, they equal the bytes in `report.json`.
* 040's success report judges a `blend_alone` criterion on a named (or the suggested) blend candidate, by
  the SAME function the blend report used — so the success report's figures equal the report's own
  `criteria` entry; a named stale report is refused `blend_report_stale`.

The rules ∪ blend branch (FTASKS 9.5, 2026-10-28) is pinned in `test_blend_rules_union.py`.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.core import runtime_settings
from src.core.config import settings
from src.services import probe_breakdown, view_columns
from src.services.probe_blend import engine
from tests.support import blend_world as bw

API = "/api/v1/probe-monitors"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


async def declare(async_session, view, column, values, size=5000, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, len(values), size).astype(np.int32)
    entry = view_columns._write_column(view_columns.view_dir(view.id), column, codes, list(values))
    view.group_columns = [column]
    view.column_snapshots = {column: entry}
    await async_session.commit()
    return codes


async def test_breakdown_of_candidate_equals_report_bytes(client, async_session, request, monkeypatch, data_dir):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    ood = world.tests[0]
    codes = await declare(async_session, ood, "source", ["a", "b", "c"])
    report_id = await bw.done_report(client, monkeypatch, world)
    report = json.loads((await client.get(f"{API}/blend-reports/{report_id}/report.json")).content)
    entry = next(c for c in report["candidates"] if c["name"] == "stack")
    stored = entry["tests"][ood.id]["breakdown"]["source"]
    rows = data_dir / "probe_monitors" / "blends" / report_id
    positions = np.load(rows / f"positions__{ood.id}.npy")
    labels = np.load(rows / f"labels__{ood.id}.npy")
    values = np.load(rows / f"c{entry['index']:02d}__{ood.id}.npy")
    fires = values >= entry["calibration"]["threshold"]
    again = probe_breakdown.groups_of(codes[positions], labels == 1, fires, values, np.ones(values.shape[0], bool),
                                      ["a", "b", "c"], 30)
    from src.services.probe_blend.report_format import round_floats

    assert round_floats(again) == stored


async def blend_world(client, async_session, request):
    criteria = [{"id": "crit-blend", "detector": "blend_alone", "subset": "all", "min_recall": 0.3}]
    return await bw.build(client, async_session, seed=request.node.nodeid, criteria=criteria)


async def test_success_report_judges_the_named_candidate(client, async_session, request, monkeypatch):
    world = await blend_world(client, async_session, request)
    report_id = await bw.done_report(client, monkeypatch, world)
    report = json.loads((await client.get(f"{API}/blend-reports/{report_id}/report.json")).content)
    made = await client.post(f"{API}/behaviours/{world.key}/success-reports",
                             json={"model_id": world.model.id, "probe_id": world.members[0].id,
                                   "blend_report_id": report_id, "candidate": "stack"})
    assert made.status_code == 200, made.text
    criterion = made.json()["report"]["criteria"][0]
    own = next(c for c in report["candidates"] if c["name"] == "stack")["criteria"][0]
    assert criterion["verdict"] == own["verdict"] in ("met", "not_met")
    assert criterion["figures"]["positives"] == own["figures"]["positives"]
    assert criterion["figures"]["flagged_positives"] == own["figures"]["flagged_positives"]
    assert criterion["blend"] == {"report_id": report_id, "candidate": "stack",
                                  "report_sha256": made.json()["report"]["criteria"][0]["blend"]["report_sha256"]}


async def test_the_suggested_blend_is_the_default(client, async_session, request, monkeypatch):
    world = await blend_world(client, async_session, request)
    report_id = await bw.done_report(client, monkeypatch, world)
    body = {"model_id": world.model.id, "probe_id": world.members[0].id}
    before = (await client.post(f"{API}/behaviours/{world.key}/success-reports", json=body)).json()
    assert before["report"]["criteria"][0]["reasons"][0]["code"] == "no_blend_report"
    await client.put(f"{API}/behaviours/{world.key}/suggested-blend",
                     json={"model": world.model.id, "report_id": report_id, "candidate": "max"})
    after = (await client.post(f"{API}/behaviours/{world.key}/success-reports", json=body)).json()
    assert after["id"] != before["id"]
    assert after["report"]["criteria"][0]["blend"]["candidate"] == "max"


async def test_stale_candidate_refused(client, async_session, request, monkeypatch):
    world = await blend_world(client, async_session, request)
    report_id = await bw.done_report(client, monkeypatch, world)
    world.members[1].decision_digest = "9" * 64
    await async_session.commit()
    response = await client.post(f"{API}/behaviours/{world.key}/success-reports",
                                 json={"model_id": world.model.id, "probe_id": world.members[0].id,
                                       "blend_report_id": report_id, "candidate": "stack"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "blend_report_stale"


async def test_the_success_report_calls_the_engines_criterion_function(client, async_session, request, monkeypatch):
    world = await blend_world(client, async_session, request)
    report_id = await bw.done_report(client, monkeypatch, world)
    calls = []
    real = engine.blend_criterion_verdict
    monkeypatch.setattr(engine, "blend_criterion_verdict", lambda *a, **k: calls.append(a[0]["id"]) or real(*a, **k))
    await client.post(f"{API}/behaviours/{world.key}/success-reports",
                      json={"model_id": world.model.id, "probe_id": world.members[0].id,
                            "blend_report_id": report_id, "candidate": "stack"})
    assert calls == ["crit-blend"]
