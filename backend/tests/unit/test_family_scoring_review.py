"""Family scoring — controls Review R1 found unpinned (Feature 041 FR-22; controls record § Review R1)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.core import runtime_settings
from src.core.config import settings
from src.services import probe_monitor_run
from tests.support import blend_world as bw

API = "/api/v1/probe-monitors"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


@pytest.fixture
def sync_db(async_engine):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(async_engine.url.set(drivername=async_engine.url.drivername.replace("+asyncpg", "")))
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def fake_score_one(db, probe_id, *, text=None, messages=None, loaded=None):
    seed = sum(map(ord, probe_id)) % 97
    return {"probe_id": probe_id, "aggregate": seed / 10.0, "n_scored": 7, "fires": seed % 2 == 0, "threshold": 0.5}


async def test_candidate_params_apply_in_the_reports_member_order(client, async_session, request, sync_db, monkeypatch):
    """B-01: the members may be named in ANY order (an agent's `member_ids`); the stack's mean, std and weights
    are per REPORT member, so the scores are arranged in the report's order, never the request's."""
    from src.models.probe_blend import ProbeBlendReport
    from src.services.probe_family_scoring import score_family

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    report = json.loads((await client.get(f"{API}/blend-reports/{report_id}/report.json")).content)
    row = sync_db.get(ProbeBlendReport, report_id)
    monkeypatch.setattr(probe_monitor_run, "score_one", fake_score_one)
    recorded = [m["probe_id"] for m in report["members"]]
    out = score_family(sync_db, list(reversed(recorded)), text="t", report=report, resolved=row.resolved,
                       candidates=["stack"], loader=lambda run: (None, None, "x"))
    stack = next(c for c in report["candidates"] if c["name"] == "stack")
    scores = np.array([fake_score_one(None, p)["aggregate"] for p in recorded])
    z = (scores - np.array(stack["params"]["mean"])) / np.array(stack["params"]["std"])
    value = float(z @ np.array(stack["params"]["weights"]) + stack["params"]["intercept"])
    assert out["candidates"][0]["value"] == pytest.approx(value, rel=1e-9)


async def test_a_report_of_another_behaviour_is_refused_before_anything_is_queued(client, async_session, request,
                                                                                   monkeypatch):
    """B-04: a family score names a blend report; one of ANOTHER behaviour would apply that behaviour's weights
    to this family. Refused 422 `report_not_of_behaviour`, nothing queued."""
    queued = []

    def gpu_delay(task, gpu):
        return lambda **kw: queued.append(kw) or type("T", (), {"id": "t"})()

    monkeypatch.setattr("src.services.gpu_dispatch.gpu_delay", gpu_delay)
    mine = await bw.build(client, async_session, seed=request.node.nodeid + "-mine")
    theirs = await bw.build(client, async_session, seed=request.node.nodeid + "-theirs")
    assert mine.key != theirs.key
    report_id = await bw.done_report(client, monkeypatch, theirs)
    response = await client.post(f"{API}/behaviours/{mine.key}/family/score-batch", json={
        "model": mine.model.id, "inputs": [{"text": "x"}], "blend_report_id": report_id, "candidates": ["stack"]})
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "report_not_of_behaviour"
    assert queued == []
