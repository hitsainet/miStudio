"""Cancelling a blend evaluation (Feature 041 OQ-5 (a); T32; FTASKS 8.5).

Scope `blend_report` (this file's name is how `test_cancel_registry_completeness.py` finds its Shape-A
test). A queued report is cancelled at once and the task then does nothing; a running one stops at the
engine's next check between folds or candidates, unlinks its staging directory and keeps every input.
"""
from __future__ import annotations

import asyncio

import pytest

from src.core import runtime_settings
from src.core.config import settings
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


@pytest.fixture
def queued(monkeypatch):
    from src.workers import probe_blend_tasks

    monkeypatch.setattr(probe_blend_tasks.run_blend_evaluation, "delay", lambda rid: type("T", (), {"id": "t"})())


async def test_cancel_queued_is_immediate(client, async_session, request, queued):
    from src.workers.probe_blend_tasks import run_blend_evaluation

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = (await client.post(f"{API}/blend-reports", json=bw.start_body(world))).json()["report_id"]
    response = await client.post(f"{API}/blend-reports/{report_id}/cancel")
    assert response.status_code == 202 and response.json()["status"] == "cancelled"
    outcome = await asyncio.to_thread(run_blend_evaluation, report_id)
    assert outcome["status"] == "cancelled"
    assert (await client.get(f"{API}/blend-reports/{report_id}")).json()["status"] == "cancelled"


async def test_cancel_between_folds_stops_and_keeps_inputs(client, async_session, request, queued, monkeypatch, data_dir):
    from src.core.cancellation import request_cancel
    from src.workers.probe_blend_tasks import run_blend_evaluation

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = (await client.post(f"{API}/blend-reports", json=bw.start_body(world))).json()["report_id"]
    checks = []
    real = engine.run

    def cancelled_mid_run(inputs, *, should_stop, beat):
        def counting():
            checks.append(1)
            if len(checks) == 3:
                request_cancel("blend_report", report_id)       # the operator, mid-run
            should_stop()

        return real(inputs, should_stop=counting, beat=beat)

    monkeypatch.setattr(engine, "run", cancelled_mid_run)
    outcome = await asyncio.to_thread(run_blend_evaluation, report_id)
    assert outcome["status"] == "cancelled" and len(checks) == 3     # stopped AT the check that saw it
    envelope = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    assert envelope["status"] == "cancelled" and envelope["report_sha256"] is None
    blends = data_dir / "probe_monitors" / "blends"
    assert not (blends / f"{report_id}.staging").exists() and not (blends / report_id).exists()
    for per_view in world.evaluations.values():                  # every input is kept
        for evaluation in per_view.values():
            assert all(__import__("pathlib").Path(p).is_file() for p in (
                evaluation.scores_path, evaluation.positions_path, evaluation.labels_path, evaluation.n_scored_path))


async def test_the_engine_checks_between_folds_and_candidates():
    """`should_stop` is called once per fold (inside each score candidate's CV) and once per candidate."""
    from tests.support import blend_fixtures as bf

    inputs = bf.build(3, candidates=["stack", "max"], folds=4)
    calls = []
    engine.run(inputs, should_stop=lambda: calls.append("stop"), beat=lambda: calls.append("beat"))
    assert calls.count("stop") == 2 + 2 * 4 and calls.count("beat") == calls.count("stop")
