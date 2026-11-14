"""Cancelling a rules ∪ probe evaluation (Feature 040 S3; FR-37; FTASKS 21.5).

Scope `"probe_rules_evaluation"` (this file's name is how `test_cancel_registry_completeness.py` finds its Shape-A
test): the REAL task body runs in-process; a cancel requested part-way stops it between chunks of rows (fewer
engine calls than rows) and leaves a `cancelled` record with no body; the operator route reaches the scope; a
cancelled row never blocks a rerun.
"""
from __future__ import annotations

import asyncio

import pytest

from src.core.config import settings
from tests.support import rules_world as rw

API = rw.API


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


async def test_cancelling_stops_between_chunks(client, async_session, data_dir, monkeypatch, request):
    import src.rule_signals as rule_signals
    from src.core.cancellation import request_cancel
    from src.services import rules_evaluation
    from src.workers.rules_evaluation_tasks import run_rules_evaluation

    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    monkeypatch.setattr(rules_evaluation, "CHUNK_ROWS", 4)
    monkeypatch.setattr(rules_evaluation, "_dispatch", lambda rid: "celery-x")
    started = (await client.post(f"{API}/probes/{world.probe.id}/rules-evaluations",
                                 json={"view_ids": [world.eval_view.id], "calibration": False})).json()
    calls = {"n": 0}
    real = rule_signals.evaluate_many

    def cancelling(*args, **kwargs):  # Feature 040 S4: the task evaluates one CHUNK of rows per call
        calls["n"] += 1
        if calls["n"] == 2:
            request_cancel("probe_rules_evaluation", started["evaluation_id"])
        return real(*args, **kwargs)

    monkeypatch.setattr(rule_signals, "evaluate_many", cancelling)
    outcome = await asyncio.to_thread(run_rules_evaluation, started["evaluation_id"])
    assert outcome["status"] == "cancelled"
    assert calls["n"] < 2 * len(rw.EVAL_ROWS) // 4, "the work went on past the cancel"
    read = (await client.get(f"{API}/rules-evaluations/{started['evaluation_id']}")).json()
    assert read["status"] == "cancelled" and read["result"] is None


async def test_the_cancel_route_reaches_the_scope(client, async_session, data_dir, monkeypatch, request):
    from src.services import rules_evaluation

    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    monkeypatch.setattr(rules_evaluation, "_dispatch", lambda rid: "celery-y")
    started = (await client.post(f"{API}/probes/{world.probe.id}/rules-evaluations",
                                 json={"view_ids": [world.eval_view.id], "calibration": False})).json()
    out = (await client.post(f"{API}/rules-evaluations/{started['evaluation_id']}/cancel")).json()
    assert out["accepted"] is True
    read = (await client.get(f"{API}/rules-evaluations/{started['evaluation_id']}")).json()
    assert read["status"] == "cancelled"
    # a cancelled row never blocks a rerun (the partial unique index covers live rows only)
    rerun = (await client.post(f"{API}/probes/{world.probe.id}/rules-evaluations",
                               json={"view_ids": [world.eval_view.id], "calibration": False})).json()
    assert rerun["evaluation_id"] != started["evaluation_id"] and rerun["duplicate"] is False
