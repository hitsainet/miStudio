"""The four S2 checklist steps follow real state (Feature 040 FR-41; FTASKS 12.4).

From nothing to a stored verdict, one fact at a time: each step is asserted `not_done` with its `next`
before the fact exists and `done` after, on real rows and arrays (`build_world`).
"""
from __future__ import annotations

import pytest

from src.core.config import settings
from tests.support.next_calls import assert_next_calls_are_live
from tests.support.random_behaviour import rng_for
from tests.unit.test_detector_success_report import build_world, held

API = "/api/v1/probe-monitors"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    return tmp_path


async def steps(client, key):
    body = (await client.get(f"{API}/behaviours/{key}/checklist")).json()
    return {s["id"]: s for s in body["steps"]}


async def test_the_detector_steps_follow_the_records(client, async_session, data_dir, request):
    from src.models.probe_monitor import ProbeMonitorEvaluation

    w = await build_world(client, async_session, data_dir, rng_for(request.node.nodeid), criteria=lambda w: [held(w)])
    s = await steps(client, w.key)
    assert s["views_registered"]["state"] == "done"
    assert s["evaluated"]["state"] == "done" and w.ood_eval.id in s["evaluated"]["evidence"]
    assert s["breakdown_available"]["state"] == "done"
    assert s["breakdown_available"]["next"][0]["tool"] == "get_probe_breakdown"
    assert s["success_report"]["state"] == "not_done"
    assert_next_calls_are_live([n for step in s.values() for n in step.get("next", [])])

    made = (await client.post(f"{API}/behaviours/{w.key}/success-reports", json={})).json()
    s = await steps(client, w.key)
    assert (s["success_report"]["state"], s["success_report"]["verdict"]) == ("done", made["report"]["overall_verdict"])

    await async_session.delete(await async_session.get(ProbeMonitorEvaluation, w.ood_eval.id))
    await async_session.commit()
    s = await steps(client, w.key)
    assert s["evaluated"]["state"] == "not_done"
    assert s["evaluated"]["next"][0]["arguments"] == {"probe_id": w.probes[0].id, "dataset_ids": [w.ood.id]}


async def test_a_sub_question_without_a_view_is_missing(client, async_session, data_dir, request):
    w = await build_world(client, async_session, data_dir, rng_for(request.node.nodeid), criteria=lambda w: [held(w)])
    await client.post(f"{API}/behaviours/{w.key}/versions", json={
        "base_version": 1, "sub_questions": [{"key": "sq-uncovered", "question": "q", "positive_meaning": "p"}]})
    s = await steps(client, w.key)
    assert s["views_registered"]["state"] == "not_done" and "sq-uncovered" in s["views_registered"]["reason"]
