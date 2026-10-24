"""Blend reports — defects Review R1 found and fixed (Feature 041; controls record § Review R1).

* A report reaped by the janitor (`time_limit`, `worker_lost`) or cancelled after the engine's last check was
  turned back into `done` by the task that was still running: the terminal state is now final.
* `GET …/scores` with a view the report does not have answered 409 `rows_pruned` (and built a file path from
  the caller's string): it is now 422 `view_not_in_report`, refused before any file is touched.
* A named `views` request with the blend_fit view also among the tests read `view_missing` (role mismatch):
  FR-14's own code, `blend_fit_overlap`, is now what it reads.
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


@pytest.fixture
def dispatched(monkeypatch):
    from src.workers import probe_blend_tasks

    calls = []
    monkeypatch.setattr(probe_blend_tasks.run_blend_evaluation, "delay",
                        lambda rid: calls.append(rid) or type("T", (), {"id": "t"})())
    return calls


async def _start(client, world):
    response = await client.post(f"{API}/blend-reports", json=bw.start_body(world))
    assert response.status_code == 202, response.text
    return response.json()["report_id"]


async def test_a_reaped_report_is_not_resurrected(client, async_session, request, queued, monkeypatch, data_dir):
    from src.core.cancellation import record_progress
    from src.workers.probe_blend_tasks import run_blend_evaluation

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await _start(client, world)
    real = engine.run

    def reaped_while_running(inputs, *, should_stop, beat):
        out = real(inputs, should_stop=should_stop, beat=beat)
        record_progress("blend_report", report_id, status="failed")      # the janitor, mid-run
        return out

    monkeypatch.setattr(engine, "run", reaped_while_running)
    outcome = await asyncio.to_thread(run_blend_evaluation, report_id)
    assert outcome["status"] == "failed"
    envelope = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    assert envelope["status"] == "failed" and envelope["report_sha256"] is None
    blends = data_dir / "probe_monitors" / "blends"
    assert not (blends / report_id).exists() and not (blends / f"{report_id}.staging").exists()


async def test_a_cancel_after_the_last_check_is_honoured(client, async_session, request, queued, monkeypatch, data_dir):
    from src.core.cancellation import request_cancel
    from src.workers.probe_blend_tasks import run_blend_evaluation

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await _start(client, world)
    real = engine.run

    def cancelled_at_the_end(inputs, *, should_stop, beat):
        out = real(inputs, should_stop=should_stop, beat=beat)
        request_cancel("blend_report", report_id)                        # the operator, after the last check
        return out

    monkeypatch.setattr(engine, "run", cancelled_at_the_end)
    outcome = await asyncio.to_thread(run_blend_evaluation, report_id)
    assert outcome["status"] == "cancelled"
    envelope = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    assert envelope["status"] == "cancelled" and envelope["report_sha256"] is None
    assert not (data_dir / "probe_monitors" / "blends" / report_id).exists()


async def test_scores_of_a_view_the_report_does_not_have_is_refused(client, async_session, request, monkeypatch):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    for view_id in ("pmd_not_here", "../../etc/passwd"):
        response = await client.get(f"{API}/blend-reports/{report_id}/scores",
                                     params={"candidate": "stack", "view_id": view_id})
        assert response.status_code == 422, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "view_not_in_report"
        assert world.blend_fit.id in detail["views"] and world.calibration.id in detail["views"]
    ok = await client.get(f"{API}/blend-reports/{report_id}/scores",
                          params={"candidate": "stack", "view_id": world.tests[0].id})
    assert ok.status_code == 200 and ok.json()["total"] > 0


async def test_named_blend_fit_among_the_tests_reads_blend_fit_overlap(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    views = {"blend_fit": world.blend_fit.id, "tests": [world.blend_fit.id, *[v.id for v in world.tests]],
             "calibration": world.calibration.id}
    response = await client.post(f"{API}/blend-reports", json={
        "behaviour_key": world.key, "model": world.model.id, "views": views,
        "calibration_sample_rows": world.sample_rows})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "blend_fit_overlap" and detail["test_views"] == [world.blend_fit.id]
    assert dispatched == []


async def test_members_of_different_scope_are_incompatible(client, async_session, request, dispatched):
    """R1-16 (FR-14): scope is the third compatibility field; dropping it from the check let a blend mix a
    whole-conversation member with an assistant-only one (precision and render were pinned, scope was not)."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    world.runs[world.members[1].id].config = {"scope": "assistant"}
    await async_session.commit()
    response = await client.post(f"{API}/blend-reports", json=bw.start_body(world))
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "members_incompatible" and detail["field"] == "scope"
    assert detail["values"][world.members[1].id] == "assistant"
    assert dispatched == []


async def test_a_new_decision_revision_with_the_same_digest_is_stale(client, async_session, request, monkeypatch):
    """R1-18 (FR-17): A → B → A brings the digest back while the revision moves on; the report recorded the
    decision IN FORCE at start by both, and staleness compares both (dropping the revision half survived)."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    member = world.members[1]
    history = list(member.calibration_history or [])
    top = max([e["revision"] for e in history if isinstance(e, dict) and isinstance(e.get("revision"), int)] or [1])
    member.calibration_history = history + [{"revision": top + 1}]
    await async_session.commit()
    body = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    assert body["stale"] is True
    assert [r["code"] for r in body["stale_reasons"]] == ["decision_changed"]
    reason = body["stale_reasons"][0]
    assert reason["probe_id"] == member.id
    assert reason["current"]["decision_revision"] == top + 1
    assert reason["current"]["decision_digest"] == reason["recorded"]["decision_digest"]


async def test_a_new_version_with_another_rule_set_is_stale(client, async_session, request, monkeypatch):
    """R1-25 (FR-17): the rule set is the first of the three version fields staleness compares; the suite
    pinned the other two only."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    document = {"format": "midataworks.rule-set/v1", "name": "r1-rules", "behaviour_key": world.key,
                "description": "d", "signals": [{"id": "s1", "kind": "pattern", "pattern": "x", "reads": "text"}],
                "expressions": {"hit": {"signal": "s1"}}, "outputs": [], "cases": []}
    made = await client.post(f"{API}/behaviours/{world.key}/versions",
                             json={"base_version": 2, "rule_set": {"document": document}})
    assert made.status_code == 201, made.text
    body = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    assert [r["code"] for r in body["stale_reasons"]] == ["version_changed"]
    assert body["stale_reasons"][0]["fields"] == ["rule_set"]


async def test_suggesting_a_report_of_another_model_is_refused(client, async_session, request, monkeypatch):
    """R1-26 (FR-21): a suggestion is per (behaviour, model); a report of this behaviour on ANOTHER model must
    not become this model's advice (the key half of the check was pinned, the model half was not)."""
    from tests.support import family_fixtures as ff

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    other = await ff.model(async_session)
    await async_session.commit()
    response = await client.put(f"{API}/behaviours/{world.key}/suggested-blend",
                                json={"model": other.id, "report_id": report_id, "candidate": "stack"})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "report_not_of_behaviour" and detail["report_model"] == world.model.id
    current = await client.get(f"{API}/behaviours/{world.key}/suggested-blend", params={"model": other.id})
    assert current.status_code == 404


async def test_the_task_writes_a_heartbeat_at_every_beat(client, async_session, request, queued, monkeypatch):
    """B-23 (T31): the engine's `beat` must reach the row. Without the write, the janitor reads a running report
    as `worker_lost` ten minutes after it started, whatever the worker is doing (the removal survived)."""
    from src.core import cancellation
    from src.workers.probe_blend_tasks import run_blend_evaluation

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await _start(client, world)
    beats, writes = [], []
    real_run, real_progress = engine.run, cancellation.record_progress

    def counting_run(inputs, *, should_stop, beat):
        def counted():
            beats.append(1)
            beat()

        return real_run(inputs, should_stop=should_stop, beat=counted)

    def spy(kind, target_id, **kw):
        if "heartbeat_at" in kw:
            writes.append((kind, target_id))
        return real_progress(kind, target_id, **kw)

    monkeypatch.setattr(engine, "run", counting_run)
    monkeypatch.setattr(cancellation, "record_progress", spy)
    outcome = await asyncio.to_thread(run_blend_evaluation, report_id)
    assert outcome["status"] == "done"
    assert beats and writes == [("blend_report", report_id)] * len(beats)


async def test_the_suggested_blend_reads_stale_when_its_report_does(client, async_session, request, monkeypatch):
    """B-26 (FR-21): the advice an application reads says when its report went stale (a member re-cut), with the
    rerun call; the envelope's `stale` flag was unpinned (the reasons list was)."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    marked = await client.put(f"{API}/behaviours/{world.key}/suggested-blend",
                              json={"model": world.model.id, "report_id": report_id, "candidate": "stack"})
    assert marked.status_code == 200, marked.text
    assert marked.json()["stale"] is False
    world.members[0].decision_digest = "d" * 64
    await async_session.commit()
    body = (await client.get(f"{API}/behaviours/{world.key}/suggested-blend", params={"model": world.model.id})).json()
    assert body["stale"] is True and [r["code"] for r in body["stale_reasons"]] == ["decision_changed"]
    assert body["next"][0]["tool"] == "rerun_blend_report" and body["next"][0]["arguments"] == {"report_id": report_id}


async def test_a_blend_calibrated_at_another_fpr_is_not_judged_against_the_criterion(client, async_session, request,
                                                                                     monkeypatch):
    """B-27 (FR-20): a criterion with no target of its own is judged at the VERSION's target; a blend calibrated
    at another rate is not assessable against it (`target_fpr_mismatch`), never silently met or not met."""
    criteria = [{"id": "crit-blend", "detector": "blend_alone", "subset": "all", "min_recall": 0.3}]
    world = await bw.build(client, async_session, seed=request.node.nodeid, criteria=criteria, target_fpr=0.05)
    report_id = await bw.done_report(client, monkeypatch, world, target_fpr=0.2)
    made = await client.post(f"{API}/behaviours/{world.key}/success-reports",
                             json={"model_id": world.model.id, "probe_id": world.members[0].id,
                                   "blend_report_id": report_id, "candidate": "stack"})
    assert made.status_code == 200, made.text
    criterion = made.json()["report"]["criteria"][0]
    assert criterion["verdict"] == "not_assessable"
    mismatch = [r for r in criterion["reasons"] if r["code"] == "target_fpr_mismatch"]
    assert len(mismatch) == 1 and mismatch[0]["criterion_target_fpr"] == pytest.approx(0.05)
    assert mismatch[0]["blend_target_fpr"] == pytest.approx(0.2)
