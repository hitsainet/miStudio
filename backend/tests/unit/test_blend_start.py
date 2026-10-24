"""Starting a blend evaluation (Feature 041 FR-6, FR-7; T26; FTASKS 7.1, 7.5, 8.1).

Through the REAL route on a real database: the report row is committed `queued` and the task is
dispatched ONCE with the report id (payload and count); the task then runs for real (`run_blend_evaluation`
called in-process, its own sync session) and writes the canonical report. View resolution, the
concurrency cap and the advisory-lock race are each pinned.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest
from sqlalchemy import select

from src.core import runtime_settings
from src.core.config import settings
from tests.support import blend_world as bw
from tests.support import family_fixtures as ff

API = "/api/v1/probe-monitors"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


class Task:
    def __init__(self, i):
        self.id = f"task-{i}"


@pytest.fixture
def dispatched(monkeypatch):
    sent = []

    def delay(report_id):
        sent.append(report_id)
        return Task(len(sent))

    from src.workers import probe_blend_tasks

    monkeypatch.setattr(probe_blend_tasks.run_blend_evaluation, "delay", delay)
    return sent


async def start(client, world, **extra):
    return await client.post(f"{API}/blend-reports", json=bw.start_body(world, **extra))


async def test_returns_before_the_fit(client, async_session, request, dispatched):
    """202 `queued` with the resolved inputs; the engine has not run (no report file yet); the task was
    dispatched once, with exactly the report id."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    response = await start(client, world)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "queued" and dispatched == [body["report_id"]]
    assert body["resolved"]["views"]["blend_fit"] == world.blend_fit.id
    assert sorted(t["id"] for t in body["resolved"]["views"]["tests"]) == sorted(v.id for v in world.tests)
    assert body["members"] == [m.id for m in world.members]
    assert "stack" in body["candidates"]
    report = (await client.get(f"{API}/blend-reports/{body['report_id']}")).json()
    assert report["status"] == "queued" and "report" not in report


async def test_the_task_writes_the_report(client, async_session, request, dispatched):
    from src.workers.probe_blend_tasks import run_blend_evaluation

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = (await start(client, world)).json()["report_id"]
    outcome = await asyncio.to_thread(run_blend_evaluation, report_id)
    assert outcome["status"] == "done", outcome
    envelope = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    assert envelope["status"] == "done" and envelope["stale"] is False and envelope["has_rung"] is False
    raw = await client.get(f"{API}/blend-reports/{report_id}/report.json")
    assert raw.headers["content-type"].startswith("application/json")
    assert hashlib.sha256(raw.content).hexdigest() == envelope["report_sha256"]
    report = envelope["report"]
    assert report["ranking"]["basis"] == "held_out_min"
    assert report["has_rung"] is False and "no_rule_set" in report["flags"]
    # A second run of the task on the same (terminal) row is not a restart.
    assert (await asyncio.to_thread(run_blend_evaluation, report_id))["status"] == "done"


async def test_views_resolve_one_per_role(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    body = (await start(client, world)).json()
    views = body["resolved"]["views"]
    assert views["calibration"] == {"id": world.calibration.id, "sample_rows": world.sample_rows}
    held = {t["id"]: t["held_out"] for t in views["tests"]}
    assert held == {world.tests[0].id: True, world.tests[1].id: False}


async def test_second_calibration_view_ambiguous(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    ds = await ff.source(async_session)
    extra = await ff.view(async_session, ds, key=world.key, role="calibration", dataset_version="dv-blend")
    await async_session.commit()
    response = await start(client, world)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "view_ambiguous" and detail["role"] == "calibration"
    assert sorted(detail["candidates"]) == sorted([world.calibration.id, extra.id])
    assert dispatched == []


async def test_named_views_bypass_resolution(client, async_session, request, dispatched):
    """FR-30: a second calibration view would make resolution ambiguous; naming the views works."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    ds = await ff.source(async_session)
    await ff.view(async_session, ds, key=world.key, role="calibration", dataset_version="dv-blend")
    await async_session.commit()
    views = {"blend_fit": world.blend_fit.id, "tests": [v.id for v in world.tests], "calibration": world.calibration.id}
    response = await client.post(f"{API}/blend-reports", json={
        "behaviour_key": world.key, "model": world.model.id, "views": views,
        "calibration_sample_rows": world.sample_rows})
    assert response.status_code == 202, response.text


async def test_third_start_over_cap_of_two_refused(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    runtime_settings.apply({"blend_max_concurrent": 2})
    assert (await start(client, world)).status_code == 202
    assert (await start(client, world)).status_code == 202
    third = await start(client, world)
    assert third.status_code == 429
    detail = third.json()["detail"]
    assert (detail["code"], detail["setting"], detail["value"], detail["active"]) == (
        "cap_exceeded", "blend_max_concurrent", 2, 2)
    assert len(dispatched) == 2
    runtime_settings.apply({"blend_max_concurrent": 3})
    assert (await start(client, world)).status_code == 202


async def test_concurrent_starts_cannot_both_pass(client, async_session, request, dispatched, async_engine):
    """T26 on Postgres: two starts at a cap of one, each in its own session and transaction — the advisory
    lock serialises count-then-insert, so exactly one row is admitted."""
    from sqlalchemy.ext.asyncio import AsyncSession as Session

    from src.models.probe_blend import ProbeBlendReport
    from src.services.probe_blend import service

    world = await bw.build(client, async_session, seed=request.node.nodeid)
    runtime_settings.apply({"blend_max_concurrent": 1})
    body = bw.start_body(world)

    async def one():
        async with Session(async_engine, expire_on_commit=False) as session:
            try:
                await service.start(session, dict(body), "operator")
                return "admitted"
            except service.BlendRefused as refusal:
                await session.rollback()
                return refusal.code

    outcomes = await asyncio.gather(one(), one())
    assert sorted(outcomes) == ["admitted", "cap_exceeded"]
    count = (await async_session.execute(select(ProbeBlendReport).where(
        ProbeBlendReport.behaviour_key == world.key))).scalars().all()
    assert len(count) == 1


async def test_the_route_dispatch_is_the_wiring(client, async_session, request, dispatched):
    """Deleting the dispatch leaves the report queued forever: the dispatch is asserted by payload."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    body = (await start(client, world)).json()
    assert dispatched == [body["report_id"]] and body["task_id"] == "task-1"
    from src.models.probe_blend import ProbeBlendReport

    row = await async_session.get(ProbeBlendReport, body["report_id"])
    await async_session.refresh(row)
    assert row.celery_task_id == "task-1"
