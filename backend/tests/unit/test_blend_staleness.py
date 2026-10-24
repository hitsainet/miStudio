"""A report goes stale on read and never recomputes itself (Feature 041 FR-17; T30; FTASKS 7.7 – 7.8).

One case per input that can move: a member re-cut (decision digest/revision), retired, its run deleted, its
key moved, and a new behaviour version changing the rule set, criteria or sub-questions. Each read path —
`get`, `list`, the suggested blend, marking and rerun — is asserted to call `stale_reasons`.
"""
from __future__ import annotations

import pytest

from src.core import runtime_settings
from src.core.config import settings
from src.services.probe_blend import service
from tests.support import blend_world as bw

API = "/api/v1/probe-monitors"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


async def report(client, async_session, request, monkeypatch):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    report_id = await bw.done_report(client, monkeypatch, world)
    envelope = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    assert envelope["stale"] is False
    return world, report_id


async def reasons(client, report_id):
    body = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    return body["stale"], [r["code"] for r in body["stale_reasons"]], body


async def test_recut_member_is_stale(client, async_session, request, monkeypatch):
    world, report_id = await report(client, async_session, request, monkeypatch)
    world.members[1].decision_digest = "f" * 64
    await async_session.commit()
    stale, codes, body = await reasons(client, report_id)
    assert stale and codes == ["decision_changed"]
    assert body["stale_reasons"][0]["probe_id"] == world.members[1].id
    assert body["next"][0] == {"tool": "rerun_blend_report", "arguments": {"report_id": report_id},
                               "why": body["next"][0]["why"]}


async def test_retired_member_is_stale(client, async_session, request, monkeypatch):
    world, report_id = await report(client, async_session, request, monkeypatch)
    await client.put(f"{API}/behaviours/{world.key}/probes/{world.members[2].id}/role", json={"role": "retired"})
    assert (await reasons(client, report_id))[1] == ["member_retired"]


async def test_deleted_run_is_stale(client, async_session, request, monkeypatch):
    world, report_id = await report(client, async_session, request, monkeypatch)
    from src.models.probe_monitor import ProbeMonitorRun

    await async_session.delete(await async_session.get(ProbeMonitorRun, world.members[2].run_id))
    await async_session.commit()
    assert (await reasons(client, report_id))[1] == ["member_gone"]


async def test_moved_key_is_stale(client, async_session, request, monkeypatch):
    world, report_id = await report(client, async_session, request, monkeypatch)
    world.members[2].behaviour = "another-key-x"
    await async_session.commit()
    assert "member_moved" in (await reasons(client, report_id))[1]


@pytest.mark.parametrize("field, change", [
    ("sub_questions", {"sub_questions": [{"key": "sq-new", "question": "q", "positive_meaning": "p"}]}),
    ("success_criteria", {"success_criteria": [{"id": "c1", "detector": "best_single", "subset": "all",
                                                 "min_recall": 0.5}]}),
])
async def test_new_version_is_stale_naming_the_field(client, async_session, request, monkeypatch, field, change):
    world, report_id = await report(client, async_session, request, monkeypatch)
    made = await client.post(f"{API}/behaviours/{world.key}/versions", json={"base_version": 2, **change})
    assert made.status_code == 201, made.text
    stale, codes, body = await reasons(client, report_id)
    assert codes == ["version_changed"] and body["stale_reasons"][0]["fields"] == [field]


async def test_a_new_version_changing_only_the_meaning_is_not_stale(client, async_session, request, monkeypatch):
    world, report_id = await report(client, async_session, request, monkeypatch)
    await client.post(f"{API}/behaviours/{world.key}/versions", json={"base_version": 2, "description": "reworded"})
    assert (await reasons(client, report_id))[0] is False


async def test_every_read_path_calls_stale_reasons(client, async_session, request, monkeypatch):
    world, report_id = await report(client, async_session, request, monkeypatch)
    calls = []
    real = service.stale_reasons

    async def spy(db, row):
        calls.append(row.id)
        return await real(db, row)

    monkeypatch.setattr(service, "stale_reasons", spy)
    await client.get(f"{API}/blend-reports/{report_id}")
    assert calls == [report_id]
    await client.get(f"{API}/blend-reports", params={"behaviour_key": world.key})
    assert calls == [report_id] * 2
    put = await client.put(f"{API}/behaviours/{world.key}/suggested-blend",
                           json={"model": world.model.id, "report_id": report_id, "candidate": "stack"})
    assert put.status_code == 200, put.text
    assert calls == [report_id] * 4            # marking refuses a stale report, then the envelope it returns
    await client.get(f"{API}/behaviours/{world.key}/suggested-blend", params={"model": world.model.id})
    assert calls == [report_id] * 5


async def test_rerun_links_both_ways_old_bytes_unchanged(client, async_session, request, monkeypatch):
    world, report_id = await report(client, async_session, request, monkeypatch)
    before = (await client.get(f"{API}/blend-reports/{report_id}/report.json")).content
    from src.workers import probe_blend_tasks

    monkeypatch.setattr(probe_blend_tasks.run_blend_evaluation, "delay", lambda rid: type("T", (), {"id": "t2"})())
    new = (await client.post(f"{API}/blend-reports/{report_id}/rerun")).json()
    old = (await client.get(f"{API}/blend-reports/{report_id}")).json()
    fresh = (await client.get(f"{API}/blend-reports/{new['report_id']}")).json()
    assert old["rerun_by"] == new["report_id"] and fresh["rerun_of"] == report_id
    assert (await client.get(f"{API}/blend-reports/{report_id}/report.json")).content == before


async def test_rerun_explicit_retired_member_refused(client, async_session, request, monkeypatch):
    world = await bw.build(client, async_session, seed=request.node.nodeid, signals=(2.0, 1.0, 0.5, 0.0))
    ids = [m.id for m in world.members[:3]]
    report_id = await bw.done_report(client, monkeypatch, world, member_ids=ids)
    await client.put(f"{API}/behaviours/{world.key}/probes/{ids[2]}/role", json={"role": "retired"})
    response = await client.post(f"{API}/blend-reports/{report_id}/rerun")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert (detail["code"], detail["members"]) == ("member_retired_or_deleted", [ids[2]])
    assert detail["next"][0]["tool"] == "evaluate_probe_blends" and "member_ids" not in detail["next"][0]["arguments"]
