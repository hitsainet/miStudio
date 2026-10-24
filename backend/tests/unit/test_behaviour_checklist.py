"""The behaviour checklist and status (Feature 037 FR-12, FR-17; D26, D49, D-T5).

Each step test flips ONE underlying record and asserts that ONLY that step changes. The "armed" step
reads miLLM through the real `millm_read` client over an `httpx.MockTransport`, asserted by request and
call count; with miLLM absent, failing, or unable to say which miStudio probe a served probe came from,
the step is "unknown" — never "not_done".

Controls are recorded in 0xcc/reviews/037_038_implementation_controls_2026-10-21.md.
"""
from __future__ import annotations

import httpx
import pytest

from src.services import millm_read
from tests.unit.test_behaviour_catalogue import API, PII, _base, _create, _probe, _prompt, _view, sync_db  # noqa: F401

pytestmark = pytest.mark.asyncio


class FakeMiLLM:
    """miLLM's probe list, as served (ApiResponse envelope)."""

    def __init__(self, rows=None, status=200):
        self.rows = rows if rows is not None else []
        self.status = status
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status >= 400:
            return httpx.Response(self.status, json={"success": False})
        return httpx.Response(200, json={"success": True, "data": self.rows})


@pytest.fixture
def millm(monkeypatch):
    from src.core.config import settings

    fake = FakeMiLLM()
    monkeypatch.setattr(settings, "millm_api_url", "http://millm.test")
    monkeypatch.setattr(millm_read, "_transport_factory", lambda: httpx.MockTransport(fake.handler))
    return fake


@pytest.fixture
def no_millm(monkeypatch):
    from src.core.config import settings

    monkeypatch.setattr(settings, "millm_api_url", "")


async def _states(client, key="pii"):
    response = await client.get(f"{API}/behaviours/{key}/checklist")
    assert response.status_code == 200, response.text
    body = response.json()
    return {s["id"]: s["state"] for s in body["steps"]}, body


BASELINE = {"defined": "done", "data": "not_done", "trained": "not_done", "rung2": "not_done",
            "prompt": "not_done", "rung3": "not_done", "exported": "not_done", "armed": "not_done",
            # Feature 040 FR-41 (S1): a 037-shaped definition declares no success criterion.
            "definition_complete": "not_done",
            # Feature 040 FR-41 (S2): the four detector steps of an empty behaviour.
            "views_registered": "not_done", "evaluated": "not_done", "breakdown_available": "not_done",
            "success_report": "not_done",
            # Feature 041 FR-5 (S1): the family steps of an empty behaviour.
            "family_trained": "not_done", "blend_report": "not_done", "family_exported": "not_done"}

#: Feature 041 FR-5: one candidate probe and miLLM absent — the family has a member, so `armed` is
#: unknown (never not_done, D-T5) and the main question has a member.
ONE_MEMBER = {"armed": "unknown", "family_trained": "done"}


async def test_an_empty_behaviour_has_only_its_definition(client, no_millm):
    await _create(client)
    states, body = await _states(client)
    assert states == BASELINE
    assert [s["id"] for s in body["steps"]] == list(BASELINE)
    assert body["status"] == "draft"


async def test_data_step(client, sync_db, no_millm):
    await _create(client)
    _, dataset = _base(sync_db)
    _view(sync_db, dataset, behaviour="pii")
    sync_db.commit()
    states, _ = await _states(client)
    assert states == {**BASELINE, "data": "done"}


async def test_trained_step_needs_a_COMPLETED_run(client, sync_db, no_millm):
    from src.models.probe_monitor import ProbeMonitorRun

    await _create(client)
    model, dataset = _base(sync_db)
    view = _view(sync_db, dataset)
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, eval_dataset_ids=[], config={},
                          status="running", environment={}, behaviour="pii")
    sync_db.add(run)
    sync_db.commit()
    assert (await _states(client))[0] == BASELINE
    run.status = "completed"
    sync_db.commit()
    assert (await _states(client))[0] == {**BASELINE, "trained": "done"}


async def test_rung_steps(client, sync_db, no_millm):
    await _create(client)
    probe = _probe(sync_db, behaviour="pii", run_behaviour=None, rung=1)
    assert (await _states(client))[0] == {**BASELINE, **ONE_MEMBER}
    probe.rung = 2
    sync_db.commit()
    assert (await _states(client))[0] == {**BASELINE, **ONE_MEMBER, "rung2": "done"}
    probe.rung = 3
    sync_db.commit()
    assert (await _states(client))[0] == {**BASELINE, **ONE_MEMBER, "rung2": "done", "rung3": "done"}


async def test_prompt_step_ignores_archived_prompts(client, sync_db, no_millm):
    from src.core.clock import utc_now

    await _create(client)
    prompt = _prompt(sync_db, behaviour="pii")
    assert (await _states(client))[0] == {**BASELINE, "prompt": "done"}
    prompt.archived_at = utc_now()
    sync_db.commit()
    assert (await _states(client))[0] == BASELINE


async def test_exported_step_reads_the_FAMILY_and_status_the_primary(client, sync_db, no_millm):
    """Feature 041 FR-5 changed 037's step: `exported` reports every member of the default family (an
    implicit candidate included). The STATUS still reads only the primary (OI-2, D36): an exported
    candidate does not raise it."""
    await _create(client)
    probe = _probe(sync_db, behaviour="pii", confirmed="pii", run_behaviour=None, rung=2)
    probe.definition_sha256 = "a" * 64
    sync_db.commit()
    states, body = await _states(client)
    assert states == {**BASELINE, **ONE_MEMBER, "rung2": "done", "exported": "done"}
    assert body["status"] == "draft", "a non-primary probe's export must not move the status"
    await client.put(f"{API}/behaviours/pii/probes/{probe.id}/role", json={"role": "primary"})
    states, body = await _states(client)
    assert states["exported"] == "done" and body["status"] == "unknown"


async def _primary(client, sync_db, rung=2):
    await _create(client)
    probe = _probe(sync_db, behaviour="pii", confirmed="pii", run_behaviour=None, rung=rung)
    probe.definition_sha256 = "b" * 64
    sync_db.commit()
    await client.put(f"{API}/behaviours/pii/probes/{probe.id}/role", json={"role": "primary"})
    return probe


async def test_armed_reads_millm_once_by_payload_and_count(client, sync_db, millm):
    probe = await _primary(client, sync_db)
    millm.rows = [{"id": "pr_1", "source_probe_id": probe.id}]
    states, body = await _states(client)
    assert states["armed"] == "done" and body["status"] == "monitoring"
    assert len(millm.requests) == 1
    request = millm.requests[0]
    assert (request.method, request.url.path, dict(request.url.params)) == ("GET", "/api/probes", {"armed": "true"})


async def test_another_armed_probe_is_not_this_one(client, sync_db, millm):
    probe = await _primary(client, sync_db)
    millm.rows = [{"id": "pr_1", "source_probe_id": "pm_someone_else"}]
    states, body = await _states(client)
    assert states["armed"] == "not_done"
    assert body["status"] == "validated" and probe.rung == 2


async def test_armed_is_unknown_without_millm(client, sync_db, no_millm):
    await _primary(client, sync_db)
    states, body = await _states(client)
    armed = next(s for s in body["steps"] if s["id"] == "armed")
    assert armed["state"] == "unknown" and "not connected" in armed["reason"]
    assert body["status"] == "unknown"


async def test_armed_is_unknown_when_millm_fails(client, sync_db, millm):
    await _primary(client, sync_db)
    millm.status = 503
    assert (await _states(client))[0]["armed"] == "unknown"


async def test_armed_is_unknown_when_millm_cannot_name_the_source_probe(client, sync_db, millm):
    """miLLM before 036 FR-36.14 serves summaries without `source_probe_id` (FTDD R-2)."""
    await _primary(client, sync_db)
    millm.rows = [{"id": "pr_1", "name": "x"}]
    states, body = await _states(client)
    armed = next(s for s in body["steps"] if s["id"] == "armed")
    assert armed["state"] == "unknown" and "source_probe_id" in armed["reason"]


async def test_the_list_carries_the_status_reading_millm_once(client, sync_db, millm):
    probe = await _primary(client, sync_db)
    await _create(client, key="other")
    millm.rows = [{"id": "pr_1", "source_probe_id": probe.id}]
    millm.requests.clear()
    listed = (await client.get(f"{API}/behaviours")).json()["behaviours"]
    assert {b["key"]: b["status"] for b in listed} == {"other": "draft", "pii": "monitoring"}
    assert len(millm.requests) == 1


async def test_an_archived_behaviour_reads_archived(client, no_millm):
    await _create(client)
    await client.post(f"{API}/behaviours/pii/archive")
    assert (await _states(client))[1]["status"] == "archived"


async def test_a_primary_that_was_never_exported_is_not_exported(client, sync_db, no_millm):
    """Control C-PRIMARYEXPORT survived the first version of this file (every primary fixture had a
    definition); this one differs from `_primary` only in `definition_sha256`."""
    probe = await _primary(client, sync_db)
    probe.definition_sha256 = None
    sync_db.commit()
    states, _ = await _states(client)
    assert states["exported"] == "not_done"


# ── Feature 039 FR-19 (D-T7): exported is done, STALE or not done ─────────────


async def _exported(client):
    _states_by_id, body = await _states(client)
    return next(step for step in body["steps"] if step["id"] == "exported"), body


async def test_a_current_export_is_done_with_its_identity(client, sync_db, no_millm):
    probe = await _primary(client, sync_db)
    probe.definition_path = "/data/probe_definitions/x.probe.json"
    probe.definition_build = {"resolved": {"detector_digest": "d" * 64, "decision_digest": "e" * 64,
                                           "decision_revision": 2, "identity_source": "built"}}
    probe.decision_digest = "e" * 64
    sync_db.commit()
    step, _ = await _exported(client)
    assert step["state"] == "done" and step["identity_source"] == "built"
    assert step["identity"] == {"detector_digest": "d" * 64, "decision_digest": "e" * 64,
                                "decision_revision": 2, "identity_current": True}


async def test_a_pre_039_export_stays_done(client, sync_db, no_millm):
    """FR-19: a definition built before digests is still exported — it is not demoted for lacking an
    identity block (requiring one would mark every existing export not done)."""
    await _primary(client, sync_db)          # definition_sha256 set, no identity anywhere
    step, _ = await _exported(client)
    assert step["state"] == "done"


async def test_an_invalidated_export_is_STALE_with_the_rebuild_as_next(client, sync_db, no_millm):
    from tests.support.next_calls import assert_next_calls_are_live

    probe = await _primary(client, sync_db)
    probe.definition_sha256 = None
    probe.definition_build = {"invalidated": {"reason": "threshold recalibrated", "cause": "decision"}}
    sync_db.commit()
    step, body = await _exported(client)
    assert step["state"] == "stale"
    assert "threshold recalibrated" in step["reason"] and "cause: decision" in step["reason"]
    assert step["next"][0]["tool"] == "build_probe_definition"
    assert step["next"][0]["arguments"] == {"probe_id": probe.id}
    assert_next_calls_are_live(step["next"])
    assert body["status"] == "validated", "a stale export is not done, so the status cannot rise on it"


async def test_a_stale_export_with_millm_unknown_is_not_status_unknown(client, sync_db, millm):
    """`status()` treats stale as NOT done: "unknown" status needs a DONE export."""
    millm.status = 500
    probe = await _primary(client, sync_db)
    probe.definition_sha256 = None
    probe.definition_build = {"invalidated": {"reason": "rung changed", "cause": "rung"}}
    sync_db.commit()
    step, body = await _exported(client)
    assert step["state"] == "stale" and body["status"] == "validated"
