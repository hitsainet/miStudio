"""The checklist reports the family (Feature 041 FR-5; FTASKS 2.6 – 2.7; OI-2 answered by D36).

`exported` and `armed` iterate the default family on each model, state per member (`partly_done` names the
rest; `unknown` stays `unknown`); three new steps (`family_trained`, `blend_report`, `family_exported`) are
read from real rows with their `next`. ⚠ STATUS STILL READS THE PRIMARY: "monitoring" means the primary is
armed (D36), whatever the family's other members are.

miLLM is the real `millm_read` client over an `httpx.MockTransport`, asserted by request and call count.
Keys are random.
"""
from __future__ import annotations

import httpx
import pytest

from src.core.config import settings
from src.services import millm_read
from tests.support import family_fixtures as ff
from tests.support.next_calls import assert_next_calls_are_live
from tests.support.random_behaviour import rng_for

API = "/api/v1/probe-monitors"


class FakeMiLLM:
    def __init__(self):
        self.rows = []
        self.requests = []

    def handler(self, request):
        self.requests.append(request)
        return httpx.Response(200, json={"success": True, "data": self.rows})


@pytest.fixture
def millm(monkeypatch):
    fake = FakeMiLLM()
    monkeypatch.setattr(settings, "millm_api_url", "http://millm.test")
    monkeypatch.setattr(millm_read, "_transport_factory", lambda: httpx.MockTransport(fake.handler))
    return fake


@pytest.fixture
def no_millm(monkeypatch):
    monkeypatch.setattr(settings, "millm_api_url", "")


@pytest.fixture
def rng(request):
    return rng_for(request.node.nodeid)


async def checklist(client, key):
    body = (await client.get(f"{API}/behaviours/{key}/checklist")).json()
    return {s["id"]: s for s in body["steps"]}, body


async def three(client, async_session, rng, **member_kwargs):
    key = await ff.catalogue(client, rng)
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    probes = [await ff.member(async_session, model, ds, key=key, role="primary", **member_kwargs),
              await ff.member(async_session, model, ds, key=key, **member_kwargs),
              await ff.member(async_session, model, ds, key=key, role="candidate", **member_kwargs)]
    await async_session.commit()
    return key, model, probes


async def test_two_of_three_armed_reads_partly_done(client, async_session, rng, millm):
    key, _model, (primary, implicit, candidate) = await three(client, async_session, rng)
    millm.rows = [{"id": "pr_1", "source_probe_id": primary.id}, {"id": "pr_2", "source_probe_id": candidate.id}]
    steps, body = await checklist(client, key)
    armed = steps["armed"]
    assert armed["state"] == "partly_done"
    assert implicit.id in armed["reason"] and primary.id not in armed["reason"].split("not armed:")[1]
    assert {m["probe_id"]: m["state"] for m in armed["members"]} == {
        primary.id: "done", implicit.id: "not_done", candidate.id: "done"}
    assert len(millm.requests) == 1
    request = millm.requests[0]
    assert (request.method, request.url.path, dict(request.url.params)) == ("GET", "/api/probes", {"armed": "true"})
    assert body["status"] == "monitoring"


async def test_every_member_armed_is_done(client, async_session, rng, millm):
    key, _model, probes = await three(client, async_session, rng)
    millm.rows = [{"id": f"pr_{i}", "source_probe_id": p.id} for i, p in enumerate(probes)]
    steps, _ = await checklist(client, key)
    assert steps["armed"]["state"] == "done"


async def test_armed_stays_unknown_without_millm(client, async_session, rng, no_millm):
    key, _model, probes = await three(client, async_session, rng)
    steps, body = await checklist(client, key)
    assert steps["armed"]["state"] == "unknown" and "not connected" in steps["armed"]["reason"]
    assert {m["state"] for m in steps["armed"]["members"]} == {"unknown"}


async def test_status_still_reads_the_primary(client, async_session, rng, millm):
    """D36 / OI-2: two non-primary members armed, the primary not → not monitoring; the primary alone
    armed → monitoring."""
    key, _model, (primary, implicit, candidate) = await three(client, async_session, rng)
    millm.rows = [{"id": "a", "source_probe_id": implicit.id}, {"id": "b", "source_probe_id": candidate.id}]
    steps, body = await checklist(client, key)
    assert steps["armed"]["state"] == "partly_done"
    assert body["status"] == "validated"
    millm.rows = [{"id": "a", "source_probe_id": primary.id}]
    steps, body = await checklist(client, key)
    assert steps["armed"]["state"] == "partly_done"
    assert body["status"] == "monitoring"


async def test_exported_reports_every_member(client, async_session, rng, no_millm):
    key, model, (primary, implicit, candidate) = await three(client, async_session, rng)
    implicit.definition_sha256 = None
    await async_session.commit()
    steps, _ = await checklist(client, key)
    exported = steps["exported"]
    assert exported["state"] == "partly_done" and implicit.id in exported["reason"]
    assert {m["probe_id"]: m["state"] for m in exported["members"]} == {
        primary.id: "done", implicit.id: "not_done", candidate.id: "done"}
    assert [n["arguments"] for n in exported["next"]] == [{"probe_id": implicit.id}]
    assert_next_calls_are_live(exported["next"])


async def test_a_declared_model_with_no_member_keeps_the_steps_from_done(client, async_session, rng, millm):
    """The version names two models; every member is on one. Exported and armed cannot be `done`."""
    other = await ff.model(async_session)
    model = await ff.model(async_session)
    await async_session.commit()
    key = await ff.catalogue(client, rng, models=[model.id, other.id])
    ds = await ff.source(async_session)
    probe = await ff.member(async_session, model, ds, key=key, role="primary")
    await async_session.commit()
    millm.rows = [{"id": "a", "source_probe_id": probe.id}]
    steps, _ = await checklist(client, key)
    assert steps["exported"]["state"] == "partly_done" and other.id in steps["exported"]["reason"]
    assert steps["armed"]["state"] == "partly_done" and other.id in steps["armed"]["reason"]
    assert {m["model_id"]: m["members"] for m in steps["armed"]["models"]} == {model.id: 1, other.id: 0}


async def test_retired_members_are_not_in_the_steps(client, async_session, rng, millm):
    key, model, probes = await three(client, async_session, rng)
    ds = await ff.source(async_session)
    retired = await ff.member(async_session, model, ds, key=key, role="retired", built=False)
    await async_session.commit()
    millm.rows = [{"id": f"pr_{i}", "source_probe_id": p.id} for i, p in enumerate(probes)]
    steps, _ = await checklist(client, key)
    assert steps["armed"]["state"] == "done" and steps["exported"]["state"] == "done"
    assert retired.id not in {m["probe_id"] for m in steps["armed"]["members"]}


# ── the three new steps ──────────────────────────────────────────────────────────────────────


async def test_family_trained_step(client, async_session, rng, no_millm):
    a, b = ff.sub_keys(rng, 2)
    key = await ff.catalogue(client, rng, sub_questions=[a, b])
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    await ff.member(async_session, model, ds, key=key, sub_question=a)
    await async_session.commit()
    steps, _ = await checklist(client, key)
    step = steps["family_trained"]
    assert step["state"] == "not_done" and step["missing"] == [{"model_id": model.id, "sub_question": b}]
    assert step["next"][0]["tool"] == "register_probe_dataset"
    assert_next_calls_are_live(step["next"])
    await ff.member(async_session, model, ds, key=key, sub_question=b)
    await async_session.commit()
    steps, _ = await checklist(client, key)
    assert steps["family_trained"]["state"] == "done"


async def test_family_trained_without_any_member(client, rng, no_millm):
    key = await ff.catalogue(client, rng)
    steps, _ = await checklist(client, key)
    assert steps["family_trained"]["state"] == "not_done"
    assert_next_calls_are_live(steps["family_trained"]["next"])


async def test_blend_report_step_is_not_done_without_a_report(client, async_session, rng, no_millm):
    key, _model, _probes = await three(client, async_session, rng)
    steps, _ = await checklist(client, key)
    assert steps["blend_report"]["state"] == "not_done"


async def test_family_exported_step(client, async_session, rng, no_millm):
    key, _model, (primary, implicit, candidate) = await three(client, async_session, rng)
    steps, _ = await checklist(client, key)
    assert steps["family_exported"]["state"] == "done"
    # A built definition whose recorded decision is behind the current bar is NOT current.
    implicit.definition_build = {"resolved": {"detector_digest": "d" * 64, "decision_digest": "e" * 64,
                                              "decision_revision": 1, "identity_source": "built"}}
    implicit.decision_digest = "f" * 64
    candidate.definition_path = None
    await async_session.commit()
    steps, _ = await checklist(client, key)
    step = steps["family_exported"]
    assert step["state"] == "partly_done"
    assert {n["arguments"]["probe_id"] for n in step["next"]} == {implicit.id, candidate.id}
    assert_next_calls_are_live(step["next"])


async def test_the_step_order_appends_after_040(client, rng, no_millm):
    key = await ff.catalogue(client, rng)
    _steps, body = await checklist(client, key)
    assert [s["id"] for s in body["steps"]][-4:] == ["success_report", "family_trained", "blend_report",
                                                      "family_exported"]
