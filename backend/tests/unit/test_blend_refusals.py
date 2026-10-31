"""Every blend refusal at the call (Feature 041 FR-14; FTDD §5.5; FTASKS 7.2 – 7.3).

Each test asserts the `code`, its data fields and its `next`, and that NO task was dispatched (a refusal
happens before a row exists, PT-5). Every `next` collected here is checked against the live registry in
`test_next_entries_are_live_calls.py`.
"""
from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import func, select

from src.core import runtime_settings
from src.core.config import settings
from tests.support import blend_world as bw
from tests.support import family_fixtures as ff
from tests.support.next_calls import assert_next_calls_are_live

API = "/api/v1/probe-monitors"

#: Every refusal body these tests produced, for the live-`next` check.
COLLECTED: list = []


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


@pytest.fixture
def dispatched(monkeypatch):
    sent = []
    from src.workers import probe_blend_tasks

    monkeypatch.setattr(probe_blend_tasks.run_blend_evaluation, "delay", lambda rid: sent.append(rid))
    return sent


async def refused(client, async_session, body, status, code, dispatched):
    from src.models.probe_blend import ProbeBlendReport

    before = (await async_session.execute(select(func.count()).select_from(ProbeBlendReport))).scalar_one()
    response = await client.post(f"{API}/blend-reports", json=body)
    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert detail["code"] == code, detail
    assert dispatched == []
    after = (await async_session.execute(select(func.count()).select_from(ProbeBlendReport))).scalar_one()
    assert after == before, "a refusal must not leave a row"
    COLLECTED.append(detail)
    # FR-26 / 7.4: every `next` of every refusal names a LIVE tool with arguments its schema accepts.
    assert_next_calls_are_live(detail.get("next") or [])
    return detail


async def test_member_scores_missing_names_the_exact_calls(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid, skip=[(1, "ood"), (2, "calibration")])
    detail = await refused(client, async_session, bw.start_body(world), 409, "member_scores_missing", dispatched)
    m1, m2 = world.members[1].id, world.members[2].id
    assert detail["next"][0]["arguments"] == {"probe_id": m1, "dataset_ids": [world.tests[0].id]}
    assert {"probe_id": m2, "dataset_ids": [world.calibration.id], "sample_rows": world.sample_rows} in \
        [n["arguments"] for n in detail["next"]]
    assert {n["tool"] for n in detail["next"]} == {"evaluate_probe_monitor"}


async def test_calibration_positions_not_recorded(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    evaluation = world.evaluations[world.members[0].id][world.calibration.id]
    evaluation.positions_path = None
    await async_session.commit()
    detail = await refused(client, async_session, bw.start_body(world), 409, "calibration_positions_not_recorded",
                           dispatched)
    assert detail["next"][0]["arguments"]["sample_rows"] == world.sample_rows


async def test_calibration_samples_differ(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    evaluation = world.evaluations[world.members[2].id][world.calibration.id]
    evaluation.metrics = {**evaluation.metrics, "sample": {"rows": 999, "seed_basis": "view+rows"}}
    await async_session.commit()
    detail = await refused(client, async_session, bw.start_body(world), 409, "calibration_samples_differ", dispatched)
    assert detail["found"][0]["sample_rows"] == 999
    assert detail["next"][0]["arguments"] == {"probe_id": world.members[2].id, "dataset_ids": [world.calibration.id],
                                              "sample_rows": world.sample_rows}


async def test_an_evaluation_without_positions_is_not_chosen(client, async_session, request, dispatched):
    """A pre-035 evaluation (no positions) never qualifies: the member's scores read missing."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    world.evaluations[world.members[0].id][world.blend_fit.id].positions_path = None
    await async_session.commit()
    await refused(client, async_session, bw.start_body(world), 409, "member_scores_missing", dispatched)


async def test_model_required(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    other = await ff.model(async_session)
    ds = await ff.source(async_session)
    await ff.member(async_session, other, ds, key=world.key)
    await async_session.commit()
    body = bw.start_body(world)
    body.pop("model")
    detail = await refused(client, async_session, body, 422, "model_required", dispatched)
    assert {m["model_id"] for m in detail["models"]} == {world.model.id, other.id}


async def test_members_incompatible(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    world.runs[world.members[1].id].environment = {"model_dtype": "float32"}
    await async_session.commit()
    detail = await refused(client, async_session, bw.start_body(world), 422, "members_incompatible", dispatched)
    assert detail["field"] == "precision"
    assert detail["values"][world.members[1].id] == "float32"


async def test_view_missing(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    detail = await refused(client, async_session, bw.start_body(world, dataset_version="dv-nowhere"), 422,
                           "view_missing", dispatched)
    assert detail["role"] == "blend_fit" and detail["next"][0]["tool"] == "register_probe_dataset"


async def test_blend_fit_overlap(client, async_session, request, dispatched):
    """Named views: the blend_fit view IS a member's training view (the check that keeps a blend from being
    judged on the rows it was fitted on)."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    run = world.runs[world.members[0].id]
    run.train_dataset_id = world.blend_fit.id
    await async_session.commit()
    views = {"blend_fit": world.blend_fit.id, "tests": [v.id for v in world.tests], "calibration": world.calibration.id}
    detail = await refused(client, async_session, {"behaviour_key": world.key, "model": world.model.id, "views": views,
                                                   "calibration_sample_rows": world.sample_rows},
                           422, "blend_fit_overlap", dispatched)
    assert detail["view"] == world.blend_fit.id


async def test_a_members_training_view_as_a_test_view_is_refused_named(client, async_session, request, dispatched):
    """D46 (operator, 2026-10-24): a test view that is a member's own training view is refused by name, before
    any evaluation work, nothing queued — on named views (ahead of the role check) and on derived views."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    member = world.members[1]
    world.runs[member.id].train_dataset_id = world.tests[0].id
    await async_session.commit()
    views = {"blend_fit": world.blend_fit.id, "tests": [v.id for v in world.tests], "calibration": world.calibration.id}
    named = await refused(client, async_session, {"behaviour_key": world.key, "model": world.model.id, "views": views,
                                                  "calibration_sample_rows": world.sample_rows},
                          422, "test_view_is_member_training_view", dispatched)
    derived = await refused(client, async_session, bw.start_body(world), 422, "test_view_is_member_training_view",
                            dispatched)
    for detail in (named, derived):
        assert detail["view"] == world.tests[0].id and detail["probe_ids"] == [member.id]


async def test_a_members_own_train_view_named_as_a_test_reads_d46_not_a_role_mismatch(client, async_session, request,
                                                                                         dispatched):
    """D46 on named views comes BEFORE the role check (as FR-14's R1 fix did): naming a member's real training
    view (role `train`) among the tests reads its own code, not `view_missing`."""
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    member = world.members[0]
    trained = world.runs[member.id].train_dataset_id
    views = {"blend_fit": world.blend_fit.id, "tests": [trained, *[v.id for v in world.tests]],
             "calibration": world.calibration.id}
    detail = await refused(client, async_session, {"behaviour_key": world.key, "model": world.model.id, "views": views,
                                                   "calibration_sample_rows": world.sample_rows},
                           422, "test_view_is_member_training_view", dispatched)
    assert detail["view"] == trained and member.id in detail["probe_ids"]


async def test_a_non_members_training_view_as_a_test_view_still_runs(client, async_session, request, monkeypatch):
    """D46: only a MEMBER's training view is refused — a family probe left out of `member_ids` may have trained
    on a test view."""
    from types import SimpleNamespace

    from src.workers import probe_blend_tasks

    dispatched = []
    monkeypatch.setattr(probe_blend_tasks.run_blend_evaluation, "delay",
                        lambda rid: dispatched.append(rid) or SimpleNamespace(id="task-d46"))
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    assert len(world.members) >= 3
    outsider = world.members[-1]
    world.runs[outsider.id].train_dataset_id = world.tests[0].id
    await async_session.commit()
    body = bw.start_body(world, member_ids=[m.id for m in world.members[:-1]])
    response = await client.post(f"{API}/blend-reports", json=body)
    assert response.status_code in (200, 202), response.text
    assert len(dispatched) == 1


async def test_members_misaligned(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid, keep=0.85)
    detail = await refused(client, async_session, bw.start_body(world), 409, "members_misaligned", dispatched)
    assert detail["min_fraction"] == 0.95 and detail["aligned"] < detail["rows"]


async def test_labels_disagree(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid, keep=1.0)
    evaluation = world.evaluations[world.members[1].id][world.tests[0].id]
    labels = np.load(evaluation.labels_path)
    labels[0] = 1 - labels[0]
    np.save(evaluation.labels_path, labels)
    detail = await refused(client, async_session, bw.start_body(world), 409, "labels_disagree", dispatched)
    assert detail["view"] == world.tests[0].id


async def test_member_scores_constant(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    evaluation = world.evaluations[world.members[2].id][world.blend_fit.id]
    scores = np.load(evaluation.scores_path)
    np.save(evaluation.scores_path, np.full(scores.shape, 0.5, dtype=scores.dtype))
    detail = await refused(client, async_session, bw.start_body(world), 422, "member_scores_constant", dispatched)
    assert detail["member"] == world.members[2].id
    assert detail["next"][0]["arguments"]["member_ids"] == [m.id for m in world.members[:2]]


async def test_cap_exceeded_members_and_candidates(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    runtime_settings.apply({"blend_max_members": 2})
    detail = await refused(client, async_session, bw.start_body(world), 422, "cap_exceeded", dispatched)
    assert (detail["setting"], detail["value"], detail["got"]) == ("blend_max_members", 2, 3)
    runtime_settings.apply({"blend_max_members": 16, "blend_max_candidates": 3})
    detail = await refused(client, async_session, bw.start_body(world), 422, "cap_exceeded", dispatched)
    assert (detail["setting"], detail["value"]) == ("blend_max_candidates", 3)


async def test_candidate_unknown(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    detail = await refused(client, async_session, bw.start_body(world, candidates=["weighted_vote"]), 422,
                           "candidate_unknown", dispatched)
    assert "stack" in detail["vocabulary"]


async def test_behaviour_archived(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    await client.post(f"{API}/behaviours/{world.key}/archive")
    detail = await refused(client, async_session, bw.start_body(world), 409, "behaviour_archived", dispatched)
    assert detail["next"][0] == {"tool": "unarchive_behaviour", "arguments": {"key": world.key},
                                 "why": detail["next"][0]["why"]}


async def test_view_archived(client, async_session, request, dispatched):
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    views = {"blend_fit": world.blend_fit.id, "tests": [v.id for v in world.tests], "calibration": world.calibration.id}
    await client.post(f"{API}/datasets/{world.tests[1].id}/archive", json={})
    detail = await refused(client, async_session, {"behaviour_key": world.key, "model": world.model.id, "views": views,
                                                   "calibration_sample_rows": world.sample_rows},
                           409, "view_archived", dispatched)
    assert detail["archived"] == [world.tests[1].id]
    assert detail["next"][0]["arguments"] == {"dataset_id": world.tests[1].id}


async def test_rules_or_candidates_refused_without_a_rule_set(client, async_session, request, dispatched):
    # 040 S3 is on main (FTASKS 6.6): a rules ∪ candidate is refused only when the version names no rule set.
    world = await bw.build(client, async_session, seed=request.node.nodeid)
    detail = await refused(client, async_session, bw.start_body(world, candidates=["rules_or_stack"]), 422,
                           "no_rule_set", dispatched)
    assert detail["candidates"] == ["rules_or_stack"]
    assert detail["next"][0]["tool"] == "new_behaviour_version" and detail["next"][0]["needs"] == ["rule_set"]
