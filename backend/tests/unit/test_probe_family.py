"""A behaviour's probe family (Feature 041 S1: FR-1 – FR-4; FTASKS 1.1 – 1.2, 2.1 – 2.5).

Through the REAL routes (`GET …/behaviours/{key}/family`, `POST /runs`) on a real database. Keys and
sub-question keys are random; a flag fixture differs from its siblings ONLY in the field under test.
Controls are recorded in `0xcc/reviews/041_implementation_controls_2026-10-24.md`.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from src.core.config import settings
from src.core import runtime_settings
from src.services import probe_family
from tests.support import family_fixtures as ff
from tests.support.next_calls import assert_next_calls_are_live
from tests.support.random_behaviour import rng_for, slug

API = "/api/v1/probe-monitors"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(settings, "millm_api_url", "", raising=False)
    runtime_settings.reset()
    yield tmp_path
    runtime_settings.reset()


@pytest.fixture
def rng(request):
    return rng_for(request.node.nodeid)


async def get_family(client, key, **params):
    return await client.get(f"{API}/behaviours/{key}/family", params=params)


# ── FR-1: run submit ─────────────────────────────────────────────────────────────────────────


class Task:
    id = "task-041"


@pytest.fixture
def no_dispatch(monkeypatch):
    sent = []
    monkeypatch.setattr("src.api.v1.endpoints.probe_monitors.gpu_delay",
                        lambda task, gpu: (lambda **kw: sent.append(kw) or Task()))
    return sent


async def test_sub_question_mismatch_refused(client, async_session, rng, no_dispatch):
    a, b = ff.sub_keys(rng, 2)
    key = await ff.catalogue(client, rng, sub_questions=[a, b])
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    train = await ff.view(async_session, ds, key=key, sub_question=a, behaviour_version=1)
    other = await ff.view(async_session, ds, key=key, role="eval", sub_question=b)
    await async_session.commit()
    response = await client.post(f"{API}/runs", json={"model_id": model.id, "train_dataset_id": train.id,
                                                      "eval_dataset_ids": [other.id], "behaviour": key})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "sub_question_mismatch"
    assert {k: detail[k] for k in ("train_view", "train_sub_question", "view", "view_sub_question")} == {
        "train_view": train.id, "train_sub_question": a, "view": other.id, "view_sub_question": b}
    assert no_dispatch == []


async def test_a_main_question_eval_view_beside_a_specialist_is_allowed(client, async_session, rng, no_dispatch):
    """OQ-1 (a): only a DIFFERENT non-null sub-question is refused."""
    (a,) = ff.sub_keys(rng, 1)
    key = await ff.catalogue(client, rng, sub_questions=[a])
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    train = await ff.view(async_session, ds, key=key, sub_question=a, behaviour_version=1)
    main = await ff.view(async_session, ds, key=key, role="eval", sub_question=None)
    await async_session.commit()
    response = await client.post(f"{API}/runs", json={"model_id": model.id, "train_dataset_id": train.id,
                                                      "eval_dataset_ids": [main.id], "behaviour": key})
    assert response.status_code == 202, response.text
    assert len(no_dispatch) == 1


async def test_sub_question_not_declared_lists_keys(client, async_session, rng, no_dispatch):
    """The run names a behaviour whose current version does not declare the training view's key."""
    a, b = ff.sub_keys(rng, 2)
    owner = await ff.catalogue(client, rng, sub_questions=[a])
    other = await ff.catalogue(client, rng, sub_questions=[b])
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    train = await ff.view(async_session, ds, key=owner, sub_question=a, behaviour_version=1)
    await async_session.commit()
    response = await client.post(f"{API}/runs", json={"model_id": model.id, "train_dataset_id": train.id,
                                                      "behaviour": other})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "sub_question_not_declared"
    assert (detail["sub_question"], detail["behaviour"], detail["version"], detail["declared"]) == (a, other, 1, [b])
    assert no_dispatch == []
    # The view's own behaviour declares it: accepted.
    accepted = await client.post(f"{API}/runs", json={"model_id": model.id, "train_dataset_id": train.id,
                                                      "behaviour": owner})
    assert accepted.status_code == 202, accepted.text


async def test_sub_question_round_trips_view_run_probe_family_report(client, async_session, rng, no_dispatch,
                                                                     data_dir, async_engine):
    """FR-1: view (registered through the route) → run (submit copies it from the TRAINING view) → probe
    (the real `_persist_probe`) → family coverage. (→ blend report: extended in S2.)"""
    import torch
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.models.probe_monitor import ProbeMonitorRun
    from src.services.probe_monitor_run import _persist_probe
    from tests.support.detector_fixtures import dataset_row, write_split

    a, b = ff.sub_keys(rng, 2)
    key = await ff.catalogue(client, rng, sub_questions=[a, b])
    model = await ff.model(async_session)
    ds = await dataset_row(async_session, write_split(data_dir, {"text": [f"r{i}" for i in range(48)],
                                                                 "label": [str(i % 2) for i in range(48)]}))
    await async_session.commit()
    registered = await client.post(f"{API}/datasets", json={
        "name": "t", "dataset_id": str(ds.id), "input_column": "text", "label_column": "label",
        "label_mapping": {"1": "positive", "0": "negative"}, "role": "train", "behaviour": key, "sub_question": a})
    assert registered.status_code == 201, registered.text
    response = await client.post(f"{API}/runs", json={"model_id": model.id, "train_dataset_id": registered.json()["id"],
                                                      "behaviour": key})
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]

    url = async_engine.url.set(drivername=async_engine.url.drivername.replace("+asyncpg", ""))
    engine = create_engine(url)
    db = sessionmaker(bind=engine)()
    try:
        run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == run_id).one()
        assert run.sub_question == a
        run.status = "completed"
        trained = SimpleNamespace(rule="mean", rule_params={}, val_auroc=0.9, best_epoch=1, epochs_run=1, history=[],
                                  weight=torch.zeros(4), mean=torch.zeros(4), std=torch.ones(4),
                                  attention_query=None, bias=0.0)
        probe = _persist_probe(db, run, trained, None, 1, SimpleNamespace(artifact_dir=data_dir), None, None)
        assert probe.sub_question == a
        probe_id = probe.id
    finally:
        db.close()
        engine.dispose()
    body = (await get_family(client, key)).json()
    assert [m["sub_question"] for m in body["members"]] == [a]
    coverage = {c["sub_question"]: c for c in body["coverage"]}
    assert coverage[a]["members"] == [probe_id] and coverage[b]["state"] == "missing"


# ── FR-2: the default family is one model ─────────────────────────────────────────────────────


async def test_default_family_one_model_three_of_four(client, async_session, rng):
    key = await ff.catalogue(client, rng)
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    primary = await ff.member(async_session, model, ds, key=key, role="primary")
    candidate = await ff.member(async_session, model, ds, key=key, role="candidate")
    implicit = await ff.member(async_session, model, ds, key=key)                  # no role row
    retired = await ff.member(async_session, model, ds, key=key, role="retired")
    await async_session.commit()
    body = (await get_family(client, key)).json()
    assert body["model"] == model.id
    assert [m["probe_id"] for m in body["members"]] == [primary.id, *sorted(
        [candidate.id, implicit.id], key=lambda pid: [candidate.id, implicit.id].index(pid))]
    assert {m["probe_id"]: m["role"] for m in body["members"]} == {
        primary.id: "primary", candidate.id: "candidate", implicit.id: "candidate"}
    assert next(m for m in body["members"] if m["probe_id"] == implicit.id)["role_recorded"] is False
    with_retired = (await get_family(client, key, include_retired="true")).json()
    assert [m["probe_id"] for m in with_retired["members"]][-1] == retired.id
    assert with_retired["total"] == 4


async def test_two_models_refused_with_counts(client, async_session, rng):
    key = await ff.catalogue(client, rng)
    m1, m2 = await ff.model(async_session), await ff.model(async_session)
    ds = await ff.source(async_session)
    for model, n in ((m1, 2), (m2, 1)):
        for _ in range(n):
            await ff.member(async_session, model, ds, key=key)
    await async_session.commit()
    response = await get_family(client, key)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "model_required"
    assert detail["models"] == sorted([{"model_id": m1.id, "members": 2}, {"model_id": m2.id, "members": 1}],
                                      key=lambda m: m["model_id"])
    assert sorted(n["arguments"]["model"] for n in detail["next"]) == sorted([m1.id, m2.id])
    assert_next_calls_are_live(detail["next"])
    one = (await get_family(client, key, model=m2.id)).json()
    assert (one["model"], one["total"]) == (m2.id, 1)
    assert {m["model_id"] for m in one["members"]} == {m2.id}


async def test_explicit_member_ids_any_role(client, async_session, rng):
    key = await ff.catalogue(client, rng)
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    retired = await ff.member(async_session, model, ds, key=key, role="retired")
    candidate = await ff.member(async_session, model, ds, key=key)
    await ff.member(async_session, model, ds, key=key)
    stranger = await ff.member(async_session, model, ds, key=await ff.catalogue(client, rng))
    await async_session.commit()
    body = (await get_family(client, key, member_ids=[retired.id, candidate.id])).json()
    assert sorted(m["probe_id"] for m in body["members"]) == sorted([retired.id, candidate.id])
    refused = await get_family(client, key, member_ids=[candidate.id, stranger.id])
    assert refused.status_code == 422
    assert refused.json()["detail"]["code"] == "member_not_in_family"
    assert refused.json()["detail"]["member_ids"] == [stranger.id]


async def test_pages_union_equals_unpaged(client, async_session, rng):
    key = await ff.catalogue(client, rng)
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    for i in range(7):
        await ff.member(async_session, model, ds, key=key, role="primary" if i == 3 else None,
                        sub_question=None)
    await async_session.commit()
    whole = (await get_family(client, key)).json()
    paged, offset = [], 0
    while offset is not None:
        page = (await get_family(client, key, offset=offset, limit=3)).json()
        paged += page["members"]
        offset = page["next_offset"]
    assert [m["probe_id"] for m in paged] == [m["probe_id"] for m in whole["members"]]
    assert whole["members"][0]["role"] == "primary"


# ── FR-3: facts and flags ────────────────────────────────────────────────────────────────────


FLAG_CASES = {
    "precision": ({"precision": "float32"}, "differs_precision"),
    "render": ({"render": None}, "differs_render"),
    "scope": ({"scope": "last_assistant"}, "differs_scope"),
    "dataset_version": ({"dataset_version": "dv-other"}, "differs_dataset_version"),
    "definition_stale": ({"built": False}, "definition_stale"),
    "sub_question_not_declared": ({"sub_question": "sq-dropped-x"}, "sub_question_not_declared"),
}


@pytest.mark.parametrize("case", sorted(FLAG_CASES))
async def test_each_flag_on_a_single_field_difference(client, async_session, rng, case):
    """Three members share every field; a CANDIDATE differs in exactly one. The flag lands on that
    candidate and on no one else — compared with the majority of ALL members, not the primary."""
    override, flag = FLAG_CASES[case]
    key = await ff.catalogue(client, rng)
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    shared = [await ff.member(async_session, model, ds, key=key, role="primary"),
              await ff.member(async_session, model, ds, key=key),
              await ff.member(async_session, model, ds, key=key)]
    odd = await ff.member(async_session, model, ds, key=key, **override)
    await async_session.commit()
    body = (await get_family(client, key)).json()
    by_id = {m["probe_id"]: m for m in body["members"]}
    assert [f["flag"] for f in by_id[odd.id]["flags"]] == [flag]
    for probe in shared:
        assert by_id[probe.id]["flags"] == [], (probe.id, by_id[probe.id]["flags"])
    entry = by_id[odd.id]["flags"][0]
    if flag.startswith("differs_"):
        assert entry["majority"] == by_id[shared[0].id][entry["field"]]
        assert entry["value"] == by_id[odd.id][entry["field"]] != entry["majority"]
    if flag == "sub_question_not_declared":
        assert (entry["sub_question"], entry["version"]) == ("sq-dropped-x", body["version"])
        assert odd.id not in body["default_blend_set"]
    assert body["flags_basis"] == "majority_of_all_members"


async def test_a_differing_PRIMARY_is_flagged_not_the_majority(client, async_session, rng):
    """The negative of comparing with the primary: when the primary is the odd one out, IT carries the
    flag and the two agreeing candidates carry none."""
    key = await ff.catalogue(client, rng)
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    primary = await ff.member(async_session, model, ds, key=key, role="primary", precision="float32")
    others = [await ff.member(async_session, model, ds, key=key) for _ in range(2)]
    await async_session.commit()
    by_id = {m["probe_id"]: m for m in (await get_family(client, key)).json()["members"]}
    assert [f["flag"] for f in by_id[primary.id]["flags"]] == ["differs_precision"]
    assert all(by_id[o.id]["flags"] == [] for o in others)


def test_flags_tie_takes_the_first_member_and_says_so():
    members = [
        {"probe_id": "p1", "precision": "a", "render_form": "r", "scope": "s", "training_dataset_version": "d",
         "definition_built": True, "identity_current": None, "identity_stale_reason": None, "sub_question": None},
        {"probe_id": "p2", "precision": "b", "render_form": "r", "scope": "s", "training_dataset_version": "d",
         "definition_built": True, "identity_current": None, "identity_stale_reason": None, "sub_question": None},
    ]
    out = probe_family.flags(members, declared=[], version=1)
    assert out["p1"] == [] and out["p2"] == [{"flag": "differs_precision", "field": "precision", "value": "b",
                                              "majority": "a", "majority_tie": True}]


async def test_member_facts(client, async_session, rng):
    (sub,) = ff.sub_keys(rng, 1)
    key = await ff.catalogue(client, rng, sub_questions=[sub])
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    probe = await ff.member(async_session, model, ds, key=key, role="primary", sub_question=sub, rung=3, layer=7,
                            rule="max", scope="assistant", dataset_version="dv-7")
    await async_session.commit()
    (member,) = (await get_family(client, key)).json()["members"]
    assert {k: member[k] for k in ("probe_id", "model_id", "role", "precision", "render_form", "scope", "layer",
                                   "rule", "sub_question", "rung", "threshold", "training_dataset_version",
                                   "definition_built", "default_blend")} == {
        "probe_id": probe.id, "model_id": model.id, "role": "primary", "precision": "bfloat16",
        "render_form": "served_bos_unrecorded", "scope": "assistant", "layer": 7, "rule": "max", "sub_question": sub,
        "rung": 3, "threshold": 0.5, "training_dataset_version": "dv-7", "definition_built": True, "default_blend": True}
    assert set(member) >= {"decision_revision", "decision_digest", "detector_digest", "identity_current", "run_id"}


async def test_precision_is_read_through_one_threadpool_call(client, async_session, rng, monkeypatch):
    """FTID §3.1: every member's precision in ONE thread hop (payload: every run id; count 1)."""
    key = await ff.catalogue(client, rng)
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    probes = [await ff.member(async_session, model, ds, key=key) for _ in range(3)]
    await async_session.commit()
    calls = []
    real = probe_family._precisions
    monkeypatch.setattr(probe_family, "_precisions", lambda ids: calls.append(sorted(ids)) or real(ids))
    await get_family(client, key)
    runs = sorted({(await async_session.get(type(probes[0]), p.id)).run_id for p in probes})
    assert calls == [runs]


# ── FR-4: coverage ───────────────────────────────────────────────────────────────────────────


async def test_missing_sub_questions_have_valid_next(client, async_session, rng):
    """Three random sub-questions, members for one: two `missing`, missing FIRST, each with a `next`
    that validates against the LIVE tool schema — a training run where a training view exists, the
    registration otherwise."""
    a, b, c = ff.sub_keys(rng, 3)
    key = await ff.catalogue(client, rng, sub_questions=[a, b, c])
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    member = await ff.member(async_session, model, ds, key=key, sub_question=a)
    main = await ff.member(async_session, model, ds, key=key)
    view_b = await ff.view(async_session, ds, key=key, sub_question=b, behaviour_version=1)
    await async_session.commit()
    body = (await get_family(client, key)).json()
    cov = body["coverage"]
    assert [(e["sub_question"], e["state"]) for e in cov] == [(b, "missing"), (c, "missing"), (a, "covered"),
                                                              (None, "covered")]
    assert cov[2]["members"] == [member.id] and cov[3]["group"] == "main_question" and cov[3]["members"] == [main.id]
    assert cov[0]["next"] == [{"tool": "start_probe_training_run", "arguments": {
        "model_id": model.id, "train_dataset_id": view_b.id, "behaviour": key}, "why": cov[0]["next"][0]["why"]}]
    assert cov[1]["next"][0]["tool"] == "register_probe_dataset"
    assert cov[1]["next"][0]["arguments"] == {"behaviour": key, "sub_question": c, "role": "train",
                                              "behaviour_version": 1}
    assert set(cov[1]["next"][0]["needs"]) >= {"dataset_id", "label_mapping"}
    assert_next_calls_are_live([n for e in cov for n in e.get("next", [])])
    assert body["declared_sub_questions"] == [a, b, c]


async def test_declared_sub_questions_come_from_the_CURRENT_version(client, async_session, rng):
    """A new version that declares another sub-question moves coverage — read from the catalogue, never a
    constant (FR-4's control)."""
    a, b = ff.sub_keys(rng, 2)
    key = await ff.catalogue(client, rng, sub_questions=[a])
    model = await ff.model(async_session)
    ds = await ff.source(async_session)
    await ff.member(async_session, model, ds, key=key, sub_question=a)
    await async_session.commit()
    assert [c["state"] for c in (await get_family(client, key)).json()["coverage"]] == ["covered", "empty"]
    made = await client.post(f"{API}/behaviours/{key}/versions", json={
        "base_version": 1, "sub_questions": [{"key": a, "question": "q", "positive_meaning": "p"},
                                             {"key": b, "question": "q", "positive_meaning": "p"}]})
    assert made.status_code == 201, made.text
    body = (await get_family(client, key)).json()
    assert [(c["sub_question"], c["state"]) for c in body["coverage"]] == [(b, "missing"), (a, "covered"),
                                                                          (None, "empty")]


# ── wiring ───────────────────────────────────────────────────────────────────────────────────


def test_family_route_in_openapi():
    from src.main import app

    paths = app.openapi()["paths"]
    assert "get" in paths["/api/v1/probe-monitors/behaviours/{key}/family"]
    params = {p["name"] for p in paths["/api/v1/probe-monitors/behaviours/{key}/family"]["get"]["parameters"]}
    assert params >= {"key", "model", "member_ids", "include_retired", "offset", "limit"}


async def test_the_route_calls_the_service_with_the_request(client, async_session, rng, monkeypatch):
    """Payload and count: the route hands every query parameter to `probe_family.family`, once."""
    key = await ff.catalogue(client, rng)
    calls = []

    async def spy(db, k, **kwargs):
        calls.append((k, kwargs))
        return {"key": k}

    monkeypatch.setattr(probe_family, "family", spy)
    await get_family(client, key, model="m_1", member_ids=["pm_a", "pm_b"], include_retired="true", offset=2, limit=3)
    assert calls == [(key, {"model": "m_1", "member_ids": ["pm_a", "pm_b"], "include_retired": True,
                            "offset": 2, "limit": 3})]


async def test_an_uncatalogued_key_is_404(client):
    response = await get_family(client, slug(rng_for("x"), prefix="zz"))
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_found"
