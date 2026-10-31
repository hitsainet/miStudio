"""The success report (Feature 040 S2 FR-19, FR-22 – FR-25; FTID §3.5; FTASKS 11.2 – 11.6).

Through the REAL route over real rows and arrays. Criteria values are random and unlike any pilot's
(min_recall 0.63 / 0.71, target FPR 0.03), so a hard-coded value cannot pass. Expected figures are
counted from the fixture's own construction: k of n positives above the bar.
"""
from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import select

from src.core.config import settings
from src.services.detector_success_report import overall
from src.services.probe_monitor_metrics import wilson_interval
from tests.support.detector_fixtures import (
    Arrays,
    dataset_row,
    declare,
    evaluation_row,
    model_row,
    run_and_probe,
    view_row,
    write_split,
)
from tests.support.next_calls import assert_next_calls_are_live
from tests.support.random_behaviour import behaviour_body, rng_for, slug

API = "/api/v1/probe-monitors"
N = 120


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def arrays(n_pos, pos_fired, n_neg, neg_fired, start=1):
    """Scores above the 0.5 bar exactly where asked; positions start..start+2·n (non-contiguous)."""
    scores = np.r_[np.full(pos_fired, 0.9), np.full(n_pos - pos_fired, 0.1),
                   np.full(neg_fired, 0.8), np.full(n_neg - neg_fired, 0.2)]
    labels = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(np.int8)
    n = n_pos + n_neg
    return Arrays(scores=scores, positions=np.arange(start, start + 2 * n, 2), n_scored=np.full(n, 12), labels=labels)


class World:
    pass


async def build_world(client, session, data_dir, rng, *, criteria, held_out=True, overlap=False,
                      n_negatives=5000, probe_target=0.03, ood_fired=30, id_fired=36, models=1):
    w = World()
    w.column = slug(rng, words=1)
    w.key = slug(rng)
    regions = [slug(rng, words=1) for _ in range(4)]
    # training rows carry regions[0..1]; test rows regions[2..3] (or regions[1] when `overlap`)
    test_values = [regions[1] if overlap else regions[2], regions[3]]
    w.train_ds = await dataset_row(session, write_split(data_dir, {
        "text": [str(i) for i in range(N)], "label": [str(i % 2) for i in range(N)],
        w.column: [regions[i % 2] for i in range(N)]}))
    w.test_ds = await dataset_row(session, write_split(data_dir, {
        "text": [str(i) for i in range(2 * N)], "label": [str(i % 2) for i in range(2 * N)],
        w.column: [test_values[i % 2] for i in range(2 * N)]}))
    w.models = [await model_row(session) for _ in range(models)]
    w.train = await view_row(session, w.train_ds, role="train", behaviour=w.key)
    w.ood = await view_row(session, w.test_ds, role="eval", distribution="out_of_distribution", behaviour=w.key)
    w.id_view = await view_row(session, w.test_ds, role="eval", distribution="in_distribution", behaviour=w.key)
    w.cal = await view_row(session, w.train_ds, role="calibration")
    for view in (w.train, w.ood):
        await declare(session, view, [w.column])
    w.probes = []
    for model in w.models:
        run, probe = await run_and_probe(session, model, w.train, eval_views=[w.ood, w.id_view], calibration_view=w.cal,
                                         behaviour=w.key, target_fpr=probe_target, n_negatives=n_negatives)
        w.ood_eval = await evaluation_row(session, probe, run, w.ood, arrays(40, ood_fired, 40, 1))
        w.id_eval = await evaluation_row(session, probe, run, w.id_view, arrays(40, id_fired, 40, 0))
        w.probes.append(probe)
        w.run = run
    await session.commit()
    body = behaviour_body(rng, key=w.key, target_fpr=0.03, success_criteria=criteria(w),
                          held_out_columns=[w.column] if held_out else None)
    made = await client.post(f"{API}/behaviours", json=body)
    assert made.status_code == 201, made.text
    return w


def held(w, **extra):
    out = {"id": "crit-held", "detector": "best_single", "subset": "all", "held_out_test": w.column, "min_recall": 0.63}
    out.update(extra)
    return out


def indist(w, **extra):
    out = {"id": "crit-indist", "detector": "best_single", "subset": "all", "min_recall": 0.71}
    out.update(extra)
    return out


async def report(client, key, **body):
    response = await client.post(f"{API}/behaviours/{key}/success-reports", json=body)
    assert response.status_code == 200, response.text
    return response.json()


async def test_two_forms_each_get_their_verdict(client, async_session, data_dir, request):
    rng = rng_for(request.node.nodeid)
    w = await build_world(client, async_session, data_dir, rng, criteria=lambda w: [
        held(w), {"id": "crit-rules", "detector": "rules_alone", "subset": "all", "min_recall": 0.5}, indist(w)])
    out = await report(client, w.key)
    criteria = {c["id"]: c for c in out["report"]["criteria"]}
    h = criteria["crit-held"]
    assert h["verdict"] == "met" and h["figures"]["recall"] == pytest.approx(30 / 40)       # 0.75 ≥ 0.63
    lo, hi = wilson_interval(30, 40)
    assert h["figures"]["recall_ci"] == [pytest.approx(lo, abs=1e-6), pytest.approx(hi, abs=1e-6)]
    assert (h["in_distribution"], h["test_view_ids"], h["target_fpr"], h["min_recall"]) == (False, [w.ood.id], 0.03, 0.63)
    assert criteria["crit-rules"]["verdict"] == "not_assessable"
    # Feature 040 S3: a version with NO rule set cannot judge `rules_alone` (no rule can see anything).
    assert [r["code"] for r in criteria["crit-rules"]["reasons"]] == ["no_rule_set"]
    i = criteria["crit-indist"]
    assert (i["verdict"], i["in_distribution"], i["test_view_ids"]) == ("met", True, [w.id_view.id])   # 0.9 ≥ 0.71
    assert out["report"]["overall_verdict"] == "not_assessable"
    assert out["report"]["probe"]["chosen_by"] == "only_run_selected"
    assert out["stale"]["stale"] is False


async def test_not_met_at_the_point_and_the_lower_bound_reported(client, async_session, data_dir, request):
    """26/40 = 0.65 ≥ 0.63 at the point, but its Wilson lower bound (0.495) is below: met, not at the bound."""
    rng = rng_for(request.node.nodeid)
    w = await build_world(client, async_session, data_dir, rng, ood_fired=26, criteria=lambda w: [held(w)])
    c = (await report(client, w.key))["report"]["criteria"][0]
    assert (c["verdict"], c["met_at_lower_bound"]) == ("met", False)
    w2 = await build_world(client, async_session, data_dir, rng, ood_fired=25, criteria=lambda w: [held(w)])
    c2 = (await report(client, w2.key))["report"]["criteria"][0]
    assert c2["verdict"] == "not_met"                                                   # 0.625 < 0.63
    assert (await report(client, w2.key))["report"]["overall_verdict"] == "not_met"


async def test_a_shared_held_out_value_is_an_overlap_naming_column_and_value(client, async_session, data_dir, request):
    rng = rng_for(request.node.nodeid)
    w = await build_world(client, async_session, data_dir, rng, overlap=True, criteria=lambda w: [held(w)])
    c = (await report(client, w.key))["report"]["criteria"][0]
    assert c["verdict"] == "not_assessable"
    reason = c["reasons"][0]
    assert (reason["code"], reason["column"], reason["test_view_id"], reason["training_view_id"]) == (
        "held_out_overlap", w.column, w.ood.id, w.train.id)


async def test_few_negatives_make_the_criterion_not_assessable(client, async_session, data_dir, request):
    rng = rng_for(request.node.nodeid)
    w = await build_world(client, async_session, data_dir, rng, n_negatives=540, criteria=lambda w: [held(w)])
    c = (await report(client, w.key))["report"]["criteria"][0]
    assert c["verdict"] == "not_assessable" and [r["code"] for r in c["reasons"]] == ["few_negatives"]
    assert c["calibration"]["sentence"] == "3% of 540 = 16 rows"


async def test_a_bar_cut_at_another_target_is_not_assessable_with_the_preview_next(client, async_session, data_dir, request):
    rng = rng_for(request.node.nodeid)
    w = await build_world(client, async_session, data_dir, rng, probe_target=0.01, criteria=lambda w: [held(w)])
    c = (await report(client, w.key))["report"]["criteria"][0]
    assert [r["code"] for r in c["reasons"]] == ["target_fpr_mismatch"]
    assert c["next"][0]["tool"] == "recalibrate_probe_monitor" and c["next"][0]["arguments"]["target_fpr"] == 0.03
    assert_next_calls_are_live(c["next"])


async def test_no_criteria_is_no_bar_set_with_figures(client, async_session, data_dir, request):
    rng = rng_for(request.node.nodeid)
    w = await build_world(client, async_session, data_dir, rng, held_out=False, criteria=lambda w: None)
    out = (await report(client, w.key))["report"]
    assert out["overall_verdict"] == "no_bar_set" and out["criteria"] == []
    assert {f["view_id"] for f in out["figures"]} == {w.ood.id, w.id_view.id}


async def test_identical_requests_return_one_report(client, async_session, data_dir, request):
    from src.models.detector_report import DetectorSuccessReport

    rng = rng_for(request.node.nodeid)
    w = await build_world(client, async_session, data_dir, rng, criteria=lambda w: [held(w)])
    a, b = await report(client, w.key), await report(client, w.key)
    assert a["id"] == b["id"] and a["body_sha256"] == b["body_sha256"]
    count = len((await async_session.execute(select(DetectorSuccessReport).where(
        DetectorSuccessReport.behaviour_key == w.key))).scalars().all())
    assert count == 1
    read = (await client.get(f"{API}/success-reports/{a['id']}")).json()
    assert read["report"] == a["report"]
    listed = (await client.get(f"{API}/behaviours/{w.key}/success-reports")).json()
    assert [r["id"] for r in listed["reports"]] == [a["id"]] and listed["total"] == 1


async def test_a_later_version_makes_the_earlier_report_stale_and_unchanged(client, async_session, data_dir, request):
    rng = rng_for(request.node.nodeid)
    w = await build_world(client, async_session, data_dir, rng, criteria=lambda w: [held(w)])
    first = await report(client, w.key)
    await client.post(f"{API}/behaviours/{w.key}/versions", json={"base_version": 1, "success_criteria": [
        held(w, min_recall=0.8)]})
    again = (await client.get(f"{API}/success-reports/{first['id']}")).json()
    assert again["report"] == first["report"] and again["body_sha256"] == first["body_sha256"]
    assert again["stale"]["stale"] is True
    assert [m["input"] for m in again["stale"]["moved"]] == ["behaviour_version"]
    assert_next_calls_are_live(again["stale"]["next"])
    second = await report(client, w.key)
    assert second["id"] != first["id"] and second["report"]["behaviour_version"] == 2
    assert second["report"]["criteria"][0]["verdict"] == "not_met"                     # 0.75 < 0.8 now


async def test_a_removed_input_is_named_on_read(client, async_session, data_dir, request):
    from src.models.probe_monitor import ProbeMonitorEvaluation

    rng = rng_for(request.node.nodeid)
    w = await build_world(client, async_session, data_dir, rng, criteria=lambda w: [held(w)])
    first = await report(client, w.key)
    await async_session.delete(await async_session.get(ProbeMonitorEvaluation, w.ood_eval.id))
    await async_session.commit()
    stale = (await client.get(f"{API}/success-reports/{first['id']}")).json()["stale"]
    assert stale["inputs_removed"] == {"evaluations": [w.ood_eval.id]}


async def test_missing_evaluation_names_the_evaluate_call(client, async_session, data_dir, request):
    from src.models.probe_monitor import ProbeMonitorEvaluation

    rng = rng_for(request.node.nodeid)
    w = await build_world(client, async_session, data_dir, rng, criteria=lambda w: [held(w)])
    await async_session.delete(await async_session.get(ProbeMonitorEvaluation, w.ood_eval.id))
    await async_session.commit()
    c = (await report(client, w.key))["report"]["criteria"][0]
    assert [r["code"] for r in c["reasons"]] == ["missing_evaluation"]
    assert c["next"] == [{"tool": "evaluate_probe_monitor", "arguments": {"probe_id": w.probes[0].id, "dataset_ids": [w.ood.id]},
                          "why": c["next"][0]["why"]}]
    assert_next_calls_are_live(c["next"])


async def test_two_models_is_ambiguous_with_one_next_per_model(client, async_session, data_dir, request):
    rng = rng_for(request.node.nodeid)
    w = await build_world(client, async_session, data_dir, rng, models=2, criteria=lambda w: [held(w)])
    response = await client.post(f"{API}/behaviours/{w.key}/success-reports", json={})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "ambiguous_model" and sorted(detail["candidates"]) == sorted(m.id for m in w.models)
    assert len(detail["next"]) == 2
    assert_next_calls_are_live(detail["next"])
    named = await report(client, w.key, model_id=w.models[1].id)
    assert named["report"]["model_id"] == w.models[1].id


@pytest.mark.parametrize("verdicts,expected", [
    (["met", "met"], "met"),
    (["met", "not_assessable"], "not_assessable"),
    (["not_assessable", "not_met"], "not_met"),
    ([], "no_bar_set"),
])
def test_the_overall_truth_table(verdicts, expected):
    assert overall(verdicts) == expected


async def test_the_floor_is_the_runtime_setting(client, async_session, data_dir, request):
    from src.core import runtime_settings

    w = await build_world(client, async_session, data_dir, rng_for(request.node.nodeid), criteria=lambda w: [held(w)])
    before = await report(client, w.key)
    try:
        runtime_settings.apply({"calibration_floor_negatives": 6000})
        after = await report(client, w.key)
    finally:
        runtime_settings.reset()
    c = after["report"]["criteria"][0]
    assert [r["code"] for r in c["reasons"]] == ["few_negatives"] and c["calibration"]["floor"] == 6000
    # the floor is part of the report's identity: a changed setting is a new report, not the earlier one back
    assert after["id"] != before["id"]
    assert before["report"]["criteria"][0]["calibration"]["floor"] != 6000


async def test_report_pages_are_the_runtime_settings(client, async_session, data_dir, request):
    from src.core import runtime_settings

    w = await build_world(client, async_session, data_dir, rng_for(request.node.nodeid), criteria=lambda w: [held(w)])
    first = await report(client, w.key)
    await client.post(f"{API}/behaviours/{w.key}/versions", json={"base_version": 1, "calibration_floor": 7})
    second = await report(client, w.key)
    try:
        runtime_settings.apply({"report_list_page_default": 1})
        page = (await client.get(f"{API}/behaviours/{w.key}/success-reports")).json()
        assert [r["id"] for r in page["reports"]] == [second["id"]] and page["next_cursor"]
        rest = (await client.get(f"{API}/behaviours/{w.key}/success-reports", params={"cursor": page["next_cursor"]})).json()
        assert [r["id"] for r in rest["reports"]] == [first["id"]] and rest["next_cursor"] is None
        runtime_settings.apply({"report_list_page_max": 1})
        over = await client.get(f"{API}/behaviours/{w.key}/success-reports", params={"limit": 2})
        assert over.status_code == 422 and over.json()["detail"]["setting"] == "report_list_page_max"
    finally:
        runtime_settings.reset()


async def test_an_oversized_report_is_refused(client, async_session, data_dir, request):
    from src.core import runtime_settings

    w = await build_world(client, async_session, data_dir, rng_for(request.node.nodeid),
                          criteria=lambda w: [held(w, id=f"crit-{i}") for i in range(12)])
    try:
        runtime_settings.apply({"success_report_max_bytes": 4096})
        response = await client.post(f"{API}/behaviours/{w.key}/success-reports", json={})
    finally:
        runtime_settings.reset()
    assert response.status_code == 422
    assert (response.json()["detail"]["code"], response.json()["detail"]["setting"]) == ("over_cap", "success_report_max_bytes")
