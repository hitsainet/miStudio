"""The success report's rule forms and rule subsets (Feature 040 S3; FR-22 (rule forms), FR-31, FR-33, FR-36; 040-T27;
FTASKS 22.4, 22.5).

Over `tests/support/rules_world.py`'s hand-computed table, after a REAL rules evaluation cached the verdicts:

* `rules_alone` → 3/6; `rules_or_blend` (the probe as the blend until 041's) → 4/6, judged at the RECORDED
  decision, with the union's realised FPR on the calibration negatives reported and, above the target,
  `not_assessable` / `union_over_budget` with a recalibration-preview `next` (040-T27);
* `rule_invisible_positives` / `rule_visible_positives` subsets select the positives a rule does / does not see;
* no completed rules evaluation → `missing_evaluation` (`rules_evaluation`) with `evaluate_detector_with_rules`;
* the success report's identity moves with the rules engine digest, and a moved engine reads `stale` (FR-38).

Controls (FTASKS 22.4): judge `rules_or_blend` at the union bar instead of the recorded decision → p5 (score 0.6,
above the recorded 0.5 bar and below the union bar) leaves the recall → red.
"""
from __future__ import annotations

import pytest

from src.core.config import settings
from tests.support import rules_world as rw
from tests.support.next_calls import assert_next_calls_are_live

API = rw.API


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def criteria():
    return [
        {"id": "c-union", "detector": "rules_or_blend", "subset": "all", "min_recall": 0.6},
        {"id": "c-rules", "detector": "rules_alone", "subset": "all", "min_recall": 0.5},
        {"id": "c-invisible", "detector": "best_single", "subset": "rule_invisible_positives", "min_recall": 0.6},
        {"id": "c-visible", "detector": "best_single", "subset": "rule_visible_positives", "min_recall": 0.5},
    ]


async def _world(client, async_session, data_dir, request, **kwargs):
    return await rw.build(client, async_session, data_dir, request.node.nodeid, distribution="in_distribution",
                          behaviour_extra={"success_criteria": criteria(), "calibration_floor": 500}, **kwargs)


async def _report(client, world):
    response = await client.post(f"{API}/behaviours/{world.key}/success-reports", json={})
    assert response.status_code == 200, response.text
    return response.json(), {c["id"]: c for c in response.json()["report"]["criteria"]}


async def test_rule_forms_and_subsets_are_judged(client, async_session, data_dir, monkeypatch, request):
    # A recorded bar of 0.985 spends 0.3% on the calibration negatives after the vetoes; the union, 0.7%.
    world = await _world(client, async_session, data_dir, request, threshold=0.985,
                         scores={"p1": 0.995, "p3": 0.995, "p4": 0.995, "n2": 0.995, "n4": 0.995})
    await rw.run_one(client, monkeypatch, world, calibration=True)
    envelope, by_id = await _report(client, world)
    union, rules = by_id["c-union"], by_id["c-rules"]
    assert (union["figures"]["flagged_positives"], union["figures"]["positives"]) == (4, 6)
    assert union["verdict"] == "met", union
    assert union["union_calibration"]["realised_union_fpr"] == pytest.approx(0.007)
    assert (rules["figures"]["flagged_positives"], rules["verdict"]) == (3, "met")
    invisible, visible = by_id["c-invisible"], by_id["c-visible"]
    assert (invisible["figures"]["flagged_positives"], invisible["figures"]["positives"], invisible["verdict"]) == (2, 3, "met")
    assert (visible["figures"]["flagged_positives"], visible["figures"]["positives"], visible["verdict"]) == (1, 3, "not_met")
    assert envelope["report"]["rules_engine_digest"] is not None


async def test_rules_or_blend_over_budget_is_not_assessable_at_the_recorded_bar(client, async_session, data_dir,
                                                                                monkeypatch, request):
    world = await _world(client, async_session, data_dir, request, scores={"p5": 0.6})
    await rw.run_one(client, monkeypatch, world, calibration=True)
    _, by_id = await _report(client, world)
    union = by_id["c-union"]
    # judged at the RECORDED 0.5 bar: p5 (0.6) fires; at the union bar (≈0.99) it would not
    assert union["figures"]["flagged_positives"] == 5
    assert union["verdict"] == "not_assessable"
    reason = next(r for r in union["reasons"] if r["code"] == "union_over_budget")
    assert reason["realised_union_fpr"] > 0.01 and reason["target_fpr"] == 0.01
    assert any(n["tool"] == "recalibrate_probe_monitor" for n in union["next"])
    assert_next_calls_are_live(union["next"])


async def test_without_a_rules_evaluation_the_criterion_names_the_call(client, async_session, data_dir, request):
    world = await _world(client, async_session, data_dir, request)
    _, by_id = await _report(client, world)
    rules = by_id["c-rules"]
    assert rules["verdict"] == "not_assessable"
    missing = next(r for r in rules["reasons"] if r["code"] == "missing_evaluation")
    assert missing["missing"] == "rules_evaluation" and missing["view_ids"] == [world.eval_view.id]
    call = next(n for n in rules["next"] if n["tool"] == "evaluate_detector_with_rules")
    assert call["arguments"]["view_ids"] == [world.eval_view.id]
    assert_next_calls_are_live(rules["next"])


async def test_a_moved_engine_marks_the_report_stale(client, async_session, data_dir, monkeypatch, request):
    import src.rule_signals as rule_signals

    world = await _world(client, async_session, data_dir, request)
    await rw.run_one(client, monkeypatch, world, calibration=True)
    envelope, _ = await _report(client, world)
    real = rule_signals.vector_digests
    monkeypatch.setattr(rule_signals, "vector_digests", lambda: {**real(), "rule-signal-cases-v1.json": "1" * 64})
    read = (await client.get(f"{API}/success-reports/{envelope['id']}")).json()
    assert read["stale"]["stale"] is True
    assert [m["input"] for m in read["stale"]["moved"]] == ["rules_engine_digest"]
    assert read["report"] == envelope["report"], "a stale report is never rewritten"


async def test_pending_operator_code_reads_not_assessable_and_another_digests_producer_is_not_used(
        client, async_session, data_dir, monkeypatch, request):
    """Review R1 (S3), FR-31/FR-35 in the report: every evaluated row depends on operator code no runner can run,
    so the rule criterion is not assessable (`rule_unknown_share`, `runner_unavailable`) — and a producer column
    written for ANOTHER digest does not stand in for miStudio's verdict (only exactly this digest may)."""
    from tests.support import rule_set_fixtures as rf
    from tests.support.random_behaviour import rng_for
    from tests.unit.test_rules_evaluation import _producer_column

    rng = rng_for(request.node.nodeid)
    code, expr, col = rf.ident(rng, "k"), rf.ident(rng, "e"), rf.ident(rng, "c")
    world = await _world(client, async_session, data_dir, request, rule_set=None)
    doc = rf.document(world.key, rng, signals=[rf.code_rule_signal(code)], expressions={expr: {"signal": code}},
                      outputs=[rf.fired(col, expr)])
    made = await client.post(f"{API}/behaviours/{world.key}/versions", json={"base_version": 1, "rule_set": {"document": doc}})
    assert made.status_code == 201, made.text
    column = "producer_" + rf.ident(rng, "m")
    await _producer_column(async_session, world, data_dir, [True] * 24, column)
    world.eval_view.rule_columns = {"rule_match": column, "rule_set_id": "s@1", "rule_set_digest": "f" * 64}
    await async_session.commit()
    await rw.run_one(client, monkeypatch, world, calibration=False)
    _, by_id = await _report(client, world)
    rules = by_id["c-rules"]
    assert rules["verdict"] == "not_assessable", rules
    codes = {r["code"] for r in rules["reasons"]}
    assert {"rule_unknown_share", "runner_unavailable"} <= codes
    assert "producer_computed" not in (rules.get("notes") or [])
