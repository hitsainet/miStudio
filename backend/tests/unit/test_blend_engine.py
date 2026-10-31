"""The blend engine (Feature 041 FR-8 – FR-13; FTASKS 6.2 – 6.7, 6.9).

Pure: synthetic member arrays with planted structure (`tests/support/blend_fixtures.py`), non-contiguous
positions and members that kept different rows. Each test names the mutation that must turn it red; the
controls are recorded in `0xcc/reviews/041_implementation_controls_2026-10-24.md`.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.services import probe_monitor_metrics, probe_monitor_trainer, probe_breakdown
from src.services.probe_blend import candidates as cand
from src.services.probe_blend import engine
from tests.support import blend_fixtures as bf


def run(inputs):
    return engine.run(inputs)


# ── alignment (T23) ──────────────────────────────────────────────────────────────────────────


def test_pairs_by_position_not_index():
    inputs = bf.build(1)
    view = "pmd_fit"
    aligned = engine.align(view, inputs.members, inputs.arrays[view], 0.9)
    for j, member in enumerate(inputs.members):
        a = inputs.arrays[view][member]
        lookup = dict(zip(a.positions.tolist(), a.scores.tolist()))
        assert [lookup[p] for p in aligned.positions.tolist()] == aligned.scores[:, j].tolist()
    assert np.all(np.diff(aligned.positions) > 0)
    # every member kept 98% of the rows: the join keeps only the shared ones and says so
    counts = aligned.counts[inputs.members[0]]
    assert counts["aligned"] == aligned.positions.size < counts["n"]


def test_misaligned_refused():
    inputs = bf.build(2, keep=0.8)
    with pytest.raises(engine.Misaligned) as caught:
        engine.align("pmd_fit", inputs.members, inputs.arrays["pmd_fit"], 0.95)
    assert caught.value.code == "members_misaligned" and caught.value.details["view"] == "pmd_fit"


def test_labels_disagree_refused():
    inputs = bf.build(3)
    a = inputs.arrays["pmd_fit"][inputs.members[1]]
    a.labels = a.labels.copy()
    shared = engine.align("pmd_fit", inputs.members, inputs.arrays["pmd_fit"], 0.9).positions
    target = int(shared[10])
    a.labels[a.positions == target] = 1 - a.labels[a.positions == target]
    with pytest.raises(engine.LabelsDisagree) as caught:
        engine.align("pmd_fit", inputs.members, inputs.arrays["pmd_fit"], 0.9)
    assert caught.value.details == {"view": "pmd_fit", "position": int(shared[10])}


def test_calibration_without_positions_refused():
    inputs = bf.build(4)
    inputs.arrays["pmd_cal"][inputs.members[0]].positions = None
    with pytest.raises(engine.PositionsMissing):
        run(inputs)


def test_member_constant_refused():
    inputs = bf.build(5)
    a = inputs.arrays["pmd_fit"][inputs.members[2]]
    a.scores = np.full(a.scores.shape, 0.25)
    with pytest.raises(engine.MemberConstant) as caught:
        run(inputs)
    assert caught.value.details == {"member": inputs.members[2]}


# ── fitting (FR-9) ───────────────────────────────────────────────────────────────────────────


def test_stack_recovers_weight_sign_and_order():
    inputs = bf.build(6, signals=(2.0, 1.0, 0.0), n_fit=2000, candidates=["stack"])
    out = run(inputs)
    weights = out["candidates"][0]["params"]["weights"]
    assert weights[0] > weights[1] > 0 and abs(weights[2]) < weights[1] / 2


def test_standardisation_is_fitted_on_blend_fit_only():
    inputs = bf.build(7, candidates=["max"])
    out = run(inputs)
    fit = engine.align("pmd_fit", inputs.members, inputs.arrays["pmd_fit"], 0.9)
    for j, member in enumerate(inputs.members):
        assert out["standardisation"][member]["mean"] == pytest.approx(float(fit.scores[:, j].mean()), rel=1e-12)
        assert out["standardisation"][member]["std"] == pytest.approx(float(fit.scores[:, j].std()), rel=1e-12)


# ── verdicts at the recorded decision (OQ-3 (a)) ─────────────────────────────────────────────


def test_member_verdict_uses_length_band():
    """A member with probe bands: a short row is judged at its band's bar, not the global one."""
    decision = {"probe_bar": 0.9, "window_entry": None, "probe_scope": "all", "window": None,
                "probe_bands": [{"min_tokens": 0, "max_tokens": 50, "threshold": 0.2},
                                {"min_tokens": 50, "max_tokens": None, "threshold": 0.9}]}
    scores = np.array([0.5, 0.5, 0.95])
    n_scored = np.array([10, 100, 100])
    np.testing.assert_array_equal(engine.member_verdicts(decision, scores, n_scored), [True, False, True])


def test_member_verdicts_call_decide_many(monkeypatch):
    calls = []
    real = probe_monitor_metrics.decide_many
    monkeypatch.setattr(probe_monitor_metrics, "decide_many", lambda *a: calls.append(a) or real(*a))
    engine.member_verdicts(bf.bar(0.4), np.array([0.1, 0.6]), np.array([3, 4]))
    assert len(calls) == 1 and calls[0][:5] == (0.4, None, None, "all", None)


# ── folds (FR-11, T25) ───────────────────────────────────────────────────────────────────────


def test_folds_never_split_a_group():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 40, 500)
    y = (rng.random(500) < 0.5).astype(int)
    fold_ids, flags, _ = engine.folds(groups, y, 5, 3, 5)
    assert flags == []
    for g in np.unique(groups):
        assert np.unique(fold_ids[groups == g]).size == 1


def test_three_groups_five_folds_reads_few_groups():
    rng = np.random.default_rng(1)
    groups = rng.integers(0, 3, 300)
    y = (rng.random(300) < 0.5).astype(int)
    fold_ids, flags, counts = engine.folds(groups, y, 5, 3, 5)
    assert "few_groups" in flags and counts["groups"] == 3
    inputs = bf.build(8, groups=3)
    out = run(inputs)
    cv = out["candidates"][0]["cv"]
    assert cv["not_assessable"] is True and "few_groups" in cv["reason"]
    assert out["candidates"][0]["tests"]["pmd_ood"]["recall"] is not None      # test figures still reported


def test_no_group_column_flags_ungrouped():
    out = run(bf.build(9, groups=None))
    assert "ungrouped_folds" in out["folds"]["flags"]


def test_cv_is_out_of_fold():
    """The stack's in-fold AUROC is computed from values each fold's model never saw: with pure-noise
    members it sits near chance (an in-sample fit would read above it)."""
    inputs = bf.build(10, signals=(0.0, 0.0, 0.0), n_fit=400, candidates=["stack"])
    cv = run(inputs)["candidates"][0]["cv"]
    assert cv["in_fold"] is True and cv["auroc"] < 0.62


# ── figures, calibration, ranking (FR-10, FR-12, FR-13) ──────────────────────────────────────


def test_one_candidate_every_figure_equals_hand_value():
    from src.services.probe_monitor_metrics import auroc, wilson_interval
    from src.services.probe_monitor_trainer import calibrate

    inputs = bf.build(11, candidates=["max"])
    out = run(inputs)
    fit = engine.align("pmd_fit", inputs.members, inputs.arrays["pmd_fit"], 0.9)
    mean, std = fit.scores.mean(axis=0), fit.scores.std(axis=0)
    cal = engine.align("pmd_cal", inputs.members, inputs.arrays["pmd_cal"], 0.9)
    cal_values = ((cal.scores - mean) / std).max(axis=1)
    cut = calibrate(cal_values.tolist(), target_fpr=inputs.target_fpr, source="calibration_set")
    entry = out["candidates"][0]
    assert entry["calibration"]["threshold"] == pytest.approx(cut.threshold, rel=1e-12)
    assert entry["calibration"]["n_negatives"] == cal.positions.size
    test = engine.align("pmd_ood", inputs.members, inputs.arrays["pmd_ood"], 0.9)
    values = ((test.scores - mean) / std).max(axis=1)
    fires = values >= cut.threshold
    pos = test.labels == 1
    hits, n_pos = int((fires & pos).sum()), int(pos.sum())
    figures = entry["tests"]["pmd_ood"]
    assert figures["recall"] == pytest.approx(hits / n_pos, rel=1e-12)
    assert figures["wilson"] == pytest.approx(list(wilson_interval(hits, n_pos)), rel=1e-12)
    assert figures["auroc"] == pytest.approx(auroc(values.tolist(), test.labels.tolist()), rel=1e-12)
    assert (figures["n_pos"], figures["n_neg"]) == (n_pos, int((~pos).sum()))


def test_verdict_candidates_report_realised_fpr_and_no_auroc():
    out = run(bf.build(12, candidates=["any_of"]))
    entry = out["candidates"][0]
    assert entry["calibration"]["threshold"] is None and entry["calibration"]["k"] == 1
    assert 0.0 <= entry["calibration"]["realised_fpr"] <= 1.0
    assert entry["tests"]["pmd_ood"]["auroc"] is None and entry["tests"]["pmd_ood"]["auroc_reason"] == "verdict_only"


def test_few_negatives_is_flagged_never_refused():
    inputs = bf.build(13, n_cal=60, candidates=["stack"])
    entry = run(inputs)["candidates"][0]
    assert entry["calibration"]["few_negatives"] is True and entry["calibration"]["n_negatives"] < 100


def test_worst_held_out_ranks_first():
    """By the WORST held-out view, not the mean: `a` has the best mean (0.775) and the worst minimum (0.6),
    so it ranks below `c` (0.7 on both); `b` ties `c` on the minimum and loses on its Wilson lower bound."""
    per = {
        "a": {"h1": {"recall": 0.95, "wilson": [0.9, 0.98]}, "h2": {"recall": 0.6, "wilson": [0.5, 0.7]},
              "i1": {"recall": 0.99, "wilson": [0.9, 1.0]}},
        "b": {"h1": {"recall": 0.7, "wilson": [0.6, 0.8]}, "h2": {"recall": 0.7, "wilson": [0.55, 0.8]},
              "i1": {"recall": 0.1, "wilson": [0.0, 0.2]}},
        "c": {"h1": {"recall": 0.7, "wilson": [0.65, 0.8]}, "h2": {"recall": 0.7, "wilson": [0.62, 0.8]},
              "i1": {"recall": 0.1, "wilson": [0.0, 0.2]}},
    }
    order, basis = engine.rank(per, ["h1", "h2"], ["i1"])
    assert (order, basis) == (["c", "b", "a"], "held_out_min")


def test_in_distribution_only_label():
    per = {"a": {"i1": {"recall": 0.4, "wilson": [0.3, 0.5]}}, "b": {"i1": {"recall": 0.6, "wilson": [0.5, 0.7]}}}
    assert engine.rank(per, [], ["i1"]) == (["b", "a"], "in_distribution_only")
    out = run(bf.build(14, tests=({"id": "pmd_ind", "held_out": False},)))
    assert out["ranking"]["basis"] == "in_distribution_only"


def test_breakdown_per_declared_column_uses_040s_groups(monkeypatch):
    inputs = bf.build(15, candidates=["stack"])
    rng = np.random.default_rng(0)
    size = int(max(a.positions.max() for a in inputs.arrays["pmd_ood"].values())) + 1
    inputs.columns = {"pmd_ood": {"kind": (rng.integers(0, 3, size), ["x", "y", "z"])}}
    calls = []
    real = probe_breakdown.groups_of
    monkeypatch.setattr(probe_breakdown, "groups_of", lambda *a: calls.append(a) or real(*a))
    out = run(inputs)
    groups = out["candidates"][0]["tests"]["pmd_ood"]["breakdown"]["kind"]
    assert len(calls) == 1 and {g["value"] for g in groups} <= {"x", "y", "z"}
    assert sum(g["rows"] for g in groups) == out["candidates"][0]["tests"]["pmd_ood"]["n_pos"] + \
        out["candidates"][0]["tests"]["pmd_ood"]["n_neg"]


def test_engine_calls_calibrate_auroc_wilson(monkeypatch):
    """6.9: the shared helpers are CALLED (payload type and count), never re-implemented."""
    calls = {"calibrate": 0, "auroc": 0, "wilson": 0}
    real_cal, real_auroc, real_wilson = probe_monitor_trainer.calibrate, probe_monitor_metrics.auroc, \
        probe_monitor_metrics.wilson_interval

    def cal(*a, **k):
        calls["calibrate"] += 1
        assert isinstance(a[0], list) and "target_fpr" in k
        return real_cal(*a, **k)

    def au(*a):
        calls["auroc"] += 1
        return real_auroc(*a)

    def wi(*a, **k):
        calls["wilson"] += 1
        return real_wilson(*a, **k)

    monkeypatch.setattr(probe_monitor_trainer, "calibrate", cal)
    monkeypatch.setattr(probe_monitor_metrics, "auroc", au)
    monkeypatch.setattr(probe_monitor_metrics, "wilson_interval", wi)
    inputs = bf.build(16, candidates=["stack"])
    run(inputs)
    # stack: 1 calibration + 1 CV operating point + 3 best-single members; 2 test views + 2×3 members' own
    assert calls["calibrate"] >= 5 and calls["auroc"] >= 3 and calls["wilson"] >= 2 + 6


# ── rules ∪ candidates: 040's union calibration, by position (6.6) ──────────────────────────────


def with_rules(inputs, *, rule_share=0.004, veto_share=0.0, seed=0, flag="low"):
    """Rule verdicts by SPLIT POSITION for every view. On the calibration view the rules flag `rule_share` of the
    aligned negatives — the LOWEST-scoring ones (`flag="low"`), which the blend alone would never flag, so a union
    calibrated without them spends the budget twice — and veto `veto_share` of the highest-scoring ones."""
    from types import SimpleNamespace

    rng = np.random.default_rng(seed)
    views = [inputs.blend_fit, *[t["id"] for t in inputs.tests], inputs.calibration]
    rules = {}
    for v in views:
        size = max(int(a.positions.max()) for a in inputs.arrays[v].values()) + 1
        match, veto = np.zeros(size, bool), np.zeros(size, bool)
        aligned = engine.align(v, inputs.members, inputs.arrays[v], inputs.min_aligned_fraction)
        if v == inputs.calibration:
            order = np.argsort(aligned.scores[:, 0], kind="stable")
            k = int(round(rule_share * aligned.positions.size))
            match[aligned.positions[order[:k] if flag == "low" else order[-k:]]] = True
            n_veto = int(round(veto_share * aligned.positions.size))
            if n_veto:
                veto[aligned.positions[order[::-1][:n_veto]]] = True
        else:
            pos = aligned.positions[aligned.labels == 1]
            match[rng.choice(pos, size=pos.size // 5, replace=False)] = True     # rules see a fifth of positives
        rules[v] = SimpleNamespace(rule_match=match, rule_veto=veto, unknown=np.zeros(size, np.uint8))
    inputs.rules, inputs.has_rule_set = rules, True
    inputs.rules_identity = {"rule_set_digest": "d" * 64, "engine_digest": "e" * 64}
    return inputs


def rules_inputs(seed, names, **kwargs):
    inputs = bf.build(seed, candidates=["stack"], n_cal=2500, target_fpr=0.01)
    inputs.candidates = [cand.parse_name(n, len(inputs.members), True) for n in names]
    return with_rules(inputs, **kwargs)


def union_on_calibration(result, inputs, name):
    """The candidate's final verdicts on the calibration negatives, read back from the run's own rows."""
    index = next(c["index"] for c in result["candidates"] if c["name"] == name)
    rows = result["rows"][inputs.calibration]
    return np.asarray(rows["values"][index])[np.asarray(rows["labels"]) == 0] >= 0.5


def test_rules_or_stack_union_realises_at_most_target():
    """FR-10: rules flag 0.4% of the calibration negatives → the union `rules_or_stack` realises ≤ 1%. The rules'
    false positives are charged FIRST; the stack's bar is cut on what is left. Control: calibrate the stack on
    all negatives ignoring the rules → the union spends 1% + 0.4% → red."""
    inputs = rules_inputs(30, ["stack", "rules_or_stack"])
    result = run(inputs)
    entry = next(c for c in result["candidates"] if c["name"] == "rules_or_stack")
    cal = entry["calibration"]
    fires = union_on_calibration(result, inputs, "rules_or_stack")
    assert cal["source"] == "union_calibration" and cal["rule_false_positives"] == round(0.004 * fires.size)
    assert fires.mean() <= 0.01, (fires.mean(), cal)
    assert cal["realised_fpr"] == pytest.approx(fires.mean(), abs=1e-6)       # 040 rounds to 6 places
    # the stack's own bar is NOT the union's: alone it spends the whole budget, the union's leaves the rules room
    stack = next(c for c in result["candidates"] if c["name"] == "stack")["calibration"]
    assert cal["threshold"] > stack["threshold"]
    assert entry["produces"] == "verdict" and entry["params"]["rules"] == inputs.rules_identity
    assert entry["params"]["weights"] == next(c for c in result["candidates"] if c["name"] == "stack")["params"]["weights"]


def test_rules_or_verdict_kind_reports_the_unions_realised_rate():
    inputs = rules_inputs(31, ["any_of", "rules_or_any_of"])
    result = run(inputs)
    entry = next(c for c in result["candidates"] if c["name"] == "rules_or_any_of")
    fires = union_on_calibration(result, inputs, "rules_or_any_of")
    base = next(c for c in result["candidates"] if c["name"] == "any_of")["calibration"]["realised_fpr"]
    assert entry["calibration"]["source"] == "realised_union"
    assert entry["calibration"]["realised_fpr"] == pytest.approx(fires.mean())
    assert fires.mean() > base                                   # the low-scoring rule hits are added, never cut


def test_rule_verdicts_are_aligned_by_position():
    """⚠ FR-20: the verdicts are read at each aligned row's SPLIT POSITION. Control: index the verdict arrays by
    row number instead → the planted rule hits land on other rows → red."""
    inputs = rules_inputs(32, ["rules_or_stack"])
    result = run(inputs)
    cal = result["candidates"][0]["calibration"]
    rows = result["rows"][inputs.calibration]
    planted = inputs.rules[inputs.calibration].rule_match[np.asarray(rows["positions"])]
    assert cal["rule_false_positives"] == int(planted.sum()) > 0
    assert result["rules"][inputs.calibration]["rule_match"] == int(planted.sum())
    assert np.array_equal(np.asarray(rows["rule_match"]), planted)


def test_rules_spend_the_budget_recorded_not_cut():
    inputs = rules_inputs(33, ["rules_or_stack"], rule_share=0.02)
    inputs.criteria = [{"id": "c", "detector": "rules_or_blend", "subset": "all", "held_out_test": True,
                        "min_recall": 0.1, "target_fpr": 0.01}]
    result = run(inputs)
    entry = result["candidates"][0]
    assert entry["calibration"]["refused"] == "rules_spend_budget" and entry["calibration"]["threshold"] is None
    assert entry["criteria"][0]["verdict"] == "not_assessable"
    assert {"code": "rules_spend_budget"} in entry["criteria"][0]["reasons"]


def test_union_calibration_calls_040s_union_cut(monkeypatch):
    """The union is 040's cut, never a copy: spy `rules_evaluation.union_cut` (payload and count)."""
    from src.services import rules_evaluation

    calls = []
    real = rules_evaluation.union_cut

    def spy(scores, match, veto, target_fpr):
        calls.append((np.asarray(scores).copy(), np.asarray(match).copy(), target_fpr))
        return real(scores, match, veto, target_fpr)

    monkeypatch.setattr(rules_evaluation, "union_cut", spy)
    inputs = rules_inputs(34, ["stack", "rules_or_stack"])
    result = run(inputs)
    assert len(calls) == 2                                       # in-fold (CV) and the calibration cut
    scores, match, target = calls[-1]
    rows = result["rows"][inputs.calibration]
    negatives = np.asarray(rows["labels"]) == 0
    stack_index = next(c["index"] for c in result["candidates"] if c["name"] == "stack")
    assert np.array_equal(scores, np.asarray(rows["values"][stack_index])[negatives])   # the stack's own values
    assert int(match.sum()) == result["candidates"][1]["calibration"]["rule_false_positives"] and target == 0.01


def test_rules_or_criteria_judged_on_the_union_and_blend_alone_on_the_base():
    inputs = rules_inputs(35, ["stack", "rules_or_stack"])
    inputs.criteria = [
        {"id": "u", "detector": "rules_or_blend", "subset": "all", "held_out_test": True, "min_recall": 0.1,
         "target_fpr": 0.01},
        {"id": "b", "detector": "blend_alone", "subset": "rule_invisible_positives", "held_out_test": True,
         "min_recall": 0.1, "target_fpr": 0.01},
    ]
    result = run(inputs)
    by = {c["name"]: c for c in result["candidates"]}
    assert [c["id"] for c in by["stack"]["criteria"]] == ["b"]
    assert [c["id"] for c in by["rules_or_stack"]["criteria"]] == ["u"]
    ood = inputs.tests[0]["id"]
    rows = result["rows"][ood]
    labels, seen = np.asarray(rows["labels"]), np.asarray(rows["rule_match"])
    # rule-invisible positives: the positives the rules do NOT see (040 FR-5), counted on the aligned rows
    assert by["stack"]["criteria"][0]["figures"]["positives"] == int(((labels == 1) & ~seen).sum())
    union = np.asarray(rows["values"][by["rules_or_stack"]["index"]]) >= 0.5
    assert by["rules_or_stack"]["criteria"][0]["figures"]["flagged_positives"] == int((union & (labels == 1)).sum())
    assert np.all(union[seen]), "a rule match is always flagged by the union"


def test_rule_subset_without_a_rule_set_reads_all_positives_and_visible_not_assessable():
    inputs = bf.build(36, candidates=["stack"])
    inputs.criteria = [{"id": c, "detector": "blend_alone", "subset": s, "held_out_test": True, "min_recall": 0.1,
                        "target_fpr": 0.05} for c, s in (("i", "rule_invisible_positives"), ("v", "rule_visible_positives"),
                                                         ("a", "all"))]
    by = {c["id"]: c for c in run(inputs)["candidates"][0]["criteria"]}
    assert by["i"]["figures"] == by["a"]["figures"] and by["i"]["notes"] == ["no_rule_set"]
    assert by["v"]["verdict"] == "not_assessable" and {"code": "no_rule_set"} in by["v"]["reasons"]


def test_rule_subset_with_a_rule_set_but_no_verdicts_is_not_assessable():
    inputs = bf.build(37, candidates=["stack"])
    inputs.has_rule_set = True
    inputs.criteria = [{"id": "i", "detector": "blend_alone", "subset": "rule_invisible_positives",
                        "held_out_test": True, "min_recall": 0.1, "target_fpr": 0.05}]
    judged = run(inputs)["candidates"][0]["criteria"][0]
    assert judged["verdict"] == "not_assessable"
    assert judged["reasons"][0]["code"] == "missing_evaluation" and judged["reasons"][0]["missing"] == "rules_evaluation"


def test_unknown_verdicts_are_no_match_and_counted():
    inputs = rules_inputs(38, ["rules_or_any_of"])
    inputs.rule_unknown_share_max = 0.05
    cal = inputs.rules[inputs.calibration]
    cal.unknown = np.where(cal.rule_match, 2, 0).astype(np.uint8)          # every rule hit is pending operator code
    result = run(inputs)
    assert result["candidates"][0]["calibration"]["rule_false_positives"] == 0
    assert result["rules"][inputs.calibration]["pending"] > 0


def test_rules_candidate_without_verdicts_refused():
    inputs = bf.build(39, candidates=["stack"])
    inputs.candidates = [cand.parse_name("rules_or_stack", len(inputs.members), True)]
    with pytest.raises(engine.EngineRefused):
        run(inputs)


def test_a_verdict_union_over_the_target_is_not_assessable():
    """040-T27 for a blend: a rules ∪ verdict kind is not cut, so its union may realise more than the target on the
    calibration negatives; a criterion is then `union_over_budget`, never met. Control: drop the check → red."""
    inputs = rules_inputs(40, ["rules_or_any_of"])
    inputs.criteria = [{"id": "u", "detector": "rules_or_blend", "subset": "all", "held_out_test": True,
                        "min_recall": 0.0, "target_fpr": 0.01}]
    entry = run(inputs)["candidates"][0]
    assert entry["calibration"]["realised_fpr"] > 0.01
    judged = entry["criteria"][0]
    assert judged["verdict"] == "not_assessable"
    assert [r["code"] for r in judged["reasons"]] == ["union_over_budget"]
