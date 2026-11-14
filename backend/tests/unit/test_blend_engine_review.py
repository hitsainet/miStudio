"""Blend engine — controls Review R1 found unpinned (Feature 041; `0xcc/reviews/041_implementation_controls_2026-10-24.md`
§ Review R1). Each test names the mutation that survived the S1 – S3 suite and now turns it red.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.services.probe_blend import candidates as cand
from src.services.probe_blend import engine
from tests.support import blend_fixtures as bf


# ── the in-fold figures are out-of-fold: standardisation AND stack refitted per training fold (FR-9, FR-11) ──


def test_cv_refits_standardisation_and_stack_on_each_training_fold_only(monkeypatch):
    """R1-01 / R1-02: fitting the in-fold standardisation or the in-fold stack on ALL `blend_fit` rows (a leak the
    pure-noise AUROC bound cannot see) changes the row count of the calls below. The first call of each is the
    whole-view fit the report records; then one per fold, on exactly that fold's training rows."""
    seen = {"standardise": [], "stack": []}
    real_std, real_stack = engine.standardise_fit, engine.fit_stack

    def spy_std(X, members):
        seen["standardise"].append(np.asarray(X).shape[0])
        return real_std(X, members)

    def spy_stack(Z, y, seed):
        seen["stack"].append(np.asarray(Z).shape[0])
        return real_stack(Z, y, seed)

    monkeypatch.setattr(engine, "standardise_fit", spy_std)
    monkeypatch.setattr(engine, "fit_stack", spy_stack)
    out = engine.run(bf.build(15, candidates=["stack"]))
    n = out["alignment"]["pmd_fit"]["rows"]
    held = [f["positives"] + f["negatives"] for f in out["folds"]["counts"]["folds"]]
    assert sum(held) == n and len(held) == 5
    train_sizes = [n - h for h in held]
    assert seen["standardise"] == [n] + train_sizes
    assert seen["stack"] == [n] + train_sizes


def test_in_fold_recall_counts_a_positive_at_the_threshold():
    """R1-14: `>=` at the out-of-fold operating point (the bar fires AT its value, as `fires_of` does)."""
    negatives = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9])
    positives = np.array([0.8, 0.85, 0.5, 0.9])
    at = engine._recall_at(positives, negatives, 0.2)              # 2 allowed → the 2nd highest negative, 0.8
    assert at["threshold"] == pytest.approx(0.8)
    assert at["recall"] == pytest.approx(3 / 4)


# ── folds (FR-11) ────────────────────────────────────────────────────────────────────────────


def test_grouped_folds_with_a_thin_class_read_few_rows():
    """R1-10: enough groups, too few positives per held-out fold → `few_rows` (never a refusal)."""
    rng = np.random.default_rng(4)
    groups = rng.integers(0, 40, 500)
    y = np.zeros(500, dtype=int)
    y[rng.choice(500, 12, replace=False)] = 1
    fold_ids, flags, counts = engine.folds(groups, y, 5, 3, 5)
    assert "few_rows" in flags
    assert any(f["positives"] < 5 for f in counts["folds"])


# ── verdict candidates' realised FPR on the calibration negatives (FR-10) ────────────────────


def test_verdict_realised_fpr_counts_n_fired_at_k():
    """R1-07: a verdict candidate fires at `n_fired >= k`; its realised FPR on the calibration negatives counts
    exactly those rows (hand value)."""
    n_fired = np.array([0, 1, 2, 1, 0, 2, 3, 0], dtype=np.float64)
    labels = np.zeros(8, dtype=np.int8)
    two = engine.calibrate_candidate(cand.parse_name("k_of_n:2", 4, False), n_fired, labels, 0.05, 100)
    assert two["realised_fpr"] == pytest.approx(3 / 8) and two["k"] == 2 and two["threshold"] is None
    one = engine.calibrate_candidate(cand.parse_name("any_of", 4, False), n_fired, labels, 0.05, 100)
    assert one["realised_fpr"] == pytest.approx(5 / 8)


# ── a blend criterion: its test views and its subset (FR-12, FR-20) ──────────────────────────


def _world():
    tests = [{"id": "h", "held_out": True}, {"id": "i", "held_out": False}]
    values = {"h": np.array([0.9, 0.1, 0.8, 0.2, 0.95]), "i": np.array([0.9, 0.2])}
    aligned = {"h": SimpleNamespace(positions=np.array([0, 2, 4, 6, 8]), labels=np.array([1, 1, 1, 0, 1])),
               "i": SimpleNamespace(positions=np.array([1, 3]), labels=np.array([1, 1]))}
    codes = np.zeros(9, dtype=np.int32)
    codes[[2, 6, 8]] = 1                      # positions 0, 4 → "x"; 2, 6, 8 → "y"
    columns = {"h": {"src": (codes, ["x", "y"])}}
    calibration = {"threshold": 0.5, "target_fpr": 0.05}
    return tests, values, aligned, columns, calibration


def test_blend_criterion_reads_only_its_subset_by_position():
    """R1-05: the subset filter is applied, by split position — x's positives are positions 0 and 4 (both fire)."""
    tests, values, aligned, columns, calibration = _world()
    criterion = {"id": "c", "held_out_test": True, "subset": {"column": "src", "value": "x"}, "min_recall": 0.9}
    out = engine.blend_criterion_verdict(criterion, cand.parse_name("max", 3, False), tests, columns, values,
                                         aligned, calibration)
    assert out["figures"]["positives"] == 2 and out["figures"]["flagged_positives"] == 2
    assert out["verdict"] == "met" and out["test_view_ids"] == ["h"]
    whole = engine.blend_criterion_verdict({**criterion, "subset": "all"}, cand.parse_name("max", 3, False), tests,
                                           columns, values, aligned, calibration)
    assert (whole["figures"]["positives"], whole["figures"]["flagged_positives"]) == (4, 3)
    assert whole["verdict"] == "not_met"


def test_blend_criterion_reads_only_the_views_its_held_out_flag_names():
    """R1-06: `held_out_test` picks the out-of-distribution views (true) or the in-distribution ones (false)."""
    tests, values, aligned, columns, calibration = _world()
    ind = engine.blend_criterion_verdict({"id": "c", "held_out_test": False, "subset": "all", "min_recall": 0.6},
                                         cand.parse_name("max", 3, False), tests, columns, values, aligned, calibration)
    assert ind["test_view_ids"] == ["i"]
    assert (ind["figures"]["positives"], ind["figures"]["flagged_positives"]) == (2, 1)
    assert ind["verdict"] == "not_met"
