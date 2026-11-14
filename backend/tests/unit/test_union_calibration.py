"""The union spends the FPR budget as a union (Feature 040 S3; FR-36; FTDD §6.5; FTASKS 22.1 – 22.3).

Pure, on hand-built arrays (the route-level check is `test_rules_evaluation_task.py::test_the_union_bar_on_the_
calibration_negatives`). The FR-36 fixture: 1,000 negatives, target 1% (budget 10), rules flag four (0.4%, low
scores), three masked negatives carry the highest scores (vetoed: removed first). The probe gets six (0.6%) and
the union realises exactly 1%. Positions are NON-CONTIGUOUS (odd split rows; every even row matches), so
aligning by index instead of position moves the figures.

Controls (FTASKS 22.3): calibrate on all negatives ignoring rules → the union exceeds the target; align by index →
the rule counts move; write the union bar onto the probe → nothing here writes (the function is pure, and the
task test reads the probe row unchanged).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.services import rules_evaluation as re_
from src.services.detector_next import DetectorRefused
from tests.support.next_calls import assert_next_calls_are_live

N = 1000


def _world(match=4, masked=3):
    """Calibration negatives on odd split rows; verdict arrays over the whole split (even rows match)."""
    scores = np.linspace(0.99, 0.01, N)
    positions = np.arange(1, 2 * N, 2)
    split = 2 * N
    rule_match = np.zeros(split, dtype=bool)
    rule_veto = np.zeros(split, dtype=bool)
    rule_match[0::2] = True                      # decoys on the even rows
    rule_match[positions[N - match:]] = True     # the four lowest-scoring negatives match
    rule_veto[positions[:masked]] = True         # the three highest are masked
    verdicts = SimpleNamespace(rule_match=rule_match, rule_veto=rule_veto, unknown=np.zeros(split, dtype=np.uint8))
    probe = SimpleNamespace(id="pm_x", length_bands=None, threshold=0.5, window_decisions=None)
    return probe, scores, positions, verdicts


def test_rules_spend_first_and_the_probe_gets_the_rest():
    probe, scores, positions, verdicts = _world()
    out = re_.union_calibration(probe, "all", scores, positions, None, verdicts, 0.01)
    g = out["global"]
    assert (g["budget"], g["rule_false_positives"], g["vetoed"], g["probe_false_positives_allowed"]) == (10, 4, 3, 6)
    assert g["probe_false_positives"] == 6
    assert g["realised_union_fpr"] == pytest.approx(0.01) and g["rule_share"] == pytest.approx(0.004)
    # the bar is the 6th highest REMAINING score: the masked three are skipped
    remaining = scores[3:N - 4]
    assert g["threshold"] == pytest.approx(remaining[5])
    assert out["aligned_by"] == "position" and out["windows"] == "not_re_cut"
    assert_next_calls_are_live(out["next"])


def test_ignoring_the_rules_overspends_the_budget():
    """What the control would produce: the probe alone cut at 1% leaves the union over the target."""
    probe, scores, positions, verdicts = _world()
    alone_bar = np.sort(scores)[::-1][9]
    match = verdicts.rule_match[positions]
    veto = verdicts.rule_veto[positions] & ~match
    union = match | ((scores >= alone_bar) & ~veto)
    assert union.sum() / N > 0.01
    out = re_.union_calibration(probe, "all", scores, positions, None, verdicts, 0.01)
    assert out["global"]["realised_union_fpr"] <= 0.01


def test_alignment_is_by_position_not_index():
    probe, scores, positions, verdicts = _world()
    by_position = re_.union_calibration(probe, "all", scores, positions, None, verdicts, 0.01)["global"]
    with pytest.raises(DetectorRefused):  # by index every even (decoy) row reads as a rule match
        re_.union_calibration(probe, "all", scores, np.arange(N), None, verdicts, 0.01)
    assert by_position["rule_false_positives"] == 4


def test_rules_that_spend_the_budget_are_refused_naming_both():
    probe, scores, positions, verdicts = _world(match=10)
    with pytest.raises(DetectorRefused) as caught:
        re_.union_calibration(probe, "all", scores, positions, None, verdicts, 0.01)
    assert caught.value.code == "rules_spend_budget"
    assert caught.value.details == {"rule_fpr": 0.01, "target_fpr": 0.01}


def test_a_banded_probe_cuts_per_band_and_a_thin_band_falls_back():
    probe, scores, positions, verdicts = _world()
    probe.length_bands = [{"min_tokens": 0, "max_tokens": 49, "threshold": 0.6},
                          {"min_tokens": 50, "max_tokens": None, "threshold": 0.4}]
    lengths = np.where(np.arange(N) < 950, 10, 80)  # 950 short, 50 long (a thin band at 1%: needs 100)
    out = re_.union_calibration(probe, "all", scores, positions, lengths, verdicts, 0.01)
    short, long_ = out["bands"]
    assert short["budget"] == 9 and short["rule_false_positives"] == 0 and short["probe_false_positives_allowed"] == 9
    assert long_["fallback"] == "global_union_bar" and long_["reason"] == "thin_band"
    assert long_["threshold"] == out["global"]["threshold"]


def test_the_union_at_the_recorded_decision_is_reported():
    probe, scores, positions, verdicts = _world()
    out = re_.union_calibration(probe, "all", scores, positions, None, verdicts, 0.01)["at_recorded_decision"]
    fires = scores >= 0.5
    match = verdicts.rule_match[positions]
    veto = verdicts.rule_veto[positions] & ~match
    assert out["realised_union_fpr"] == pytest.approx((match | (fires & ~veto)).sum() / N, abs=1e-6)
    assert out["over_budget"] is True


def test_an_unknown_verdict_is_no_match_and_no_veto():
    probe, scores, positions, verdicts = _world()
    verdicts.unknown[positions[:3]] = 2   # the three masked rows: their veto is unknown
    g = re_.union_calibration(probe, "all", scores, positions, None, verdicts, 0.01)["global"]
    assert g["vetoed"] == 0 and g["probe_false_positives_allowed"] == 6


@pytest.mark.parametrize("allowed,expected_rank", [(1, 0), (6, 5), (0, None)])
def test_the_cut_is_the_order_statistic(allowed, expected_rank):
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    bar = re_._cut(scores, allowed)
    assert bar == (None if expected_rank is None else pytest.approx(np.sort(scores)[::-1][expected_rank]))
