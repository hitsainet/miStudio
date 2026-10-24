"""The spot-check report (038 FR-10, D32): numbers equal hand-computed values on a 7/2/1/10 fixture.

Hand computation, n = 20 (tp 7, fp 2, fn 1, tn 10): agreement = 17/20 = 0.85; probe fires 9/20 = 0.45,
judge positive 8/20 = 0.40; chance agreement = 0.45·0.40 + 0.55·0.60 = 0.51; kappa = (0.85 − 0.51)/0.49.
Controls: C-FPFN (swap fp/fn in `agreement_table`), C-STATEMENT (statement removed).
"""
from types import SimpleNamespace

import pytest

from src.services import probe_spot_check as spot
from src.services.probe_monitor_metrics import agreement_ci


def _item(fired, positive, stratum="fired", **over):
    base = dict(probe_id="pm_1", millm_probe_id="pr_1", probe_fired=fired, judge_positive=positive,
                not_recorded=False, parse_failed=False, judge_saw="same_window", stratum=stratum,
                threshold_revision=1)
    base.update(over)
    return SimpleNamespace(**base)


def _fixture():
    return ([_item(True, True)] * 7 + [_item(True, False)] * 2 + [_item(False, True, "not_fired")]
            + [_item(False, False, "not_fired")] * 10)


def test_hand_computed_7_2_1_10():
    report = spot.report(_fixture(), None, seed=42)
    overall = report["probes"]["pm_1"]["overall"]
    assert (overall["n"], overall["tp"], overall["fp"], overall["fn"], overall["tn"]) == (20, 7, 2, 1, 10)
    assert overall["agreement"] == pytest.approx(0.85)
    assert overall["kappa"] == pytest.approx((0.85 - 0.51) / 0.49)
    pairs = [(i.probe_fired, i.judge_positive) for i in _fixture()]
    assert overall["agreement_ci"] == agreement_ci(pairs, resamples=1000, seed=42), "reproduced from the seed"
    assert overall["agreement_ci"]["low"] <= 0.85 <= overall["agreement_ci"]["high"]
    assert report["probes"]["pm_1"]["strata"]["fired"]["n"] == 9
    assert report["probes"]["pm_1"]["disagreements"] == 3


def test_statement_is_present():
    report = spot.report(_fixture(), None, seed=1)
    assert report["statement"] == (
        "The judge is a second opinion, not ground truth. Disagreement means one of the two is wrong, not which."
    )
    assert "sampling design" in report["overall_note"]


def test_parse_failures_and_unrecorded_turns_are_counted_not_tabled():
    items = _fixture() + [_item(True, None, parse_failed=True), _item(False, None, not_recorded=True)]
    probe = spot.report(items, None, seed=1)["probes"]["pm_1"]
    assert probe["overall"]["n"] == 20 and probe["n_parse_failed"] == 1 and probe["n_not_recorded"] == 1


def test_newest_turn_probes_are_reported_apart():
    items = _fixture() + [_item(True, True, probe_id="pm_2", judge_saw="newest_turn")]
    report = spot.report(items, None, seed=1)
    assert set(report["probes"]) == {"pm_1"} and set(report["newest_turn_probes"]) == {"pm_2"}
