"""The spread between a probe's shipped threshold and each set's own is reported, not left implicit.

⚠ THIS IS NOT A FIX AND IS NOT PRESENTED AS ONE. A linear probe ranks well within a distribution
while its absolute scale shifts between them; there is no threshold that serves every
distribution, and calibrating on the one being served is the correct response. What was missing
is that a reader could not SEE the spread without assembling it by hand from five ROC curves.

Measured on the first shipped probe (Llama-3.1-8B, L11, `mean`, calibrated on chat at 1%):

    set                       own 1% threshold    at the shipped 11.91
    mental_health_balanced              18.77     recall 0.748, FPR 0.426
    anthropic_hh_balanced                5.10     recall 0.219
    mt_balanced                          0.08     recall 0.020
    aya_redteaming_balanced             -1.80     recall 0.013
    toolace_balanced                    -5.44     recall 0.000  (max score 1.17)

A 24-point span, and on `toolace` the shipped threshold is above every score the set produces, so
recall there is exactly zero however well the probe ranks — it scores AUROC 0.858 on that set.
The probe reported one honest AUROC and one honest threshold; read together they suggest one
detector with one operating point, which is incomplete in a way only visible per set.
"""

from __future__ import annotations

import pytest

from src.services.probe_monitor_metrics import threshold_transfer


def _roc(points):
    """(threshold, tpr, fpr) triples as the evaluator stores them."""
    return [{"threshold": t, "tpr": tpr, "fpr": fpr} for t, tpr, fpr in points]


def _evaluation(name, *, own_1pct, roc, auroc=0.9, scored=True):
    return {
        "metrics": {
            "name": name,
            "scored": scored,
            "auroc": auroc,
            "operating_points": [
                {"target_fpr": 0.01, "threshold": own_1pct, "recall": 0.4, "realised_fpr": 0.01}
            ],
            "roc": _roc(roc),
        }
    }


class TestItReportsWhatTheShippedThresholdDoes:
    def test_recall_and_fpr_at_the_shipped_threshold(self):
        ev = _evaluation("setA", own_1pct=5.0, roc=[(20.0, 0.1, 0.0), (10.0, 0.5, 0.02), (1.0, 0.9, 0.3)])
        out = threshold_transfer(10.0, "calibration_set", [ev])

        entry = out["per_set"][0]
        assert entry["name"] == "setA"
        assert entry["recall_at_shipped"] == pytest.approx(0.5)
        assert entry["fpr_at_shipped"] == pytest.approx(0.02)
        assert entry["unreachable"] is False

    def test_it_uses_the_nearest_reachable_point_not_the_extreme(self):
        """Picking the highest threshold at-or-above would report the recall of a far stricter
        cut than the one actually shipped."""
        ev = _evaluation("setA", own_1pct=1.0, roc=[(50.0, 0.01, 0.0), (11.0, 0.6, 0.05), (2.0, 0.95, 0.4)])
        out = threshold_transfer(10.0, "calibration_set", [ev])
        assert out["per_set"][0]["recall_at_shipped"] == pytest.approx(0.6)

    def test_each_sets_OWN_threshold_is_carried_beside_it(self):
        out = threshold_transfer(
            10.0, "calibration_set",
            [_evaluation("a", own_1pct=18.77, roc=[(20.0, 0.2, 0.0)]),
             _evaluation("b", own_1pct=-5.44, roc=[(1.0, 0.9, 0.5)])],
        )
        owns = {e["name"]: e["own_threshold_at_1pct"] for e in out["per_set"]}
        assert owns == {"a": 18.77, "b": -5.44}


class TestTheUnreachableCase:
    """The sharp one: the shipped threshold is above every score the set produces."""

    def test_a_set_whose_scores_never_reach_the_threshold(self):
        # toolace's real shape: AUROC 0.858, and a maximum score of 1.17.
        ev = _evaluation("toolace", own_1pct=-5.44, auroc=0.858,
                         roc=[(1.17, 0.02, 0.0), (0.0, 0.5, 0.2), (-5.44, 0.9, 0.6)])
        out = threshold_transfer(11.91, "calibration_set", [ev])

        entry = out["per_set"][0]
        assert entry["unreachable"] is True
        assert entry["recall_at_shipped"] == 0.0
        assert entry["max_score"] == pytest.approx(1.17)
        assert out["unreachable_sets"] == ["toolace"]

    def test_the_caution_names_the_set_and_says_what_it_costs(self):
        """A flag nobody can act on is decoration. The wording has to say that ranking quality
        does not rescue it."""
        ev = _evaluation("toolace", own_1pct=-5.44, roc=[(1.17, 0.02, 0.0)])
        out = threshold_transfer(11.91, "calibration_set", [ev])
        assert "toolace" in out["caution"]
        assert "zero" in out["caution"]

    def test_a_reachable_set_is_NOT_flagged(self):
        """⚠ Specificity. If every set were flagged the flag would carry nothing."""
        ev = _evaluation("ok", own_1pct=5.0, roc=[(20.0, 0.3, 0.0), (10.0, 0.6, 0.05)])
        out = threshold_transfer(10.0, "calibration_set", [ev])
        assert out["per_set"][0]["unreachable"] is False
        assert out["unreachable_sets"] == []
        assert out["caution"] is None


class TestTheSpread:
    def test_it_measures_the_span_of_the_sets_own_thresholds(self):
        out = threshold_transfer(
            11.91, "calibration_set",
            [_evaluation("a", own_1pct=18.77, roc=[(20.0, 0.5, 0.1)]),
             _evaluation("b", own_1pct=5.10, roc=[(20.0, 0.5, 0.1)]),
             _evaluation("c", own_1pct=-5.44, roc=[(20.0, 0.5, 0.1)])],
        )
        assert out["own_threshold_spread"] == pytest.approx(18.77 - (-5.44))

    def test_a_single_set_has_no_spread(self):
        """One set cannot disagree with itself; None is the honest answer, not 0.0."""
        out = threshold_transfer(10.0, "calibration_set",
                                 [_evaluation("a", own_1pct=5.0, roc=[(20.0, 0.5, 0.1)])])
        assert out["own_threshold_spread"] is None

    def test_a_wide_spread_cautions_even_when_every_set_is_reachable(self):
        out = threshold_transfer(
            5.0, "calibration_set",
            [_evaluation("a", own_1pct=18.0, roc=[(20.0, 0.5, 0.1)]),
             _evaluation("b", own_1pct=-5.0, roc=[(20.0, 0.5, 0.1)])],
        )
        assert out["caution"] is not None
        assert "does not transfer" in out["caution"]


class TestItDegradesHonestly:
    def test_an_unscored_set_is_skipped_rather_than_counted_as_zero(self):
        """A refused evaluation contributes no threshold. Treating it as 0.0 would invent a
        data point and widen the spread with a number nobody measured."""
        out = threshold_transfer(
            10.0, "calibration_set",
            [_evaluation("good", own_1pct=5.0, roc=[(20.0, 0.5, 0.1)]),
             {"metrics": {"name": "refused", "scored": False, "reason": "too few of a class"}}],
        )
        assert [e["name"] for e in out["per_set"]] == ["good"]

    def test_a_probe_with_no_threshold_still_reports_the_sets(self):
        """A probe that ranks without deciding has no shipped threshold; the per-set thresholds
        are still worth showing."""
        out = threshold_transfer(None, None,
                                 [_evaluation("a", own_1pct=5.0, roc=[(20.0, 0.5, 0.1)])])
        assert out["shipped_threshold"] is None
        assert out["per_set"][0]["own_threshold_at_1pct"] == 5.0
        assert "recall_at_shipped" not in out["per_set"][0]

    def test_no_evaluations_gives_an_empty_report_not_an_error(self):
        out = threshold_transfer(10.0, "calibration_set", [])
        assert out["per_set"] == []
        assert out["own_threshold_spread"] is None
        assert out["caution"] is None

    def test_the_source_is_carried_through(self):
        """`calibration_set` and `validation_negatives` mean very different things about who the
        threshold was fitted for, so it travels beside the number."""
        out = threshold_transfer(10.0, "validation_negatives", [])
        assert out["threshold_source"] == "validation_negatives"


class TestTheReportSurfacesIt:
    def test_the_endpoint_assembles_it(self):
        """Wiring: a report nobody serves is the shape this whole arc keeps hitting."""
        import ast
        import inspect

        from src.api.v1.endpoints import probe_monitors

        tree = ast.parse(inspect.getsource(probe_monitors.get_probe_report))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "threshold_transfer" in called, (
            f"get_probe_report does not call threshold_transfer; calls seen: {sorted(called)}"
        )

    def test_the_schema_carries_the_field(self):
        from src.schemas.probe_monitor import ProbeReport

        assert "threshold_transfer" in ProbeReport.model_fields
