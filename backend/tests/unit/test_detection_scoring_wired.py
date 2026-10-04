"""The trial must actually SCORE, not just label.

labeling_detection_scorer.py and detection_metrics.py were fully implemented and
unit-tested with ZERO production callers — nothing in src/ imported either one.
Feature 30's headline capability, ranking templates by an automated detection
score, did not exist for any caller. The trial produced labels and a human read
them, which is the method the trial was built to replace.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.services.labeling_trial_service import LabelingTrialService


def _svc():
    with patch.object(LabelingTrialService, "__init__", lambda self: None):
        s = LabelingTrialService()
    s.db = MagicMock()
    return s


def _feature(fid, ext="ext1"):
    f = MagicMock()
    f.id = fid
    f.extraction_job_id = ext
    return f


def _positives(n=6):
    return [{"prime_token": "▁FDA", "sample_index": i, "max_activation": 1.0 - i * 0.1}
            for i in range(n)]


def _ok(fid, specific, description="d"):
    return {"feature_id": fid, "status": "ok", "specific": specific,
            "description": description, "neuron_index": 0}


class TestTheScorerIsReached:
    def test_score_panel_is_actually_called(self):
        svc = _svc()
        run = MagicMock(id="ltr_x", panel_id="pnl_x")
        feats = [_feature("f1"), _feature("f2")]
        results = [_ok("f1", "fda_regulatory_actions"), _ok("f2", "legal_proceedings")]
        ex = {"f1": _positives(), "f2": _positives()}

        with patch("src.services.labeling_detection_scorer.sample_negatives",
                   return_value=([{"x": 1}], [{"y": 1}])), \
             patch("src.services.labeling_detection_scorer.assemble_items",
                   return_value=[{"text": "t", "truth": 1}]), \
             patch("src.services.labeling_detection_scorer.negative_ceiling",
                   return_value=0.5), \
             patch("src.services.labeling_detection_scorer.run_gate",
                   return_value={"passed": True}) as gate, \
             patch("src.services.labeling_detection_scorer.score_panel",
                   return_value={"panel_ba": 0.81}) as sp, \
             patch.object(LabelingTrialService, "_build_judge",
                          return_value=lambda p: "[1]"):
            out = svc._score_detection(run=run, features=feats, results=results,
                                       examples_by_feature=ex, labeler=MagicMock())

        assert sp.call_count == 1, (
            "score_panel was never called; the trial produces labels and no score"
        )
        assert gate.call_count == 1, "the judge sanity gate was skipped"
        assert out["panel_ba"] == 0.81

    def test_the_gate_result_is_passed_into_score_panel(self):
        """score_panel refuses an ABSENT gate; passing None would silently
        disable the whole sanity check."""
        svc = _svc()
        with patch("src.services.labeling_detection_scorer.sample_negatives",
                   return_value=([{"x": 1}], [])), \
             patch("src.services.labeling_detection_scorer.assemble_items",
                   return_value=[{"text": "t"}]), \
             patch("src.services.labeling_detection_scorer.negative_ceiling",
                   return_value=None), \
             patch("src.services.labeling_detection_scorer.run_gate",
                   return_value={"passed": True, "marker": "GATE"}), \
             patch("src.services.labeling_detection_scorer.score_panel",
                   return_value={}) as sp, \
             patch.object(LabelingTrialService, "_build_judge",
                          return_value=lambda p: "[1]"):
            svc._score_detection(
                run=MagicMock(id="r", panel_id="p"), features=[_feature("f1")],
                results=[_ok("f1", "a_label")],
                examples_by_feature={"f1": _positives()}, labeler=MagicMock())
        assert sp.call_args.kwargs["gate"]["marker"] == "GATE"


class TestCoverageIsPartOfTheResult:
    """A template that refuses everything must not look excellent.

    The substitution-test candidate labelled 1 of 31 features on a real panel.
    Scored on that one feature alone it would have reported a near-perfect
    number against the baseline's eighteen.
    """

    def test_refusals_are_excluded_and_counted(self):
        svc = _svc()
        feats = [_feature(f"f{i}") for i in range(3)]
        results = [
            _ok("f0", "fda_regulatory_actions"),
            _ok("f1", "uninterpretable"),
            _ok("f2", "noise"),
        ]
        ex = {f"f{i}": _positives() for i in range(3)}

        captured = {}

        def _sp(features, judge, **kw):
            captured["n"] = len(features)
            return {}

        with patch("src.services.labeling_detection_scorer.sample_negatives",
                   return_value=([{"x": 1}], [])), \
             patch("src.services.labeling_detection_scorer.assemble_items",
                   return_value=[{"text": "t"}]), \
             patch("src.services.labeling_detection_scorer.negative_ceiling",
                   return_value=None), \
             patch("src.services.labeling_detection_scorer.run_gate",
                   return_value={"passed": True}), \
             patch("src.services.labeling_detection_scorer.score_panel", _sp), \
             patch.object(LabelingTrialService, "_build_judge",
                          return_value=lambda p: "[1]"):
            out = svc._score_detection(run=MagicMock(id="r", panel_id="p"),
                                       features=feats, results=results,
                                       examples_by_feature=ex, labeler=MagicMock())

        assert captured["n"] == 1, "a refusal was scored as though it were a label"
        assert out["coverage"] == {"scored": 1, "skipped": 2, "panel_size": 3}

    def test_a_panel_of_only_refusals_reports_not_scored(self):
        svc = _svc()
        out = svc._score_detection(
            run=MagicMock(id="r", panel_id="p"), features=[_feature("f0")],
            results=[_ok("f0", "uninterpretable")],
            examples_by_feature={"f0": _positives()}, labeler=MagicMock())
        assert out["scored"] is False
        assert out["coverage"]["scored"] == 0


class TestScoringNeverDiscardsLabels:
    def test_a_scoring_failure_is_recorded_not_raised(self):
        """Labels cost real GPU time; a broken judge must not throw them away."""
        svc = _svc()
        with patch("src.services.labeling_detection_scorer.sample_negatives",
                   side_effect=RuntimeError("judge exploded")):
            out = svc._score_detection(
                run=MagicMock(id="r", panel_id="p"), features=[_feature("f0")],
                results=[_ok("f0", "a_label")],
                examples_by_feature={"f0": _positives()}, labeler=MagicMock())
        assert out["scored"] is False
        assert "judge exploded" in out["reason"]

    def test_no_score_and_a_bad_score_are_distinguishable(self):
        svc = _svc()
        with patch("src.services.labeling_detection_scorer.sample_negatives",
                   side_effect=RuntimeError("boom")):
            out = svc._score_detection(
                run=MagicMock(id="r", panel_id="p"), features=[_feature("f0")],
                results=[_ok("f0", "a_label")],
                examples_by_feature={"f0": _positives()}, labeler=MagicMock())
        assert "scored" in out and out["scored"] is False
        assert "panel_ba" not in out, (
            "a failed scoring run produced a number; a broken judge would read "
            "as a bad template"
        )
