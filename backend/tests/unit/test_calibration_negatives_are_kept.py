"""The negative scores a threshold was cut from survive the run.

⚠ THEY DID NOT, AND THE ASYMMETRY WAS INVISIBLE BECAUSE NOTHING FAILED. The calibrating stage
computed the negatives, handed them to `calibrate()`, and dropped them. The evaluation stage has
always saved its score arrays. So the threshold was correct and simply unrepeatable: answering
"what would a 5% false-positive budget look like on our traffic" required a fresh ~70-minute run
that re-captured activations in order to recompute a percentile over numbers already computed
once.

A threshold is the `(1 - target_fpr)` quantile of exactly these scores. Keeping them turns the
operating point from a training-time decision into a dial — which matters because the operating
point is the part most likely to need changing after watching real traffic, and the part least
deserving of a GPU run.

Recorded 2026-09-29 after the first shipped probe caught 0.200 at its threshold while ranking at
0.8841 AUROC: the ranking was good and the cut was in the wrong place, and moving it cost an hour.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.services.probe_monitor_run import persist_calibration_negatives


class TestTheScoresAreWritten:
    def test_it_writes_a_readable_array(self, tmp_path):
        scores = [1.5, -2.25, 9.0, 0.0]
        path = persist_calibration_negatives(tmp_path, "pm_abc", scores)

        assert path is not None
        back = np.load(path)
        assert list(back) == pytest.approx(scores)

    def test_the_file_is_named_for_its_probe(self, tmp_path):
        """Two probes in one run share an artifact directory, so the name has to separate them —
        otherwise the second overwrites the first's negatives and both report the same
        re-derivable threshold."""
        a = persist_calibration_negatives(tmp_path, "pm_aaa", [1.0])
        b = persist_calibration_negatives(tmp_path, "pm_bbb", [2.0])
        assert a != b
        assert "pm_aaa" in a and "pm_bbb" in b
        assert np.load(a).tolist() == [1.0]
        assert np.load(b).tolist() == [2.0]

    def test_it_lands_inside_the_run_artifact_directory(self, tmp_path):
        """Deleting a run removes its artifact directory, so these go with it rather than
        accumulating unowned files — the retention defect this estate already paid for."""
        path = persist_calibration_negatives(tmp_path, "pm_abc", [1.0])
        assert str(tmp_path) in path

    def test_the_array_is_float32(self, tmp_path):
        """Matching the evaluation arrays, so a reader can treat the two the same way."""
        path = persist_calibration_negatives(tmp_path, "pm_abc", [1.0, 2.0])
        assert np.load(path).dtype == np.float32

    def test_an_empty_sequence_still_writes(self, tmp_path):
        """An empty array is a true statement — "calibrated on nothing" — and distinguishable
        from None, which means the write failed."""
        path = persist_calibration_negatives(tmp_path, "pm_abc", [])
        assert path is not None
        assert np.load(path).tolist() == []


class TestItNeverCostsAProbeItsThreshold:
    def test_an_unwritable_directory_returns_None_rather_than_raising(self, tmp_path):
        """⚠ The threshold is already computed by this point. Losing a convenience file must not
        lose the probe."""
        missing = tmp_path / "does" / "not" / "exist"
        assert persist_calibration_negatives(missing, "pm_abc", [1.0]) is None

    def test_the_failure_is_logged_with_what_it_costs(self, tmp_path, caplog):
        """A silent None reads downstream as "this probe predates the feature". The log is the
        only thing that distinguishes a failure from an old probe."""
        import logging

        with caplog.at_level(logging.WARNING):
            persist_calibration_negatives(tmp_path / "nope" / "nope", "pm_abc", [1.0])
        # getMessage() renders the lazy %-args. My first version did `r.message % r.args`, which
        # raises on any record whose args were already applied.
        rendered = " ".join(r.getMessage() for r in caplog.records)
        assert "calibration negatives" in rendered
        assert "pm_abc" in rendered, "the log does not name the probe that lost its file"
        assert "new run" in rendered, "the log does not say what the failure costs"


class TestTheThresholdIsRederivableFromTheFile:
    """The point of keeping them, asserted end to end against the real calibrator."""

    def test_the_saved_scores_reproduce_the_shipped_threshold(self, tmp_path):
        from src.services.probe_monitor_trainer import calibrate

        rng = np.random.default_rng(1337)
        negatives = (rng.normal(0.0, 1.0, size=2000) * 3.0).tolist()

        shipped = calibrate(negatives, target_fpr=0.01, source="calibration_set")
        path = persist_calibration_negatives(tmp_path, "pm_abc", negatives)
        rederived = calibrate(np.load(path).tolist(), target_fpr=0.01, source="calibration_set")

        # float32 on the way to disk, so not bit-identical — but the same operating point.
        assert rederived.threshold == pytest.approx(shipped.threshold, abs=1e-4)
        assert rederived.realised_fpr == pytest.approx(shipped.realised_fpr, abs=1e-6)

    def test_a_DIFFERENT_target_gives_a_different_threshold_from_the_same_file(self, tmp_path):
        """⚠ The capability, not just the storage. This is the question that used to need a
        70-minute run: a looser budget must be answerable from the file alone."""
        from src.services.probe_monitor_trainer import calibrate

        rng = np.random.default_rng(7)
        negatives = (rng.normal(0.0, 1.0, size=2000) * 3.0).tolist()
        path = persist_calibration_negatives(tmp_path, "pm_abc", negatives)
        saved = np.load(path).tolist()

        one = calibrate(saved, target_fpr=0.01, source="calibration_set")
        five = calibrate(saved, target_fpr=0.05, source="calibration_set")

        assert five.threshold < one.threshold, (
            "a looser budget must place a LOWER threshold; got "
            f"1%={one.threshold} 5%={five.threshold}"
        )
        assert five.realised_fpr > one.realised_fpr


class TestTheStagePersistsThem:
    """Wiring. The function existing and being tested is the state this file exists to prevent."""

    @staticmethod
    def _calls():
        import ast
        import inspect

        from src.services import probe_monitor_run

        tree = ast.parse(inspect.getsource(probe_monitor_run._calibrate_probes))
        return {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    def test_the_calibrating_stage_calls_it(self):
        assert "persist_calibration_negatives" in self._calls()

    def test_the_scan_can_see_a_known_call(self):
        """A source scan that matches nothing asserts nothing."""
        assert "calibrate" in self._calls()

    def test_the_result_is_assigned_to_the_column(self):
        """⚠ Calling it and discarding the path would leave the file on disk and unreachable —
        exactly the shape of the original defect, one layer along."""
        import ast
        import inspect

        from src.services import probe_monitor_run

        tree = ast.parse(inspect.getsource(probe_monitor_run._calibrate_probes))
        assigned = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "persist_calibration_negatives"
            and any(
                isinstance(t, ast.Attribute) and t.attr == "calibration_scores_path"
                for t in node.targets
            )
        ]
        assert assigned, (
            "the stage does not assign the returned path to probe.calibration_scores_path, so "
            "the file would be written and never referenced"
        )


class TestTheColumnExists:
    def test_the_orm_declares_it(self):
        from src.models.probe_monitor import ProbeMonitor

        assert hasattr(ProbeMonitor, "calibration_scores_path")

    def test_it_is_nullable(self):
        """Every probe trained before this feature keeps NULL, which correctly reads as
        "its threshold cannot be recomputed without a new run"."""
        from src.models.probe_monitor import ProbeMonitor

        assert ProbeMonitor.__table__.c.calibration_scores_path.nullable is True
