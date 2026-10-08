"""Moving an operating point without a GPU, and refusing to move it halfway.

⚠ THE DIAL EXISTED FOR A WEEK BEFORE ANYTHING TURNED IT. `persist_calibration_negatives` has
kept every probe's negative scores since September, with a docstring opening "THIS IS WHAT MAKES
THE OPERATING POINT A DIAL RATHER THAN A RERUN" — and nothing ever read the file back. The only
line in the estate that wrote `probe.threshold` was inside the `calibrating` stage of a full
training run, so changing a bar cost ~2.6 hours of GPU. That is the gap this module closes, and
these tests are mostly about the three ways a cheap re-cut can lie.

The controls, named so a reader can tell what each is for:

* **Exact reproduction.** Re-cutting at the target the run used must return the shipped threshold
  BIT FOR BIT. If it does not, the re-cut and the original disagree and nothing else here means
  anything. This is also what pins `calibrate` against `np.quantile`: the quantile interpolates
  between order statistics and is wrong by up to 0.99 on the shipped probe's own array.
* **Affordability.** `calibrate` answers an unaffordable budget with `threshold=None`, which means
  FIRE ON NOTHING. Silently setting a monitor to that is going dark.
* **The silent no-op.** Both `window_for` and `threshold_for_length` PREFER a derived bar over the
  global one, so moving only the global on a probe that has windows changes nothing a consumer
  serves while the row, the report and the UI all show the new number.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.services import probe_recalibration as R


# The shipped L11 probe's own operating point, read off the production row on 2026-10-02.
SHIPPED_THRESHOLD = 11.914362907409668
SHIPPED_TARGET = 0.01


def _array(tmp_path: Path, name: str, values) -> str:
    path = tmp_path / name
    np.save(path, np.asarray(list(values), dtype=np.float32))
    return str(path)


def _probe(tmp_path: Path, *, negatives, lengths=None, windows=None, bands=None, **over):
    """A probe stub carrying only what `propose` reads, plus whatever a test overrides."""
    fields = dict(
        id="pm_test",
        calibration_scores_path=_array(tmp_path, "neg.npy", negatives),
        calibration_lengths_path=_array(tmp_path, "len.npy", lengths) if lengths else None,
        threshold=None,
        target_fpr=SHIPPED_TARGET,
        realised_fpr=SHIPPED_TARGET,
        threshold_source="calibration_set",
        window_decisions=windows,
        length_bands=bands,
        published=None,
        calibration_history=None,
        definition_path=None,
        # ⚠ NO `updated_at`. `ProbeMonitor` HAS NO SUCH COLUMN, and declaring it here made this
        # stub MORE FORGIVING THAN THE ORM: `apply` read `(probe.updated_at or probe.created_at)`
        # and every test here passed over a line that cannot run, until the first live commit
        # returned a 500. A stand-in that invents a field cannot disagree with code that reads it.
        # `test_probe_definition_reads_real_columns.py` now walks this module's AST for exactly
        # that, which is the guard a fixture cannot provide.
        created_at=None,
    )
    fields.update(over)
    return SimpleNamespace(**fields)


@pytest.fixture
def shipped_negatives():
    """The real 2,000-negative array from `pm_36d1a65f7953`, regenerated deterministically.

    ⚠ NOT THE REAL FILE. A fixture copied from `/data` would make the suite depend on a volume
    that only one machine has, and the property under test is arithmetic, not those exact floats.
    The EXACT-reproduction control against the production value runs separately, on hardware, and
    is recorded in the review notes — this fixture pins the same property on a synthetic array
    whose answer is computable by hand.
    """
    rng = np.random.default_rng(1337)
    return sorted(rng.normal(0.0, 5.0, 2000).tolist(), reverse=True)


class TestTheRecutReproducesWhatTheRunCut:
    """The control everything else rests on."""

    def test_the_same_target_returns_the_same_threshold_bit_for_bit(self, tmp_path):
        """⚠ EXACT, NOT APPROXIMATE — and the reason it can be exact is worth stating.

        The stored array is float32, so a round trip through it is only lossless for values that
        are float32-representable. Real aggregates are: they come out of a float32 torch tensor,
        which is why re-cutting the production probe `pm_36d1a65f7953` at its own 0.01 returns
        `11.914362907409668` bit for bit (measured against the live row, 2026-10-02). This
        fixture therefore uses float32-representable values, because a float64 fixture would
        fail for a reason that cannot happen in production and would teach the wrong lesson.
        """
        from src.services.probe_monitor_trainer import calibrate

        # Exactly what a real aggregate is: float32 widened to Python float.
        values = [float(v) for v in np.linspace(-10.0, 30.0, 2000, dtype=np.float32)]
        shipped = calibrate(values, target_fpr=0.01, source="calibration_set")
        probe = _probe(tmp_path, negatives=values, threshold=shipped.threshold)
        again = R.propose(probe, [], target_fpr=0.01)["proposed"]
        assert again["threshold"] == shipped.threshold, (
            "re-cutting at the target the run used returned a different number, so the re-cut "
            "and the original calibration disagree"
        )
        assert again["realised_fpr"] == shipped.realised_fpr

    def test_the_float32_STORE_is_what_bounds_that_exactness(self, tmp_path):
        """⚠ A CAVEAT PINNED SO IT CANNOT BE DISCOVERED LATER AS A SURPRISE.

        `persist_calibration_negatives` casts to float32. If an aggregate were ever computed in
        float64 — a different pooling rule, a future accumulation in double — the re-cut would
        differ from the shipped threshold in the last bits, and a test asserting bit-equality
        would fail for a reason that has nothing to do with the re-cut being wrong. This test
        states the bound: the dial is exact to float32, and no further.
        """
        from src.services.probe_monitor_trainer import calibrate

        f64 = [float(v) for v in np.linspace(-10.0, 30.0, 2000)]
        shipped = calibrate(f64, target_fpr=0.01, source="calibration_set")
        again = R.propose(_probe(tmp_path, negatives=f64), [], target_fpr=0.01)["proposed"]
        assert again["threshold"] != shipped.threshold, (
            "float64 negatives survived a float32 store unchanged — if that is now true, the "
            "store changed and this test's premise needs re-deriving, not deleting"
        )
        assert again["threshold"] == pytest.approx(shipped.threshold, rel=1e-6), (
            "the disagreement is larger than a float32 round trip, so something other than the "
            "store's precision is moving the number"
        )

    def test_it_is_the_ORDER_STATISTIC_not_an_interpolated_quantile(self, tmp_path):
        """⚠ THIS IS THE np.quantile TRAP, PINNED.

        `calibrate` takes the `int(target_fpr * n)`-th highest negative. `np.quantile` interpolates
        between order statistics, and on the shipped probe's own array the two disagree by 0.08 at
        1% and by 0.99 at 0.1%. A re-cut written as a quantile would be plausible, close, and
        wrong — and would never reproduce the threshold the run shipped.
        """
        values = [float(v) for v in range(2000)]
        got = R.propose(_probe(tmp_path, negatives=values), [], target_fpr=0.01)["proposed"]
        # The 20th highest of 0..1999 is 1980.
        assert got["threshold"] == 1980.0
        assert got["threshold"] != float(np.quantile(values, 0.99))

    def test_a_different_target_moves_it_in_the_right_direction(self, tmp_path, shipped_negatives):
        probe = _probe(tmp_path, negatives=shipped_negatives)
        tighter = R.propose(probe, [], target_fpr=0.001)["proposed"]
        looser = R.propose(probe, [], target_fpr=0.05)["proposed"]
        assert tighter["threshold"] > looser["threshold"], (
            "a tighter false-positive budget must raise the bar, not lower it"
        )
        assert tighter["realised_fpr"] < looser["realised_fpr"]

    def test_the_realised_rate_never_overspends_the_target(self, tmp_path, shipped_negatives):
        probe = _probe(tmp_path, negatives=shipped_negatives)
        for target in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1):
            got = R.propose(probe, [], target_fpr=target)["proposed"]
            assert got["realised_fpr"] <= target, (
                f"at target {target} the re-cut realises {got['realised_fpr']}, spending more "
                f"than the budget — the wrong direction for a monitor"
            )


class TestAnUnaffordableTargetIsRefusedRatherThanGoingDark:
    """⚠ `threshold=None` MEANS FIRE ON NOTHING, AND IT IS A REAL OPERATING POINT.

    `calibrate` returns it when `int(target_fpr * n) <= 0`, deliberately not `inf`. A recalibrate
    endpoint that passed that through would answer "0.0001 please" with 200 OK and a monitor that
    never fires again — silence being the failure mode this estate names as its worst, because it
    is indistinguishable from nothing being wrong.
    """

    def test_the_finest_affordable_rate_is_one_over_n(self):
        assert R.finest_affordable_fpr(2000) == 0.0005
        assert R.finest_affordable_fpr(620) == pytest.approx(1 / 620)

    def test_a_target_below_it_is_refused(self, tmp_path, shipped_negatives):
        with pytest.raises(R.RecalibrationRefused) as caught:
            R.propose(_probe(tmp_path, negatives=shipped_negatives), [], target_fpr=0.0001)
        assert caught.value.code == "target_fpr_unaffordable"
        assert caught.value.status == 422

    def test_the_refusal_NAMES_the_rate_that_would_work(self, tmp_path, shipped_negatives):
        """A refusal that does not say what to ask for instead sends the caller guessing."""
        with pytest.raises(R.RecalibrationRefused) as caught:
            R.propose(_probe(tmp_path, negatives=shipped_negatives), [], target_fpr=0.0001)
        assert "0.0005" in caught.value.detail
        assert "2000 negatives" in caught.value.detail

    def test_the_boundary_itself_is_AFFORDABLE(self, tmp_path, shipped_negatives):
        """1/n buys exactly one negative, so it is the finest rate that is not fire-on-nothing."""
        got = R.propose(_probe(tmp_path, negatives=shipped_negatives), [], target_fpr=0.0005)
        assert got["proposed"]["threshold"] is not None
        assert got["proposed"]["fires_on_nothing"] is False

    def test_it_can_be_asked_for_EXPLICITLY_and_says_so(self, tmp_path, shipped_negatives):
        got = R.propose(
            _probe(tmp_path, negatives=shipped_negatives),
            [], target_fpr=0.0001, allow_fire_on_nothing=True,
        )
        assert got["proposed"]["threshold"] is None
        assert got["proposed"]["fires_on_nothing"] is True, (
            "a fire-on-nothing bar must announce itself; a None threshold read as 'no opinion' "
            "is the NULL-is-not-zero confusion the probe model warns about"
        )

    def test_a_target_outside_the_open_interval_is_refused(self, tmp_path, shipped_negatives):
        probe = _probe(tmp_path, negatives=shipped_negatives)
        for bad in (0.0, 1.0, 1.5, -0.1):
            with pytest.raises(R.RecalibrationRefused) as caught:
                R.propose(probe, [], target_fpr=bad, allow_fire_on_nothing=True)
            assert caught.value.status == 422


class TestADerivedBarIsNeverLeftBehind:
    """⚠ THE LOAD-BEARING REFUSAL. THIS IS THE ONE THAT STOPS A SILENT NO-OP.

    A probe can carry a bar per contract window and a bar per length band, and `window_for` and
    `threshold_for_length` both take precedence over the global threshold. Moving only the global
    on such a probe would change nothing a consumer serves — while the row, the report and the UI
    all showed the new number. Each stale entry also carries its own `target_fpr`, so it would
    claim a budget the probe no longer has.
    """

    def _windows(self, tmp_path, *, with_arrays: bool):
        entry = {"threshold": 7.96, "target_fpr": 0.01, "realised_fpr": 0.01,
                 "n_negatives": 2000, "scope": "input"}
        if with_arrays:
            entry["scores_path"] = _array(tmp_path, "w.npy", [float(v) for v in range(2000)])
        return {"prompt": entry}

    def test_a_probe_whose_window_arrays_are_MISSING_is_refused(self, tmp_path, shipped_negatives):
        probe = _probe(
            tmp_path, negatives=shipped_negatives,
            windows=self._windows(tmp_path, with_arrays=False),
        )
        with pytest.raises(R.RecalibrationRefused) as caught:
            R.propose(probe, [], target_fpr=0.005)
        assert caught.value.code == "derived_bars_not_recuttable"
        assert "prompt" in caught.value.detail
        assert "recut_windows=true" in caught.value.detail, (
            "the refusal must say how to proceed, or it is a dead end rather than a gate"
        )

    def test_the_refusal_EXPLAINS_why_moving_only_the_global_is_not_enough(
        self, tmp_path, shipped_negatives
    ):
        probe = _probe(
            tmp_path, negatives=shipped_negatives,
            windows=self._windows(tmp_path, with_arrays=False),
        )
        with pytest.raises(R.RecalibrationRefused) as caught:
            R.propose(probe, [], target_fpr=0.005)
        assert "precedence" in caught.value.detail

    def test_a_probe_whose_window_arrays_ARE_kept_moves_every_bar(self, tmp_path, shipped_negatives):
        probe = _probe(
            tmp_path, negatives=shipped_negatives,
            windows=self._windows(tmp_path, with_arrays=True),
        )
        got = R.propose(probe, [], target_fpr=0.005)
        assert got["window_decisions"]["prompt"]["target_fpr"] == 0.005, (
            "the per-window bar kept the old target, so a consumer would judge the prompt window "
            "against a 1% bar while the probe claims 0.5%"
        )
        assert got["window_decisions"]["prompt"]["threshold"] != 7.96
        # The scope is carried through, not re-derived: it records what `prompt` MEANT here.
        assert got["window_decisions"]["prompt"]["scope"] == "input"

    def test_a_probe_with_NO_windows_is_not_refused(self, tmp_path, shipped_negatives):
        got = R.propose(_probe(tmp_path, negatives=shipped_negatives), [], target_fpr=0.005)
        assert got["window_decisions"] is None, (
            "None means 'never attempted' and must survive a re-cut; {} would claim the windows "
            "were calibrated and came back empty"
        )

    def test_a_probe_whose_LENGTHS_are_missing_is_refused(self, tmp_path, shipped_negatives):
        probe = _probe(
            tmp_path, negatives=shipped_negatives,
            bands=[{"min_tokens": 0, "max_tokens": None, "threshold": 9.9,
                    "threshold_source": "band", "target_fpr": 0.01,
                    "realised_fpr": 0.01, "n_negatives": 2000}],
        )
        with pytest.raises(R.RecalibrationRefused) as caught:
            R.propose(probe, [], target_fpr=0.005)
        assert caught.value.code == "derived_bars_not_recuttable"
        assert "per-length" in caught.value.detail

    def test_a_probe_whose_lengths_ARE_kept_re_cuts_its_bands(self, tmp_path, shipped_negatives):
        lengths = [50 + (i % 800) for i in range(2000)]
        probe = _probe(
            tmp_path, negatives=shipped_negatives, lengths=lengths,
            bands=[{"min_tokens": 0, "max_tokens": None, "threshold": 9.9,
                    "threshold_source": "band", "target_fpr": 0.01,
                    "realised_fpr": 0.01, "n_negatives": 2000}],
        )
        got = R.propose(probe, [], target_fpr=0.05)
        assert got["length_bands"], "the band table came back empty rather than re-cut"
        assert all(b["target_fpr"] == 0.05 for b in got["length_bands"])
        # Still contiguous from zero with an open-ended tail — the contract's own requirement.
        assert got["length_bands"][0]["min_tokens"] == 0
        assert got["length_bands"][-1]["max_tokens"] is None

    def test_arrays_of_DIFFERENT_SIZES_are_refused_rather_than_zipped(
        self, tmp_path, shipped_negatives
    ):
        """Two arrays that do not describe the same rows would pair a score with another row's
        length, producing a plausible table of bars for lengths nothing was measured at."""
        probe = _probe(
            tmp_path, negatives=shipped_negatives, lengths=[10, 20, 30],
            bands=[{"min_tokens": 0, "max_tokens": None, "threshold": 9.9,
                    "threshold_source": "band", "target_fpr": 0.01,
                    "realised_fpr": 0.01, "n_negatives": 2000}],
        )
        with pytest.raises(R.RecalibrationRefused) as caught:
            R.propose(probe, [], target_fpr=0.005)
        assert caught.value.code == "calibration_arrays_disagree"


class TestAProbeWithNoStoredArrayCannotBeRecutAtAll:
    def test_a_null_path_is_refused(self, tmp_path):
        probe = _probe(tmp_path, negatives=[1.0])
        probe.calibration_scores_path = None
        with pytest.raises(R.RecalibrationRefused) as caught:
            R.propose(probe, [], target_fpr=0.01)
        assert caught.value.code == "no_calibration_array"
        assert "new run" in caught.value.detail

    def test_a_recorded_path_that_is_GONE_is_refused_the_same_way(self, tmp_path):
        probe = _probe(tmp_path, negatives=[1.0])
        probe.calibration_scores_path = str(tmp_path / "vanished.npy")
        with pytest.raises(R.RecalibrationRefused) as caught:
            R.propose(probe, [], target_fpr=0.01)
        assert caught.value.code == "no_calibration_array"

    def test_a_CORRUPT_array_is_not_mistaken_for_an_absent_one(self, tmp_path):
        """Different statements: "nothing was kept" and "what was kept is unreadable"."""
        bad = tmp_path / "corrupt.npy"
        bad.write_bytes(b"not a numpy file at all")
        probe = _probe(tmp_path, negatives=[1.0])
        probe.calibration_scores_path = str(bad)
        with pytest.raises(R.RecalibrationRefused) as caught:
            R.propose(probe, [], target_fpr=0.01)
        assert caught.value.code == "calibration_array_unreadable"


class _FakeDb:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


class TestThePreviewAnswersWhatTheMoveWOULDDO:
    """⚠ A DIAL WITH NO FEEDBACK IS WHAT TURNED THIS INTO AN ARGUMENT RATHER THAN A MEASUREMENT.

    `threshold_transfer` already looks ANY threshold up against each evaluation set's stored ROC
    and returns the recall, the realised FPR, whether the set is unreachable, and a `caution`.
    Passing the CANDIDATE instead of the shipped bar makes the per-set consequence of a move free
    — no GPU, no new metric code. Without it an operator moves a number and learns nothing.
    """

    @staticmethod
    def _evaluation(name: str):
        roc = [{"threshold": float(t), "tpr": min(1.0, t / 20.0), "fpr": max(0.0, 1 - t / 20.0)}
               for t in range(0, 21)]
        return SimpleNamespace(metrics={
            "name": name, "scored": True, "auroc": 0.9, "roc": roc,
            "operating_points": [{"target_fpr": 0.01, "threshold": 15.0, "recall": 0.4}],
        })

    def test_it_reports_the_candidate_bar_AND_the_one_being_served(self, tmp_path, shipped_negatives):
        probe = _probe(tmp_path, negatives=shipped_negatives, threshold=5.0)
        got = R.propose(probe, [self._evaluation("set_a")], target_fpr=0.001)
        assert got["transfer_current"]["shipped_threshold"] == 5.0
        assert got["transfer_proposed"]["shipped_threshold"] == got["proposed"]["threshold"]
        assert got["transfer_proposed"]["shipped_threshold"] != 5.0, (
            "the proposal reported the consequences of the bar already in force, so a preview "
            "would show no change whatever target was asked for"
        )

    def test_it_carries_the_per_set_consequence_of_the_candidate(self, tmp_path, shipped_negatives):
        probe = _probe(tmp_path, negatives=shipped_negatives, threshold=5.0)
        got = R.propose(probe, [self._evaluation("set_a")], target_fpr=0.001)
        entry = got["transfer_proposed"]["per_set"][0]
        assert entry["name"] == "set_a"
        assert "recall_at_shipped" in entry and "fpr_at_shipped" in entry

    def test_a_preview_WRITES_NOTHING(self, tmp_path, shipped_negatives):
        probe = _probe(tmp_path, negatives=shipped_negatives, threshold=5.0)
        before = (probe.threshold, probe.target_fpr, probe.calibration_history)
        R.propose(probe, [], target_fpr=0.001)
        assert (probe.threshold, probe.target_fpr, probe.calibration_history) == before

    def test_a_published_probe_is_told_its_copies_go_stale(self, tmp_path, shipped_negatives):
        """A published definition is append-only and cannot be reached. Stating it on the
        PROPOSAL rather than only on the commit puts it in the decision."""
        plain = _probe(tmp_path, negatives=shipped_negatives)
        assert R.propose(plain, [], target_fpr=0.005)["published_copies_go_stale"] is False
        published = _probe(
            tmp_path, negatives=shipped_negatives,
            published={"repo_id": "mistudio/probe", "at": "2026-09-27"},
        )
        assert R.propose(published, [], target_fpr=0.005)["published_copies_go_stale"] is True


class TestTheCommitRecordsWhereTheBarCameFrom:
    def _applied(self, tmp_path, negatives, **over):
        probe = _probe(tmp_path, negatives=negatives, threshold=5.0, **over)
        db = _FakeDb()
        proposal = R.propose(probe, [], target_fpr=0.005)
        return probe, db, R.apply(db, probe, proposal, reason="asked for a tighter budget")

    def test_it_writes_every_field_of_the_operating_point(self, tmp_path, shipped_negatives):
        probe, db, out = self._applied(tmp_path, shipped_negatives)
        assert probe.threshold == out["proposed"]["threshold"]
        assert probe.target_fpr == 0.005
        assert probe.realised_fpr == out["proposed"]["realised_fpr"]
        assert probe.threshold_source == "calibration_set"
        assert db.commits >= 1, "the row was never committed, so the move exists only in session"

    def test_the_history_SEEDS_revision_1_before_overwriting_it(self, tmp_path, shipped_negatives):
        """⚠ WITHOUT THE SEED, AN EVENT STAMPED `revision 1` IS UNANSWERABLE.

        The row carries only the current bar, so the bar the training run cut has to be recorded
        before it is replaced — otherwise a consumer holding a verdict judged at revision 1 has
        no way to learn what number that was.
        """
        probe, _db, _out = self._applied(tmp_path, shipped_negatives)
        history = probe.calibration_history
        assert len(history) == 2, f"expected a seed plus the re-cut, got {len(history)} entries"
        assert history[0]["revision"] == 1
        assert history[0]["to"]["threshold"] == 5.0, (
            "the seed does not record the bar the run cut, so revision 1 is unanswerable"
        )
        assert history[0]["from"] is None
        assert history[0]["reason"] == "cut by the training run"

    def test_the_recut_entry_records_BOTH_ends_of_the_move(self, tmp_path, shipped_negatives):
        probe, _db, out = self._applied(tmp_path, shipped_negatives)
        entry = probe.calibration_history[-1]
        assert entry["revision"] == 2
        assert entry["from"]["threshold"] == 5.0
        assert entry["to"]["threshold"] == out["proposed"]["threshold"]
        assert entry["reason"] == "asked for a tighter budget"
        assert entry["n_negatives"] == 2000

    def test_a_second_recut_appends_rather_than_replacing(self, tmp_path, shipped_negatives):
        probe, db, _ = self._applied(tmp_path, shipped_negatives)
        R.apply(db, probe, R.propose(probe, [], target_fpr=0.02), reason="looser")
        revisions = [e["revision"] for e in probe.calibration_history]
        assert revisions == [1, 2, 3], f"history is {revisions}, so a revision was lost"

    def test_the_revision_is_derived_from_the_history_not_a_counter(self, tmp_path, shipped_negatives):
        """A stored counter can disagree with the entries a reader uses to answer
        "what was revision 3". Deriving it makes that impossible."""
        probe = _probe(tmp_path, negatives=shipped_negatives, threshold=5.0)
        assert R.next_revision(probe) == 2
        probe.calibration_history = [{"revision": 1}, {"revision": 2}]
        assert R.next_revision(probe) == 3


class TestAMovedBarStalesWhatStatedTheOldOne:
    """⚠ AN EXPORTED DEFINITION CARRIES THE OPERATING POINT AND A CONSUMER CANNOT TELL IT MOVED.

    The run's own calibrating stage already invalidates for exactly this, with exactly this reason
    string. A re-cut that skipped it would leave a cached definition shipping the old bar as
    current — the stalest possible lie, because the file validates.
    """

    def _apply_with_definition(self, tmp_path, negatives, monkeypatch, *, target_fpr, path):
        calls = []
        from src.services import probe_definition_builder

        monkeypatch.setattr(
            probe_definition_builder, "invalidate_definition",
            lambda db, probe_id, *, reason: calls.append((probe_id, reason)) or True,
        )
        probe = _probe(tmp_path, negatives=negatives, threshold=5.0, definition_path=path)
        db = _FakeDb()
        R.apply(db, probe, R.propose(probe, [], target_fpr=target_fpr), reason="")
        return calls

    def test_a_moved_bar_invalidates_the_cached_definition(
        self, tmp_path, shipped_negatives, monkeypatch
    ):
        calls = self._apply_with_definition(
            tmp_path, shipped_negatives, monkeypatch, target_fpr=0.005, path="/data/d.json"
        )
        assert calls, "the cached definition was left stating the old threshold"
        assert calls[0][1] == "threshold recalibrated", (
            "the reason string must match the run's own call; a reader greps for one phrase"
        )

    def test_a_bar_that_did_NOT_move_does_not_churn_the_cache(
        self, tmp_path, shipped_negatives, monkeypatch
    ):
        """Re-cutting at the target already in force is a no-op, and invalidating a definition
        that still states the truth would force a GPU rebuild for nothing."""
        # The bar AS THE STORE WOULD RETURN IT, not as a float64 calibrate would compute it —
        # see `test_the_float32_STORE_is_what_bounds_that_exactness`. Using the float64 value
        # here would make the bar "move" by one float32 ulp and test the wrong thing.
        probe = _probe(
            tmp_path, negatives=shipped_negatives, threshold=0.0, definition_path="/data/d.json"
        )
        probe.threshold = R.propose(probe, [], target_fpr=0.01)["proposed"]["threshold"]
        calls = []
        from src.services import probe_definition_builder

        monkeypatch.setattr(
            probe_definition_builder, "invalidate_definition",
            lambda db, probe_id, *, reason: calls.append(reason) or True,
        )
        R.apply(_FakeDb(), probe, R.propose(probe, [], target_fpr=0.01), reason="")
        assert not calls

    def test_a_probe_with_no_cached_definition_needs_no_invalidation(
        self, tmp_path, shipped_negatives, monkeypatch
    ):
        calls = self._apply_with_definition(
            tmp_path, shipped_negatives, monkeypatch, target_fpr=0.005, path=None
        )
        assert not calls


def _calls_in(fn) -> set:
    """Every name called inside one function, by AST.

    Scoped to ONE function on purpose: a module-wide scan has twice been satisfied by the wrong
    occurrence here, including once by a comment describing the code it scanned for.
    """
    tree = ast.parse(inspect.cleandoc(inspect.getsource(fn)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


class TestTheCapabilityIsREACHABLE:
    """⚠ A CAPABILITY IS NOT SHIPPED UNTIL A TEST FAILS WHEN ITS WIRING IS REMOVED.

    This estate's signature failure was 16 MCP tools, fully implemented, unit-tested and
    documented, never registered with the server — every test passed by importing the module
    directly. Everything below asserts the CALL, not that the module imports.
    """

    def test_the_endpoint_calls_the_service(self):
        from src.api.v1.endpoints import probe_monitors

        assert "recalibrate_probe" in _calls_in(probe_monitors.recalibrate_probe_endpoint), (
            "the endpoint does not call recalibrate_probe, so the route answers without ever "
            "re-cutting anything"
        )

    def test_the_scan_can_see_a_known_call(self):
        """A source scan that matches nothing asserts nothing — twice-learned here."""
        from src.api.v1.endpoints import probe_monitors

        assert "run_in_threadpool" in _calls_in(probe_monitors.recalibrate_probe_endpoint)

    def test_the_endpoint_runs_the_SYNC_service_off_the_event_loop(self):
        """`recalibrate_probe` opens a sync session. Called directly from an async handler it
        would block the loop for the whole query — the reason every sibling uses a threadpool."""
        from src.api.v1.endpoints import probe_monitors

        calls = _calls_in(probe_monitors.recalibrate_probe_endpoint)
        assert "run_in_threadpool" in calls and "SyncSessionLocal" in calls

    def test_the_refusals_reach_the_caller_as_their_own_status(self):
        from src.api.v1.endpoints import probe_monitors

        source = inspect.getsource(probe_monitors.recalibrate_probe_endpoint)
        assert "RecalibrationRefused" in source
        assert "refusal.status" in source, (
            "a service refusal flattened to one status code would report an unaffordable target "
            "and a stale per-window bar as the same thing"
        )

    def test_the_route_is_served_at_the_path_the_tools_use(self):
        """Read off the built app, not the decorator: a router included under the wrong prefix
        serves a path nobody calls while every unit test passes."""
        from src.main import app

        paths = set(app.openapi()["paths"].keys())
        assert "/api/v1/probe-monitors/probes/{probe_id}/recalibrate" in paths, (
            f"the recalibrate route is not in the served app; probe paths are "
            f"{sorted(p for p in paths if 'probe-monitors' in p)}"
        )

    def test_the_route_answers_200_not_202(self):
        """⚠ NOT A STYLE POINT. 202 would mean this went through a queue, and a threshold re-cut
        is microseconds of arithmetic over a stored array — queueing it behind an hour-long
        capture is what made the operating point feel fixed in the first place."""
        from src.main import app

        op = app.openapi()["paths"]["/api/v1/probe-monitors/probes/{probe_id}/recalibrate"]["post"]
        assert "200" in op["responses"], f"responses are {sorted(op['responses'])}"
        assert "202" not in op["responses"]

    def test_the_GPU_arm_is_a_REGISTERED_celery_task(self):
        from src.core.celery_app import celery_app

        name = "src.workers.probe_monitor_tasks.recut_probe_windows"
        assert name in celery_app.tasks, (
            "the window re-cut task is not in the live registry, so the endpoint's 202 arm "
            "dispatches to nothing"
        )

    def test_the_GPU_arm_is_ROUTED_to_the_gpu_queue(self):
        """⚠ ROUTES HERE ARE EXACT TASK NAMES, NOT GLOBS, and this estate has already shipped a
        task that silently used the default queue because its entry was missing."""
        from src.core.celery_app import celery_app

        name = "src.workers.probe_monitor_tasks.recut_probe_windows"
        route = celery_app.conf.task_routes.get(name)
        assert route is not None, f"{name} has no route, so it falls to the default queue"
        assert route["queue"] == "extraction"

    def test_the_GPU_arm_calls_the_service_that_rescores_AND_recuts(self):
        from src.workers.probe_monitor_tasks import recut_probe_windows

        calls = _calls_in(recut_probe_windows)
        assert "recut_probe_windows_on_gpu" in calls
        assert "_require_row" in calls, "an unknown probe id would reach the GPU path"

    def test_the_gpu_service_rescores_the_windows_and_then_applies(self):
        from src.services.probe_monitor_run import recut_probe_windows_on_gpu

        calls = _calls_in(recut_probe_windows_on_gpu)
        for needed in ("_calibrate_windows", "propose", "apply"):
            assert needed in calls, (
                f"{needed} is not called, so the GPU arm would re-score and then not move the bar"
            )


class TestTheRunKeepsWhatARECUTNEEDS:
    """⚠ FOUR MUTATIONS IN THE 033 ARC SURVIVED FIRST TIME AND ALL FOUR WERE THE SAME GAP: a
    well-covered helper whose CALLER was covered by nothing. `persist_calibration_negatives` has
    its own tests; the lines that CALL it with the lengths and with each window's negatives are
    what ships, and they are what these assert.
    """

    def test_the_stage_persists_the_LENGTHS(self):
        from src.services import probe_monitor_run

        tree = ast.parse(inspect.getsource(probe_monitor_run._calibrate_probes))
        named = [
            kw.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "persist_calibration_negatives"
            for kw in node.keywords
            if kw.arg == "name" and isinstance(kw.value, ast.Constant)
        ]
        assert "lengths" in named, (
            "the calibrating stage does not persist the per-row token counts, so `length_bands` "
            "stays the one bar that needs a GPU pass to move"
        )

    def test_the_lengths_path_is_ASSIGNED_to_the_column(self):
        """Saving the file and discarding the path leaves it on disk and unreachable — the
        original defect, one layer along, which is why the base array has this guard too."""
        from src.services import probe_monitor_run

        tree = ast.parse(inspect.getsource(probe_monitor_run._calibrate_probes))
        assigned = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "calibration_lengths_path"
                for t in node.targets
            )
        ]
        assert assigned, "probe.calibration_lengths_path is never assigned"

    def test_each_WINDOW_persists_its_own_negatives(self):
        from src.services import probe_monitor_run

        src = inspect.getsource(probe_monitor_run._calibrate_windows)
        tree = ast.parse(inspect.cleandoc(src))
        persisted = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "persist_calibration_negatives"
        ]
        assert persisted, (
            "_calibrate_windows scores each window's negatives and throws them away, so moving "
            "a per-window bar to another target still needs the model"
        )

    def test_the_window_path_lands_in_the_DECISION_a_consumer_reads(self):
        """A saved file whose path is not in `window_decisions[window]` is unreachable: the
        re-cut looks it up there, and nowhere else."""
        from src.services import probe_monitor_run

        src = inspect.getsource(probe_monitor_run._calibrate_windows)
        assert '"scores_path"' in src

    def test_the_columns_exist_and_are_NULLABLE(self):
        """NULL means "this probe predates the persistence", which is true of every existing
        row — and a server default would make each of them claim a provenance nobody recorded."""
        from src.models.probe_monitor import ProbeMonitor

        for column in ("calibration_lengths_path", "calibration_history"):
            assert hasattr(ProbeMonitor, column), f"{column} is not on the ORM"
            assert ProbeMonitor.__table__.c[column].nullable is True
            assert ProbeMonitor.__table__.c[column].server_default is None, (
                f"{column} has a server default, which would make every historical probe claim "
                f"a calibration provenance it has not got — the chat_format mistake"
            )


class TestTheGpuArmLeavesAProbeFULLYRecuttable:
    """⚠ "ONE PAYMENT PER PROBE" WAS HALF TRUE, AND THAT HALF WAS THE EXPENSIVE ONE.

    The GPU re-cut arm persisted each window's negatives and NOT the per-row lengths. On a probe
    that also carries a length table — which both production probes do (`pm_36d1a65f7953` and
    `pm_f463a8a235ae`, 4 bands each, `calibration_lengths_path` NULL) — it would re-score three
    windows, save them, and then be refused by `propose` over the lengths it had not kept.
    Twenty minutes of GPU spent on a refusal.

    Found by being asked whether the arm is a one-time action, not by any test: every existing
    test of the arm asserted that it CALLS the right things, and calling `_calibrate_windows` is
    exactly what it did. The gap was between what the arm writes and what `propose` then demands.

    ⚠ THE SCOPE EQUALITY IN THE FIX IS A CORRECTNESS GUARD, NOT AN OPTIMISATION. `length_bands`
    is cut from the base pass's lengths, which run at `context.scope`. Lengths from another window
    describe a different token span, so persisting those would build a plausible band table whose
    boundaries were measured over spans nothing was scored on — and unlike an absent table, a
    wrong one does not refuse.
    """

    @staticmethod
    def _calibrate(artifact_dir, *, run_scope: str, lengths_by_scope: dict):
        """Run `_calibrate_windows` over per-scope scores carrying per-scope lengths — the
        arm scores only the contract windows' scopes, exactly as `recut_probe_windows_on_gpu` does."""
        from types import SimpleNamespace

        from src.services import probe_monitor_run as run_mod
        from src.services.probe_monitor_render import CONTRACT_WINDOW_SCOPES

        scores = {
            scope: run_mod.CalibrationScores(
                negatives=[1.0, 2.0, 3.0], n_rows=3, lengths=lengths_by_scope.get(scope, [])
            )
            for scope in CONTRACT_WINDOW_SCOPES.values()
        }
        probe = SimpleNamespace(
            id="pm_x", calibration_lengths_path=None, window_decisions=None
        )
        context = SimpleNamespace(
            # Three negatives per window: 0.34 is a budget they can place; at 0.01 each window is
            # too thin and places no bar (review round 1, M2).
            target_fpr=0.34, scope=run_scope, artifact_dir=Path(artifact_dir)
        )
        decisions = run_mod._calibrate_windows(probe, context, "calibration_set", scores)
        return probe, decisions

    def test_it_persists_the_lengths_from_the_pass_that_IS_the_base_pass(self, tmp_path):
        probe, _d = self._calibrate(
            tmp_path,
            run_scope="all",
            lengths_by_scope={"all": [10, 20, 30], "input": [4, 5, 6], "last_assistant": [1, 2, 3]},
        )
        assert probe.calibration_lengths_path, (
            "the window loop scored the base pass and threw its lengths away, so the GPU re-cut "
            "arm still cannot move a per-length bar"
        )
        assert [int(v) for v in np.load(probe.calibration_lengths_path)] == [10, 20, 30]

    def test_it_persists_from_the_run_scope_even_when_that_is_NOT_all(self, tmp_path):
        """A run scoped `last_assistant` is reproduced by the `response` window, so that is the
        pass whose lengths are the base pass's."""
        probe, _d = self._calibrate(
            tmp_path,
            run_scope="last_assistant",
            lengths_by_scope={"all": [99, 99, 99], "input": [4, 5, 6], "last_assistant": [7, 8, 9]},
        )
        assert [int(v) for v in np.load(probe.calibration_lengths_path)] == [7, 8, 9]

    def test_it_persists_NOTHING_when_no_window_reproduces_the_run_scope(self, tmp_path):
        """⚠ THE CORRECTNESS HALF. A run scoped `assistant` is matched by no contract window, so
        there is no pass whose lengths describe the base pass's rows. Writing another window's
        would be worse than writing none — an absent table refuses, a wrong one does not."""
        probe, _d = self._calibrate(
            tmp_path,
            run_scope="assistant",
            lengths_by_scope={"all": [10, 20, 30], "input": [4, 5, 6], "last_assistant": [1, 2, 3]},
        )
        assert probe.calibration_lengths_path is None

    def test_it_persists_nothing_when_the_pass_returned_no_lengths(self, tmp_path):
        probe, _d = self._calibrate(
            tmp_path, run_scope="all", lengths_by_scope={"all": [], "input": [], "last_assistant": []}
        )
        assert probe.calibration_lengths_path is None

    def test_every_window_still_keeps_its_OWN_negatives(self, tmp_path):
        """The lengths are additional, not a replacement — the per-window arrays are what make a
        per-window bar re-cuttable, and they were already correct."""
        _probe, decisions = self._calibrate(
            tmp_path, run_scope="all", lengths_by_scope={"all": [10, 20, 30]}
        )
        from src.services.probe_monitor_render import CONTRACT_WINDOW_SCOPES

        assert set(decisions) == set(CONTRACT_WINDOW_SCOPES)
        for window, entry in decisions.items():
            assert entry["scores_path"], f"{window} kept no negatives of its own"


class TestTheGpuArmRefusesBeforeSpendingTheGpu:
    """⚠ A PRECONDITION CHECKED AFTER THE COST HAS BEEN PAID IS NOT A PRECONDITION.

    A run scoped `assistant` or `user` is reproduced by no contract window, so re-scoring the
    windows cannot recover the lengths its length table was cut from. The arm must say so before
    loading the model, not after twenty minutes.
    """

    @staticmethod
    def _arm(run_scope: str, *, bands, lengths_path, precision_refusal=None):
        from types import SimpleNamespace
        from unittest.mock import patch

        from src.services import probe_monitor_run as run_mod

        probe = SimpleNamespace(
            id="pm_x", run_id="pmr_x", length_bands=bands,
            calibration_lengths_path=lengths_path, calibration_dataset_id="pmd_cal",
            calibration_scores_path=None,
            threshold_source="calibration_set", window_decisions={},
            # Trained under the served render (2026-10-08) with its start-of-text record; the render
            # gate is tested on its own.
            render_form={"generation_prompt": True, "add_special_tokens": False, "bos_handling": {"template_wrote_bos": True, "tokenizer_added_bos": False, "bos_count": 1}},
        )
        run = SimpleNamespace(id="pmr_x", calibration_dataset_id="pmd_cal")

        class _Q:
            def __init__(self, row): self._row = row
            def filter(self, *_a, **_k): return self
            def first(self): return self._row

        class _Db:
            def query(self, model):
                return _Q(probe if getattr(model, "__name__", "") == "ProbeMonitor" else run)

        loaded = {"model": False}

        def _never_load(*_a, **_k):
            loaded["model"] = True
            raise AssertionError("the model was loaded despite an unsatisfiable precondition")

        # The precision check is its own precondition (tested below); these tests are about the
        # length one, so it answers "no refusal" unless a test says otherwise.
        precision = {"recorded": "bfloat16", "loads_at": "bfloat16", "refusal": precision_refusal}
        with patch.object(run_mod, "context_from_row",
                          return_value=SimpleNamespace(scope=run_scope, target_fpr=0.01)), \
             patch.object(run_mod, "probe_precision", return_value=precision), \
             patch.object(run_mod, "_load_model_for_run", side_effect=_never_load):
            return run_mod.recut_probe_windows_on_gpu(_Db(), "pm_x", target_fpr=0.005)

    def test_an_unreachable_run_scope_refuses_without_loading_the_model(self):
        bands = [{"min_tokens": 0, "max_tokens": None, "threshold": 1.0}]
        with pytest.raises(R.RecalibrationRefused) as caught:
            self._arm("assistant", bands=bands, lengths_path=None)
        assert caught.value.code == "lengths_not_recoverable"
        assert "assistant" in caught.value.detail
        assert "new run" in caught.value.detail

    def test_the_refusal_names_what_CAN_still_be_recut(self):
        """A dead end and a partial capability are different answers, and an operator acts on
        them differently."""
        bands = [{"min_tokens": 0, "max_tokens": None, "threshold": 1.0}]
        with pytest.raises(R.RecalibrationRefused) as caught:
            self._arm("user", bands=bands, lengths_path=None)
        assert "per-window" in caught.value.detail

    def test_a_probe_whose_lengths_ARE_kept_is_not_refused_on_this_ground(self):
        """It proceeds past the precondition — reaching the model load, which this stub refuses
        loudly so the test cannot pass by never getting there."""
        bands = [{"min_tokens": 0, "max_tokens": None, "threshold": 1.0}]
        with pytest.raises(AssertionError, match="model was loaded"):
            self._arm("assistant", bands=bands, lengths_path="/data/len.npy")

    def test_a_probe_with_NO_length_table_is_not_refused_on_this_ground(self):
        with pytest.raises(AssertionError, match="model was loaded"):
            self._arm("assistant", bands=None, lengths_path=None)

    def test_a_run_at_another_precision_is_refused_before_the_model_loads(self):
        """⚠ Review round 1, H1: re-cut bars are cut from negatives scored NOW. A run fitted at
        float16 re-scored at bfloat16 would get bars from another distribution, silently."""
        with pytest.raises(R.RecalibrationRefused) as caught:
            self._arm("all", bands=None, lengths_path=None,
                      precision_refusal="run pmr_x predates precision recording")
        assert caught.value.code == "precision_mismatch"


class TestTheHANDOFFBetweenTheArmAndThePropose:
    """⚠ THE JUNCTION, WHICH IS WHERE THE DEFECT LIVED.

    `_calibrate_windows` persisting the arrays is tested above. `propose` refusing on a missing
    array is tested further up. **Both passed while the arm was broken**, because nothing asserted
    that what the first writes is what the second then accepts — the "well-covered helpers, naked
    junction" shape that produced four surviving mutations in the 033 arc and the pod-with-no-
    database defect in this one.

    So this runs the REAL `_calibrate_windows` over a stubbed scorer, then the REAL `propose` on
    the probe it produced, and requires success. A change to either side that breaks the handoff
    goes red here even if both sides' own tests stay green.
    """

    @staticmethod
    def _after_the_arm(tmp_path, *, run_scope="all", bands):
        """A probe as `_calibrate_windows` leaves it — windows re-scored, lengths kept."""
        from types import SimpleNamespace

        from src.services import probe_monitor_run as run_mod

        # 2,000 negatives and 2,000 lengths, as a real calibration pass returns.
        negatives = [float(v) for v in np.linspace(-5.0, 25.0, 2000, dtype=np.float32)]
        lengths = [50 + (i % 800) for i in range(2000)]

        from src.services.probe_monitor_render import CONTRACT_WINDOW_SCOPES

        scores = {
            scope: run_mod.CalibrationScores(
                negatives=negatives, n_rows=len(negatives),
                lengths=lengths if scope == run_scope else [],
            )
            for scope in CONTRACT_WINDOW_SCOPES.values()
        }

        probe = SimpleNamespace(
            id="pm_handoff",
            # The base array is on disk already — that half always worked.
            calibration_scores_path=_array(tmp_path, "neg.npy", negatives),
            calibration_lengths_path=None,
            window_decisions={"all": {}, "prompt": {}, "response": {}},
            length_bands=bands,
            threshold=11.9144, target_fpr=0.01, realised_fpr=0.01,
            threshold_source="calibration_set",
            published=None, calibration_history=None, definition_path=None,
            created_at=None,
        )
        context = SimpleNamespace(
            target_fpr=0.01, scope=run_scope, artifact_dir=Path(tmp_path)
        )
        probe.window_decisions = run_mod._calibrate_windows(
            probe, context, "calibration_set", scores
        )
        return probe

    def test_a_probe_the_arm_has_processed_is_FULLY_recuttable(self, tmp_path):
        """The acceptance the arm's docstring claims: one payment, then every bar moves."""
        bands = [{"min_tokens": 0, "max_tokens": None, "threshold": 9.9,
                  "threshold_source": "band", "target_fpr": 0.01,
                  "realised_fpr": 0.01, "n_negatives": 2000}]
        probe = self._after_the_arm(tmp_path, bands=bands)

        got = R.propose(probe, [], target_fpr=0.005)

        assert got["proposed"]["target_fpr"] == 0.005
        assert got["window_decisions"] and all(
            e["target_fpr"] == 0.005 for e in got["window_decisions"].values()
        ), "the per-window bars did not move, so the arm left the probe half re-cuttable"
        assert got["length_bands"] and all(
            b["target_fpr"] == 0.005 for b in got["length_bands"]
        ), "the per-length table did not move — this is the defect the arm shipped with"

    def test_WITHOUT_the_lengths_the_same_handoff_is_REFUSED(self, tmp_path):
        """The negative half: this is precisely what the arm used to produce, and `propose`
        refusing it is why twenty minutes bought nothing."""
        bands = [{"min_tokens": 0, "max_tokens": None, "threshold": 9.9,
                  "threshold_source": "band", "target_fpr": 0.01,
                  "realised_fpr": 0.01, "n_negatives": 2000}]
        probe = self._after_the_arm(tmp_path, bands=bands)
        probe.calibration_lengths_path = None  # as the arm left it before the fix

        with pytest.raises(R.RecalibrationRefused) as caught:
            R.propose(probe, [], target_fpr=0.005)
        assert caught.value.code == "derived_bars_not_recuttable"
        assert "per-length" in caught.value.detail

    def test_a_probe_with_no_length_table_still_hands_off_cleanly(self, tmp_path):
        probe = self._after_the_arm(tmp_path, bands=None)
        got = R.propose(probe, [], target_fpr=0.005)
        assert got["length_bands"] is None
        assert all(e["target_fpr"] == 0.005 for e in got["window_decisions"].values())
