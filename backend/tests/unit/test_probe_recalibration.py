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
        # Feature 039: the real columns `apply` now reads and writes (the stored decision snapshot)
        # and the run it resolves a calibration reference from.
        run_id="pmr_test",
        decision_snapshot=None,
        decision_digest=None,
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
    """A session with no rows: `apply` (Feature 039) resolves the new bar's calibration reference
    from the probe's run, and a stub run that does not exist is a run with no calibration view.

    Feature 039 (C2): `apply` re-reads the row under a lock through the `_lock_row` seam; for this
    stub session the autouse fixture below hands back the stub itself. The REAL lock is proven
    only against the database (`test_probe_recalibration_locks.py`)."""

    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def query(self, _model):
        class _NoRows:
            def filter(self, *_a, **_k):
                return self

            def first(self):
                return None

        return _NoRows()


@pytest.fixture(autouse=True)
def _stub_sessions_lock_their_stub(monkeypatch):
    """`_lock_row` returns the stub for a `_FakeDb` session; any other session gets the real lock."""
    real = R._lock_row
    monkeypatch.setattr(
        R, "_lock_row", lambda db, probe: probe if isinstance(db, _FakeDb) else real(db, probe)
    )


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
        # ⚠ A GAPPED HISTORY (an entry pruned) is where "the length" and "the maximum" part — the
        # two-entry case above agrees with `len(history) + 1` by construction, so it never pinned the
        # earlier fix (039 control C4.11 survived it; this line is its regression).
        probe.calibration_history = [{"revision": 1}, {"revision": 2}, {"revision": 4}]
        assert R.next_revision(probe) == 5


class TestAMovedBarStalesWhatStatedTheOldOne:
    """⚠ AN EXPORTED DEFINITION CARRIES THE OPERATING POINT AND A CONSUMER CANNOT TELL IT MOVED.

    The run's own calibrating stage already invalidates for exactly this, with exactly this reason
    string. A re-cut that skipped it would leave a cached definition shipping the old bar as
    current — the stalest possible lie, because the file validates.

    Feature 039 (intended change, FPRD §13): the invalidation is `mark_invalidated` on the LOCKED row
    inside the re-cut's ONE commit (it used to be `invalidate_definition`, committing on its own
    after the re-cut's commit), it carries `cause: "decision"`, and an identical re-cut (FR-12) is
    not a commit at all — so it can never churn the cache.
    """

    def _with_definition(self, tmp_path, negatives, *, path, **over):
        return _probe(
            tmp_path, negatives=negatives, threshold=5.0, definition_path=path,
            definition_sha256=("a" * 64) if path else None, definition_built_at="t" if path else None,
            definition_build={"resolved": {"detector_digest": "d" * 64}} if path else None,
            built_detector_digest="d" * 64 if path else None, **over,
        )

    def test_a_moved_bar_invalidates_the_cached_definition(self, tmp_path, shipped_negatives):
        probe = self._with_definition(tmp_path, shipped_negatives, path="/data/d.json")
        db = _FakeDb()
        out = R.apply(db, probe, R.propose(probe, [], target_fpr=0.005), reason="")
        assert out["definition_invalidated"] is True
        assert probe.definition_path is None, "the cached definition was left stating the old bar"
        invalidated = probe.definition_build["invalidated"]
        assert invalidated["reason"] == "threshold recalibrated", (
            "the reason string must match the run's own call; a reader greps for one phrase"
        )
        assert invalidated["cause"] == "decision"
        assert db.commits == 1, "the invalidation must land in the re-cut's own single commit"

    def test_a_bar_that_did_NOT_move_does_not_churn_the_cache(self, tmp_path, shipped_negatives):
        """Re-cutting at the target already in force is a no-op, and invalidating a definition
        that still states the truth would force a GPU rebuild for nothing."""
        # The bar AS THE STORE WOULD RETURN IT, not as a float64 calibrate would compute it —
        # see `test_the_float32_STORE_is_what_bounds_that_exactness`.
        from src.services.probe_decision import write_decision_snapshot

        probe = self._with_definition(tmp_path, shipped_negatives, path="/data/d.json")
        probe.threshold = R.propose(probe, [], target_fpr=0.01)["proposed"]["threshold"]
        # A probe cut after 039 carries its decision snapshot (the run's stage wrote it).
        write_decision_snapshot(probe, None)
        db = _FakeDb()
        out = R.apply(db, probe, R.propose(probe, [], target_fpr=0.01), reason="")
        assert out["committed"] is False and out["reason"] == "unchanged"
        assert probe.definition_path == "/data/d.json"
        assert probe.calibration_history is None and db.commits == 0

    def test_a_probe_with_no_cached_definition_needs_no_invalidation(
        self, tmp_path, shipped_negatives
    ):
        probe = self._with_definition(tmp_path, shipped_negatives, path=None)
        out = R.apply(_FakeDb(), probe, R.propose(probe, [], target_fpr=0.005), reason="")
        assert out["committed"] is True and out["definition_invalidated"] is False
        assert "invalidated" not in (probe.definition_build or {})

    def test_window_only_recut_invalidates(self, tmp_path, shipped_negatives):
        """FR-11 (review R1 debt): a re-cut that moves ONLY a per-window bar appends a revision, so
        the cached definition — whose `identity.decision_digest` names the old decision — is stale
        and must be invalidated. The old rule ("only when the global threshold moved") kept it."""
        windows = {"prompt": {"threshold": 0.0, "target_fpr": 0.01, "realised_fpr": 0.01,
                              "n_negatives": 2000, "scope": "user",
                              "scores_path": _array(tmp_path, "w.npy", shipped_negatives),
                              "length_bands": None}}
        probe = self._with_definition(tmp_path, shipped_negatives, path="/data/d.json",
                                      windows=windows)
        probe.threshold = R.propose(probe, [], target_fpr=0.01)["proposed"]["threshold"]
        proposal = R.propose(probe, [], target_fpr=0.01)
        assert proposal["proposed"]["threshold"] == probe.threshold, "the global bar must not move"
        assert proposal["window_decisions"]["prompt"]["threshold"] != 0.0
        out = R.apply(_FakeDb(), probe, proposal, reason="")
        assert out["committed"] is True and out["definition_invalidated"] is True
        assert probe.definition_build["invalidated"]["cause"] == "decision"


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
            # Feature 039: a decision snapshot is already stored (a probe cut after 039), so the
            # arm's `ensure_decision_snapshot` has nothing to form on this stub session.
            decision_snapshot={"decision": None, "digest": None}, decision_digest=None,
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


# ── Feature 039: the invalidation split (D-T3, FR-11, E7) ─────────────────────


def test_invalidating_fields_is_gone():
    """`INVALIDATING_FIELDS` listed `rung` beside decision fields and had no reader; 039 deleted it
    (E7). A constant nothing reads is a guarantee nothing enforces."""
    from src.services import probe_definition_builder

    assert not hasattr(probe_definition_builder, "INVALIDATING_FIELDS")


def test_mark_invalidated_does_not_commit_and_keeps_what_consumers_hold():
    from src.services.probe_definition_builder import mark_invalidated

    probe = SimpleNamespace(
        definition_path="/data/probe_definitions/pm_x.probe.json", definition_built_at="t",
        definition_sha256="f" * 64, built_detector_digest="d" * 64,
        definition_build={"resolved": {"detector_digest": "d" * 64, "decision_digest": "e" * 64,
                                       "decision_revision": 2, "identity_source": "built"}},
    )
    assert mark_invalidated(probe, reason="threshold recalibrated", cause="decision") is True
    invalidated = probe.definition_build["invalidated"]
    assert invalidated["cause"] == "decision" and invalidated["reason"] == "threshold recalibrated"
    assert invalidated["was_sha256"] == "f" * 64
    assert invalidated["was_identity"] == {"detector_digest": "d" * 64, "decision_digest": "e" * 64,
                                           "decision_revision": 2, "identity_source": "built"}
    assert probe.definition_path is None and probe.definition_sha256 is None
    assert probe.built_detector_digest == "d" * 64
    # Nothing to drop: no change, no claim.
    assert mark_invalidated(probe, reason="again", cause="rung") is False


def test_rung_invalidation_records_cause_and_was_identity(sync_db, tmp_path):  # noqa: F811
    """The rung change's committing wrapper, on a real row: `cause: "rung"`, the identity the file
    carried, and `built_detector_digest` kept."""
    from src.models.probe_monitor import ProbeMonitor
    from src.services.probe_definition_builder import invalidate_definition

    rows = make_rows(sync_db, tmp_path)
    probe = rows.probe
    probe.definition_path = str(tmp_path / "pm.probe.json")
    probe.definition_sha256 = "a" * 64
    probe.built_detector_digest = "d" * 64
    probe.definition_build = {"resolved": {"detector_digest": "d" * 64, "decision_digest": "e" * 64,
                                           "decision_revision": 1, "identity_source": "built"}}
    sync_db.commit()
    assert invalidate_definition(sync_db, probe.id, reason="rung changed 2 -> 3", cause="rung")
    sync_db.expire_all()
    row = sync_db.query(ProbeMonitor).filter(ProbeMonitor.id == probe.id).one()
    invalidated = row.definition_build["invalidated"]
    assert invalidated["cause"] == "rung" and invalidated["reason"] == "rung changed 2 -> 3"
    assert invalidated["was_identity"]["decision_digest"] == "e" * 64
    assert invalidated["was_sha256"] == "a" * 64
    assert row.definition_path is None and row.built_detector_digest == "d" * 64


from tests.unit.test_probe_decision import make_rows, second_session, sync_db  # noqa: E402,F401


# ── Feature 039 Slice 2a: the re-cut reports and protects identity (FR-10 – FR-14, C2, C3) ────
#
# Real rows in `mistudio_test` (the `make_rows` fixture of `test_probe_decision.py`), behaviour keys
# random. The stub-session tests above cannot see a digest: a stub has no stored snapshot.


def _recut(db, probe_id, target, *, commit=True, **kw):
    from src.services.probe_recalibration import recalibrate_probe

    return recalibrate_probe(db, probe_id, target_fpr=target, commit=commit, **kw)


def _fresh(db, probe_id):
    from src.models.probe_monitor import ProbeMonitor

    db.expire_all()
    return db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).one()


@pytest.fixture
def no_millm(monkeypatch):
    """miLLM not configured (the standalone state), stated rather than inherited from the env."""
    from src.services import millm_read

    monkeypatch.setattr(millm_read, "imported_copies",
                        lambda pid: millm_read.NotConfigured("not connected"))


def test_preview_reports_identity_on_both_paths(sync_db, tmp_path, monkeypatch, no_millm):
    """FR-10: the preview states the digest and revision of BOTH ends, the last built detector
    digest and `detector_unchanged`; the GPU arm's result states the same identity. The preview's
    promised digest is the one the commit stores (FR-8)."""
    from src.schemas.probe_identity import decision_digest_of
    from src.services import probe_decision as P

    rows = make_rows(sync_db, tmp_path)
    rows.probe.built_detector_digest = "d" * 64
    sync_db.commit()
    preview = _recut(sync_db, rows.probe.id, 0.02, commit=False)
    probe = _fresh(sync_db, rows.probe.id)
    assert preview["current"]["decision_digest"] == probe.decision_digest is not None
    assert preview["current"]["decision_revision"] == preview["current"]["revision"] == 1
    assert preview["proposed"]["decision_revision"] == preview["proposed"]["revision"] == 2
    assert preview["detector_digest"] == "d" * 64 and preview["detector_unchanged"] is True
    assert preview["unchanged"] is False
    assert preview["proposed"]["decision_digest"] not in (None, probe.decision_digest)
    committed = _recut(sync_db, rows.probe.id, 0.02)
    probe = _fresh(sync_db, rows.probe.id)
    assert committed["current"]["decision_digest"] == preview["proposed"]["decision_digest"] == (
        probe.decision_digest
    ), "the preview promised a digest the commit did not store"
    assert committed["previous"]["decision_digest"] == preview["current"]["decision_digest"]
    assert decision_digest_of(P.decision_for_build(sync_db, probe)) == probe.decision_digest

    # The GPU arm: the same identity on its result (it commits through `apply`).
    from src.core import config as config_module
    from src.services import probe_monitor_run as run_mod

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
    gpu = make_rows(sync_db, tmp_path, windows=True)
    _arm_stubs(monkeypatch, run_mod, gpu)
    before = _fresh(sync_db, gpu.probe.id)
    P.ensure_decision_snapshot(sync_db, before)
    old_digest = before.decision_digest
    out = run_mod.recut_probe_windows_on_gpu(sync_db, gpu.probe.id, target_fpr=0.02, reason="t")
    after = _fresh(sync_db, gpu.probe.id)
    assert out["committed"] is True
    assert out["previous"]["decision_digest"] == old_digest
    assert out["current"]["decision_digest"] == out["proposed"]["decision_digest"] == (
        after.decision_digest
    ) != old_digest
    assert out["current"]["decision_revision"] == 2 and out["detector_unchanged"] is True


def _arm_stubs(monkeypatch, run_mod, rows, *, during_score=None):
    """The GPU arm with its model and re-score stubbed (the `test_probe_decision` pattern)."""
    run_mod.artifact_dir_for(rows.run.id).mkdir(parents=True, exist_ok=True)
    negatives = [float(v) for v in np.linspace(-8.0, 28.0, 2000, dtype=np.float32)]
    lengths = [20 + (i % 300) for i in range(2000)]

    def fake_group(db, prepared, group, scopes, *a, **k):
        if during_score is not None:
            during_score()
        return [{scope: run_mod.CalibrationScores(negatives=negatives, n_rows=2000,
                                                  lengths=lengths if scope == "all" else [])
                 for scope in scopes}]

    monkeypatch.setattr(run_mod, "_load_model_for_run", lambda run: (None, None, "llama"))
    monkeypatch.setattr(run_mod, "load_probe", lambda db, pid: (
        SimpleNamespace(layer=1), SimpleNamespace(rule="mean", rule_params={})))
    monkeypatch.setattr(run_mod, "_prepare_calibration_rows", lambda *a, **k: ([], []))
    monkeypatch.setattr(run_mod, "_score_calibration_group", fake_group)
    monkeypatch.setattr(run_mod, "probe_precision", lambda db, run: {"refusal": None})
    monkeypatch.setattr(run_mod, "probe_render", lambda run, probe: {"refusal": None})


def test_identical_recut_appends_nothing(sync_db, tmp_path, no_millm):
    """FR-12: re-cutting to the decision in force commits nothing — no revision, no snapshot
    change, the cached definition kept."""
    rows = make_rows(sync_db, tmp_path)
    rows.probe.definition_path = str(tmp_path / "pm.probe.json")
    rows.probe.definition_sha256 = "a" * 64
    sync_db.commit()
    out = _recut(sync_db, rows.probe.id, 0.01)
    probe = _fresh(sync_db, rows.probe.id)
    assert out["committed"] is False and out["reason"] == "unchanged"
    assert out["unchanged"] is True and out["next"] == []
    assert probe.calibration_history is None
    assert probe.definition_path == str(tmp_path / "pm.probe.json")
    assert out["current"]["decision_digest"] == probe.decision_digest


def test_not_formable_identical_recut_is_unchanged(sync_db, tmp_path, no_millm):
    """D-T5: a bar with no digest (its calibration set has no HuggingFace id) re-cut to itself is
    still recognised as unchanged — by the blocks without `calibration`."""
    rows = make_rows(sync_db, tmp_path, hf_repo_id=None)
    out = _recut(sync_db, rows.probe.id, 0.01)
    probe = _fresh(sync_db, rows.probe.id)
    assert probe.decision_digest is None and probe.decision_snapshot["decision"] is None
    assert out["committed"] is False and out["reason"] == "unchanged"
    assert probe.calibration_history is None
    moved = _recut(sync_db, rows.probe.id, 0.02)
    assert moved["committed"] is True, "a real move of a not-formable bar must still commit"


def test_no_two_consecutive_entries_share_a_digest(sync_db, tmp_path, no_millm):
    """FR-12 narrowed (R-4): A → A is no entry; A → B → A is three entries, never two equal
    neighbours."""
    rows = make_rows(sync_db, tmp_path)
    for target in (0.02, 0.02, 0.005, 0.005, 0.01, 0.02):
        _recut(sync_db, rows.probe.id, target)
    history = _fresh(sync_db, rows.probe.id).calibration_history
    digests = [entry["decision_digest"] for entry in history]
    assert [e["revision"] for e in history] == [1, 2, 3, 4, 5]
    assert all(a != b for a, b in zip(digests, digests[1:])), digests
    assert digests[0] == digests[3], "A → B → A must give A at two revisions"


def test_new_entries_carry_digests_old_read_null(sync_db, tmp_path, no_millm):
    """FR-13: entries written by a 039 re-cut carry the digest a build at that revision carries;
    entries written before 039 are NEVER back-filled — they read null."""
    from src.schemas.probe_identity import decision_digest_of
    from src.services import probe_decision as P

    rows = make_rows(sync_db, tmp_path)
    old = [{"revision": 1, "to": {"threshold": rows.probe.threshold}, "reason": "cut by the run"},
           {"revision": 2, "to": {"threshold": rows.probe.threshold}, "reason": "pre-039 re-cut"}]
    rows.probe.calibration_history = old
    sync_db.commit()
    built = []
    for target in (0.02, 0.005):
        _recut(sync_db, rows.probe.id, target)
        probe = _fresh(sync_db, rows.probe.id)
        built.append(decision_digest_of(P.decision_for_build(sync_db, probe)))
    history = _fresh(sync_db, rows.probe.id).calibration_history
    assert [e["revision"] for e in history] == [1, 2, 3, 4]
    assert "decision_digest" not in history[0] and "decision_digest" not in history[1], (
        "a pre-039 entry was back-filled with a digest nobody ever stored"
    )
    assert [history[2]["decision_digest"], history[3]["decision_digest"]] == built


def test_commit_next_names_live_tools(sync_db, tmp_path, monkeypatch):
    """FR-14: the commit's `next` names tools in the LIVE registry with arguments that validate —
    with and without publications and imported copies; unreachable miLLM → `millm_list_probes`;
    not configured → no miLLM step at all."""
    from src.services import millm_read
    from tests.support.next_calls import assert_next_calls_are_live

    rows = make_rows(sync_db, tmp_path)
    rows.probe.definition_path = str(tmp_path / "pm.probe.json")
    rows.probe.published = [{"repo_id": "org/one", "private": True},
                            {"repo_id": "org/two", "private": False}]
    sync_db.commit()
    monkeypatch.setattr(millm_read, "imported_copies", lambda pid: ["pr_a", "pr_b"])
    out = _recut(sync_db, rows.probe.id, 0.02)
    assert_next_calls_are_live(out["next"])
    tools = [c["tool"] for c in out["next"]]
    assert tools == ["build_probe_definition", "publish_probe_definition",
                     "millm_recalibrate_probe", "millm_recalibrate_probe"], tools
    assert out["next"][1]["arguments"] == {"probe_id": rows.probe.id, "repo_id": "org/two",
                                           "private": False}
    push = out["next"][2]["arguments"]
    assert push["probe_id"] == "pr_a" and push["mistudio_probe_id"] == rows.probe.id
    assert push["decision_digest"] == out["current"]["decision_digest"]
    assert push["decision_revision"] == out["current"]["decision_revision"] == 2
    assert out["millm_copies"] == ["pr_a", "pr_b"]

    monkeypatch.setattr(millm_read, "imported_copies",
                        lambda pid: millm_read.Unknown("miLLM at http://x could not be reached"))
    out = _recut(sync_db, rows.probe.id, 0.005)
    assert_next_calls_are_live(out["next"])
    assert [c["tool"] for c in out["next"]] == [
        "build_probe_definition", "publish_probe_definition", "millm_list_probes"]
    assert out["millm_copies"] == {"unknown": "miLLM at http://x could not be reached"}

    monkeypatch.setattr(millm_read, "imported_copies",
                        lambda pid: millm_read.NotConfigured("not connected"))
    out = _recut(sync_db, rows.probe.id, 0.02)
    assert_next_calls_are_live(out["next"])
    assert not any(c["tool"].startswith("millm_") for c in out["next"])
    assert out["millm_copies"] == {"not_configured": "not connected"}

    bare = make_rows(sync_db, tmp_path)
    out = _recut(sync_db, bare.probe.id, 0.02)
    assert out["next"] == [], "nothing was built or published, and miLLM holds no copy"


def test_a_build_below_rung_2_names_the_acknowledgement_it_needs(sync_db, tmp_path, no_millm):
    """Review R1 (4.7): for a probe below rung 2 the build call "as it stands" would be refused, so
    it names `acknowledge_below_rung2_reason` under `needs` — and never invents one."""
    from src.services.next_calls import after_build_race
    from tests.support.next_calls import assert_next_calls_are_live

    rows = make_rows(sync_db, tmp_path)
    rows.probe.rung = 1
    rows.probe.definition_path = str(tmp_path / "pm.probe.json")
    sync_db.commit()
    out = _recut(sync_db, rows.probe.id, 0.02)
    assert_next_calls_are_live(out["next"])
    assert out["next"][0]["needs"] == ["acknowledge_below_rung2_reason"]
    raced = after_build_race(rows.probe.id, rung=1)
    assert_next_calls_are_live(raced)
    assert raced[0]["needs"] == ["acknowledge_below_rung2_reason"]
    assert "needs" not in after_build_race(rows.probe.id, rung=2)[0]


def test_imported_copies_runs_after_the_commit_without_the_lock(
    sync_db, second_session, tmp_path, monkeypatch
):
    """FR-14: miLLM is asked AFTER the commit, outside the row lock — the row is updatable from
    another session during the miLLM call (a held lock would make that update time out)."""
    from sqlalchemy import text

    from src.models.probe_monitor import ProbeMonitor
    from src.services import millm_read

    rows = make_rows(sync_db, tmp_path)
    seen = {}

    def copies(pid):
        second_session.execute(text("SET lock_timeout = '2s'"))
        row = second_session.query(ProbeMonitor).filter(ProbeMonitor.id == pid).one()
        seen["threshold"] = row.threshold
        row.rung = 3
        second_session.commit()
        return []

    monkeypatch.setattr(millm_read, "imported_copies", copies)
    out = _recut(sync_db, rows.probe.id, 0.02)
    assert out["committed"] is True
    assert seen["threshold"] == out["current"]["threshold"], "miLLM was asked before the commit"
    assert _fresh(sync_db, rows.probe.id).rung == 3


def test_decision_changed_is_logged_and_refused(sync_db, tmp_path, caplog, no_millm):
    """C3 + 8.4: `expect_decision_digest` naming another decision is refused before any write,
    with `expected`, `current` and a live `next`, and one structured log line."""
    import logging

    from tests.support.next_calls import assert_next_calls_are_live

    rows = make_rows(sync_db, tmp_path)
    with caplog.at_level(logging.INFO, logger="src.services.probe_recalibration"):
        with pytest.raises(R.RecalibrationRefused) as refused:
            _recut(sync_db, rows.probe.id, 0.02, expect_decision_digest="e" * 64)
    probe = _fresh(sync_db, rows.probe.id)
    assert refused.value.code == "decision_changed" and refused.value.status == 409
    assert refused.value.extra["expected"] == "e" * 64
    assert refused.value.extra["current"] == probe.decision_digest
    assert_next_calls_are_live(refused.value.extra["next"])
    assert probe.calibration_history is None, "a refused commit wrote a revision"
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("decision_changed")]
    assert lines == [f"decision_changed probe={probe.id} expected={'e' * 64} "
                     f"current={probe.decision_digest}"]


def test_gpu_recut_refused_by_apply_restores_arrays(sync_db, second_session, tmp_path,
                                                    monkeypatch, no_millm):
    """X1/TI6: a `decision_changed` raised by `apply` on the GPU arm (a re-cut committed while it
    scored) rolls back the swapped table AND restores the re-scored arrays on disk."""
    from src.core import config as config_module
    from src.services import probe_monitor_run as run_mod
    from src.services import probe_decision as P

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
    rows = make_rows(sync_db, tmp_path, windows=True)
    P.ensure_decision_snapshot(sync_db, rows.probe)
    from src.services.probe_monitor_render import CONTRACT_WINDOW_SCOPES

    artifact = run_mod.artifact_dir_for(rows.run.id)
    artifact.mkdir(parents=True, exist_ok=True)
    # The arrays the re-score OVERWRITES under fixed names (`negatives_window_<scope>`), present
    # before the job with known bytes — so "restored" is checked against what the job really wrote.
    before = {}
    for scope in set(CONTRACT_WINDOW_SCOPES.values()):
        marker = artifact / f"{rows.probe.id}__calibration_negatives_window_{scope}.npy"
        np.save(marker, np.asarray([1.0, 2.0], dtype=np.float32))
        before[marker.name] = marker.read_bytes()

    def recut_meanwhile():
        _recut(second_session, rows.probe.id, 0.005)

    _arm_stubs(monkeypatch, run_mod, rows, during_score=recut_meanwhile)
    with pytest.raises(R.RecalibrationRefused) as refused:
        run_mod.recut_probe_windows_on_gpu(sync_db, rows.probe.id, target_fpr=0.02, reason="t")
    assert refused.value.code == "decision_changed"
    after = {p.name: p.read_bytes() for p in artifact.glob(f"{rows.probe.id}__calibration_*.npy")}
    assert after == before, "the re-scored arrays were left on disk (or new ones were not removed)"
    probe = _fresh(sync_db, rows.probe.id)
    assert probe.target_fpr == 0.005 and all(
        e["target_fpr"] == 0.005 for e in probe.window_decisions.values()
    ), "the other session's committed re-cut was overwritten"
    assert [e["revision"] for e in probe.calibration_history] == [1, 2]


def test_a_failure_after_the_commit_never_restores_the_old_arrays(sync_db, tmp_path, monkeypatch,
                                                                   no_millm):
    """Review R1 (Slice 2): the GPU arm's rollback wraps `apply`, and restores the pre-re-score
    arrays on ANY exception. `apply` asked miLLM and formed `next` AFTER its commit, so an exception
    there put the old arrays back on disk under a committed row whose bars were cut from the new
    ones. Nothing after the commit may raise: the re-cut is reported committed, its arrays kept."""
    from src.core import config as config_module
    from src.services import next_calls
    from src.services import probe_decision as P
    from src.services import probe_monitor_run as run_mod
    from src.services.probe_monitor_render import CONTRACT_WINDOW_SCOPES

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
    rows = make_rows(sync_db, tmp_path, windows=True)
    P.ensure_decision_snapshot(sync_db, rows.probe)
    artifact = run_mod.artifact_dir_for(rows.run.id)
    artifact.mkdir(parents=True, exist_ok=True)
    before = {}
    for scope in set(CONTRACT_WINDOW_SCOPES.values()):
        marker = artifact / f"{rows.probe.id}__calibration_negatives_window_{scope}.npy"
        np.save(marker, np.asarray([1.0, 2.0], dtype=np.float32))
        before[marker.name] = marker.read_bytes()

    def broken(*args, **kwargs):
        raise RuntimeError("follow-up formation failed")

    monkeypatch.setattr(next_calls, "after_recut", broken)
    _arm_stubs(monkeypatch, run_mod, rows)
    out = run_mod.recut_probe_windows_on_gpu(sync_db, rows.probe.id, target_fpr=0.02, reason="t")
    assert out["committed"] is True and out["next"] == []
    assert "unknown" in out["millm_copies"]
    after = {name: (artifact / name).read_bytes() for name in before}
    assert after != before, "the committed re-cut's re-scored arrays were replaced by the old ones"
    probe = _fresh(sync_db, rows.probe.id)
    assert [e["revision"] for e in probe.calibration_history] == [1, 2]


def test_a_pre_039_gpu_recut_seeds_revision_1_with_the_digest_of_the_bar_it_replaces(
        sync_db, tmp_path, monkeypatch, no_millm):
    """Review R1 (Slice 2): the GPU arm forms a pre-039 probe's snapshot BEFORE it swaps the table.
    Without that line (control R1-M4) the re-cut still committed, but the seeded revision 1 recorded
    `decision_digest: null` (FR-13) — the run's own bar, which every pre-039 event was judged by,
    could no longer be found by its digest."""
    from src.core import config as config_module
    from src.services import probe_decision as P
    from src.services import probe_monitor_run as run_mod

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
    rows = make_rows(sync_db, tmp_path, windows=True)
    assert rows.probe.decision_snapshot is None, "the fixture must be a probe cut before 039"
    expected = P.form_snapshot(rows.probe, P.calibration_from_run(
        sync_db, rows.probe, rows.probe.threshold_source))["digest"]
    assert expected is not None
    _arm_stubs(monkeypatch, run_mod, rows)
    out = run_mod.recut_probe_windows_on_gpu(sync_db, rows.probe.id, target_fpr=0.02, reason="t")
    assert out["committed"] is True
    probe = _fresh(sync_db, rows.probe.id)
    assert probe.calibration_history[0]["revision"] == 1
    assert probe.calibration_history[0]["decision_digest"] == expected
    assert out["previous"]["decision_digest"] == expected


def test_a_preview_of_a_probe_with_no_threshold_source_promises_the_digest_the_commit_stores(
        sync_db, tmp_path, no_millm):
    """Review R1 (Slice 2): a probe whose `threshold_source` is NULL GAINS `calibration_set` on its
    first re-cut, so the preview's calibration reference needs the run row — `recalibrate_probe`
    passes the session to `propose` for exactly this. Without it (control R1-M7) the preview reported
    the candidate NOT FORMABLE (digest null) while the commit stored a real digest."""
    rows = make_rows(sync_db, tmp_path, threshold_source=None)
    preview = _recut(sync_db, rows.probe.id, 0.02, commit=False)
    committed = _recut(sync_db, rows.probe.id, 0.02)
    assert preview["proposed"]["decision_digest"] is not None
    assert preview["proposed"]["decision_digest"] == committed["current"]["decision_digest"]


def test_gpu_identical_recut_keeps_bookkeeping(sync_db, tmp_path, monkeypatch, no_millm):
    """D-T6: a GPU window re-cut whose contract decision is unchanged appends no revision but keeps
    the re-scored arrays' bookkeeping, so the next re-cut needs no GPU."""
    from src.core import config as config_module
    from src.services import probe_monitor_run as run_mod
    from src.services import probe_decision as P

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
    rows = make_rows(sync_db, tmp_path, windows=True)
    _arm_stubs(monkeypatch, run_mod, rows)
    first = run_mod.recut_probe_windows_on_gpu(sync_db, rows.probe.id, target_fpr=0.02, reason="a")
    assert first["committed"] is True
    probe = _fresh(sync_db, rows.probe.id)
    digest, history = probe.decision_digest, list(probe.calibration_history)
    # Drop the bookkeeping, as a probe calibrated before its window arrays were kept has none.
    probe.window_decisions = {w: {k: v for k, v in e.items() if k != "scores_path"}
                              for w, e in probe.window_decisions.items()}
    sync_db.commit()
    again = run_mod.recut_probe_windows_on_gpu(sync_db, rows.probe.id, target_fpr=0.02, reason="b")
    probe = _fresh(sync_db, rows.probe.id)
    assert again["committed"] is False and again["reason"] == "unchanged"
    assert again["bookkeeping_updated"] is True
    assert probe.decision_digest == digest and probe.calibration_history == history
    assert all(e.get("scores_path") for e in probe.window_decisions.values()), (
        "the re-scored arrays' paths were discarded, so the next re-cut pays the GPU again"
    )
    assert P.decision_for_build(sync_db, probe) is not None


def test_gpu_recut_carries_expect_digest(monkeypatch):
    """X6/TI7: the route's `recut_windows=true` branch passes `expect_decision_digest` to the GPU
    task — asserted by payload and call count."""
    import asyncio
    from unittest.mock import MagicMock

    from src.api.v1.endpoints import probe_monitors as ep
    from src.schemas.probe_monitor import RecalibrateRequest

    async def no_refusal(_pid):
        return None

    sent = []
    task = MagicMock(id="task-1")
    monkeypatch.setattr(ep, "refuse_precision_mismatch", no_refusal)
    monkeypatch.setattr(ep, "refuse_render_mismatch", no_refusal)
    monkeypatch.setattr(ep, "gpu_delay", lambda fn, gpu: (lambda **kw: sent.append(kw) or task))
    request = RecalibrateRequest(target_fpr=0.02, preview=False, reason="r",
                                 expect_decision_digest="c" * 64)
    out = asyncio.run(ep.recalibrate_probe_endpoint("pm_x", request, recut_windows=True))
    assert out["task_id"] == "task-1"
    assert sent == [{"probe_id": "pm_x", "target_fpr": 0.02, "reason": "r",
                     "expect_decision_digest": "c" * 64}]
    from src.workers.probe_monitor_tasks import recut_probe_windows

    assert "expect_decision_digest" in __import__("inspect").signature(
        recut_probe_windows.run).parameters


def test_the_gpu_task_threads_the_digest_to_the_service(monkeypatch):
    """The worker passes `expect_decision_digest` through to `recut_probe_windows_on_gpu`."""
    from contextlib import contextmanager

    from src.services import probe_monitor_run as run_mod
    from src.workers import probe_monitor_tasks as tasks

    calls = []
    monkeypatch.setattr(run_mod, "recut_probe_windows_on_gpu",
                        lambda db, pid, **kw: calls.append((pid, kw)) or {"ok": True})
    monkeypatch.setattr(tasks, "_require_row", lambda *a, **k: None)

    @contextmanager
    def fake_db(*_a, **_k):
        yield object()

    monkeypatch.setattr(tasks.recut_probe_windows, "get_db", fake_db, raising=False)
    tasks.recut_probe_windows.run("pm_x", 0.02, "r", expect_decision_digest="c" * 64)
    assert calls == [("pm_x", {"target_fpr": 0.02, "reason": "r",
                               "expect_decision_digest": "c" * 64})]


def _route(monkeypatch, fake):
    """Call the REAL recalibrate route function with the service replaced by `fake` and the
    session factory stubbed (no database: the route's own wiring is what is under test)."""
    import asyncio

    from src.api.v1.endpoints import probe_monitors as ep
    from src.core import database
    from src.schemas.probe_monitor import RecalibrateRequest
    from src.services import probe_recalibration

    monkeypatch.setattr(probe_recalibration, "recalibrate_probe", fake)
    monkeypatch.setattr(database, "SyncSessionLocal", lambda: SimpleNamespace(close=lambda: None))
    request = RecalibrateRequest(target_fpr=0.02, preview=False, expect_decision_digest="c" * 64)
    return asyncio.run(ep.recalibrate_probe_endpoint("pm_x", request, recut_windows=False))


def test_the_sync_route_passes_expect_decision_digest(monkeypatch):
    calls = []
    _route(monkeypatch, lambda db, pid, **kw: calls.append((pid, kw)) or {"ok": True})
    assert len(calls) == 1 and calls[0][1]["expect_decision_digest"] == "c" * 64
    assert calls[0][1]["commit"] is True


def test_decision_changed_reaches_the_caller_with_expected_current_and_next(monkeypatch):
    """TI4: the refusal's `extra` is merged into the HTTP detail beside `code` and `message`."""
    from fastapi import HTTPException

    def refuse(db, pid, **kw):
        raise R.RecalibrationRefused(
            "decision_changed", "moved", status=409,
            extra={"expected": "c" * 64, "current": "d" * 64, "next": [{"tool": "t"}]},
        )

    with pytest.raises(HTTPException) as raised:
        _route(monkeypatch, refuse)
    assert raised.value.status_code == 409
    assert raised.value.detail == {"code": "decision_changed", "message": "moved",
                                   "expected": "c" * 64, "current": "d" * 64,
                                   "next": [{"tool": "t"}]}


def test_the_gpu_arm_refuses_a_stale_expect_digest_BEFORE_the_model_loads(
    sync_db, tmp_path, monkeypatch, no_millm
):
    """C3 on the GPU arm, checked before the twenty minutes of scoring: a caller naming a decision
    other than the one in force is refused with nothing loaded and nothing re-scored."""
    from src.core import config as config_module
    from src.services import probe_monitor_run as run_mod

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
    rows = make_rows(sync_db, tmp_path, windows=True)
    _arm_stubs(monkeypatch, run_mod, rows)
    loads = []
    monkeypatch.setattr(run_mod, "_load_model_for_run", lambda run: loads.append(run) or (None, None, "llama"))
    with pytest.raises(R.RecalibrationRefused) as refused:
        run_mod.recut_probe_windows_on_gpu(sync_db, rows.probe.id, target_fpr=0.02,
                                           expect_decision_digest="e" * 64)
    assert refused.value.code == "decision_changed" and loads == []
    assert refused.value.extra["current"] == _fresh(sync_db, rows.probe.id).decision_digest


def test_a_moved_not_formable_gpu_recut_reports_changed(sync_db, tmp_path, monkeypatch, no_millm):
    """The `unchanged` a result states is the COMMIT's decision, not the preview's: the GPU arm's
    proposal is formed over the swapped table, so only `apply` (on the locked row) can say."""
    from src.core import config as config_module
    from src.services import probe_monitor_run as run_mod

    monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
    rows = make_rows(sync_db, tmp_path, windows=True, hf_repo_id=None)
    _arm_stubs(monkeypatch, run_mod, rows)
    # At the run's own target the re-scored table IS what `propose` re-cuts it to, so a preview
    # formed over the swapped table would call this "unchanged" — while the row's windows (cut from
    # other negatives) do move, and `apply` commits a revision.
    out = run_mod.recut_probe_windows_on_gpu(sync_db, rows.probe.id, target_fpr=0.01, reason="t")
    assert out["committed"] is True and out["unchanged"] is False
    assert [e["revision"] for e in _fresh(sync_db, rows.probe.id).calibration_history] == [1, 2]
