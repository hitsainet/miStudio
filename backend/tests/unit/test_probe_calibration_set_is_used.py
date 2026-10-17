"""The configured calibration set actually places the threshold.

⚠ IT DID NOT. `calibration_dataset_id` was accepted by the API, validated against the training
set, stored on the run row, and read by exactly ONE consumer — the export builder, which named it
in `decision.calibration`. The calibrating stage scored the training set's own validation split
and hardcoded `source="validation_negatives"`.

So a run configured with a calibration set produced a threshold from plain training prose while
its exported document told a consumer the operating point came from the calibration set. The
document contradicted itself: `decision.calibration` naming one distribution beside a
`threshold_source` naming another.

Found 2026-09-28 while chasing a real false positive — a `high-stakes` probe fired on
*"What is the capital of France?"* — whose cause is exactly this: the 1% budget was spent on
plain-prose negatives and then applied to chat traffic. A false-positive rate is a property of the
negative distribution the monitor will actually see.

The three things asserted here are the three that were wrong: the set is READ, the source SAYS SO,
and the export cannot claim a calibration it did not use.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from src.schemas.probe_monitor import (
    DEFAULT_CALIBRATION_MAX_ROWS,
    ProbeRunConfig,
)
from src.services.probe_monitor_run import subsample_indices


class TestTheSampleIsSeededRandomNotTheFirstRows:
    """⚠ THE CALIBRATION CORPUS IS SOURCE-ORDERED, so head-N calibrates on one source.

    OpenHermes-2.5 is written one source at a time — 14 of its 15 labelled sources occupy a
    single contiguous row range. This estate has already shipped the head-N version of this
    mistake: an extraction read 7,000 of 189,087 blocks and saw 2 of 15 sources, never reaching
    the 18.2% that is glaive-code-assist nor the 49.6% unlabelled remainder.
    """

    def test_it_returns_None_when_everything_fits(self):
        """`None` means "take all of it" — not an empty sample, which would score nothing."""
        assert subsample_indices(100, 2000, seed=1337) is None
        assert subsample_indices(2000, 2000, seed=1337) is None

    def test_it_draws_the_requested_count(self):
        keep = subsample_indices(1_000_000, 2000, seed=1337)
        assert keep is not None
        assert len(keep) == 2000
        assert len(set(keep)) == 2000, "the sample must not repeat a row"

    def test_it_is_ascending(self):
        """So the sample indexes the underlying columns in one forward pass."""
        keep = subsample_indices(1_000_000, 2000, seed=1337)
        assert keep == sorted(keep)

    def test_it_is_reproducible_from_the_seed(self):
        assert subsample_indices(1_000_000, 500, seed=1337) == subsample_indices(
            1_000_000, 500, seed=1337
        )

    def test_a_different_seed_gives_a_different_sample(self):
        assert subsample_indices(1_000_000, 500, seed=1337) != subsample_indices(
            1_000_000, 500, seed=7
        )

    def test_IT_IS_NOT_THE_FIRST_N(self):
        """⚠ The assertion that matters. A source-ordered corpus sampled head-N calibrates on
        whatever source happens to sit at the top of the file."""
        keep = subsample_indices(1_000_000, 2000, seed=1337)
        assert keep != list(range(2000))
        # And it must actually SPREAD: a sample confined to the first 1% of a source-ordered
        # corpus is the defect even if it is not literally range(N).
        assert max(keep) > 900_000, f"sample stops at {max(keep)} of 1,000,000"
        assert min(keep) < 100_000

    def test_it_spans_the_corpus_roughly_evenly(self):
        """Each tenth of a 1M-row corpus should get roughly a tenth of a 5,000-row sample."""
        keep = subsample_indices(1_000_000, 5000, seed=1337)
        deciles = [0] * 10
        for i in keep:
            deciles[min(i // 100_000, 9)] += 1
        assert all(350 < c < 650 for c in deciles), f"uneven coverage: {deciles}"

    def test_a_nonpositive_limit_raises(self):
        with pytest.raises(ValueError, match="limit must be positive"):
            subsample_indices(100, 0, seed=1)


class TestTheCapMustAffordTheTarget:
    """⚠ A cap too small for the target FPR places NO threshold — silently.

    `calibrate` allows `int(target_fpr * n)` false positives and returns `threshold=None`
    ("fire on nothing") when that rounds to zero. Legitimate when the negatives say so, a
    misconfiguration when it is just arithmetic.
    """

    def test_the_default_affords_the_default(self):
        config = ProbeRunConfig(stride=5)
        assert int(config.target_fpr * config.calibration_max_rows) >= 1
        assert config.calibration_max_rows == DEFAULT_CALIBRATION_MAX_ROWS

    @pytest.mark.parametrize(
        "target,cap",
        # Each is affordable by the field bounds (cap >= 100) but unaffordable for its target,
        # so the model_validator is what refuses it rather than the `ge=100` floor.
        # int(target * cap) must be 0: 0.2, 0.999 and 0.5 respectively.
        [(0.0001, 2000), (0.001, 999), (0.005, 100)],
    )
    def test_an_unaffordable_combination_is_refused(self, target, cap):
        with pytest.raises(ValueError, match="cannot afford"):
            ProbeRunConfig(stride=5, target_fpr=target, calibration_max_rows=cap)

    def test_the_refusal_names_the_number_of_rows_needed(self):
        """An error that does not say what to do instead is a dead end."""
        with pytest.raises(ValueError) as exc:
            ProbeRunConfig(stride=5, target_fpr=0.001, calibration_max_rows=500)
        assert str(math.ceil(1 / 0.001)) in str(exc.value)

    @pytest.mark.parametrize("target,cap", [(0.01, 100), (0.001, 1000), (0.05, 100)])
    def test_the_boundary_is_allowed(self, target, cap):
        """Exactly affordable must pass — an off-by-one here would refuse valid configs."""
        assert ProbeRunConfig(stride=5, target_fpr=target, calibration_max_rows=cap)


class TestTheExportCannotClaimACalibrationItDidNotUse:
    """The half that made the old state worse than a missing feature."""

    @staticmethod
    def _gate(threshold_source, calibration_dataset_id):
        """The condition as written in `probe_definition_builder`, exercised directly."""
        return bool(calibration_dataset_id and threshold_source == "calibration_set")

    def test_it_is_named_when_it_was_used(self):
        assert self._gate("calibration_set", "pmd_x") is True

    def test_it_is_NOT_named_when_the_threshold_came_from_validation_negatives(self):
        """⚠ The exact reported state: a run carrying a calibration id whose threshold came
        from somewhere else."""
        assert self._gate("validation_negatives", "pmd_x") is False

    def test_it_is_not_named_when_there_is_no_calibration_set(self):
        assert self._gate("validation_negatives", None) is False

    def test_the_builder_gates_on_threshold_source_by_AST(self):
        """⚠ Asserted on the SOURCE OF THE CONDITION, because the value is assembled inside a
        large pydantic constructor that a unit test cannot cheaply reach. Reads the AST for a
        comparison against the literal, not the text for a name — the comment above the code
        mentions `threshold_source` repeatedly and a substring search would match it.
        """
        import ast
        import inspect

        from src.services import probe_definition_builder

        tree = ast.parse(inspect.getsource(probe_definition_builder))
        compares = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and node.left.attr == "threshold_source"
            and any(
                isinstance(c, ast.Constant) and c.value == "calibration_set"
                for c in node.comparators
            )
        ]
        assert compares, (
            "the export does not compare probe.threshold_source against 'calibration_set', so "
            "decision.calibration can again name a set the threshold did not come from"
        )


class TestTheDecisionItself:
    """⚠ BEHAVIOURAL, because the scrape version FAILED OPEN.

    The first guard here read `execute_probe_run`'s AST for a call to the calibration scorer.
    Replacing `if calibration_id:` with `if False:` — which restores the original defect exactly —
    left the call present inside a dead branch and all 23 tests passed. So the branch moved into
    `resolve_calibration_negatives`, whose behaviour is run: which producer fired, and what source
    came back.
    """

    @staticmethod
    def _producers():
        """Two recording producers, so WHICH path ran is observable rather than inferred."""
        calls = []

        def from_set():
            calls.append("calibration_set")
            return [1.0, 2.0, 3.0], 3

        def from_validation():
            calls.append("validation_negatives")
            return [9.0]

        return calls, from_set, from_validation

    def test_a_configured_set_is_used_and_named(self):
        from src.services.probe_monitor_run import resolve_calibration_negatives

        calls, from_set, from_validation = self._producers()
        scores, source = resolve_calibration_negatives(
            "pmd_abc",
            from_calibration_set=from_set,
            from_validation_negatives=from_validation,
        )
        assert calls == ["calibration_set"], (
            f"the configured calibration set was not read; producers called: {calls}"
        )
        assert scores == [1.0, 2.0, 3.0]
        assert source == "calibration_set"

    def test_without_one_it_falls_back_and_says_so(self):
        from src.services.probe_monitor_run import resolve_calibration_negatives

        calls, from_set, from_validation = self._producers()
        scores, source = resolve_calibration_negatives(
            None,
            from_calibration_set=from_set,
            from_validation_negatives=from_validation,
        )
        assert calls == ["validation_negatives"]
        assert scores == [9.0]
        assert source == "validation_negatives"

    def test_the_unused_producer_is_NOT_called(self):
        """⚠ Both paths score a model. Calling both would silently double the calibration cost
        of every run, and the wasted pass is over a corpus sample."""
        from src.services.probe_monitor_run import resolve_calibration_negatives

        calls, from_set, from_validation = self._producers()
        resolve_calibration_negatives(
            "pmd_abc",
            from_calibration_set=from_set,
            from_validation_negatives=from_validation,
        )
        assert "validation_negatives" not in calls

    @pytest.mark.parametrize("empty", ["", None])
    def test_an_empty_id_is_not_a_calibration_set(self, empty):
        """An empty string is absence, not a set named ''. Treating it as present would send
        the scorer after a view that cannot exist."""
        from src.services.probe_monitor_run import resolve_calibration_negatives

        calls, from_set, from_validation = self._producers()
        _scores, source = resolve_calibration_negatives(
            empty,
            from_calibration_set=from_set,
            from_validation_negatives=from_validation,
        )
        assert source == "validation_negatives"
        assert calls == ["validation_negatives"]

    def test_the_scores_and_the_source_cannot_disagree(self):
        """⚠ The invariant the export depends on. `probe_definition_builder` gates
        `decision.calibration` on `threshold_source`, so a path returning one distribution's
        scores under the other's label would put a false provenance in the document."""
        from src.services.probe_monitor_run import resolve_calibration_negatives

        for cal_id, expected_scores, expected_source in (
            ("pmd_abc", [1.0, 2.0, 3.0], "calibration_set"),
            (None, [9.0], "validation_negatives"),
        ):
            _calls, from_set, from_validation = self._producers()
            scores, source = resolve_calibration_negatives(
                cal_id,
                from_calibration_set=from_set,
                from_validation_negatives=from_validation,
            )
            assert (scores, source) == (expected_scores, expected_source)


class TestTheStageCallsTheDecision:
    """The wiring half only. The BEHAVIOUR is covered above, which is the division that was
    missing when a scrape was doing both jobs."""

    @staticmethod
    def _calls_in_stage():
        import ast
        import inspect

        from src.services import probe_monitor_run

        tree = ast.parse(inspect.getsource(probe_monitor_run._calibrate_probes))
        return {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    def test_the_stage_calls_it(self):
        assert "resolve_calibration_negatives" in self._calls_in_stage()

    def test_the_scan_can_see_a_known_call(self):
        assert "calibrate" in self._calls_in_stage()


