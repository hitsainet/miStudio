"""
Paper-consistent intervention scoring.

The first implementation measured the mean absolute displacement of a
transported activation — a quantity the source paper never reports as an effect
size, computed somewhere the paper never measures. The transport is linear and
`apply_additive` is `h + s*v`, so `J(h + s*v) - J(h) = s*J(v)` and the
activation cancels. Verified on hardware: two unrelated prompts returned
0.01739214 to seven significant figures, with `positions: [5]` reported as
though it had mattered.

The paper instead perturbs, "allow[s] the forward pass to continue", and reports
"the fraction of trials in which the swap places the target-appropriate answer
at the top of the model's output distribution" with "Wilson 95% CIs".

MUTATION CONTROLS:
  * normal-approximation interval instead of Wilson -> "zero successes" fails
  * return (0.0, 0.0) for n == 0                    -> "no trials" fails
  * compare rates instead of intervals              -> "noise is not a finding" fails
  * drop the baseline from the summary              -> "baseline is carried" fails
"""

import pytest

from src.services.jlens_causal import CausalReport, Trial, wilson_interval


class TestWilsonInterval:
    def test_zero_successes_does_not_claim_certainty(self):
        """0/20 is not "0.000 +/- 0.000".

        The normal approximation reports exactly that, claiming certainty from
        the observation that carries least. Wilson keeps an upper bound.
        """
        lo, hi = wilson_interval(0, 20)
        assert lo == 0.0
        assert hi > 0.1, f"upper bound {hi} implies 0/20 nearly rules out an effect"

    def test_all_successes_does_not_claim_certainty_either(self):
        lo, hi = wilson_interval(20, 20)
        assert hi == 1.0 and lo < 0.9

    def test_no_trials_is_MAXIMAL_uncertainty_not_zero(self):
        """n=0 must not read as a measured zero.

        MUTATION CONTROL: return (0.0, 0.0) and this fails — an unrun
        experiment would then be indistinguishable from one that found nothing.
        """
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_the_interval_narrows_as_trials_accumulate(self):
        narrow = wilson_interval(50, 100)
        wide = wilson_interval(5, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_the_interval_stays_inside_zero_and_one(self):
        for s, n in ((0, 3), (3, 3), (1, 4), (99, 100)):
            lo, hi = wilson_interval(s, n)
            assert 0.0 <= lo <= hi <= 1.0


def _report(intervened, control, baseline=None, target=" dog"):
    """Ranks as lists; None means "outside the examined top-k"."""
    base = baseline if baseline is not None else [None] * len(intervened)
    return CausalReport(
        trials=[
            Trial(prompt=f"p{i}", baseline_rank=b, intervened_rank=i_, control_rank=c)
            for i, (b, i_, c) in enumerate(zip(base, intervened, control))
        ],
        target_token=target,
        primitive="additive",
        layers=[9, 10],
        strength=1.0,
    )


class TestTheReportIsTheComparison:
    def test_it_reports_top1_and_top5_for_all_three_arms(self):
        r = _report(
            intervened=[0, 0, 3, 7, None],
            control=[9, None, None, 4, None],
            baseline=[None, None, None, None, None],
        ).summary()
        assert r["intervened_top1"]["hits"] == 2
        assert r["intervened_top5"]["hits"] == 3  # ranks 0, 0, 3
        assert r["control_top1"]["hits"] == 0
        assert r["control_top5"]["hits"] == 1     # rank 4

    def test_the_BASELINE_is_carried_so_a_no_op_cannot_look_like_an_effect(self):
        """An intervention that "achieves" what the model already did moved nothing.

        MUTATION CONTROL: drop the baseline arm from `summary()` and this fails.
        Without it, a prompt set the model already answers correctly manufactures
        a 100% intervened rate.
        """
        r = _report(
            intervened=[0, 0, 0, 0],
            control=[9, 9, 9, 9],
            baseline=[0, 0, 0, 0],
        ).summary()
        assert r["baseline_top1"]["rate"] == 1.0
        assert r["intervened_top1"]["rate"] == r["baseline_top1"]["rate"], (
            "the intervention changed nothing and the report must show it"
        )

    def test_a_clear_effect_separates_from_its_control(self):
        r = _report(
            intervened=[0] * 30,
            control=[None] * 30,
        )
        assert r.separated_from_control() is True

    def test_a_SMALL_GAP_on_FEW_TRIALS_is_not_a_finding(self):
        """6/10 versus 5/10 is noise, and reporting it as an effect is how a
        null becomes a result.

        MUTATION CONTROL: compare the raw rates instead of the intervals and
        this fails.
        """
        r = _report(
            intervened=[0, 0, 0, 0, 0, 0, None, None, None, None],
            control=[0, 0, 0, 0, 0, None, None, None, None, None],
        )
        assert r.separated_from_control() is False
        summary = r.summary()
        # The difference is still REPORTED — hiding it would be its own dishonesty.
        assert summary["excess_top1_over_control"] == pytest.approx(0.1)
        assert summary["separated_from_control"] is False

    def test_an_identical_arm_never_separates(self):
        r = _report(intervened=[0, 1, 2, None] * 5, control=[0, 1, 2, None] * 5)
        assert r.separated_from_control() is False

    def test_a_rank_outside_the_examined_topk_is_None_not_a_big_number(self):
        """`None` and "rank 10000" are different claims.

        A cutoff reported as a rank implies a search that never happened.
        """
        r = _report(intervened=[None, None], control=[None, None]).summary()
        assert r["intervened_top1"]["hits"] == 0
        assert r["intervened_top5"]["hits"] == 0

    def test_the_summary_names_the_target_and_the_recipe(self):
        """A number with no target token or layer list cannot be reproduced."""
        r = _report(intervened=[0], control=[None], target=" cat").summary()
        assert r["target_token"] == " cat"
        assert r["layers"] == [9, 10]
        assert r["primitive"] == "additive"
        assert r["n_trials"] == 1

    def test_one_trial_yields_an_interval_that_admits_it_knows_nothing(self):
        """The honest rendering of a single observation.

        A single prompt was what the old implementation reported, with no
        interval at all.
        """
        r = _report(intervened=[0], control=[None]).summary()
        width = r["intervened_top1"]["ci95_high"] - r["intervened_top1"]["ci95_low"]
        assert width > 0.7, f"a single trial reported a {width:.2f}-wide interval"
        assert r["separated_from_control"] is False

class TestSeparationMustBeATTAINABLEBeforeItIsReported:
    """Below four trials no outcome separates. That is a fact about the sample.

    Both UI paths sent a single prompt, so `separated_from_control: false` was
    the only verdict either could ever produce — and it reads as a finding about
    the direction. The distinction is the whole point, and it shipped with no
    test: replacing the body with `return True` left the suite green.

    MUTATION CONTROLS:
      * `return True`            -> "one trial cannot" fails
      * `return False`           -> "four trials can" fails
      * `>=` instead of `>`      -> "three trials cannot" fails
      * drop the caveat          -> "says WHY" fails
    """

    @staticmethod
    def _report(n: int, *, intervened: int = 0, control: int = 9):
        from src.services.jlens_causal import CausalReport, Trial

        return CausalReport(
            trials=[
                Trial(
                    prompt=f"p{i}",
                    baseline_rank=9,
                    intervened_rank=intervened,
                    control_rank=control,
                )
                for i in range(n)
            ],
            target_token=" Paris",
            primitive="additive",
            layers=[9],
            strength=1.0,
        )

    def test_ONE_trial_cannot_separate_however_perfect_the_arms(self):
        """The best case: every intervened trial a hit, every control a miss."""
        r = self._report(1)
        assert r.separation_attainable() is False
        # AND THE BEST CASE REALLY IS PERFECT, so this is not passing because
        # the fixture happened to be weak.
        s = r.summary()
        assert s["intervened_top1"]["rate"] == 1.0
        assert s["control_top1"]["rate"] == 0.0
        assert s["separated_from_control"] is False

    def test_THREE_trials_cannot_either(self):
        """The boundary from below. 3/3 gives [0.4385, 1.0]; 0/3 [0.0, 0.5615]."""
        assert self._report(3).separation_attainable() is False

    def test_FOUR_trials_can(self):
        """The boundary from above. [0.5101, 1.0] against [0.0, 0.4899]."""
        r = self._report(4)
        assert r.separation_attainable() is True
        # And at four the perfect case DOES separate, so the constant is the
        # real threshold rather than one short of it.
        assert r.summary()["separated_from_control"] is True

    def test_ZERO_trials_cannot(self):
        """`wilson_interval` returns (0, 1) for n=0 — maximal uncertainty."""
        assert self._report(0).separation_attainable() is False

    def test_the_summary_says_WHY_not_merely_that_it_did_not_separate(self):
        s = self._report(1).summary()
        assert s["separation_attainable"] is False
        assert s["min_trials_for_separation"] == 4
        assert "not attainable" in s["caveat"]
        # THE REMEDY, NAMED. "add prompts" is the action; without it the reader
        # is told the result is uninformative and not what to do about it.
        assert "prompts" in s["caveat"]

    def test_an_ATTAINABLE_run_carries_no_sample_size_caveat(self):
        """Otherwise every run would carry it and it would stop meaning anything."""
        s = self._report(8).summary()
        assert s["separation_attainable"] is True
        assert "not attainable" not in (s.get("caveat") or "")

    def test_the_constant_is_DERIVED_and_still_matches_the_arithmetic(self):
        """The docstring states 3 overlaps and 4 does not. Recomputed here.

        A constant whose justification lives only in prose drifts silently the
        moment the interval changes — a different z, a different estimator.
        """
        from src.services.jlens_causal import (
            MIN_TRIALS_FOR_SEPARATION,
            wilson_interval,
        )

        def attainable(n: int) -> bool:
            return wilson_interval(n, n)[0] > wilson_interval(0, n)[1]

        assert not attainable(MIN_TRIALS_FOR_SEPARATION - 1)
        assert attainable(MIN_TRIALS_FOR_SEPARATION)

