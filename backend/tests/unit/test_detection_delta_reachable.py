"""The paired detection comparison must actually be reached.

WHY THIS EXISTS
---------------
`detection_metrics.compare_panels` — the paired bootstrap, MIN_MEANINGFUL_DELTA,
the minimum detectable effect, the dropout counters — had **no production
caller**. Only `tests/unit/test_detection_scoring.py` imported it.

Meanwhile `LabelingTrialService.compare` shipped a LABEL-STRING DIFF as its
verdict: `b_differs` or `identical`. That is not a quality measurement. A
template that renames every feature scores the same as one that fixes every
feature, and a template that changes nothing scores `identical` whether it is
excellent or useless.

So the entire statistical apparatus this repo built for comparing prompt
templates was dead code, and the one thing a user could actually run answered a
different question than the one they were asking. That is this codebase's
signature failure mode — implemented, unit-tested, documented, unreachable.

A capability is not shipped until a test FAILS when its wiring is removed.

MUTATION CONTROLS:
  C60 remove the `_detection_delta` call from `compare`'s return dict
       -> test_the_paired_comparison_is_reached_from_compare
  C61 drop the prompt_version equality check
       -> test_a_moved_ruler_refuses_instead_of_subtracting
  C62 drop the coverage-gap guard
       -> test_a_coverage_gap_confounds_the_verdict
  C63 ignore a failed gate
       -> test_a_failed_gate_is_not_an_absence_of_overlap
"""

from unittest.mock import patch

from src.services.labeling_trial_service import LabelingTrialService

PANEL = "panel_abc"
VERSION = "detv1"


def _payload(scores, *, coverage=None, version=VERSION, gate=None, panel=PANEL,
             midrange=None):
    """A trial payload carrying detection scores for the given features.

    `midrange` adds the second ruler's block. Omitted by default so the
    single-ruler behaviour stays covered: a trial run before the second ruler
    existed carries only `detection`, and must still compare.
    """
    payload = {
        "panel": {"panel_id": panel},
        "results": [
            {"feature_id": f, "status": "ok", "specific": f"name_{f}",
             "category": "semantic"}
            for f in scores
        ],
        "detection": {
            "per_feature": {
                f: {"balanced_accuracy": ba} for f, ba in scores.items()
            },
            "coverage": coverage or {
                "scored": len(scores), "skipped": 0, "panel_size": len(scores),
            },
            "gate": gate if gate is not None else {"passed": True},
            "prompt_version": version,
        },
    }
    if midrange is not None:
        payload["detection_midrange"] = {
            "per_feature": {
                f: {"balanced_accuracy": ba} for f, ba in midrange.items()
            },
            "coverage": {"scored": len(midrange), "skipped": 0,
                         "panel_size": len(midrange)},
            "gate": {"passed": True},
            "prompt_version": version,
        }
    return payload


class TestTheComparisonIsReached:
    def test_the_paired_comparison_is_reached_from_compare(self):
        """C60. Not "compare_panels works" — "compare CALLS it".

        Patched at the point of use and asserted on the PAYLOAD and the CALL
        COUNT: "was called" would pass against a call sending the wrong arms,
        which is how an earlier reachability harness in this repo let three
        mutations through.
        """
        a = _payload({"f1": 0.60, "f2": 0.55})
        b = _payload({"f1": 0.80, "f2": 0.75})

        with patch(
            "src.services.detection_metrics.compare_panels",
            wraps=__import__(
                "src.services.detection_metrics", fromlist=["compare_panels"]
            ).compare_panels,
        ) as spy:
            out = LabelingTrialService.compare(a, b)

        # ONE CALL PER RULER, and no more.
        #
        # Two rulers means two bootstraps. A third would mean the verdict is
        # recomputing deltas the payload already reports, which is both wasted
        # work and a way for the two to drift apart.
        assert spy.call_count == 1, (
            f"compare_panels called {spy.call_count} times for a payload with "
            f"ONE ruler; a run predating the second ruler must still compare"
        )
        baseline, candidate = spy.call_args_list[0].args
        assert baseline == {"f1": 0.60, "f2": 0.55}, (
            "the BASELINE arm's scores were not passed as the baseline"
        )
        assert candidate == {"f1": 0.80, "f2": 0.75}, (
            "the CANDIDATE arm's scores were not passed as the candidate"
        )

        assert out["detection_delta"] is not None
        assert out["detection_delta"]["compared"] == 2
        assert out["detection_delta"]["mean_delta"] > 0

    def test_a_trial_without_detection_scores_reports_not_measured(self):
        """Negative control: None means "not measured", not "no difference".

        A `_detection_delta` that returned a dict unconditionally would pass the
        test above and destroy the distinction between an unmeasured trial and
        an inconclusive one.
        """
        a = {"panel": {"panel_id": PANEL},
             "results": [{"feature_id": "f1", "status": "ok",
                          "specific": "x", "category": "semantic"}]}
        b = {"panel": {"panel_id": PANEL},
             "results": [{"feature_id": "f1", "status": "ok",
                          "specific": "y", "category": "semantic"}]}

        out = LabelingTrialService.compare(a, b)
        assert out["detection_delta"] is None
        # The string diff still works — this change adds a verdict, it does not
        # replace the existing one.
        assert out["verdict"] == "b_differs"


class TestTheRulerMustNotMove:
    def test_a_moved_ruler_refuses_instead_of_subtracting(self):
        """C61. PADR IDL-48: an editable ruler invalidates every prior score.

        Two arms graded under different detection prompts are two different
        measurements. Subtracting them yields a number with no meaning, and a
        number with no meaning is worse than a refusal because someone will act
        on it.
        """
        a = _payload({"f1": 0.60}, version="detv1")
        b = _payload({"f1": 0.90}, version="detv2")

        delta = LabelingTrialService.compare(a, b)["detection_delta"]

        assert delta["verdict"] is None
        assert delta["compared"] == 0
        assert "different rulers" in delta["reason"]

    def test_a_matching_ruler_still_compares(self):
        """Negative control for C61."""
        a = _payload({"f1": 0.60}, version="detv1")
        b = _payload({"f1": 0.90}, version="detv1")
        delta = LabelingTrialService.compare(a, b)["detection_delta"]
        assert delta["compared"] == 1


class TestCoverageConfounds:
    def test_a_coverage_gap_confounds_the_verdict(self):
        """C62. Scoring fewer features and scoring better are not the same.

        An arm that refuses the hard features is graded on the easy ones it
        chose to answer. Without this guard, refusing more looks like improving.
        """
        a = _payload(
            {f"f{i}": 0.50 for i in range(10)},
            coverage={"scored": 10, "skipped": 0, "panel_size": 10},
        )
        # b answered only 5 of 10 — and scored better on those five.
        b = _payload(
            {f"f{i}": 0.90 for i in range(5)},
            coverage={"scored": 5, "skipped": 5, "panel_size": 10},
        )

        delta = LabelingTrialService.compare(a, b)["detection_delta"]

        assert delta["confounded_by_coverage"] is True
        assert delta["verdict"] is None, (
            "a 50% coverage gap produced a verdict; the delta measures which "
            "features were answered, not how well"
        )
        assert delta["coverage_gap"] == 0.5

    def test_equal_coverage_is_not_confounded(self):
        """Negative control for C62: the guard must not refuse everything."""
        a = _payload({f"f{i}": 0.50 for i in range(10)})
        b = _payload({f"f{i}": 0.90 for i in range(10)})

        delta = LabelingTrialService.compare(a, b)["detection_delta"]
        assert delta["confounded_by_coverage"] is False
        assert delta["compared"] == 10


class TestAFailedGateIsNotAnAbsence:
    def test_a_failed_gate_is_not_an_absence_of_overlap(self):
        """C63. "The judge cannot grade" must not read as "no data".

        `score_panel` withholds scores entirely on a failed gate, so
        `per_feature` is empty — and an empty overlap would otherwise report
        "no overlapping scored features", blaming the panel for the judge's
        incapacity. That is the exact misattribution the gate exists to prevent.
        """
        a = _payload({"f1": 0.60})
        b = _payload({}, gate={"passed": False, "reason": "judge too weak"})

        delta = LabelingTrialService.compare(a, b)["detection_delta"]

        assert delta["verdict"] is None
        assert "sanity gate" in delta["reason"], (
            f"a failed gate was reported as {delta['reason']!r}, which blames "
            f"the panel for the judge"
        )


class TestEveryReturnPathCarriesTheKey:
    """`detection_delta` must be present on refusals too, with the same keys.

    WHY THIS EXISTS
    ---------------
    `compare`'s docstring promises ONE SHAPE FOR EVERY RETURN PATH, and cites
    `compare_panels`' own history: dropout counters were added to the success
    branch only, so a consumer got a KeyError *precisely when the comparison was
    refused* — the moment it most needed them.

    Round 1 found the promise unguarded on both axes. The three refusal branches
    of `compare` carried the key and nothing asserted it; and `_detection_delta`'s
    own refusals returned three keys where a verdict returns twelve, while the
    MCP tool advertises `mean_delta` / `ci` / `minimum_detectable_effect` as part
    of that payload. Deleting the key from a refusal branch left the suite green,
    recreating the exact KeyError the docstring is about.

    MUTATION CONTROLS:
      C94 delete `"detection_delta": None` from any refusal branch of `compare`
           -> test_every_compare_refusal_carries_the_key
      C95 return a short dict from a `_detection_delta` refusal
           -> test_a_refused_delta_has_the_same_keys_as_a_verdict
    """

    #: DERIVED FROM A REAL VERDICT, not hand-listed.
    #:
    #: The first version listed nine keys and omitted `baseline_total`,
    #: `candidate_total` and `dropped` — the very dropout counters the
    #: docstring above cites as the original sin. Deleting them from `_refused`
    #: left the guard green. Taking the key set from an actual verdict means a
    #: key added to the success branch is required of the refusals too, which
    #: is the invariant, rather than a list that has to be remembered.
    #:
    #: `coverage_gap` / `confounded_by_coverage` are excluded: they exist only
    #: when both arms reported coverage, which a refusal by definition may not
    #: have.
    _COVERAGE_ONLY = {"coverage_gap", "confounded_by_coverage"}

    def test_every_compare_refusal_carries_the_key(self):
        """C94. All three refusal branches, driven for real."""
        scored = _payload({"f1": 0.6})

        # 1. different panels
        other = _payload({"f1": 0.9}, panel="panel_other")
        out = LabelingTrialService.compare(scored, other)
        assert out["comparable"] is False
        assert "detection_delta" in out, (
            "the different-panels refusal omits detection_delta; a consumer "
            "reading it gets a KeyError exactly when the comparison is refused"
        )

        # 2. no overlapping features
        empty = {"panel": {"panel_id": PANEL}, "results": [],
                 "detection": scored["detection"]}
        out = LabelingTrialService.compare(scored, empty)
        assert "detection_delta" in out

        # 3. every overlapping feature errored
        errored = {
            "panel": {"panel_id": PANEL},
            "results": [{"feature_id": "f1", "status": "error"}],
            "detection": scored["detection"],
        }
        out = LabelingTrialService.compare(scored, errored)
        assert out["verdict"] == "inconclusive"
        assert "detection_delta" in out

    def test_a_refused_delta_has_the_same_keys_as_a_verdict(self):
        """C95. A refusal is a different ANSWER, not a different SHAPE."""
        verdict = LabelingTrialService.compare(
            _payload({"f1": 0.6, "f2": 0.5}),
            _payload({"f1": 0.8, "f2": 0.7}),
        )["detection_delta"]
        promised = set(verdict) - self._COVERAGE_ONLY
        # SELF-CHECK: if a verdict stopped carrying the statistics, every
        # assertion below would pass vacuously.
        for essential in ("mean_delta", "ci", "minimum_detectable_effect",
                          "dropped", "baseline_total", "candidate_total"):
            assert essential in promised, (
                f"a verdict no longer reports {essential!r}; this guard is inert"
            )

        for label, a, b in (
            ("moved ruler",
             _payload({"f1": 0.6}, version="v1"),
             _payload({"f1": 0.9}, version="v2")),
            ("failed gate",
             _payload({"f1": 0.6}),
             _payload({}, gate={"passed": False, "reason": "weak judge"})),
        ):
            refused = LabelingTrialService.compare(a, b)["detection_delta"]
            missing = sorted(promised - set(refused))
            assert not missing, (
                f"the {label} refusal omits {missing}; the MCP tool documents "
                f"those as part of this payload, so every reader breaks on "
                f"exactly the cases this branch exists to report"
            )
            assert refused["verdict"] is None
            assert refused["reason"]

    def test_a_refusal_reports_what_was_actually_dropped(self):
        """The VALUES, not just the keys.

        The guard added when `dropped: 0` was first corrected asserted key
        PRESENCE only, so reverting the computation to the literal `0` left the
        suite green — the fix was unprotected by the test written for it.

        And the first correction was itself wrong: it used the SCORED sets on
        both sides, while `compare_panels` counts everything PRESENT that did
        not make the overlap. Since scored is a subset of present, that
        undercounted — in the flattering direction, the same shape as this
        arc's filter_suppressed_neurons bug.
        """
        # Arm a scored 2 of 3 present; arm b was never scored at all.
        det_a = {
            "per_feature": {
                "f1": {"balanced_accuracy": 0.6},
                "f2": {"balanced_accuracy": 0.7},
                "f3": {"balanced_accuracy": None},   # attempted, ungradeable
            },
            "coverage": {"scored": 2, "skipped": 1, "panel_size": 3},
            "gate": {"passed": True},
            "prompt_version": VERSION,
        }
        det_b = {
            "per_feature": {},
            "coverage": {"scored": 0, "skipped": 3, "panel_size": 3},
            "gate": {"passed": False, "reason": "weak judge"},
            "prompt_version": VERSION,
        }
        a = {"panel": {"panel_id": PANEL},
             "results": [{"feature_id": "f1", "status": "ok",
                          "specific": "x", "category": "semantic"}],
             "detection": det_a}
        b = {"panel": {"panel_id": PANEL},
             "results": [{"feature_id": "f1", "status": "ok",
                          "specific": "y", "category": "semantic"}],
             "detection": det_b}

        delta = LabelingTrialService.compare(a, b)["detection_delta"]

        assert delta["dropped"] == 3, (
            f"reported dropped={delta['dropped']}; three features were on the "
            f"panel and none made the comparison"
        )
        assert delta["baseline_total"] == 3, (
            "baseline_total must count features PRESENT, as compare_panels "
            "does — reporting the scored count here makes one key mean two "
            "different things depending on the branch"
        )
        assert delta["candidate_total"] == 0

    def test_an_unscored_arm_is_not_called_a_moved_ruler(self):
        """"Never measured" and "measured with a different ruler" differ.

        `_score_detection`'s early returns carry no `prompt_version` at all, so
        comparing against an unscorable arm reported a version mismatch —
        blaming a moved ruler for an arm that was never scored. That is the same
        confusion the scorer elsewhere works hard to avoid: "no score" and "a
        bad score" are different facts.
        """
        a = _payload({"f1": 0.6})
        never_scored = {
            "panel": {"panel_id": PANEL},
            "results": [{"feature_id": "f1", "status": "ok",
                         "specific": "x", "category": "semantic"}],
            "detection": {"reason": "no feature carried a testable label"},
        }

        delta = LabelingTrialService.compare(a, never_scored)["detection_delta"]

        assert delta["verdict"] is None
        assert "never measured" in delta["reason"], (
            f"an unscored arm was reported as {delta['reason']!r}"
        )
        assert "different rulers" not in delta["reason"]


class TestTheTwoRulerVerdict:
    """One ruler cannot detect the thing the experiment is testing for.

    Detection asks whether a label lets a judge pick the feature's passages out
    of a pool. Over TOP-K positives that is maximised by the NARROWEST label
    still covering the top decile — and a wider example spread exists to produce
    labels broader than the top decile. On one ruler the instrument and the
    hypothesis point in opposite directions, and the experiment can only return
    "worse" or "no difference".

    Worked case: a feature whose top decile is legal boilerplate and whose ranks
    20-100 are formal institutional prose. `legal_contract_language` is narrower
    and false of the feature; `formal_institutional_register` is truer and fires
    on more of a general corpus, so it scores ~0.175 WORSE on top-K. A
    single-ruler experiment says "do not ship" about the better label.

    Pinning to mid-range instead would just invert the bias. Both, with
    asymmetric thresholds, is the only formulation where "truer across the
    range" is rewardable while an arm that stopped describing the strongest
    evidence is still caught.

    MUTATION CONTROLS:
      C113 make the verdict read only the top-K delta
            -> test_a_midrange_win_with_a_tolerable_top_k_loss_ships
      C114 drop the top-K tolerance check
            -> test_a_large_top_k_loss_is_not_shipped
    """

    def test_a_midrange_win_with_a_tolerable_top_k_loss_ships(self):
        """C113. The case a single top-K ruler would reject outright."""
        a = _payload({f"f{i}": 0.80 for i in range(12)},
                     midrange={f"f{i}": 0.55 for i in range(12)})
        b = _payload({f"f{i}": 0.79 for i in range(12)},
                     midrange={f"f{i}": 0.75 for i in range(12)})

        out = LabelingTrialService.compare(a, b)
        verdict = out["two_ruler_verdict"]

        assert verdict["verdict"] == "ship", verdict
        assert verdict["midrange_mean_delta"] > 0
        # And a top-K-only reading would have called this a loss.
        assert out["detection_delta"]["mean_delta"] < 0

    def test_a_large_top_k_loss_is_not_shipped(self):
        """C114. A broader label is not automatically a better one.

        Giving up the top decile entirely is a real regression, and the
        mid-range ruler alone cannot see it.
        """
        a = _payload({f"f{i}": 0.95 for i in range(12)},
                     midrange={f"f{i}": 0.55 for i in range(12)})
        b = _payload({f"f{i}": 0.60 for i in range(12)},
                     midrange={f"f{i}": 0.80 for i in range(12)})

        verdict = LabelingTrialService.compare(a, b)["two_ruler_verdict"]
        assert verdict["verdict"] == "traded_top_k", verdict

    def test_a_midrange_regression_is_reported_as_worse(self):
        a = _payload({f"f{i}": 0.80 for i in range(12)},
                     midrange={f"f{i}": 0.80 for i in range(12)})
        b = _payload({f"f{i}": 0.80 for i in range(12)},
                     midrange={f"f{i}": 0.50 for i in range(12)})

        verdict = LabelingTrialService.compare(a, b)["two_ruler_verdict"]
        assert verdict["verdict"] == "worse", verdict

    def test_a_single_ruler_run_yields_no_two_ruler_verdict(self):
        """Negative control, and back-compatibility.

        A trial run before the second ruler existed carries only `detection`.
        It must still compare on that ruler and simply decline the combined
        verdict, rather than inventing one from half the evidence.
        """
        a = _payload({"f1": 0.6, "f2": 0.5})
        b = _payload({"f1": 0.8, "f2": 0.7})

        out = LabelingTrialService.compare(a, b)
        assert out["detection_delta"]["compared"] == 2
        assert out["detection_delta_midrange"] is None
        assert out["two_ruler_verdict"]["verdict"] is None
        assert "no comparison" in out["two_ruler_verdict"]["reason"]


class TestThePanelResolutionPreflight:
    """A threshold below the panel's resolution is unreachable.

    The plan pre-registers a 0.02 improvement as the ship criterion. On ~80
    features the real resolution is nearer 0.047, so the experiment could not
    have returned a positive result at all — and a null could not be
    distinguished from "underpowered".

    Running the baseline TWICE measures it: the true delta is zero by
    construction, so the spread of per-feature differences is pure noise.

    MUTATION CONTROL:
      C115 have panel_resolution read the mean delta instead of the MDE
            -> test_the_resolution_comes_from_a_null_comparison
    """

    def test_the_resolution_comes_from_a_null_comparison(self):
        """Two runs of the same arm: delta ~0, resolution > 0."""
        import random

        rng = random.Random(20260910)
        # Same underlying quality, different draws — i.e. judge noise.
        a = _payload({f"f{i}": 0.70 + rng.gauss(0, 0.08) for i in range(30)})
        b = _payload({f"f{i}": 0.70 + rng.gauss(0, 0.08) for i in range(30)})

        out = LabelingTrialService.compare(a, b)
        resolution = LabelingTrialService.panel_resolution(out)

        assert resolution is not None
        assert resolution > 0, (
            "a null comparison reported infinite resolution; that is a claim "
            "no sample supports"
        )
        # The point of the pre-flight: the resolution is materially larger than
        # a naively pre-registered 0.02.
        assert resolution > 0.02, (
            f"resolution {resolution:.4f} — if this really is below the "
            f"pre-registered threshold the panel is adequately powered, which "
            f"would be worth knowing too"
        )
        assert out["detection_delta"]["verdict"] == "indistinguishable", (
            "a null comparison produced a verdict; the arms are the same arm"
        )

    def test_it_returns_none_when_there_is_nothing_to_measure(self):
        """No comparison, no resolution — rather than a reassuring zero."""
        assert LabelingTrialService.panel_resolution({}) is None
        assert LabelingTrialService.panel_resolution(
            {"detection_delta": None}) is None

    def test_the_resolution_is_not_the_mean_delta(self):
        """C115. They are different quantities and one is near zero here."""
        a = _payload({f"f{i}": 0.70 for i in range(12)})
        b = _payload({f"f{i}": 0.70 for i in range(12)})

        out = LabelingTrialService.compare(a, b)
        # Zero variance: there is no evidence about resolution, so there is no
        # number — rather than 0.000, which would claim infinite resolution.
        assert LabelingTrialService.panel_resolution(out) is None
        assert out["detection_delta"]["mean_delta"] == 0.0
