"""032 task 1.7 — probe evaluation metrics.

Four things are pinned, and each is pinned against something OUTSIDE this module,
because a metric checked only against itself is a tautology:

  * **AUROC against sklearn**, on random data and on the tie-heavy cases where a
    hand-rolled rank implementation usually parts company with the definition;
  * **the ROC curve against hand arithmetic** on six points anyone can verify by
    reading;
  * **the tie rule** for the operating point, which the task fixes as
    "ties → lower FPR";
  * **the refusal** below MIN_PER_CLASS, which must be a reason and not a number.
"""
import pytest
from sklearn.metrics import roc_auc_score, roc_curve

from src.services.probe_monitor_metrics import (
    MIN_PER_CLASS,
    RocPoint,
    auroc,
    auroc_ci,
    best_point_within,
    check_scoreable,
    evaluate,
    length_bands,
    mean_auroc_across_sets,
    operating_point_at_fpr,
    recall_at_fpr,
    roc_points,
    threshold_at_fpr,
)


def _labelled(seed: int, n: int = 200, separation: float = 1.0):
    """Scores with a known signal, so AUROC is well above chance but not 1.0."""
    import random

    rng = random.Random(seed)
    labels = [1 if i % 2 == 0 else 0 for i in range(n)]
    scores = [rng.gauss(separation if y == 1 else 0.0, 1.0) for y in labels]
    return scores, labels


# ── AUROC against sklearn ─────────────────────────────────────────────────────

class TestAurocMatchesSklearn:

    @pytest.mark.parametrize("seed", [1, 2, 3, 17, 99])
    def test_on_continuous_scores(self, seed):
        scores, labels = _labelled(seed)
        assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)

    def test_with_heavy_ties(self):
        """Ties are where a rank implementation goes wrong, and a saturating probe
        produces them exactly when it looks best."""
        scores = [0.5] * 10 + [0.9] * 10 + [0.5] * 10 + [0.1] * 10
        labels = [1] * 20 + [0] * 20
        assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)

    def test_when_every_score_is_identical(self):
        """AUROC is exactly 0.5 — every pair is a tie, none is a win."""
        scores = [0.7] * 40
        labels = [1] * 20 + [0] * 20
        assert auroc(scores, labels) == pytest.approx(0.5)
        assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))

    def test_perfect_and_inverted_separation(self):
        scores, labels = [0.0, 0.1, 0.9, 1.0], [0, 0, 1, 1]
        assert auroc(scores, labels) == pytest.approx(1.0)
        assert auroc(scores, [1, 1, 0, 0]) == pytest.approx(0.0)

    def test_a_missing_class_returns_none_rather_than_a_number(self):
        assert auroc([0.1, 0.2], [1, 1]) is None
        assert auroc([0.1, 0.2], [0, 0]) is None

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="against"):
            auroc([0.1, 0.2], [1])


# ── the ROC curve, by hand ────────────────────────────────────────────────────

class TestRocCurveByHand:
    """Six points, three positive and three negative, verifiable by reading:

        score  0.9  0.8  0.7  0.6  0.5  0.4
        label   1    0    1    1    0    0
    """

    SCORES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    LABELS = [1, 0, 1, 1, 0, 0]

    def test_the_curve_is_what_arithmetic_says(self):
        got = [(p.fpr, p.tpr) for p in roc_points(self.SCORES, self.LABELS)]
        assert got == pytest.approx([
            (0.0, 0.0),        # threshold above everything
            (0.0, 1 / 3),      # >= 0.9  : TP 1
            (1 / 3, 1 / 3),    # >= 0.8  : FP 1
            (1 / 3, 2 / 3),    # >= 0.7  : TP 2
            (1 / 3, 1.0),      # >= 0.6  : TP 3
            (2 / 3, 1.0),      # >= 0.5  : FP 2
            (1.0, 1.0),        # >= 0.4  : FP 3
        ])

    def test_it_starts_at_the_origin_so_a_caller_need_not_add_it(self):
        first = roc_points(self.SCORES, self.LABELS)[0]
        assert (first.fpr, first.tpr) == (0.0, 0.0)

    def test_the_points_match_sklearns_curve(self):
        """`drop_intermediate=False` is required: sklearn drops collinear points by
        default, which would make a 7-point curve look like a 5-point disagreement."""
        fpr, tpr, _ = roc_curve(self.LABELS, self.SCORES, drop_intermediate=False)
        theirs = {(round(f, 12), round(t, 12)) for f, t in zip(fpr.tolist(), tpr.tolist())}
        mine = {(round(p.fpr, 12), round(p.tpr, 12)) for p in roc_points(self.SCORES, self.LABELS)}
        assert mine == theirs

    def test_a_tie_group_is_one_point_not_several(self):
        """Splitting a tie would invent operating points no threshold can reach."""
        points = roc_points([0.5, 0.5, 0.5, 0.1], [1, 1, 0, 0])
        # The origin's threshold is None ("above every score"), not a value.
        thresholds = [p.threshold for p in points if p.threshold is not None]
        assert thresholds == [0.5, 0.1]


# ── the operating point and its tie rule ──────────────────────────────────────

class TestOperatingPoint:

    def test_it_never_exceeds_the_budget(self):
        scores, labels = _labelled(5, n=400)
        for target in (0.01, 0.05, 0.10):
            point = operating_point_at_fpr(scores, labels, target)
            assert point is not None
            assert point.realised_fpr <= target + 1e-12

    def test_it_reports_what_was_SPENT_not_what_was_asked_for(self):
        """A realised FPR of 0 against a 10% budget must not be reported as 10%."""
        point = operating_point_at_fpr([0.9, 0.8, 0.1, 0.05], [1, 1, 0, 0], 0.10)
        assert point.target_fpr == 0.10
        assert point.realised_fpr == pytest.approx(0.0)
        assert point.recall == pytest.approx(1.0)

    def test_ties_on_recall_go_to_the_LOWER_fpr(self):
        """The rule task 1.6 fixes. Two thresholds reach recall 1.0 inside a 50%
        budget: one spends 0 false positives, one spends 1. Take the cheaper."""
        scores = [0.9, 0.8, 0.4, 0.3]
        labels = [1, 1, 0, 0]
        point = operating_point_at_fpr(scores, labels, 0.5)
        assert point.recall == pytest.approx(1.0)
        assert point.realised_fpr == pytest.approx(0.0), (
            "a threshold spending a false positive was chosen over one spending none"
        )
        assert point.threshold == pytest.approx(0.8)

    def test_the_choice_does_not_depend_on_input_order(self):
        """Determinism comes from (recall, -fpr) being UNIQUE over ROC points —
        every point consumes at least one row, so no two share both tp and fp. The
        first version added a third tie-break on the threshold and documented it as
        the determinism guarantee; it was unreachable, and inverting it left the
        suite green, so its test could not fail. This checks the property instead
        of a branch that cannot run."""
        scores = [0.9, 0.8, 0.2, 0.1]
        labels = [1, 1, 0, 0]
        first = operating_point_at_fpr(scores, labels, 0.0)
        second = operating_point_at_fpr(list(reversed(scores)), list(reversed(labels)), 0.0)
        assert (first.threshold, first.recall, first.realised_fpr) == (
            second.threshold, second.recall, second.realised_fpr
        )

    def test_a_zero_budget_is_a_real_request(self):
        point = operating_point_at_fpr([0.9, 0.8, 0.1], [1, 1, 0], 0.0)
        assert point.realised_fpr == pytest.approx(0.0)
        assert point.recall == pytest.approx(1.0)

    def test_recall_and_threshold_helpers_agree_with_the_point(self):
        scores, labels = _labelled(11, n=200)
        point = operating_point_at_fpr(scores, labels, 0.05)
        assert recall_at_fpr(scores, labels, 0.05) == pytest.approx(point.recall)
        assert threshold_at_fpr(scores, labels, 0.05) == pytest.approx(point.threshold)

    def test_an_out_of_range_target_is_refused(self):
        with pytest.raises(ValueError, match=r"target_fpr must be in \[0, 1\]"):
            operating_point_at_fpr([0.1], [1], 1.5)

    def test_recall_at_a_target_is_monotone_in_the_budget(self):
        scores, labels = _labelled(23, n=300)
        recalls = [recall_at_fpr(scores, labels, t) for t in (0.01, 0.05, 0.10, 0.25)]
        assert recalls == sorted(recalls), "a bigger budget cannot buy less recall"


# ── the refusal ───────────────────────────────────────────────────────────────

class TestRefusalBelowTwenty:

    def test_the_boundary_is_exactly_twenty(self):
        assert MIN_PER_CLASS == 20
        scores = [0.9] * 20 + [0.1] * 20
        labels = [1] * 20 + [0] * 20
        assert check_scoreable(scores, labels) is None

    def test_nineteen_of_a_class_refuses(self):
        scores = [0.9] * 19 + [0.1] * 40
        labels = [1] * 19 + [0] * 40
        refusal = check_scoreable(scores, labels)
        assert refusal is not None
        assert refusal.n_positive == 19 and refusal.n_negative == 40

    def test_the_refusal_says_why_and_carries_no_number(self):
        result = evaluate([0.9] * 5 + [0.1] * 5, [1] * 5 + [0] * 5)
        assert result["scored"] is False
        assert "fewer than 20" in result["reason"]
        assert "auroc" not in result, "a refusal must not carry a score at all"

    def test_evaluate_scores_when_both_classes_clear_the_bar(self):
        scores, labels = _labelled(7, n=100)
        result = evaluate(scores, labels, lengths=list(range(100)), target_fpr=0.02)
        assert result["scored"] is True
        assert result["auroc"] == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)
        assert result["ci"]["low"] <= result["auroc"] <= result["ci"]["high"]
        assert any(op["target_fpr"] == 0.02 for op in result["operating_points"])
        assert len(result["length_bands"]) == 4


# ── the interval ──────────────────────────────────────────────────────────────

class TestTheInterval:

    def test_it_brackets_the_point_estimate(self):
        scores, labels = _labelled(3, n=200)
        ci = auroc_ci(scores, labels, resamples=500)
        assert ci["low"] <= auroc(scores, labels) <= ci["high"]

    def test_it_is_reproducible_for_a_given_seed(self):
        scores, labels = _labelled(4, n=120)
        a = auroc_ci(scores, labels, resamples=300, seed=99)
        b = auroc_ci(scores, labels, resamples=300, seed=99)
        assert a == b

    def test_a_different_seed_moves_it(self):
        scores, labels = _labelled(4, n=120)
        a = auroc_ci(scores, labels, resamples=300, seed=1)
        b = auroc_ci(scores, labels, resamples=300, seed=2)
        assert (a["low"], a["high"]) != (b["low"], b["high"])

    def test_a_larger_sample_narrows_it(self):
        small = auroc_ci(*_labelled(8, n=60), resamples=400)
        large = auroc_ci(*_labelled(8, n=600), resamples=400)
        assert (large["high"] - large["low"]) < (small["high"] - small["low"])

    def test_chance_data_does_not_clear_the_rung_threshold(self):
        """The interval's job: on scores with no signal, the lower bound must not
        sit above 0.5, because FR-13 promotes a probe on exactly that."""
        import random

        rng = random.Random(0)
        labels = [i % 2 for i in range(400)]
        scores = [rng.random() for _ in labels]
        ci = auroc_ci(scores, labels, resamples=800)
        assert ci["low"] <= 0.5, f"chance data produced a lower bound of {ci['low']}"

    def test_too_few_rows_returns_none(self):
        assert auroc_ci([0.9, 0.1], [1, 0]) is None


# ── the mean across sets, which is where bootstrap_ci belongs ─────────────────

class TestMeanAcrossSets:

    def test_it_averages_the_scored_sets(self):
        result = mean_auroc_across_sets({"a": 0.8, "b": 0.6, "c": 0.7})
        assert result["mean_auroc"] == pytest.approx(0.7)
        assert result["sets_scored"] == 3

    def test_an_unscored_set_is_excluded_and_named_not_imputed(self):
        result = mean_auroc_across_sets({"a": 0.9, "b": None})
        assert result["mean_auroc"] == pytest.approx(0.9)
        assert result["sets_scored"] == 1 and result["sets_total"] == 2
        assert result["unscored"] == ["b"]

    def test_nothing_scored_is_a_refusal_not_a_comfortable_half(self):
        result = mean_auroc_across_sets({"a": None, "b": None})
        assert result["scored"] is False
        assert result["mean_auroc"] is None
        assert "no evaluation set produced a score" in result["reason"]

    def test_it_carries_an_interval_over_SETS(self):
        result = mean_auroc_across_sets({f"s{i}": 0.6 + i * 0.02 for i in range(6)})
        assert result["ci"] is not None
        assert result["ci"]["low"] <= result["mean_auroc"] <= result["ci"]["high"]


# ── length bands ──────────────────────────────────────────────────────────────

class TestLengthBands:

    def test_there_are_four_bands_cut_at_the_observed_quartiles(self):
        scores, labels = _labelled(13, n=400)
        lengths = list(range(400))
        bands = length_bands(scores, labels, lengths)
        assert [b["band"] for b in bands] == [0, 1, 2, 3]
        assert sum(b["n"] for b in bands) == 400
        assert bands[0]["min_length"] == 0 and bands[3]["max_length"] == 399

    def test_a_band_that_cannot_be_scored_says_so_with_its_counts(self):
        """A silently dropped band hides the very variation bands exist to show."""
        scores = [0.9] * 30 + [0.1] * 30
        labels = [1] * 30 + [0] * 30
        lengths = [1] * 30 + [1000] * 30      # each band ends up single-class
        bands = length_bands(scores, labels, lengths)
        unscored = [b for b in bands if b["scored"] is False]
        assert unscored, "single-class bands must refuse"
        populated = [b for b in unscored if b["n"] > 0]
        assert populated, "this fixture is meant to produce populated single-class bands"
        assert all("fewer than 20" in b["reason"] for b in populated)

    def test_an_EMPTY_band_is_not_reported_as_an_under_sampled_one(self):
        """Two different problems, and the same message sent a reader after data
        for bands that cannot exist.

        With only two distinct lengths the quartile cuts collapse, so two of the
        four bands hold no rows at all. `check_scoreable` described them as "fewer
        than 20 examples of a class (0 positive, 0 negative)" — which reads as a
        band that exists and is too small. On a packed corpus, where every row is
        the same width, three bands are empty and the previous wording reported
        three under-sampled bands.
        """
        scores = [0.9] * 30 + [0.1] * 30
        labels = [1] * 30 + [0] * 30
        lengths = [1] * 30 + [1000] * 30
        bands = length_bands(scores, labels, lengths)
        empty = [b for b in bands if b["n"] == 0]
        assert empty, "this fixture is meant to collapse the quartile cuts"
        for band in empty:
            assert band["scored"] is False
            assert band["reason"] == "no examples fall in this length band"
            assert "fewer than" not in band["reason"]

    def test_a_scored_band_carries_its_auroc_VALUE(self):
        """A mutation dropping "auroc" from the scored branch left 40/40 green:
        nothing asserted the number, only the shape."""
        scores = [0.9] * 40 + [0.1] * 40 + [0.8] * 40 + [0.2] * 40
        labels = [1] * 40 + [0] * 40 + [1] * 40 + [0] * 40
        lengths = [1] * 80 + [1000] * 80
        bands = length_bands(scores, labels, lengths)
        scored = [b for b in bands if b["scored"]]
        assert scored, "this fixture is meant to produce scoreable bands"
        for band in scored:
            assert "auroc" in band, "a scored band without its value is not a result"
            assert band["auroc"] == pytest.approx(1.0)

    def test_the_per_band_floor_needs_about_160_balanced_rows_and_says_so(self):
        """The floor applies PER BAND: 4 bands x 20 per class x 2 classes = 160.
        A balanced 100-row set therefore yields four refusals and no numbers. That
        is deliberate — the alternative is a weaker standard for a band than for
        the set it came from — and each band reports the `minimum` it applied so a
        reader can see why."""
        scores, labels = _labelled(77, n=100)
        bands = length_bands(scores, labels, list(range(100)))
        assert all(b["scored"] is False for b in bands)
        assert all(b["minimum"] == MIN_PER_CLASS for b in bands)

        wide_scores, wide_labels = _labelled(78, n=400)
        wide = length_bands(wide_scores, wide_labels, list(range(400)))
        assert any(b["scored"] for b in wide), "160+ balanced rows must score"

    def test_a_caller_can_lower_the_floor_but_must_ask(self):
        scores, labels = _labelled(79, n=100)
        bands = length_bands(scores, labels, list(range(100)), minimum=5)
        assert any(b["scored"] for b in bands)
        assert all(b["minimum"] == 5 for b in bands)

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="must match"):
            length_bands([0.1, 0.2], [1, 0], [5])

    def test_no_rows_gives_no_bands(self):
        assert length_bands([], [], []) == []


# ── the report must survive serialisation ─────────────────────────────────────

class TestTheResultIsJsonSerialisable:
    """Review finding: the ROC origin and an empty operating point carried
    `float("inf")`. Starlette renders JSON with `allow_nan=False` and jsonb
    rejects it, so the report endpoint would 500 — and the 1%-FPR point lands
    there whenever the top-scoring row is negative, common at 20 negatives."""

    def test_a_scored_result_is_strict_json(self):
        import json

        scores, labels = _labelled(31, n=120)
        result = evaluate(scores, labels, lengths=list(range(120)), name="dev")
        json.dumps(result, allow_nan=False)      # raises on inf/nan

    def test_the_roc_origin_is_none_not_infinity(self):
        points = roc_points([0.9, 0.1], [1, 0])
        assert points[0].threshold is None

    def test_a_zero_budget_with_a_negative_on_top_serialises(self):
        """The case that produces the empty operating point: no threshold admits
        anything inside the budget, so recall 0 at FPR 0 — and that must be
        representable."""
        import json

        scores = [0.99] + [0.5] * 20 + [0.4] * 20
        labels = [0] + [1] * 20 + [0] * 20
        point = operating_point_at_fpr(scores, labels, 0.0)
        assert point.threshold is None and point.recall == 0.0
        json.dumps(evaluate(scores, labels), allow_nan=False)


# ── distinct FPR targets must not collide ─────────────────────────────────────

class TestTargetsDoNotCollide:
    """Review finding: keying operating points by `f"{target:.2f}"` mapped 0.011
    and 0.01 onto one key, so a probe calibrated at 1.1% silently overwrote the
    1% point FR-9 mandates — and the survivor reported target_fpr 0.011."""

    def test_a_nearby_calibrated_target_does_not_displace_the_standard_one(self):
        scores, labels = _labelled(41, n=200)
        result = evaluate(scores, labels, target_fpr=0.011)
        targets = [op["target_fpr"] for op in result["operating_points"]]
        assert 0.01 in targets and 0.011 in targets
        assert len(targets) == len(set(targets))

    def test_the_three_standard_targets_are_always_present(self):
        scores, labels = _labelled(42, n=200)
        targets = [op["target_fpr"] for op in evaluate(scores, labels)["operating_points"]]
        assert {0.01, 0.05, 0.10} <= set(targets)

    def test_a_target_equal_to_a_standard_one_is_not_duplicated(self):
        scores, labels = _labelled(43, n=200)
        targets = [op["target_fpr"] for op in evaluate(scores, labels, target_fpr=0.05)["operating_points"]]
        assert targets.count(0.05) == 1


# ── labels outside {0,1} ──────────────────────────────────────────────────────

class TestUnexpectedLabelsAreRefused:
    """Review finding: rows whose label was neither 0 nor 1 were silently dropped,
    so an AUROC over 80 rows could be presented as one over 100."""

    def test_an_excluded_label_is_refused_rather_than_dropped(self):
        with pytest.raises(ValueError, match="labels must be 0 or 1"):
            auroc([0.1, 0.2, 0.3], [0, 1, 2])

    def test_the_refusal_says_where_excluded_rows_belong(self):
        with pytest.raises(ValueError, match="dropped by the caller"):
            check_scoreable([0.1, 0.2], [1, 7])

    def test_evaluate_refuses_too_rather_than_reporting_short_counts(self):
        scores = [0.9] * 20 + [0.1] * 20 + [0.5] * 5
        labels = [1] * 20 + [0] * 20 + [2] * 5
        with pytest.raises(ValueError, match="labels must be 0 or 1"):
            evaluate(scores, labels)


# ── round 2: the fixes, pinned ─────────────────────────────────────────────────


class TestLabelsAreValidatedBEFORETruncation:
    """`{int(y) for y in labels} - {0, 1}` validates the wrong thing.

    Round 1 replaced a silent drop of non-0/1 labels with a refusal. The refusal
    was built on `int(y)`, so it truncated before it validated: a column of
    PROBABILITIES passed, with 0.9 counted as a negative.
    """

    def test_a_probability_column_is_refused_not_rounded(self):
        scores = [0.5] * 4
        with pytest.raises(ValueError, match="labels must be 0 or 1"):
            auroc(scores, [0.9, 0.1, 1.0, 0.0])

    def test_a_rating_above_one_is_refused(self):
        with pytest.raises(ValueError, match="labels must be 0 or 1"):
            auroc([0.1, 0.2, 0.3], [1, 0, 1.5])

    def test_the_offending_value_is_NAMED_not_just_counted(self):
        """A reader must be able to see it was 0.9, not 'a bad label somewhere'."""
        with pytest.raises(ValueError) as caught:
            auroc([0.1, 0.2], [0.9, 0])
        assert "0.9" in str(caught.value)

    def test_the_forms_that_ARE_zero_and_one_still_pass(self):
        """0.0/1.0 and bools are labels, not mistakes — the guard must not overreach."""
        scores = [0.9] * 3 + [0.1] * 3
        assert auroc(scores, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) == 1.0
        assert auroc(scores, [True, True, True, False, False, False]) == 1.0


class TestThresholdAtFprDoesNotConflateTwoAnswers:
    """None meant both "fire on nothing" and "could not score" — opposite orders.

    `threshold or 0.0` then turns the first into a threshold of 0.0, which on a
    centred score fires on about half of all inputs: a monitor inverted from silent
    to noisy by a defaulting expression.
    """

    def test_it_RAISES_when_a_class_is_missing(self):
        with pytest.raises(ValueError, match="one class is missing"):
            threshold_at_fpr([0.1, 0.2, 0.3], [1, 1, 1], 0.05)

    def test_None_means_fire_on_nothing_and_ONLY_that(self):
        # top-scoring row is negative, so a 0% budget admits no firing threshold
        scores = [0.99] + [0.5] * 10 + [0.1] * 10
        labels = [0] + [1] * 10 + [0] * 10
        assert threshold_at_fpr(scores, labels, 0.0) is None

    def test_a_real_threshold_still_comes_back_as_a_number(self):
        scores = [0.9] * 10 + [0.1] * 10
        labels = [1] * 10 + [0] * 10
        assert threshold_at_fpr(scores, labels, 0.05) == pytest.approx(0.9)


class TestTheTieBreakIsReachableNow:
    """Round 1 removed one unreachable tie-break and left a second in that state.

    `(p.tpr, -p.fpr)` was delivered by `roc_points` emitting points in increasing
    FPR order plus `max` keeping the first maximal element — so deleting `-p.fpr`
    left the whole suite green, and a later change to `roc_points` could have
    reversed the documented rule silently. The decision is now a pure function
    over a point LIST, tested where ordering cannot rescue it.
    """

    def test_ties_go_to_the_lower_fpr_even_when_the_points_arrive_worst_first(self):
        points = [
            RocPoint(threshold=0.1, fpr=0.08, tpr=0.7),   # same recall, more spent
            RocPoint(threshold=0.5, fpr=0.02, tpr=0.7),   # the right answer
            RocPoint(threshold=None, fpr=0.0, tpr=0.0),
        ]
        assert best_point_within(points, 0.10).threshold == 0.5

    def test_and_when_they_arrive_best_first(self):
        points = [
            RocPoint(threshold=0.5, fpr=0.02, tpr=0.7),
            RocPoint(threshold=0.1, fpr=0.08, tpr=0.7),
        ]
        assert best_point_within(points, 0.10).threshold == 0.5

    def test_recall_still_beats_a_cheaper_fpr(self):
        """The tie-break is a TIE-break: it must not outrank recall."""
        points = [
            RocPoint(threshold=0.9, fpr=0.01, tpr=0.3),
            RocPoint(threshold=0.4, fpr=0.09, tpr=0.8),
        ]
        assert best_point_within(points, 0.10).threshold == 0.4

    def test_nothing_inside_the_budget_is_None(self):
        assert best_point_within([RocPoint(threshold=0.1, fpr=0.5, tpr=0.9)], 0.1) is None

    def test_the_wrapper_uses_it(self):
        """Reachability: the pure function must be what the public entry point calls."""
        import ast
        import inspect

        from src.services import probe_monitor_metrics as module

        tree = ast.parse(inspect.getsource(module))
        target = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "operating_point_at_fpr"
        )
        called = {
            node.func.id
            for node in ast.walk(target)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "best_point_within" in called, (
            "operating_point_at_fpr no longer CALLS best_point_within, so the tests "
            "above pin a function nothing uses"
        )


class TestAMeanOverOneSetHasNoInterval:
    """`scored: True` with `ci: None` raises in a consumer reading `ci["low"]`."""

    def test_one_scored_set_names_why_the_interval_is_absent(self):
        result = mean_auroc_across_sets({"a": 0.8, "b": None})
        assert result["scored"] is True
        assert result["ci"] is None
        assert "at least 2" in result["ci_unavailable"]

    def test_two_scored_sets_get_an_interval_and_no_excuse(self):
        result = mean_auroc_across_sets({"a": 0.8, "b": 0.9})
        assert result["ci"] is not None
        assert result["ci_unavailable"] is None

    def test_the_key_is_present_even_when_nothing_scored(self):
        """A stable shape, so a consumer never has to `.get` its way around it."""
        result = mean_auroc_across_sets({"a": None})
        assert "ci_unavailable" in result
