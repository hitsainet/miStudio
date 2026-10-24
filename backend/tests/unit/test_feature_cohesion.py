"""Embedding cohesion arithmetic.

Deterministic by construction — the point of cohesion is that it replaces an
unstable model judgement with a computed number, so these tests use hand-built
vectors with known answers rather than anything sampled.
"""

import math

import pytest

from src.services.feature_cohesion import (
    cohesion_score,
    mean_pairwise_cosine,
    summarise,
)


class TestMeanPairwiseCosine:
    def test_identical_vectors_are_perfectly_cohesive(self):
        assert mean_pairwise_cosine([[1, 0], [1, 0], [1, 0]]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert mean_pairwise_cosine([[1, 0], [0, 1]]) == pytest.approx(0.0)

    def test_opposed_vectors_score_minus_one(self):
        assert mean_pairwise_cosine([[1, 0], [-1, 0]]) == pytest.approx(-1.0)

    def test_magnitude_is_irrelevant_only_direction_counts(self):
        """Passage length must not masquerade as similarity."""
        same_dir = mean_pairwise_cosine([[1, 0], [1000, 0]])
        assert same_dir == pytest.approx(1.0)

    def test_it_averages_over_all_distinct_pairs(self):
        # three vectors: two identical, one orthogonal -> pairs are 1, 0, 0
        got = mean_pairwise_cosine([[1, 0], [1, 0], [0, 1]])
        assert got == pytest.approx(1.0 / 3.0)

    def test_result_stays_within_bounds(self):
        v = [[0.6, 0.8], [0.6, 0.8], [0.8, 0.6]]
        got = mean_pairwise_cosine(v)
        assert -1.0 <= got <= 1.0


class TestUnmeasurableIsNoneNotZero:
    def test_a_single_passage_has_no_pairs(self):
        """1.0 would rank a feature with one stored example as maximally coherent."""
        assert mean_pairwise_cosine([[1, 0]]) is None

    def test_empty_input(self):
        assert mean_pairwise_cosine([]) is None

    def test_zero_vectors_are_dropped_not_scored_as_dissimilar(self):
        """A zero embedding has no direction; cosine against it is undefined."""
        assert mean_pairwise_cosine([[0, 0], [0, 0]]) is None
        # One good pair survives alongside a zero vector.
        assert mean_pairwise_cosine([[1, 0], [1, 0], [0, 0]]) == pytest.approx(1.0)

    def test_non_finite_norms_are_dropped(self):
        assert mean_pairwise_cosine([[float("inf"), 0], [1, 0]]) is None


class TestTheContrastIsApplied:
    def test_score_subtracts_the_corpus_baseline(self):
        assert cohesion_score(0.80, 0.55) == pytest.approx(0.25)

    def test_a_feature_no_better_than_random_scores_zero(self):
        """The case the contrast exists for: all-news-prose similarity."""
        assert cohesion_score(0.62, 0.62) == pytest.approx(0.0)

    def test_a_feature_less_alike_than_random_goes_negative(self):
        assert cohesion_score(0.40, 0.60) == pytest.approx(-0.20)

    @pytest.mark.parametrize("own,base", [(None, 0.5), (0.5, None), (None, None)])
    def test_none_propagates_rather_than_becoming_zero(self, own, base):
        """0.0 means 'measured, exactly average'. None means 'not measured'."""
        assert cohesion_score(own, base) is None


class TestSummarise:
    def test_it_counts_unscored_features_separately(self):
        s = summarise({"a": 0.1, "b": None, "c": 0.3})
        assert s["n"] == 2 and s["unscored"] == 1

    def test_all_unscored(self):
        assert summarise({"a": None})["n"] == 0

    def test_percentiles_are_ordered(self):
        s = summarise({str(i): i / 10.0 for i in range(10)})
        assert s["min"] <= s["p25"] <= s["median"] <= s["p75"] <= s["max"]
