"""Unit tests for the feature grouping algorithm (Feature 010).

Covers the pure-computation parts of FeatureGroupingService with synthetic
data — no database. The DB path is covered by the API integration tests.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.models.feature_grouping import FeatureTokenIndex
from src.services.feature_grouping_service import (
    DEFAULT_PARAMS,
    FeatureGroupingService,
    params_hash,
)


def example(prime, prefix=None, suffix=None, activation=1.0):
    return SimpleNamespace(
        prime_token=prime,
        prefix_tokens=prefix or [],
        suffix_tokens=suffix or [],
        max_activation=activation,
    )


def run_stub():
    return SimpleNamespace(id="run-1", extraction_id="ext-1")


def index_row(token, bag, feature_id="f", raw=None):
    row = FeatureTokenIndex(
        run_id="run-1",
        extraction_id="ext-1",
        feature_id=feature_id,
        neuron_index=0,
        raw_token=raw or token,
        normalized_token=token,
        token_rank=1,
        weight=1.0,
        context_tokens=bag,
    )
    return row


class TestParamsHash:
    def test_stable_and_order_independent(self):
        a = params_hash({"b": 2, "a": 1})
        b = params_hash({"a": 1, "b": 2})
        assert a == b
        assert a != params_hash({"a": 1, "b": 3})


class TestIndexRowsForFeature:
    def setup_method(self):
        self.svc = FeatureGroupingService()

    def test_rank1_token_is_weighted_winner(self):
        examples = [
            example("▁love", activation=5.0),
            example("▁love", activation=4.0),
            example("▁heart", activation=1.0),
        ]
        rows = self.svc._index_rows_for_feature(run_stub(), "feat-1", 7, examples, window=5)
        assert rows[0].normalized_token == "love"
        assert rows[0].token_rank == 1
        assert rows[0].raw_token == "▁love"
        assert rows[0].neuron_index == 7

    def test_low_weight_secondary_tokens_dropped(self):
        # "heart" share = 1/101 < 0.2 → only rank 1 kept
        examples = [example("▁love", activation=100.0), example("▁heart", activation=1.0)]
        rows = self.svc._index_rows_for_feature(run_stub(), "f", 0, examples, window=5)
        assert [r.normalized_token for r in rows] == ["love"]

    def test_punctuation_prime_tokens_excluded(self):
        rows = self.svc._index_rows_for_feature(
            run_stub(), "f", 0, [example("..."), example("▁")], window=5
        )
        assert rows == []

    def test_context_bag_is_normalized_and_windowed(self):
        examples = [
            example(
                "▁love",
                prefix=["▁I", "▁really", "▁truly", "▁do", "▁so", "▁much", "▁very"],
                suffix=["▁you", "..."],
            )
        ]
        rows = self.svc._index_rows_for_feature(run_stub(), "f", 0, examples, window=2)
        # window=2: last 2 prefix + first 2 suffix, normalized, punctuation dropped
        assert rows[0].context_tokens == ["much", "very", "you"]

    def test_snippet_attached(self):
        rows = self.svc._index_rows_for_feature(
            run_stub(), "f", 0, [example("▁love", prefix=["▁I"], suffix=["▁you"])], window=5
        )
        assert "*▁love*" in rows[0]._snippet


class TestContextComponents:
    def test_similar_contexts_group_together(self):
        rows = [
            index_row("love", ["i", "really", "you", "so", "much"], "f1"),
            index_row("love", ["i", "really", "you", "very", "much"], "f2"),
            index_row("love", ["would", "to", "see", "the", "menu"], "f3"),
        ]
        components, sims = FeatureGroupingService._context_components(rows, threshold=0.35)
        comp_sets = [set(c) for c in components]
        assert {0, 1} in comp_sets  # romantic pair grouped
        assert {2} in comp_sets     # "would love to" split off

    def test_empty_bags_form_single_group(self):
        rows = [index_row("x", [], "f1"), index_row("x", [], "f2")]
        components, sims = FeatureGroupingService._context_components(rows, threshold=0.35)
        assert components == [[0, 1]]
        assert sims.shape == (2, 2)


class TestCohesion:
    def test_singleton_is_perfect(self):
        cohesion, member = FeatureGroupingService._cohesion(np.ones((3, 3)), [1])
        assert cohesion == 1.0 and member == [1.0]

    def test_mean_pairwise(self):
        sims = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]])
        cohesion, member = FeatureGroupingService._cohesion(sims, [0, 1, 2])
        assert cohesion == pytest.approx((0.5 + 0.0 + 0.5) / 3)
        assert member[1] == pytest.approx(0.5)  # middle member closest to both


class TestDefaults:
    def test_default_params(self):
        assert DEFAULT_PARAMS == {
            "context_window": 5,
            "similarity_threshold": 0.35,
            "top_examples": 10,
            "min_group_size": 2,
        }
