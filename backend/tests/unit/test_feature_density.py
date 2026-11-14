"""The dense tail of the feature distribution, which nothing ever looked at.

WHY THIS FILE EXISTS. Training kept a per-feature EMA and read it only at the
DEAD end (`< 0.01`). An SAE where 5% of latents fire on 80% of tokens — the
signature of polysemantic features — produced a healthy aggregate L0 and raised
nothing at all.

The pre-existing EMA also cannot be read as a frequency: it accumulates a
batch-level indicator with an unnormalised update, so its steady state is
`window / batch_size`. This module tracks a real rate; these tests pin that
distinction, because reusing the old quantity would look right and mean nothing.
"""

import pytest
import torch

from src.services.feature_density import (
    DENSE_THRESHOLD,
    density_summary,
    describe,
    update_firing_rate,
)


class TestFiringRateIsARate:

    def test_it_is_the_fraction_of_tokens_not_a_batch_indicator(self):
        """The distinction the old EMA got wrong.

        Feature 0 fires on 1 of 4 tokens. A batch-level indicator would record
        1.0 ("it fired"); a rate records 0.25.
        """
        z = torch.zeros(4, 3)
        z[0, 0] = 1.0
        rate = update_firing_rate(None, z)
        assert rate[0].item() == pytest.approx(0.25)
        assert rate[1].item() == pytest.approx(0.0)

    def test_a_feature_firing_everywhere_converges_to_one(self):
        z = torch.ones(8, 2)
        rate = None
        for _ in range(500):
            rate = update_firing_rate(rate, z, momentum=0.9)
        assert rate.max().item() == pytest.approx(1.0, abs=1e-3)

    def test_a_silent_feature_converges_to_zero(self):
        z = torch.zeros(8, 2)
        rate = torch.ones(2)
        for _ in range(500):
            rate = update_firing_rate(rate, z, momentum=0.9)
        assert rate.max().item() == pytest.approx(0.0, abs=1e-3)

    def test_the_rate_is_bounded_by_construction(self):
        """Unlike the old accumulator, whose steady state was window/batch_size."""
        z = (torch.rand(16, 32) > 0.5).float()
        rate = None
        for _ in range(200):
            rate = update_firing_rate(rate, z)
        assert rate.min() >= 0.0 and rate.max() <= 1.0


class TestDensitySummary:

    def test_it_finds_features_doing_too_much_work(self):
        """The case that previously raised nothing: a few latents on everything."""
        rates = torch.full((100,), 0.01)
        rates[:5] = 0.8
        summary = density_summary(rates)
        assert summary["dense_features"] == 5
        assert summary["dense_fraction"] == pytest.approx(0.05)
        assert summary["density_max"] == pytest.approx(0.8)

    def test_a_healthy_dictionary_flags_nothing(self):
        """NEGATIVE CONTROL — the alarm must be able to stay silent."""
        summary = density_summary(torch.full((100,), 0.02))
        assert summary["dense_features"] == 0
        assert summary["top1pct_share"] < 0.05

    def test_concentration_is_visible_in_the_top_percentile_share(self):
        concentrated = torch.full((1000,), 1e-5)
        concentrated[:10] = 1.0
        spread = torch.full((1000,), 0.01)
        assert density_summary(concentrated)["top1pct_share"] > 0.9
        assert density_summary(spread)["top1pct_share"] < 0.05

    def test_silent_features_are_counted_separately_from_dense_ones(self):
        rates = torch.cat([torch.zeros(40), torch.full((60,), 0.5)])
        summary = density_summary(rates)
        assert summary["silent_features"] == 40
        assert summary["dense_features"] == 60

    def test_percentiles_are_ordered(self):
        summary = density_summary(torch.rand(500))
        assert summary["density_p50"] <= summary["density_p90"] <= summary["density_p99"]
        assert summary["density_p99"] <= summary["density_max"]

    def test_an_empty_tensor_yields_nothing_rather_than_crashing(self):
        assert density_summary(torch.empty(0)) == {}

    def test_the_threshold_is_the_documented_one(self):
        """A feature on >30% of tokens is the stated concern."""
        rates = torch.tensor([DENSE_THRESHOLD + 0.01, DENSE_THRESHOLD - 0.01])
        assert density_summary(rates)["dense_features"] == 1


class TestDescription:

    def test_it_states_the_numbers_an_operator_would_act_on(self):
        text = describe(density_summary(torch.full((10,), 0.5)))
        assert "dense" in text and "top1%share" in text

    def test_no_features_is_said_plainly(self):
        assert "no features" in describe({})
