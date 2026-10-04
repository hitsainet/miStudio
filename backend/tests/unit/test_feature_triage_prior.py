"""The no-LLM interpretability prior.

Fixtures mirror a REAL nlp_analysis blob sampled from
extr_20260828_080834_sae_sae_39cc_002, including its awkward parts:
fragment_percentage is 0-100 rather than 0-1, token_types keys are SPARSE (a
real blob carried only capitalized/punctuation/content_words), and
num_examples_analyzed is 100 rather than the 10 examples a labeling prompt sees.
Getting any of those wrong yields a plausible number that is silently wrong.
"""

import pytest

from src.services.feature_triage_prior import (
    DEFAULT_WEIGHTS,
    extract_signals,
    score_signals,
    triage_prior,
)


def _blob(**over):
    """A dispersed feature — the real sampled blob."""
    b = {
        "num_examples_analyzed": 100,
        "prime_token_analysis": {
            "concentration_ratio": 0.03,
            "unique_count": 94,
            "total_count": 100,
            "fragment_percentage": 36.0,
            "token_types": {"capitalized": 14, "punctuation": 5, "content_words": 79},
        },
        "activation_stats": {"coefficient_of_variation": 0.0348, "skewness": 1.88},
        "semantic_clusters": [{"size": 39}, {"size": 20}],
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(b.get(k), dict):
            b[k] = {**b[k], **v}
        else:
            b[k] = v
    return b


class TestSignalExtraction:
    def test_fragment_percentage_is_read_as_a_percentage(self):
        """It is 0-100. Treating it as 0-1 makes every feature look like debris."""
        s = extract_signals(_blob())
        assert s["whole_word_ratio"] == pytest.approx(0.64), (
            "36.0 means 36%, so 64% whole words; a 0-1 reading gives 0.0 and "
            "ranks every feature as pure BPE fragments"
        )

    def test_token_purity_inverts_diversity(self):
        s = extract_signals(_blob())
        assert s["token_purity"] == pytest.approx(1 - 94 / 100)

    def test_cluster_dominance_uses_num_examples_not_cluster_count(self):
        s = extract_signals(_blob())
        assert s["cluster_dominance"] == pytest.approx(39 / 100)

    def test_content_ratio_uses_total_count_not_the_sum_of_token_types(self):
        """token_types is sparse; summing it would make the ratio ~1.0 always."""
        s = extract_signals(_blob())
        assert s["content_ratio"] == pytest.approx(79 / 100)
        # The sample's token_types sums to 98, not 100 — proving the denominator
        # matters and is not simply the dict total.
        assert sum(_blob()["prime_token_analysis"]["token_types"].values()) != 100

    def test_a_focused_feature_scores_higher_than_a_dispersed_one(self):
        dispersed = triage_prior(_blob())
        focused = triage_prior(_blob(
            prime_token_analysis={
                "concentration_ratio": 0.85, "unique_count": 4,
                "total_count": 100, "fragment_percentage": 2.0,
                "token_types": {"content_words": 96},
            },
            semantic_clusters=[{"size": 88}],
        ))
        assert focused > dispersed, (focused, dispersed)
        assert focused > 0.7 and dispersed < 0.4


class TestMissingDataIsUnknownNotZero:
    def test_absent_blob_yields_none(self):
        assert triage_prior(None) is None
        assert triage_prior({}) is None
        assert all(v is None for v in extract_signals(None).values())

    @pytest.mark.parametrize("junk", ["", 0, [], "a string", 3.5])
    def test_non_dict_input_does_not_raise(self, junk):
        assert triage_prior(junk) is None

    def test_partial_analysis_is_scored_on_what_it_has(self):
        """Renormalised, not penalised for missing fields."""
        partial = {"prime_token_analysis": {"concentration_ratio": 0.9}}
        p = triage_prior(partial)
        assert p == pytest.approx(0.9), (
            "a feature with only one available signal should be scored on that "
            "signal, not diluted toward zero by absent ones"
        )

    def test_a_missing_signal_is_not_treated_as_worst_case(self):
        full = _blob()
        without = _blob(semantic_clusters=[])
        # Removing a mid-valued signal must not crater the score the way a 0.0
        # substitution would.
        assert triage_prior(without) is not None
        assert abs(triage_prior(without) - triage_prior(full)) < 0.25

    def test_zero_denominators_do_not_raise_or_mislead(self):
        s = extract_signals(_blob(
            prime_token_analysis={"unique_count": 0, "total_count": 0},
            num_examples_analyzed=0,
        ))
        assert s["token_purity"] is None
        assert s["cluster_dominance"] is None


class TestBoundsAndShape:
    def test_prior_is_bounded(self):
        for b in (_blob(), _blob(prime_token_analysis={"concentration_ratio": 5.0}),
                  _blob(prime_token_analysis={"fragment_percentage": 500.0})):
            p = triage_prior(b)
            assert p is None or 0.0 <= p <= 1.0

    def test_activation_cv_is_reported_but_unweighted(self):
        """Its direction is genuinely unknown; the harness decides, not a guess."""
        assert "activation_cv" in extract_signals(_blob())
        assert "activation_cv" not in DEFAULT_WEIGHTS

    def test_score_signals_carries_every_signal_plus_the_prior(self):
        out = score_signals(_blob())
        assert "prior" in out
        for k in ("concentration", "token_purity", "cluster_dominance",
                  "content_ratio", "whole_word_ratio", "activation_cv"):
            assert k in out, f"{k} missing — the harness cannot evaluate it"
