"""032 task 1.5 — the probe head and its six rules.

THE LOAD-BEARING TEST IS BATCH vs ONLINE. Every streamable rule has two
implementations of one definition: a batched form for training and evaluation,
and an incremental form for a serving runtime scoring tokens as they arrive. Two
implementations of one definition is the shape that drifts, and the drift would
be silent — both forms return a plausible float. So they are run against each
other to 1e-6 on random sequences, including the adversarial cases where the
online softmax's running-max rescaling is what keeps them equal.

The other two things pinned here are mask handling (a rule that folds padding
into a mean reports a number about the padding, and ~48% of a padded batch here
has been padding before) and that `last` REFUSES to stream rather than returning
the newest score.
"""
import math

import pytest
import torch

from src.ml.probe_monitor_model import (
    DEFAULT_TAU,
    DEFAULT_WINDOW,
    RULES,
    STREAMABLE,
    OnlineRule,
    ProbeHead,
    combine,
    combine_sequence,
    is_streamable,
    rule_parameters,
)

TOL = 1e-6


def _sequence(n: int, *, seed: int, scale: float = 1.0) -> list[float]:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, generator=g, dtype=torch.float64) * scale).tolist()


# ── the rule set itself ───────────────────────────────────────────────────────

class TestTheRuleSet:

    def test_all_six_rules_from_fr6_exist(self):
        assert set(RULES) == {
            "mean", "max", "last", "softmax", "attention", "rolling_mean_max"
        }

    def test_everything_but_last_is_streamable(self):
        assert STREAMABLE == set(RULES) - {"last"}
        assert not is_streamable("last")

    @pytest.mark.parametrize("rule", sorted(STREAMABLE))
    def test_each_streamable_rule_says_so(self, rule):
        assert is_streamable(rule) is True

    def test_an_unknown_rule_raises_rather_than_reading_as_not_streamable(self):
        """A typo that returned False would silently disable streaming."""
        with pytest.raises(ValueError, match="unknown combining rule"):
            is_streamable("meaan")

    def test_only_the_rules_that_use_a_parameter_record_one(self):
        assert rule_parameters("softmax", tau=0.5) == {"tau": 0.5}
        assert rule_parameters("rolling_mean_max", window=8) == {"window": 8}
        assert rule_parameters("mean") == {}
        assert rule_parameters("max") == {}
        assert rule_parameters("last") == {}


# ── batch vs online, every streamable rule ────────────────────────────────────

class TestBatchAndOnlineAgree:

    @pytest.mark.parametrize("rule", sorted(STREAMABLE))
    @pytest.mark.parametrize("n", [1, 2, 5, 17, 64])
    def test_they_agree_to_1e6_on_random_sequences(self, rule, n):
        scores = _sequence(n, seed=n * 31 + len(rule))
        logits = _sequence(n, seed=n * 17 + 5) if rule == "attention" else None

        batch = combine_sequence(
            rule, scores, attention_logits=logits, tau=DEFAULT_TAU, window=DEFAULT_WINDOW
        )

        online = OnlineRule(rule, tau=DEFAULT_TAU, window=DEFAULT_WINDOW)
        value = None
        for i, s in enumerate(scores):
            value = online.update(s, attention_logit=None if logits is None else logits[i])

        assert value == pytest.approx(batch, abs=TOL), f"{rule} n={n}: {value} vs {batch}"

    @pytest.mark.parametrize("rule", ["softmax", "attention"])
    def test_they_agree_when_logits_are_large_enough_to_overflow_a_naive_sum(self, rule):
        """exp(800) is inf. A naive online sum returns nan; the batch form, which
        max-shifts, does not. This is the case the rescaling exists for."""
        scores = [1.0, 2.0, 3.0, 4.0]
        logits = [700.0, 800.0, 750.0, 10.0] if rule == "attention" else None
        if rule == "softmax":
            scores = [700.0, 800.0, 750.0, 10.0]

        batch = combine_sequence(rule, scores, attention_logits=logits)
        online = OnlineRule(rule)
        for i, s in enumerate(scores):
            online.update(s, attention_logit=None if logits is None else logits[i])

        assert math.isfinite(online.value)
        assert online.value == pytest.approx(batch, abs=1e-6)

    @pytest.mark.parametrize("rule", ["softmax", "attention"])
    def test_they_agree_when_the_maximum_arrives_last(self, rule):
        """Forces a rebase on the final token — the branch a monotonically
        decreasing sequence never exercises."""
        scores = [0.0, 1.0, 2.0, 50.0]
        logits = [0.0, 1.0, 2.0, 50.0] if rule == "attention" else None
        batch = combine_sequence(rule, scores, attention_logits=logits)
        online = OnlineRule(rule)
        for i, s in enumerate(scores):
            online.update(s, attention_logit=None if logits is None else logits[i])
        assert online.value == pytest.approx(batch, abs=TOL)

    def test_rolling_mean_max_agrees_when_the_sequence_is_shorter_than_the_window(self):
        """Both forms score a short row as one window over its real tokens, so the
        rule equals `mean` there rather than being undefined."""
        scores = _sequence(5, seed=99)
        batch = combine_sequence("rolling_mean_max", scores, window=16)
        online = OnlineRule("rolling_mean_max", window=16)
        for s in scores:
            online.update(s)
        assert online.value == pytest.approx(batch, abs=TOL)
        assert batch == pytest.approx(sum(scores) / len(scores), abs=TOL)

    def test_rolling_mean_max_agrees_exactly_at_the_window_boundary(self):
        scores = _sequence(DEFAULT_WINDOW, seed=7)
        batch = combine_sequence("rolling_mean_max", scores)
        online = OnlineRule("rolling_mean_max")
        for s in scores:
            online.update(s)
        assert online.value == pytest.approx(batch, abs=TOL)

    @pytest.mark.parametrize("rule", sorted(STREAMABLE))
    def test_the_online_value_after_k_tokens_equals_the_batch_value_of_those_k(self, rule):
        """A streaming verdict must be the verdict for what has been seen — that is
        the property `STREAMABLE` claims."""
        scores = _sequence(12, seed=404)
        logits = _sequence(12, seed=405) if rule == "attention" else None
        online = OnlineRule(rule)
        for k in range(1, len(scores) + 1):
            online.update(scores[k - 1], attention_logit=None if logits is None else logits[k - 1])
            expected = combine_sequence(
                rule, scores[:k], attention_logits=None if logits is None else logits[:k]
            )
            assert online.value == pytest.approx(expected, abs=TOL), f"{rule} at k={k}"


# ── `last` refuses to stream ──────────────────────────────────────────────────

class TestLastRefusesToStream:

    def test_constructing_an_online_last_raises(self):
        with pytest.raises(ValueError, match="cannot be computed while tokens stream"):
            OnlineRule("last")

    def test_the_refusal_explains_why_rather_than_just_refusing(self):
        with pytest.raises(ValueError) as exc:
            OnlineRule("last")
        message = str(exc.value)
        assert "not defined until the sequence ends" in message

    def test_last_still_works_in_the_batch_form(self):
        assert combine_sequence("last", [1.0, 2.0, 9.0]) == pytest.approx(9.0)


# ── masking ───────────────────────────────────────────────────────────────────

class TestMasking:

    def test_mean_ignores_padding(self):
        scores = torch.tensor([[1.0, 3.0, 100.0]], dtype=torch.float64)
        mask = torch.tensor([[1, 1, 0]])
        assert float(combine("mean", scores, mask=mask)) == pytest.approx(2.0)

    def test_max_ignores_padding(self):
        scores = torch.tensor([[1.0, 3.0, 100.0]], dtype=torch.float64)
        mask = torch.tensor([[1, 1, 0]])
        assert float(combine("max", scores, mask=mask)) == pytest.approx(3.0)

    def test_last_is_the_last_REAL_token_not_the_last_column(self):
        scores = torch.tensor([[1.0, 3.0, 100.0]], dtype=torch.float64)
        mask = torch.tensor([[1, 1, 0]])
        assert float(combine("last", scores, mask=mask)) == pytest.approx(3.0)

    def test_softmax_puts_no_weight_on_padding(self):
        """A pad score far above every real one would dominate an unmasked softmax."""
        scores = torch.tensor([[1.0, 2.0, 50.0]], dtype=torch.float64)
        mask = torch.tensor([[1, 1, 0]])
        masked = float(combine("softmax", scores, mask=mask))
        unmasked = float(combine("softmax", scores))
        assert masked == pytest.approx(combine_sequence("softmax", [1.0, 2.0]), abs=TOL)
        assert unmasked > 40.0, "sanity: without the mask the pad dominates"

    def test_attention_puts_no_weight_on_padding(self):
        scores = torch.tensor([[1.0, 2.0, 99.0]], dtype=torch.float64)
        logits = torch.tensor([[0.0, 0.0, 50.0]], dtype=torch.float64)
        mask = torch.tensor([[1, 1, 0]])
        got = float(combine("attention", scores, attention_logits=logits, mask=mask))
        assert got == pytest.approx(1.5, abs=TOL)

    def test_rolling_mean_max_ignores_padding(self):
        scores = torch.tensor([[1.0, 1.0, 1.0, 99.0]], dtype=torch.float64)
        mask = torch.tensor([[1, 1, 1, 0]])
        assert float(combine("rolling_mean_max", scores, mask=mask, window=2)) == pytest.approx(1.0)

    def test_a_row_with_no_real_tokens_is_refused_not_scored(self):
        """mean would be 0/0 and max -inf; both look like numbers."""
        scores = torch.zeros(1, 4, dtype=torch.float64)
        mask = torch.zeros(1, 4, dtype=torch.long)
        with pytest.raises(ValueError, match="no unmasked tokens"):
            combine("mean", scores, mask=mask)

    def test_last_is_correct_under_LEFT_padding(self):
        """Review finding: `_last_valid_index` counted real tokens, which is the
        last token only when padding is on the RIGHT. Under left padding it
        returned a PAD position — while the docstring claimed left-padding
        correctness and every other mask test used right padding."""
        scores = torch.tensor([[7.0, 7.0, 1.0, 2.0]], dtype=torch.float64)
        mask = torch.tensor([[0, 0, 1, 1]])
        assert float(combine("last", scores, mask=mask)) == pytest.approx(2.0)

    @pytest.mark.parametrize("rule", sorted(set(RULES)))
    def test_every_rule_ignores_padding_on_either_side(self, rule):
        """The same two real tokens, padded left and padded right, must score alike."""
        left = torch.tensor([[0.0, 0.0, 1.0, 3.0]], dtype=torch.float64)
        left_mask = torch.tensor([[0, 0, 1, 1]])
        right = torch.tensor([[1.0, 3.0, 0.0, 0.0]], dtype=torch.float64)
        right_mask = torch.tensor([[1, 1, 0, 0]])
        logits = torch.tensor([[0.0, 0.0, 0.5, 1.5]], dtype=torch.float64)
        right_logits = torch.tensor([[0.5, 1.5, 0.0, 0.0]], dtype=torch.float64)
        kw = {"attention_logits": logits} if rule == "attention" else {}
        rkw = {"attention_logits": right_logits} if rule == "attention" else {}
        a = float(combine(rule, left, mask=left_mask, window=2, **kw))
        b = float(combine(rule, right, mask=right_mask, window=2, **rkw))
        assert a == pytest.approx(b, abs=TOL), f"{rule}: left {a} vs right {b}"

    def test_a_mismatched_mask_is_refused(self):
        with pytest.raises(ValueError, match="mask shape"):
            combine("mean", torch.zeros(1, 4, dtype=torch.float64), mask=torch.ones(1, 3))

    def test_rows_in_a_batch_are_independent(self):
        scores = torch.tensor([[1.0, 3.0, 0.0], [5.0, 7.0, 9.0]], dtype=torch.float64)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        got = combine("mean", scores, mask=mask).tolist()
        assert got == pytest.approx([2.0, 7.0])


# ── the head ──────────────────────────────────────────────────────────────────

class TestProbeHead:

    def test_score_is_w_dot_z_plus_b(self):
        head = ProbeHead(weight=torch.tensor([1.0, 2.0], dtype=torch.float64), bias=0.5)
        z = torch.tensor([[[1.0, 1.0], [2.0, 0.0]]], dtype=torch.float64)
        assert head.token_scores(z)[0].tolist() == pytest.approx([3.5, 2.5])

    def test_standardisation_uses_the_training_statistics(self):
        head = ProbeHead(
            weight=torch.tensor([1.0, 0.0], dtype=torch.float64),
            mean=torch.tensor([10.0, 0.0], dtype=torch.float64),
            std=torch.tensor([2.0, 1.0], dtype=torch.float64),
        )
        z = torch.tensor([[[12.0, 0.0]]], dtype=torch.float64)
        assert float(head.token_scores(z)[0, 0]) == pytest.approx(1.0)

    def test_a_constant_dimension_does_not_produce_an_infinite_score(self):
        """std 0 is a feature that never varied; it must contribute nothing."""
        head = ProbeHead(
            weight=torch.tensor([1.0], dtype=torch.float64),
            mean=torch.tensor([5.0], dtype=torch.float64),
            std=torch.tensor([0.0], dtype=torch.float64),
        )
        score = float(head.token_scores(torch.tensor([[[5.0]]], dtype=torch.float64))[0, 0])
        assert math.isfinite(score) and score == pytest.approx(0.0)

    def test_a_normalisation_of_the_wrong_width_is_refused(self):
        with pytest.raises(ValueError, match="must match its weights"):
            ProbeHead(
                weight=torch.tensor([1.0, 2.0]),
                mean=torch.tensor([0.0, 0.0, 0.0]),
            )

    def test_activations_of_the_wrong_width_are_refused(self):
        head = ProbeHead(weight=torch.tensor([1.0, 2.0], dtype=torch.float64))
        with pytest.raises(ValueError, match="d_model"):
            head.token_scores(torch.zeros(1, 3, 5, dtype=torch.float64))

    def test_the_attention_query_lives_on_the_head_so_it_can_be_exported(self):
        """IDL-53 requires the query inline. There was nowhere to put it, so an
        `attention` probe could be swept and then never exported."""
        head = ProbeHead(
            weight=torch.tensor([1.0, 0.0], dtype=torch.float64),
            attention_query=torch.tensor([0.0, 2.0], dtype=torch.float64),
        )
        z = torch.tensor([[[1.0, 3.0], [2.0, 1.0]]], dtype=torch.float64)
        assert head.attention_logits(z)[0].tolist() == pytest.approx([6.0, 2.0])

    def test_attention_logits_refuse_when_no_query_was_trained(self):
        head = ProbeHead(weight=torch.tensor([1.0, 0.0], dtype=torch.float64))
        with pytest.raises(ValueError, match="no attention_query"):
            head.attention_logits(torch.zeros(1, 2, 2, dtype=torch.float64))

    def test_a_query_of_the_wrong_width_is_refused(self):
        with pytest.raises(ValueError, match="must match its weights"):
            ProbeHead(weight=torch.tensor([1.0, 2.0]), attention_query=torch.tensor([1.0]))

    def test_the_query_is_standardised_with_the_scores_statistics(self):
        """It was trained in that space; using raw activations is a different rule."""
        head = ProbeHead(
            weight=torch.tensor([1.0, 0.0], dtype=torch.float64),
            mean=torch.tensor([0.0, 10.0], dtype=torch.float64),
            std=torch.tensor([1.0, 2.0], dtype=torch.float64),
            attention_query=torch.tensor([0.0, 1.0], dtype=torch.float64),
        )
        z = torch.tensor([[[0.0, 14.0]]], dtype=torch.float64)
        assert float(head.attention_logits(z)[0, 0]) == pytest.approx(2.0)

    def test_a_two_dimensional_weight_is_refused(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            ProbeHead(weight=torch.zeros(2, 2))


# ── rule semantics worth pinning ──────────────────────────────────────────────

class TestRuleSemantics:

    def test_softmax_at_low_temperature_approaches_max(self):
        scores = [0.0, 1.0, 5.0]
        assert combine_sequence("softmax", scores, tau=0.01) == pytest.approx(5.0, abs=1e-6)

    def test_softmax_at_tau_one_is_a_self_weighted_mean_above_the_plain_mean(self):
        scores = [0.0, 1.0, 5.0]
        assert combine_sequence("softmax", scores) > sum(scores) / 3

    def test_a_non_positive_temperature_is_refused(self):
        with pytest.raises(ValueError, match="temperature must be > 0"):
            combine_sequence("softmax", [1.0, 2.0], tau=0.0)

    def test_attention_weights_by_the_logits_not_by_the_scores(self):
        """Passing scores as logits would make this softmax under another name."""
        scores = [0.0, 10.0]
        by_logits = combine_sequence("attention", scores, attention_logits=[10.0, 0.0])
        by_scores = combine_sequence("softmax", scores)
        assert by_logits < 1.0, "the low-scoring token was weighted, as its logit said"
        assert by_logits != pytest.approx(by_scores)

    def test_attention_without_logits_is_refused(self):
        with pytest.raises(ValueError, match="needs `attention_logits`"):
            combine("attention", torch.zeros(1, 3, dtype=torch.float64))

    def test_rolling_mean_max_finds_a_burst_a_plain_mean_would_dilute(self):
        scores = [0.0] * 50 + [5.0] * 4
        assert combine_sequence("rolling_mean_max", scores, window=4) == pytest.approx(5.0)
        assert combine_sequence("mean", scores) < 0.5

    def test_a_window_below_one_is_refused(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            combine_sequence("rolling_mean_max", [1.0, 2.0], window=0)

    def test_an_unknown_rule_is_refused_by_combine(self):
        with pytest.raises(ValueError, match="unknown combining rule"):
            combine("median", torch.zeros(1, 3, dtype=torch.float64))


# ── round 2: a degenerate channel must contribute nothing ─────────────────────


class TestAZeroVarianceChannelIsZeroedNotAmplified:
    """`std.clamp_min(1e-6)` amplified by a million where it claimed to neutralise.

    The docstring said a clamped dimension "contributes nothing, which is correct
    for a feature that never varied". The code divided by 1e-6, so a drift of 0.001
    on that channel reached the dot product as 1000.0 — larger than every real
    channel combined, on the one channel known to carry no signal. A probe that
    fires is a probe that is believed, so this is the dangerous direction.
    """

    def _head(self, std_values):
        return ProbeHead(
            weight=torch.tensor([1.0, 1.0]),
            mean=torch.tensor([0.0, 0.0]),
            std=torch.tensor(std_values),
        )

    def test_a_drift_on_a_constant_channel_does_not_reach_the_score(self):
        head = self._head([1.0, 0.0])
        activations = torch.tensor([[0.5, 1e-3]])
        # channel 1 contributes 0, so the score is channel 0 alone
        assert head.token_scores(activations).item() == pytest.approx(0.5, abs=TOL)

    def test_the_clamping_version_would_have_scored_a_THOUSAND(self):
        """The magnitude, stated, so the fix cannot be read as cosmetic."""
        head = self._head([1.0, 0.0])
        activations = torch.tensor([[0.5, 1e-3]])
        clamped = 1e-3 / max(0.0, head.eps)      # what clamp_min(eps) produced
        assert clamped == pytest.approx(1000.0)
        assert head.token_scores(activations).item() < 1.0

    def test_on_TRAINING_data_the_two_treatments_agree_exactly(self):
        """A constant channel is exactly its mean in training, so centring gives 0.

        The change is therefore invisible on the data the statistics came from and
        bites only off it — which is the point.
        """
        head = ProbeHead(
            weight=torch.tensor([2.0, 3.0]),
            mean=torch.tensor([1.0, 7.0]),
            std=torch.tensor([1.0, 0.0]),
        )
        activations = torch.tensor([[1.5, 7.0]])          # channel 1 at its mean
        assert head.token_scores(activations).item() == pytest.approx(1.0, abs=TOL)

    def test_a_std_BELOW_eps_counts_as_degenerate_too(self):
        head = self._head([1.0, 1e-9])
        assert head.token_scores(torch.tensor([[0.0, 1.0]])).item() == pytest.approx(0.0, abs=TOL)

    def test_a_healthy_small_std_still_divides(self):
        """The guard must not swallow a real, merely small, standard deviation."""
        head = self._head([1.0, 0.01])
        assert head.token_scores(torch.tensor([[0.0, 0.02]])).item() == pytest.approx(2.0, abs=1e-5)

    def test_the_attention_query_sees_the_same_treatment(self):
        """Both readouts standardise, so both must neutralise the same channel."""
        head = ProbeHead(
            weight=torch.tensor([1.0, 1.0]),
            mean=torch.tensor([0.0, 0.0]),
            std=torch.tensor([1.0, 0.0]),
            attention_query=torch.tensor([0.0, 1.0]),
        )
        logits = head.attention_logits(torch.tensor([[[0.0, 1e-3], [0.0, 5e-3]]]))
        assert torch.allclose(logits, torch.zeros_like(logits), atol=TOL)
