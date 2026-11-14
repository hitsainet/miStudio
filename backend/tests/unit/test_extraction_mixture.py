"""A feature's examples come from every corpus the SAE trained on.

WHY THIS FILE EXISTS. Feature extraction read ONE dataset while the SAEs were
trained on five at 35/30/15/10/10. `activation_frequency` is
`feature_activation_counts / len(dataset)` and it gates dead-neuron deletion at
0.001, so a feature firing on 40% of code rows scores ~0 against OpenWebText
alone and is deleted WITH ITS EXAMPLES. On the mixture it scores 0.15 x 0.40 =
0.06 and survives. The single-corpus default was destroying the domain features
the estate exists to find.

The real corpora these numbers come from (blocks, 2026-09-18): OpenWebText
548,716 · OpenHermes 189,086 · Bloomberg 138,400 · small-the_pile 74,358 ·
codeparrot 55,926. The 10:1 spread between the largest and smallest is why the
no-weights default is EQUAL rather than proportional.

MUTATION CONTROLS (2026-09-18; each applied alone, source restored by bytes and
the sha256 verified):
  X1 `extraction_quotas` passes `None` through to `allocate_tokens`
        -> test_no_weights_is_equal_not_proportional
  X2 `batch_spans` iterates the whole range instead of per span
        -> test_a_batch_never_straddles_two_corpora
  X3 `calibration_indices` returns `range(batch_size)`
        -> test_calibration_draws_from_every_corpus
  X4 `plan_spans` drops empty spans
        -> test_a_corpus_allocated_nothing_still_has_a_span
  X5 single-span identity broken (any of the three)
        -> the TestOneCorpusIsExactlyTheOldBehaviour class
"""

import pytest

from src.services.extraction_mixture import (
    batch_spans,
    calibration_indices,
    describe_spans,
    extraction_quotas,
    plan_spans,
    span_of,
)

# The five corpora, in block counts measured on disk.
CORPORA = ["OpenWebText-2M", "OpenHermes-2.5", "Bloomberg", "small-the_pile", "codeparrot"]
IDS = ["ds_owt", "ds_oh", "ds_bloom", "ds_pile", "ds_code"]
BLOCKS = [548_716, 189_086, 138_400, 74_358, 55_926]
TRAINING_MIX = [0.30, 0.35, 0.10, 0.10, 0.15]


class TestQuotas:

    def test_no_weights_is_equal_not_proportional(self):
        """THE DEPARTURE. `allocate_tokens`' own default is proportional to
        availability; extraction wants equal, because the examples are a
        statement about every source. Proportional would give codeparrot 1.8% of
        the evidence for a dictionary that spent 15% of its training on code."""
        quotas = extraction_quotas(BLOCKS, 10_000)
        assert quotas == [2000, 2000, 2000, 2000, 2000]

        # And explicitly NOT the proportional answer.
        assert quotas[0] != pytest.approx(10_000 * 548_716 / sum(BLOCKS), abs=1)

    def test_weights_are_honoured(self):
        quotas = extraction_quotas(BLOCKS, 10_000, TRAINING_MIX)
        assert quotas == [3000, 3500, 1000, 1000, 1500]
        assert sum(quotas) == 10_000

    def test_a_small_corpus_is_never_over_drawn(self):
        """Asking a 500-row corpus for 2,000 rows must not invent rows."""
        quotas = extraction_quotas([500, 100_000], 4_000)
        assert quotas[0] == 500
        assert sum(quotas) == 4_000, "the shortfall goes to the corpus that has it"

    def test_a_length_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="one-to-one"):
            extraction_quotas([100, 100], 50, [1.0, 1.0, 1.0])

    def test_degenerate_inputs(self):
        assert extraction_quotas([], 100) == []
        assert extraction_quotas([100, 100], 0) == [0, 0]


class TestSpans:

    def test_spans_are_contiguous_and_global(self):
        spans = plan_spans(CORPORA, IDS, BLOCKS, 10_000, TRAINING_MIX)
        assert spans[0].start == 0
        # NOT strict: pairwise iteration zips a list against its own tail, so
        # the operands differ in length by one BY DESIGN. `strict=True` raises
        # here on every non-empty list. The three strict zips in the module
        # proper are a different case — there the operands must correspond.
        for earlier, later in zip(spans, spans[1:]):  # noqa: B905
            assert later.start == earlier.stop, "a gap would misalign sample_index"
        assert spans[-1].stop == 10_000

    def test_spans_stay_aligned_with_the_datasets_given(self):
        spans = plan_spans(CORPORA, IDS, BLOCKS, 10_000, TRAINING_MIX)
        assert [s.dataset_id for s in spans] == IDS
        assert [s.label for s in spans] == CORPORA

    def test_a_corpus_allocated_nothing_still_has_a_span(self):
        """Positional alignment is the contract: per-corpus statistics need an
        entry for every corpus the operator selected, including an empty one."""
        spans = plan_spans(CORPORA, IDS, BLOCKS, 10_000, [1, 1, 1, 1, 0])
        assert len(spans) == 5
        assert spans[4].rows == 0
        assert spans[4].start == spans[4].stop

    def test_start_sample_applies_per_corpus(self):
        """Each corpus contributes its own slice from that offset — the blocks
        are already permuted or measured unordered, so a prefix of each is
        representative of it."""
        spans = plan_spans(["a", "b"], ["ds_a", "ds_b"], [100, 100], 40, None, start_sample=90)
        # Only 10 rows are reachable in each after the offset.
        assert sum(s.rows for s in spans) == 20

    def test_mismatched_inputs_are_refused(self):
        with pytest.raises(ValueError, match="one-to-one"):
            plan_spans(["a"], ["ds_a", "ds_b"], [10, 10], 5)


class TestBatching:

    def test_a_batch_never_straddles_two_corpora(self):
        """THE CORRECTNESS REQUIREMENT. `batch_process_features` sums
        `fired_counts` over a whole call, so a straddling batch could not be
        attributed and the per-corpus frequencies would be unrecoverable."""
        spans = plan_spans(CORPORA, IDS, BLOCKS, 10_000, TRAINING_MIX)
        for start, stop in batch_spans(spans, 512):
            assert span_of(spans, start) == span_of(spans, stop - 1), (
                f"batch [{start}:{stop}] crosses a corpus boundary"
            )

    def test_every_allocated_row_is_read_exactly_once(self):
        spans = plan_spans(CORPORA, IDS, BLOCKS, 10_000, TRAINING_MIX)
        seen = [i for start, stop in batch_spans(spans, 512) for i in range(start, stop)]
        assert sorted(seen) == list(range(10_000))
        assert len(seen) == len(set(seen))

    def test_an_empty_corpus_contributes_no_batches(self):
        spans = plan_spans(["a", "b"], ["ds_a", "ds_b"], [100, 100], 50, [1, 0])
        assert all(span_of(spans, s) == 0 for s, _ in batch_spans(spans, 16))

    def test_a_non_positive_batch_size_is_refused(self):
        spans = plan_spans(["a"], ["ds_a"], [100], 50)
        with pytest.raises(ValueError):
            batch_spans(spans, 0)


class TestCalibration:

    def test_calibration_draws_from_every_corpus(self):
        """Calibrating on the first batch fixes `log_threshold` from whichever
        corpus sorts first, so a code feature's firing threshold would be set
        from chat activations and it would then read as dead on code."""
        spans = plan_spans(CORPORA, IDS, BLOCKS, 10_000, TRAINING_MIX)
        indices = calibration_indices(spans, 64)
        hit = {span_of(spans, i) for i in indices}
        assert hit == {0, 1, 2, 3, 4}, "every corpus must be represented"

    def test_it_stays_within_the_batch_budget(self):
        spans = plan_spans(CORPORA, IDS, BLOCKS, 10_000, TRAINING_MIX)
        assert len(calibration_indices(spans, 64)) <= 64

    def test_every_index_is_inside_an_allocated_span(self):
        spans = plan_spans(CORPORA, IDS, BLOCKS, 10_000, TRAINING_MIX)
        assert all(span_of(spans, i) >= 0 for i in calibration_indices(spans, 64))

    def test_an_empty_corpus_is_skipped_not_indexed(self):
        spans = plan_spans(["a", "b"], ["ds_a", "ds_b"], [100, 100], 50, [1, 0])
        assert {span_of(spans, i) for i in calibration_indices(spans, 16)} == {0}

    def test_no_populated_corpus_gives_no_indices(self):
        spans = plan_spans(["a"], ["ds_a"], [0], 10)
        assert calibration_indices(spans, 16) == []


class TestOneCorpusIsExactlyTheOldBehaviour:
    """Backward compatibility rests entirely on these three.

    Every existing extraction test drives a single dataset. If one corpus does
    not reproduce the old arithmetic byte for byte, the arc breaks them all.
    """

    def test_the_span_is_the_old_select_range(self):
        # The line replaced: end = min(start_sample + evaluation_samples, size)
        for size, evaluation_samples, start_sample in [
            (1000, 200, 0), (1000, 5000, 0), (1000, 200, 50), (100, 5000, 90),
        ]:
            spans = plan_spans(["a"], ["ds_a"], [size], evaluation_samples, None, start_sample)
            expected = min(evaluation_samples, max(0, size - start_sample))
            assert len(spans) == 1
            assert spans[0].start == 0
            assert spans[0].rows == expected

    def test_the_batches_are_the_old_range_loop(self):
        spans = plan_spans(["a"], ["ds_a"], [1000], 1000)
        n = spans[0].rows
        for batch_size in (1, 7, 64, 512, 4096):
            expected = [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]
            assert batch_spans(spans, batch_size) == expected

    def test_the_calibration_slice_is_the_old_head(self):
        spans = plan_spans(["a"], ["ds_a"], [1000], 1000)
        for batch_size in (8, 64, 4096):
            assert calibration_indices(spans, batch_size) == list(
                range(min(batch_size, 1000))
            )


class TestDescription:

    def test_it_states_what_will_actually_be_read(self):
        spans = plan_spans(CORPORA, IDS, BLOCKS, 10_000, TRAINING_MIX)
        text = describe_spans(spans)
        assert "OpenHermes-2.5=3,500 (35.0%)" in text
        assert "codeparrot=1,500 (15.0%)" in text
