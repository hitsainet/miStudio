"""Training batches: length bucketing, an exact full-batch gradient, and a forecast.

⚠ WHY THIS FILE EXISTS. The first Stage 1 acceptance run was OOMKilled (exit 137) at the
training stage, after the 17-minute layer sweep and the 14-minute token capture had both
succeeded. `_pack` right-padded every row to the longest one and trained full-batch over
the result. Measured on the real data — 8,000 rows of `Arrrlex/models-under-pressure`,
d=4096, 1,003,703 scored tokens, median 89, p99 476, max 1384:

    6,800 rows x 1,384 x 4,096 x 4 B = 154.2 GB     for 16.4 GB of actual activations

an ELEVEN-FOLD padding waste on a 124 GB node. Nothing in the suite could see it: the
trainer's tests use a handful of rows of 6 to 12 tokens, where padding to the longest row
costs nothing and one batch is the whole dataset. The fixtures agreed with the design by
construction, which is this repo's most expensive recurring shape.

THE LOAD-BEARING PROPERTY IS THAT BATCHING CHANGED NOTHING. Gradient accumulation with a
summed loss divided by the row count is exactly the full-batch gradient, so the result
must not depend on how the rows were grouped. That is asserted directly, by training the
same data twice under different budgets, because a per-batch `mean` loss — the obvious way
to write this — is a DIFFERENT objective that would weight a short batch as heavily as a
long one and would still look like it worked.

MUTATION CONTROLS (each verified to fail this file):
  B1  `reduction="sum"` back to the default mean       → the anchored first-epoch loss
  B2  the `/ n_train` division dropped                 → the anchored first-epoch loss
  B3  buckets built in row order instead of by length  → the waste-factor test
  B4  the forecast ceiling removed                     → the refusal test
  B5  predictions returned in batch order              → the row-order test
  B6  padding left non-zero after standardisation      → the pad-is-zero test
  B7  `slots_for` ignoring d_model (the 1000x error)   → the budget-units test

⚠ WHY THE COMPARISON IS ON THE FIRST EPOCH'S LOSS AND NOT ON THE FINAL WEIGHTS. Measured:
the accumulated gradient agrees with the single-batch gradient to 1.8e-7, which is float
noise from a different summation order — but AdamW divides by sqrt(v), so that noise
becomes a visible parameter difference within ~25 epochs (0.81 vs 0.99 on one weight).
Comparing final weights would therefore be a flaky test of floating-point associativity.
The first epoch's loss from a zero init is exact, deterministic, and has a closed-form
value — ln 2 — so it pins the reduction absolutely rather than relatively.
"""
import numpy as np
import pytest
import torch

from src.services.probe_monitor_trainer import (
    DEFAULT_BATCH_BYTES,
    MAX_BATCH_BYTES,
    ProbeTrainingTooLarge,
    _standardisation,
    _standardised_batches,
    forecast_batches,
    plan_length_buckets,
    slots_for,
    train_rule,
)

#: The measured shape of the real training set, scaled down in `d` so the test is fast:
#: the same length distribution is what produced the 11x waste.
#: 40 padded slots at d_model=8, fp32 — small enough that the 24-row fixture below really
#: becomes several batches. Asserted, not assumed, by `test_the_split_really_did_produce…`.
_TINY_BUDGET = 40 * 8 * 4

MEASURED = {"median": 89, "p99": 476, "max": 1384, "rows": 8000, "tokens": 1003703}


def _lengths_like_the_real_data(rows=400, seed=0):
    """A long-tailed distribution with the measured median and maximum."""
    rng = np.random.default_rng(seed)
    lengths = rng.lognormal(mean=np.log(MEASURED["median"]), sigma=0.9, size=rows)
    lengths = np.clip(lengths.astype(int), 1, MEASURED["max"])
    lengths[0] = MEASURED["max"]          # the one long row that used to set the width
    return [int(v) for v in lengths]


def _rows(lengths, d_model=8, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n, d_model)).astype(np.float32) for n in lengths]


class TestBucketingRemovesThePaddingWaste:
    def test_the_waste_at_the_real_scale_is_near_one(self):
        """8,000 rows is the acceptance set's size, and the size matters: with fewer rows
        each batch spans a wider slice of the length distribution, so the waste is higher
        (about 1.9x at 400 rows). The claim is about the scale this runs at."""
        lengths = _lengths_like_the_real_data(rows=6800)
        forecast = forecast_batches(lengths, 4096)
        assert forecast["waste_factor"] < 1.2, forecast
        assert forecast["batches"] > 1, forecast

    def test_row_order_grouping_would_waste_an_order_of_magnitude_more(self):
        """The control for the claim above: the same budget WITHOUT sorting is much worse,
        so the sort is doing the work rather than the budget."""
        lengths = _lengths_like_the_real_data(rows=6800)
        sorted_forecast = forecast_batches(lengths, 4096)
        # One batch, the shape `_pack` produced.
        unsorted_slots = len(lengths) * max(lengths)
        unsorted_waste = unsorted_slots / sum(lengths)
        assert unsorted_waste > 5 * sorted_forecast["waste_factor"], (
            f"unsorted waste {unsorted_waste:.2f}x vs sorted "
            f"{sorted_forecast['waste_factor']:.2f}x"
        )

    def test_every_row_appears_exactly_once(self):
        lengths = _lengths_like_the_real_data(rows=137)
        buckets = plan_length_buckets(lengths, budget_slots=5000)
        flat = [index for batch in buckets for index in batch]
        assert sorted(flat) == list(range(len(lengths)))

    def test_batches_respect_the_budget_except_for_a_single_oversized_row(self):
        lengths = [10, 10, 10, 900]
        buckets = plan_length_buckets(lengths, budget_slots=100)
        for batch in buckets:
            width = max(lengths[i] for i in batch)
            assert len(batch) * width <= 100 or len(batch) == 1, (batch, width)

    def test_a_row_wider_than_the_budget_is_kept_not_dropped(self):
        lengths = [5, 5, 4096]
        buckets = plan_length_buckets(lengths, budget_slots=10)
        assert sorted(i for b in buckets for i in b) == [0, 1, 2]

    def test_no_rows_is_no_batches(self):
        assert plan_length_buckets([], budget_slots=10) == []

    def test_a_zero_budget_is_refused(self):
        with pytest.raises(ValueError):
            plan_length_buckets([3, 4], budget_slots=0)


class TestTheForecastRefusesBeforeAllocating:
    """An OOM kill writes no status, takes the worker down and leaves the row claiming to
    run until a janitor notices 90 minutes later. A forecast is the same information an
    hour earlier, with the numbers in the message."""

    def test_the_real_shape_is_forecast_accurately(self):
        lengths = [89] * 10
        forecast = forecast_batches(lengths, 4096)
        assert forecast["padded_slots"] == 890
        assert forecast["bytes"] == 890 * 4096 * 4
        assert forecast["rows"] == 10
        assert forecast["widest_row"] == 89

    def test_an_oversized_plan_raises_with_the_numbers_in_it(self):
        rows = _rows([4] * 4, d_model=8)
        mean, std = _standardisation(rows)
        with pytest.raises(ProbeTrainingTooLarge) as caught:
            _standardised_batches(rows, mean, std, torch.device("cpu"), max_bytes=8)
        message = str(caught.value)
        assert "GiB" in message and "padded slots" in message
        assert "max_length" in message, "the refusal must say what the operator can change"

    def test_the_ceiling_is_below_the_nodes_memory(self):
        """48 GiB on a 124 GB node shared with a 16 GB model, five containers and the page
        cache that makes the token memmap readable at all."""
        assert MAX_BATCH_BYTES < 64 * 1024 ** 3

    def test_the_old_single_tensor_would_have_been_refused(self):
        """The regression stated as a number: the 154 GB allocation is above the ceiling,
        so even without bucketing this run would now fail fast instead of being killed."""
        assert 6800 * 1384 * 4096 * 4 > MAX_BATCH_BYTES


class TestPaddingCarriesNoValue:
    def test_pad_positions_are_zero_after_standardisation(self):
        """`combine` masks them, but a non-zero pad would still reach `max` through any
        mask bug, and zero is the honest filler."""
        rows = _rows([2, 7], d_model=4)
        mean, std = _standardisation(rows)
        batches = _standardised_batches(rows, mean, std, torch.device("cpu"))
        for values, mask, _indices in batches:
            padded = values[~mask]
            if padded.numel():
                assert torch.all(padded == 0), padded

    def test_the_mask_marks_exactly_the_real_tokens(self):
        lengths = [3, 11, 5]
        rows = _rows(lengths, d_model=4)
        mean, std = _standardisation(rows)
        batches = _standardised_batches(rows, mean, std, torch.device("cpu"))
        seen = {}
        for _values, mask, indices in batches:
            for position, row_index in enumerate(indices):
                seen[row_index] = int(mask[position].sum().item())
        assert seen == {0: 3, 1: 11, 2: 5}


class TestTheResultDoesNotDependOnTheBATCHING:
    """⚠ THE LOAD-BEARING TEST. Gradient accumulation with a summed loss divided by the row
    count is EXACTLY the full-batch gradient, so grouping the rows differently must change
    nothing. A per-batch `mean` loss — the obvious way to write the loop — is a different
    objective, and it would still train, still report a plausible AUROC, and be impossible
    to notice without this comparison.
    """

    @staticmethod
    def _data(rows=24, d_model=6, seed=7):
        rng = np.random.default_rng(seed)
        lengths = [int(v) for v in rng.integers(2, 25, size=rows)]
        labels = [index % 2 for index in range(rows)]
        data = []
        for length, label in zip(lengths, labels):
            block = rng.normal(size=(length, d_model)).astype(np.float32)
            block[:, 0] += 2.0 * label            # a real, learnable signal
            data.append(block)
        split = rows // 2
        return data[:split], labels[:split], data[split:], labels[split:]

    @pytest.mark.parametrize("rule", ["mean", "max", "last", "attention"])
    def test_one_batch_and_many_batches_agree(self, rule):
        train_rows, train_labels, val_rows, val_labels = self._data()
        whole = train_rule(
            rule, train_rows, train_labels, val_rows, val_labels,
            epochs=25, patience=25, seed=11, batch_bytes=1 << 30,
        )
        split = train_rule(
            rule, train_rows, train_labels, val_rows, val_labels,
            epochs=25, patience=25, seed=11, batch_bytes=_TINY_BUDGET,
        )
        # The decisive comparison: the first epoch's loss, from a zero init, before any
        # parameter update. Grouping-invariant if and only if the reduction is a sum over
        # rows divided by the row count.
        assert whole.history[0]["loss"] == pytest.approx(
            split.history[0]["loss"], abs=1e-6
        ), (
            f"{rule}: first-epoch loss {whole.history[0]['loss']} in one batch vs "
            f"{split.history[0]['loss']} split — the reduction is not full-batch"
        )
        # And the absolute anchor, independent of any comparison: at w=0, b=0 every
        # aggregate is 0, so BCE-with-logits is ln 2 per row for any label.
        for trained in (whole, split):
            assert trained.history[0]["loss"] == pytest.approx(np.log(2.0), abs=1e-6), (
                f"{rule}: first-epoch loss {trained.history[0]['loss']} is not ln 2, so "
                f"the loss is not the mean over rows"
            )
        # Behaviour, loosely: AdamW amplifies the 1e-7 summation-order difference, so the
        # final weights are NOT expected to match bit-for-bit — only the probe's quality.
        assert whole.val_auroc == pytest.approx(split.val_auroc, abs=0.05), (
            f"{rule}: {whole.val_auroc} in one batch vs {split.val_auroc} split"
        )

    def test_the_split_really_did_produce_several_batches(self):
        """Without this the test above could pass by producing one batch either way."""
        train_rows, *_ = self._data()
        mean, std = _standardisation(train_rows)
        batches = _standardised_batches(
            train_rows, mean, std, torch.device("cpu"), budget_bytes=_TINY_BUDGET
        )
        assert len(batches) > 3, f"only {len(batches)} batch(es); the comparison is vacuous"

    def test_predictions_come_back_in_ROW_order(self):
        """The batches are sorted by length, so returning them in batch order would pair
        every prediction with another row's label — an AUROC over a permutation, which
        reads as a bad probe rather than as a bug."""
        train_rows, train_labels, val_rows, val_labels = self._data()
        trained = train_rule(
            "mean", train_rows, train_labels, val_rows, val_labels,
            epochs=40, patience=40, seed=3, batch_bytes=_TINY_BUDGET,
        )
        # The signal is strong and separable, so a correctly ordered evaluation is far
        # above chance; a permuted one sits at chance.
        assert trained.val_auroc is not None and trained.val_auroc > 0.9, trained.val_auroc


class TestTheBudgetIsInBytes:
    """⚠ THE FIRST VERSION OF THIS FIX DID NOTHING, and its own comment was the reason.

    It budgeted 32,000,000 padded SLOTS and claimed that was "512 MB at d=4096". It is
    524 GB. At that budget every one of the 6,800 rows still fell into ONE batch, the
    length bucketing never engaged, and the forecast still read 154 GB — a fix that
    changed nothing, found by asserting the waste factor rather than by reading the code.
    """

    def test_the_slot_count_scales_inversely_with_d_model(self):
        assert slots_for(4096) == DEFAULT_BATCH_BYTES // (4096 * 4)
        assert slots_for(8) == DEFAULT_BATCH_BYTES // (8 * 4)
        assert slots_for(8) > slots_for(4096) * 100

    def test_the_default_budget_really_bounds_a_batch(self):
        """The property the slot version lost: one batch's activations fit the budget."""
        lengths = _lengths_like_the_real_data(rows=6800)
        forecast = forecast_batches(lengths, 4096)
        assert forecast["batches"] > 1, forecast
        biggest = forecast["slots_per_batch"] * 4096 * 4
        assert biggest <= DEFAULT_BATCH_BYTES * 1.001, biggest

    def test_the_real_plan_fits_the_ceiling(self):
        """The regression, restated: 6,800 rows at d=4096 must now forecast well under the
        ceiling that the 154 GB single tensor blew through."""
        lengths = _lengths_like_the_real_data(rows=6800)
        forecast = forecast_batches(lengths, 4096)
        assert forecast["bytes"] < MAX_BATCH_BYTES, forecast
        assert forecast["waste_factor"] < 1.2, forecast

    def test_a_zero_d_model_is_refused(self):
        with pytest.raises(ValueError):
            slots_for(0)

    def test_at_least_one_slot_survives_an_absurd_d_model(self):
        assert slots_for(1 << 30, budget_bytes=16) == 1


class TestTheStandardisationIsStreamed:
    """⚠ IT USED TO CONCATENATE EVERY TRAINING TOKEN INTO ONE ARRAY.

    850,425 x 4096 x 4 B = 13.9 GB on the acceptance set, and the list comprehension
    feeding `np.concatenate` held a second full copy while the concatenation ran — about
    28 GB peak, on top of the 8 GB the rows occupy and the 14 GB the batches will. Same
    class of allocation as the 154 GB padded tensor, just smaller.

    The replacement must be EXACT, not approximately right, because these numbers travel
    with the head: a probe standardised with the wrong statistics is a different detector.

    MUTATION CONTROLS:
      B8   one-pass `E[x^2] - E[x]^2`                      → the large-mean test
      B9   the sample std (ddof=1) instead of population   → the exactness test
      B10  the degenerate-channel floor dropped            → the degenerate test
    """

    def test_it_matches_a_concatenated_computation_exactly(self):
        rng = np.random.default_rng(5)
        rows = [rng.normal(size=(int(n), 9)).astype(np.float32) for n in rng.integers(1, 40, 25)]
        mean, std = _standardisation(rows)
        stacked = np.concatenate([np.asarray(r, dtype=np.float32) for r in rows], axis=0)
        assert np.allclose(mean.numpy(), stacked.mean(axis=0), atol=1e-5)
        assert np.allclose(std.numpy(), stacked.std(axis=0), atol=1e-5)

    def test_it_is_stable_when_the_mean_dwarfs_the_spread(self):
        """The reason for two passes. `E[x^2] - E[x]^2` on values around 1e4 with a spread
        of 1e-2 cancels catastrophically in float32 and can even go negative."""
        rng = np.random.default_rng(6)
        rows = [
            (10_000.0 + rng.normal(scale=0.01, size=(50, 4))).astype(np.float32)
            for _ in range(20)
        ]
        _mean, std = _standardisation(rows)
        assert np.all(np.isfinite(std.numpy()))
        assert np.all(std.numpy() > 0)
        assert np.allclose(std.numpy(), 0.01, rtol=0.25), std

    def test_a_degenerate_channel_gets_one_not_a_floor(self):
        rows = [np.concatenate(
            [np.full((6, 1), 3.0, dtype=np.float32), np.random.randn(6, 1).astype(np.float32)],
            axis=1,
        )]
        _mean, std = _standardisation(rows)
        assert std.numpy()[0] == 1.0, "a constant channel must get std 1.0, never a tiny floor"

    def test_empty_rows_are_skipped_not_counted(self):
        rows = [np.zeros((0, 3), dtype=np.float32), np.ones((4, 3), dtype=np.float32)]
        mean, std = _standardisation(rows)
        assert np.allclose(mean.numpy(), 1.0)
        assert np.all(std.numpy() == 1.0)      # constant channel → 1.0

    def test_no_rows_at_all_is_refused(self):
        with pytest.raises(ValueError):
            _standardisation([])

    def test_all_empty_rows_are_refused(self):
        with pytest.raises(ValueError):
            _standardisation([np.zeros((0, 3), dtype=np.float32)])

    def test_nothing_concatenates_the_whole_set(self):
        """Asserted structurally: a concatenate over a comprehension of every row is the
        allocation this replaced, and it would pass every numeric test above."""
        import ast
        import inspect

        from src.services import probe_monitor_trainer

        tree = ast.parse(inspect.getsource(probe_monitor_trainer._standardisation).lstrip())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            assert node.func.attr not in {"concatenate", "vstack", "stack"}, (
                f"_standardisation calls np.{node.func.attr}, which materialises every "
                f"training token in one array"
            )
