"""A 16-bit model's probe batches are held as stored float16 and standardised at use (ENH-006 G24).

On 2026-11-04 the general SSN run was refused before approval: 68.39 GiB needed against 68.06 usable,
62.95 GiB of it batches that widened every stored float16 activation to float32. Holding them as
float16 halves that. The claim this file pins is that it changes NOTHING a probe learns: float16 to
float32 is exact and the standardisation is one function (`standardise_batch`), so the weights must be
bit-identical (`torch.equal`, never `allclose`) to the old float32 path on the same values.

⚠ THE FIXTURE IS FLOAT16 FROM THE START and the float32 rows are its exact upcast, so both paths see the
same numbers. Rows drawn in float32 and rounded to float16 for one side only would compare two datasets.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.services import probe_monitor_trainer as trainer
from src.services.probe_monitor_trainer import (
    _standardisation,
    _standardised_batches,
    forecast_batches,
    ready_batch,
    standardisation_terms,
    stored_as_float16,
    train_rule,
)

D_MODEL = 24
RULES = ["mean", "max", "last", "softmax", "attention", "rolling_mean_max"]


def make_rows(n=64, seed=3):
    """float16 rows of varied length, with a channel offset large enough that standardising matters
    and one constant (degenerate) channel."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=D_MODEL)
    rows, labels = [], []
    for i in range(n):
        y = i % 2
        length = int(rng.integers(4, 50))
        block = rng.normal(3.0, 2.0, size=(length, D_MODEL)) + (0.5 * direction if y else 0.0)
        block[:, 5] = 7.0
        rows.append(block.astype(np.float16))
        labels.append(y)
    return rows, labels


@pytest.fixture
def data16():
    rows, labels = make_rows()
    split = int(len(rows) * 0.75)
    return rows[:split], labels[:split], rows[split:], labels[split:]


def upcast(rows):
    return [r.astype(np.float32) for r in rows]


SMALL = 24 * 4 * 120  # a budget small enough that several length buckets form at test scale


class TestTheFixtureCanFail:
    def test_the_rows_form_several_buckets_and_one_channel_is_degenerate(self, data16):
        tr, _, _, _ = data16
        mean, std = _standardisation(tr)
        batches = _standardised_batches(tr, mean, std, torch.device("cpu"), budget_bytes=SMALL)
        assert len(batches) > 2
        assert float(std[5]) == 1.0  # the degenerate channel takes the zeroing branch

    def test_the_upcast_is_exact(self, data16):
        tr, _, _, _ = data16
        assert all(np.array_equal(a.astype(np.float32).astype(np.float16), a) for a in tr)


class TestHeldRaw:
    def test_float16_rows_are_held_as_float16_and_float32_rows_are_standardised(self, data16):
        tr, _, _, _ = data16
        mean, std = _standardisation(tr)
        raw = _standardised_batches(tr, mean, std, torch.device("cpu"), budget_bytes=SMALL)
        eager = _standardised_batches(
            upcast(tr), mean, std, torch.device("cpu"), budget_bytes=SMALL
        )
        assert stored_as_float16(tr) and not stored_as_float16(upcast(tr))
        assert {x.dtype for x, _m, _i in raw} == {torch.float16}
        assert {x.dtype for x, _m, _i in eager} == {torch.float32}

    def test_held_raw_is_half_the_memory_and_the_forecast_says_so(self, data16):
        tr, _, _, _ = data16
        mean, std = _standardisation(tr)
        raw = _standardised_batches(tr, mean, std, torch.device("cpu"), budget_bytes=SMALL)
        eager = _standardised_batches(
            upcast(tr), mean, std, torch.device("cpu"), budget_bytes=SMALL
        )
        held = sum(x.numel() * x.element_size() for x, _m, _i in raw)
        assert held * 2 == sum(x.numel() * x.element_size() for x, _m, _i in eager)
        widths = [r.shape[0] for r in tr]
        forecast = forecast_batches(
            widths, D_MODEL, budget_bytes=SMALL, element_bytes=2
        )
        assert forecast["bytes"] == held

    def test_ready_batch_gives_exactly_the_standardised_batch(self, data16):
        tr, _, _, _ = data16
        mean, std = _standardisation(tr)
        raw = _standardised_batches(tr, mean, std, torch.device("cpu"), budget_bytes=SMALL)
        eager = _standardised_batches(
            upcast(tr), mean, std, torch.device("cpu"), budget_bytes=SMALL
        )
        terms = standardisation_terms(mean, std, torch.device("cpu"))
        for (x, m, i), (y, n, j) in zip(raw, eager, strict=True):
            assert i == j and torch.equal(m, n)
            assert torch.equal(ready_batch(x, m, terms), y)
        assert ready_batch(eager[0][0], eager[0][1], terms) is eager[0][0]

    def test_the_ceiling_is_held_to_the_raw_bytes_not_float32(self, data16):
        """A 16-bit plan that fits at 2 B must not be refused against its float32 figure."""
        tr, _, _, _ = data16
        mean, std = _standardisation(tr)
        widths = [r.shape[0] for r in tr]
        held = forecast_batches(widths, D_MODEL, budget_bytes=SMALL, element_bytes=2)["bytes"]
        _standardised_batches(
            tr, mean, std, torch.device("cpu"), budget_bytes=SMALL, max_bytes=held
        )
        with pytest.raises(trainer.ProbeTrainingTooLarge, match=f"x {D_MODEL} dims x 2 B"):
            _standardised_batches(
                tr, mean, std, torch.device("cpu"), budget_bytes=SMALL, max_bytes=held - 1
            )

    @pytest.mark.parametrize("raw", [True, False])
    def test_a_degenerate_channel_is_zeroed_even_where_it_moved(self, raw):
        """A channel constant in TRAINING can differ in validation rows; it must still read zero, as
        `ProbeHead.standardise` serves it. (In the fixture above it never moves, so this is the case
        that can tell a skipped fill.)"""
        mean, std = torch.full((3,), 7.0), torch.tensor([2.0, 0.0, 2.0])
        terms = standardisation_terms(mean, std, torch.device("cpu"))
        x = torch.tensor([[[9.0, 9.0, 9.0], [5.0, 3.0, 1.0]]])
        mask = torch.tensor([[True, True]])
        out = (
            ready_batch(x.to(torch.float16), mask, terms)
            if raw
            else trainer.standardise_batch(x.clone(), mask, terms)
        )
        assert torch.equal(out, torch.tensor([[[1.0, 0.0, 1.0], [-1.0, 0.0, -3.0]]]))

    def test_a_batch_of_any_other_dtype_is_refused(self):
        x = torch.zeros((1, 2, 3), dtype=torch.float64)
        with pytest.raises(TypeError):
            ready_batch(
                x,
                torch.ones((1, 2), dtype=torch.bool),
                standardisation_terms(torch.zeros(3), torch.ones(3), torch.device("cpu")),
            )

    def test_the_bucket_plan_does_not_move_with_the_element_size(self, data16):
        widths = [r.shape[0] for r in data16[0]]
        two = forecast_batches(widths, D_MODEL, budget_bytes=SMALL, element_bytes=2)
        four = forecast_batches(widths, D_MODEL, budget_bytes=SMALL, element_bytes=4)
        assert (two["batches"], two["padded_slots"]) == (four["batches"], four["padded_slots"])
        assert two["bytes"] * 2 == four["bytes"]


class TestTheWeightsDoNotMove:
    @pytest.mark.parametrize("rule", RULES)
    def test_held_raw_trains_bit_identical_weights(self, data16, rule):
        tr, trl, va, val = data16
        held = train_rule(rule, tr, trl, va, val, seed=1337, epochs=20, batch_bytes=SMALL, hold_raw=True)
        eager = train_rule(
            rule, upcast(tr), trl, upcast(va), val, seed=1337, epochs=20, batch_bytes=SMALL
        )
        assert torch.equal(held.weight, eager.weight)
        assert held.bias == eager.bias
        assert held.val_auroc == eager.val_auroc and held.best_epoch == eager.best_epoch
        assert [h["loss"] for h in held.history] == [h["loss"] for h in eager.history]
        if rule == "attention":
            assert torch.equal(held.attention_query, eager.attention_query)

    def test_prebuilt_raw_batches_as_the_run_passes_them(self, data16):
        tr, trl, va, val = data16
        mean, std = _standardisation(tr)
        device = torch.device("cpu")
        prebuilt = (
            mean,
            std,
            _standardised_batches(tr, mean, std, device, budget_bytes=SMALL),
            _standardised_batches(va, mean, std, device, budget_bytes=SMALL),
        )
        held = train_rule("mean", tr, trl, va, val, seed=1337, epochs=20, prebuilt=prebuilt)
        eager = train_rule(
            "mean", upcast(tr), trl, upcast(va), val, seed=1337, epochs=20, batch_bytes=SMALL
        )
        assert torch.equal(held.weight, eager.weight) and held.bias == eager.bias

    def test_the_comparison_can_tell_a_different_standardisation(self, data16, monkeypatch):
        """Specificity: standardising raw batches with other statistics must move the weights."""
        tr, trl, va, val = data16
        eager = train_rule(
            "mean", upcast(tr), trl, upcast(va), val, seed=1337, epochs=20, batch_bytes=SMALL
        )
        real = trainer.standardisation_terms

        def shifted(mean, std, device):
            return real(mean + 0.25, std, device)

        monkeypatch.setattr(trainer, "standardisation_terms", shifted)
        moved = train_rule("mean", tr, trl, va, val, seed=1337, epochs=20, batch_bytes=SMALL, hold_raw=True)
        assert not torch.equal(moved.weight, eager.weight)
