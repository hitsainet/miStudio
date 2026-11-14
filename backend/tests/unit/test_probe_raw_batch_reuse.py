"""A held-raw batch is widened into ONE reused buffer, a cache-sized run at a time (ENH-006 G25).

G24 held a 16-bit model's batches as float16 and standardised each at use, which was right for memory
and cost the training stage five times its time: arm A trained 9,184 s against arm B's 1,848 s for the
same epochs (42 attention, 41 max) on the node, because every use made a fresh float32 copy and then
passed over it four times in main memory. G24's benchmark timed the standardise step on the
workstation, never the stage.

What this file pins:
- the chunked path gives the SAME BITS as standardising the whole widened batch, at every run size,
  including runs that do not divide the batch, a degenerate channel and padding;
- `train_rule` widens every held-raw batch into the one buffer it allocates once: asserted on every
  call (what is passed, what comes back, how many calls), not on the helper alone;
- the runs are cache-sized: a batch larger than a run is done in several, each within the budget.
The weights staying bit-identical to the float32 path is pinned by `test_probe_batches_held_raw.py`,
whose training tests now run through this path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.services import probe_monitor_trainer as trainer
from src.services.probe_monitor_trainer import (
    _standardisation,
    _standardised_batches,
    raw_batch_scratch,
    ready_batch,
    standardisation_terms,
    standardise_batch,
    standardise_into,
    train_rule,
)

D_MODEL = 24
SMALL = D_MODEL * 4 * 120  # several length buckets at test scale
CPU = torch.device("cpu")


def make_rows(n=48, seed=11):
    """float16 rows of varied length (so batches carry padding), with an offset that makes
    standardising matter and one constant, degenerate channel."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for i in range(n):
        length = int(rng.integers(3, 40))
        block = rng.normal(2.0, 3.0, size=(length, D_MODEL)) + (0.7 if i % 2 else 0.0)
        block[:, 7] = -4.0
        rows.append(block.astype(np.float16))
        labels.append(i % 2)
    return rows, labels


@pytest.fixture
def batches():
    rows, _labels = make_rows()
    mean, std = _standardisation(rows)
    held = _standardised_batches(rows, mean, std, CPU, budget_bytes=SMALL)
    # `_standardisation` gives a constant channel std 1.0, so the zeroing branch is reached only by
    # statistics handed in directly; a zero here makes the fixture exercise it.
    std = std.clone()
    std[7] = 0.0
    return held, standardisation_terms(mean, std, CPU)


def bits(t):
    return t.contiguous().view(torch.int32)


class TestTheFixtureCanFail:
    def test_batches_are_raw_padded_and_one_channel_is_degenerate(self, batches):
        held, terms = batches
        assert all(x.dtype == torch.float16 for x, _m, _i in held)
        assert any(not bool(m.all()) for _x, m, _i in held), "no padding to mask"
        assert bool(terms[2].any()), "no degenerate channel"
        assert len(held) > 1


class TestTheSameBits:
    @pytest.mark.parametrize("slots_per_run", [1, 7, 64, 10_000])
    def test_any_run_size_gives_the_whole_batch_bits(self, batches, monkeypatch, slots_per_run):
        held, terms = batches
        monkeypatch.setattr(trainer, "STANDARDISE_CHUNK_BYTES", slots_per_run * D_MODEL * 4)
        for x, mask, _i in held:
            whole = standardise_batch(x.to(torch.float32), mask, terms)
            out = torch.full(x.shape, float("nan"), dtype=torch.float32)
            standardise_into(out, x, mask, terms)
            assert torch.equal(bits(out), bits(whole))

    def test_ready_batch_with_scratch_gives_the_whole_batch_bits(self, batches):
        held, terms = batches
        scratch = raw_batch_scratch(held, CPU)
        for x, mask, _i in held:
            whole = standardise_batch(x.to(torch.float32), mask, terms)
            assert torch.equal(bits(ready_batch(x, mask, terms, scratch)), bits(whole))


class TestTheArithmeticItself:
    """⚠ Every comparison above is against `standardise_batch`, which shares `_standardise_in_place`
    with the chunked path, so a change to the arithmetic moves both sides at once and they still
    agree (mutation controls M6, M7 survived until this class existed). This spells the formula out
    independently: (x - mean) / std, a degenerate channel 0, a padded slot 0."""

    def test_ready_batch_is_the_formula_written_out(self, batches):
        held, terms = batches
        mean, divisor, degenerate = terms
        scratch = raw_batch_scratch(held, CPU)
        for x, mask, _i in held:
            expected = (x.to(torch.float32) - mean) / divisor
            expected[..., degenerate] = 0.0
            expected = torch.where(mask.unsqueeze(-1), expected, torch.zeros(()))
            got = ready_batch(x, mask, terms, scratch)
            # Values, not bits: the mask multiplies a negative pad to -0.0, which `where` writes as
            # +0.0; equal as numbers, and only the number reaches the forward.
            assert torch.equal(got, expected)

    def test_a_padded_slot_is_exactly_zero(self, batches):
        held, terms = batches
        scratch = raw_batch_scratch(held, CPU)
        padded = 0
        for x, mask, _i in held:
            got = ready_batch(x, mask, terms, scratch)
            assert bool((got[~mask] == 0).all())
            padded += int((~mask).sum())
        assert padded > 0


class TestTheBufferIsReused:
    def test_scratch_is_the_largest_raw_batch_and_none_without_one(self, batches):
        held, _terms = batches
        scratch = raw_batch_scratch(held, CPU)
        assert scratch.dtype == torch.float32
        assert scratch.numel() == max(x.numel() for x, _m, _i in held)
        widened = [(x.to(torch.float32), m, i) for x, m, i in held]
        assert raw_batch_scratch(widened, CPU) is None

    def test_ready_batch_returns_a_view_of_the_scratch(self, batches):
        held, terms = batches
        scratch = raw_batch_scratch(held, CPU)
        for x, mask, _i in held:
            assert ready_batch(x, mask, terms, scratch).data_ptr() == scratch.data_ptr()

    def test_train_rule_widens_every_raw_batch_into_one_buffer(self, monkeypatch):
        rows, labels = make_rows(n=64)
        tr, trl, va, val = rows[:48], labels[:48], rows[48:], labels[48:]
        real_ready = trainer.ready_batch
        real_scratch = trainer.raw_batch_scratch
        scratches, calls = [], []

        def spy_scratch(batches_, device):
            buffer = real_scratch(batches_, device)
            scratches.append(buffer)
            return buffer

        def spy_ready(x, mask, terms, scratch=None):
            out = real_ready(x, mask, terms, scratch)
            calls.append((x.dtype, scratch, out.data_ptr()))
            return out

        monkeypatch.setattr(trainer, "raw_batch_scratch", spy_scratch)
        monkeypatch.setattr(trainer, "ready_batch", spy_ready)
        epochs = 4
        train_rule(
            "mean", tr, trl, va, val, seed=1337, epochs=epochs, patience=100, batch_bytes=SMALL,
            hold_raw=True,
        )

        assert len(scratches) == 1 and scratches[0] is not None
        buffer = scratches[0]
        n_train = len(_standardised_batches(tr, *_standardisation(tr), CPU, budget_bytes=SMALL))
        mean, std = _standardisation(tr)
        n_val = len(_standardised_batches(va, mean, std, CPU, budget_bytes=SMALL))
        assert len(calls) == epochs * (n_train + n_val)
        for dtype, scratch, pointer in calls:
            assert dtype == torch.float16
            assert scratch is buffer
            assert pointer == buffer.data_ptr()


class TestRunsAreCacheSized:
    def test_the_run_budget_is_cache_sized(self):
        assert 256 * 1024 <= trainer.STANDARDISE_CHUNK_BYTES <= 16 * 1024**2

    def test_a_batch_larger_than_a_run_is_done_in_several_within_budget(self, batches, monkeypatch):
        held, terms = batches
        slots_per_run = 16
        monkeypatch.setattr(trainer, "STANDARDISE_CHUNK_BYTES", slots_per_run * D_MODEL * 4)
        real_run = trainer._standardise_run
        runs = []

        def spy_run(out, raw, keep, terms_):
            runs.append(out.shape[0])
            real_run(out, raw, keep, terms_)

        monkeypatch.setattr(trainer, "_standardise_run", spy_run)
        x, mask, _i = max(held, key=lambda b: b[0].numel())
        slots = x.shape[0] * x.shape[1]
        assert slots > slots_per_run
        standardise_into(torch.empty(x.shape, dtype=torch.float32), x, mask, terms)
        assert len(runs) == math.ceil(slots / slots_per_run)
        assert sum(runs) == slots and max(runs) <= slots_per_run
