"""The on-the-fly path's buffer: whole blocks, a shared permutation, weights, no repeats, resumable.

SAE training remediation item 3 (2026-09-15). Training with no extractions drew
``batch_size`` random ROWS with replacement every step, kept ``batch_size``
random tokens from them, gave each layer its own random positions, ignored
``dataset_weights`` and crashed at step 0. ``ModelActivationSource`` replaces
that with the rolling buffer's contract and the base model as the reader.

Every token a fake capture returns ENCODES its origin — source, row and
position in its first column, the layer in the second — so what the buffer
serves is checked exactly, not statistically. The tiny real model runs are in
``test_training_on_the_fly_e2e.py``.

MUTATION CONTROLS (2026-09-15). Each broke one line of
src/services/model_activation_source.py or src/services/activation_mask.py, ran the listed
tests, and was restored from the original bytes with the sha256 and `git diff` re-checked.
  D15  the second layer written through a reversed permutation
         -> test_one_permutation_serves_every_layer, and the E2E same-positions test
  D16  no shuffle (torch.arange for the permutation)
         -> test_a_batch_holds_tokens_from_many_rows, test_a_batch_draws_from_both_sources,
            and two E2E tests (4)
  D17  rows drawn at random with replacement instead of by RowCycle
         -> no-repeat, held-out, padding, round-trip and retry tests, two E2E tests (15)
  D19  the capture flattened without the mask (pads written as tokens)
         -> test_no_pad_position_is_ever_served, test_no_token_repeats_..., two E2E tests
  D23  the served selection recorded AFTER the refill, not before it
         -> test_a_loaded_source_serves_exactly_what_the_original_would[x8]
  D24  load_state_dict resumes at position 0
         -> the round-trip test at every non-zero position (6)
  D25  one forward over every planned row (micro-batching ignored)
         -> test_no_forward_takes_more_rows_than_the_micro_batch[x3], the context-manager test
  D26  a context-manager capture never entered around the refill
         -> test_a_context_manager_capture_is_entered_around_each_refill
  D30  split_documents tests the flat index, not its document, against the held set
         -> three test_activation_mask split tests, test_split_rows_holds_out_what_split_documents_would
  D31  split_rows seeds its choice differently from split_documents
         -> test_split_rows_holds_out_what_split_documents_would
  D33  the row digest dropped from the state checks
         -> FIRST RUN: SURVIVED. The `rows` case held out an extra row, which changed the
            row COUNT, so the cycle's length check refused it anyway — the fixture agreed
            with the defect. Added `same_count_other_rows` (one split swapped for another of
            the same size). Re-run: KILLED by
            test_a_state_for_another_configuration_is_refused[same_count_other_rows]
  D35  load_state_dict replays without restoring the saved selection
         -> the round-trip test (5)
  D38  a failed refill no longer restores the cycles and generator
         -> test_a_forward_that_fails_is_retried_on_the_same_rows
The training task's wiring of this source is controlled in test_training_data_path_e2e.py.
"""

import io
import logging

import numpy as np
import pytest
import torch

from src.services import activation_mask, dataset_mixture
from src.services import model_activation_source as MAS

CPU = torch.device("cpu")
KEYS = [(3, "residual"), (5, "residual")]
PAD = 7_777_777
D = 4


def _id(source: int, row: int, pos: int) -> int:
    return source * 1_000_000 + row * 1_000 + pos


def _origin(value: float):
    value = int(round(float(value)))
    return value // 1_000_000, (value // 1_000) % 1_000, value % 1_000


class _Rows:
    """A tokenized dataset stand-in: rows pre-padded to ``width``, as Arrow returns them."""

    def __init__(self, source: int, lengths, width: int, with_mask: bool = True):
        self.source, self.lengths, self.width, self.with_mask = source, list(lengths), width, with_mask
        self.reads = 0

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, row):
        self.reads += 1
        real = self.lengths[row]
        ids = [_id(self.source, row, p) for p in range(real)] + [PAD] * (self.width - real)
        item = {"input_ids": ids}
        if self.with_mask:
            item["attention_mask"] = [1] * real + [0] * (self.width - real)
        return item


class _Capture:
    """Fake model forward: each (layer, hook) output encodes the token id and the layer."""

    def __init__(self, keys=KEYS):
        self.keys = keys
        self.calls = []

    def __call__(self, padded, masks):
        ids = torch.tensor(padded, dtype=torch.float64)
        self.calls.append(len(padded))
        out = {}
        for layer, hook in self.keys:
            acts = torch.zeros(ids.shape[0], ids.shape[1], D, dtype=torch.float64)
            acts[..., 0] = ids
            acts[..., 1] = layer
            out[(layer, hook)] = acts.float()
        return out


def _source(source=0, rows=40, length=16, width=16, lengths=None, holdout_rows=()):
    lengths = list(lengths) if lengths is not None else [length] * rows
    data = _Rows(source, lengths, width)
    train = np.array([r for r in range(len(lengths)) if r not in set(holdout_rows)], dtype=np.int64)
    return MAS.TokenRowSource(label=f"ds_{source}", dataset=data, rows=train, max_row_tokens=width)


def _build(sources, quotas, *, seed=0, micro=3, capture=None, prefill=True, keys=KEYS):
    return MAS.ModelActivationSource(
        sources, keys, quotas, capture=capture or _Capture(keys), hidden_dim=D, seed=seed,
        storage_device=CPU, train_device=CPU, micro_batch_rows=micro, pad_token_id=PAD, prefill=prefill,
    )


def _origins(tensor):
    return [_origin(v) for v in tensor[:, 0].tolist()]


def _buffer_rows(buf, key=KEYS[0]):
    return {(s, r) for s, r, _ in _origins(buf.tensors[key])}


class TestBatchesMixManyBlocks:
    def test_a_batch_holds_tokens_from_many_rows(self):
        buf = _build([_source(rows=40)], [10 * 16])
        batch = buf.next_batch(32)[KEYS[0]]
        rows = {r for _, r, _ in _origins(batch)}
        assert len(rows) >= 6, f"a 32-token batch came from {len(rows)} rows: the buffer is not shuffled"

    def test_the_buffer_is_shuffled_not_served_in_row_order(self):
        buf = _build([_source(rows=40)], [10 * 16])
        ids = buf.tensors[KEYS[0]][:, 0].tolist()
        assert ids != sorted(ids)


class TestNoBlockRepeatsWithinAPass:
    def test_every_row_is_read_once_before_any_is_read_again(self):
        src = _source(rows=40)
        buf = _build([src], [4 * 16])  # 4 rows per refill, 10 refills per pass
        seen = []
        for refill in range(10):
            if refill:
                buf.refill()
            rows = _buffer_rows(buf)
            assert len(rows) == 4
            assert not rows & set(seen), f"refill {refill} repeated rows {sorted(rows & set(seen))}"
            seen.extend(rows)
            assert buf.epochs_completed() == [0]
        assert len(seen) == 40
        buf.refill()
        assert buf.epochs_completed() == [1], "the source was not reshuffled after it was exhausted"

    def test_no_token_repeats_before_every_token_has_been_served(self):
        """Checked per buffer: the refill that finishes a pass also starts the next
        (RowCycle wraps inside a take), so the invariant is that the first buffer
        holding a repeat is also one by whose end every token has been served."""
        lengths = [3 + (7 * r) % 13 for r in range(30)]
        src = _source(rows=30, lengths=lengths, width=16)
        buf = _build([src], [50])
        expected = {(0, r, p) for r in range(30) for p in range(lengths[r])}
        seen, repeats = set(), 0
        for _ in range(3 * 30):
            tokens = _origins(buf.tensors[KEYS[0]])
            assert len(tokens) == len(set(tokens)), "a token appears twice in one buffer"
            if set(tokens) & seen:
                repeats += 1
                assert seen | set(tokens) == expected, (
                    f"a token was served again while {len(expected - seen - set(tokens))} had never been served"
                )
                seen = set(tokens) - (expected - seen)
            else:
                seen |= set(tokens)
            buf.refill()
        assert repeats >= 2, "the fixture never completed a pass"

    def test_held_out_rows_are_never_served(self):
        held = {2, 11, 17, 29}
        src = _source(rows=30, holdout_rows=held)
        buf = _build([src], [5 * 16])
        rows = set()
        for _ in range(12):
            rows |= {r for _, r in _buffer_rows(buf)}
            buf.refill()
        assert rows == set(range(30)) - held


class TestEveryLayerSeesTheSamePositions:
    def test_one_permutation_serves_every_layer(self):
        buf = _build([_source(rows=40)], [10 * 16])
        for _ in range(12):  # crosses refills
            batch = buf.next_batch(24)
            assert torch.equal(batch[KEYS[0]][:, 0], batch[KEYS[1]][:, 0]), "layers saw different tokens"
            assert set(batch[KEYS[0]][:, 1].tolist()) == {3.0} and set(batch[KEYS[1]][:, 1].tolist()) == {5.0}


class TestPaddingIsExcluded:
    def test_no_pad_position_is_ever_served(self):
        lengths = [1 + (5 * r) % 16 for r in range(40)]
        buf = _build([_source(rows=40, lengths=lengths, width=16)], [120])
        for _ in range(8):
            for s, r, p in _origins(buf.tensors[KEYS[0]]):
                assert p < lengths[r], f"row {r} position {p} is padding (real length {lengths[r]})"
            buf.refill()

    def test_all_padding_rows_are_consumed_but_never_forwarded(self):
        lengths = [0 if r % 4 == 0 else 8 for r in range(40)]
        capture = _Capture()
        buf = _build([_source(rows=40, lengths=lengths, width=8)], [80], capture=capture, micro=1)
        rows = _buffer_rows(buf)
        assert all(r % 4 for _, r in rows)
        assert sum(capture.calls) == len(rows), "an all-padding row was run through the model"


class TestTheMixtureFollowsTheWeights:
    def test_each_refill_takes_each_sources_quota(self):
        a, b = _source(0, rows=200), _source(1, rows=200)
        quotas = dataset_mixture.allocate_tokens([3200, 3200], 640, [3.0, 1.0])
        assert quotas == [480, 160]
        buf = _build([a, b], quotas)
        for _ in range(5):
            per = {0: 0, 1: 0}
            for s, _, _ in _origins(buf.tensors[KEYS[0]]):
                per[s] += 1
            assert per == {0: 480, 1: 160}, per
            buf.refill()
        # The first fill and five refills.
        assert buf.refills == 6 and buf.tokens_loaded == [6 * 480, 6 * 160]

    def test_a_batch_draws_from_both_sources(self):
        buf = _build([_source(0, rows=200), _source(1, rows=200)], [480, 160])
        sources = [s for s, _, _ in _origins(buf.next_batch(64)[KEYS[0]])]
        assert 0 < sources.count(1) < sources.count(0)


class TestTheForwardIsMicroBatched:
    @pytest.mark.parametrize("micro", [1, 3, 8])
    def test_no_forward_takes_more_rows_than_the_micro_batch(self, micro):
        capture = _Capture()
        _build([_source(rows=40)], [10 * 16], capture=capture, micro=micro)
        assert capture.calls and max(capture.calls) <= micro and sum(capture.calls) == 10

    def test_a_context_manager_capture_is_entered_around_each_refill(self):
        events = []

        class Session(_Capture):
            def __enter__(self):
                events.append("enter")
                return self

            def __exit__(self, *exc):
                events.append("exit")

            def __call__(self, padded, masks):
                events.append("call")
                return super().__call__(padded, masks)

        buf = _build([_source(rows=40)], [6 * 16], capture=Session(), micro=2)
        buf.refill()
        assert events == ["enter", "call", "call", "call", "exit"] * 2


class TestAFailedRefillIsRetriedNotSkipped:
    """The rolling buffer's rule, on the model path: a refill that fails part-way (an
    OOM in the forward, which the training loop retries) is retried on the SAME rows.
    Otherwise the cycles have already moved past rows that were never served, and
    they are skipped for the whole pass."""

    def test_a_forward_that_fails_is_retried_on_the_same_rows(self):
        reference = _build([_source(rows=40)], [4 * 16], seed=3)
        reference.refill()
        expected = reference.tensors[KEYS[0]].clone()

        capture = _Capture()
        armed = {"fail": False}

        def flaky(padded, masks):
            if armed["fail"]:
                armed["fail"] = False
                raise RuntimeError("CUDA out of memory")
            return capture(padded, masks)

        buf = _build([_source(rows=40)], [4 * 16], seed=3, capture=flaky, micro=2)
        armed["fail"] = True
        with pytest.raises(RuntimeError, match="out of memory"):
            buf.refill()
        assert buf.size == 0, "a half-written buffer is still being served"
        buf.refill()
        assert torch.equal(buf.tensors[KEYS[0]], expected), "the retry read different rows or a different order"
        assert buf.refills == reference.refills and buf.tokens_loaded == reference.tokens_loaded


class TestBookkeepingAndLogs:
    def test_each_refill_logs_the_sources_and_their_passes(self, caplog):
        with caplog.at_level(logging.INFO, logger=MAS.logger.name):
            _build([_source(0, rows=10), _source(1, rows=10)], [32, 32])
        assert "source passes completed [0, 0]" in caplog.text
        assert "ds_0=32 (2 rows)" in caplog.text and "ds_1=32 (2 rows)" in caplog.text

    def test_the_unserved_tail_is_counted_as_dropped(self):
        buf = _build([_source(rows=40)], [4 * 16])
        buf.next_batch(40)
        buf.next_batch(40)  # 24 left over: refills
        assert buf.tokens_dropped == 24 and buf.refills == 2

    def test_a_refill_of_nothing_but_padding_refuses(self):
        with pytest.raises(ValueError, match="no real tokens"):
            _build([_source(rows=4, lengths=[0, 0, 0, 0], width=8)], [8])

    def test_quotas_must_match_the_sources(self):
        with pytest.raises(ValueError, match="quotas"):
            _build([_source(rows=4)], [8, 8])


class TestAShortSourceIsNeverSilent:
    """R1-B F2 (2026-09-15). A source whose quota exceeds its real tokens is read WHOLE on
    every refill — a full pass per buffer — and supplies less than its quota, so the buffer's
    mixture falls short of what was asked. Nothing said so: measured, 90/10 was asked, the
    source was re-read every refill and delivered 7.9%, with no warning at all.

    It happens whenever the real-token ESTIMATE overshoots, and the estimate read 64 rows: on a
    corpus of mixed-length rows (10% full, 90% one token) it ranged 0.39x-1.42x of the truth
    over 20 seeds.

    MUTATION CONTROLS (R1-B):
      C5  the shortfall warning removed          -> test_a_source_that_cannot_fill_its_quota_warns_once
      C5b the warn-once set never updated        -> test_a_source_that_cannot_fill_its_quota_warns_once
    The estimate fix (64 -> 1,024 rows) was backed out at the coordinator's instruction: it
    changes quota planning, which the integrator's R1D-1/R1D-2 storage-plan fix owns. Its
    control (C6) is retired, and the estimate test is a strict xfail until that lands.
    """

    def test_a_source_that_cannot_fill_its_quota_warns_once(self, caplog):
        with caplog.at_level(logging.WARNING, logger=MAS.logger.name):
            buf = _build([_source(0, rows=200), _source(1, rows=10, length=16)], [900, 400])
            for _ in range(4):
                buf.refill()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        short = [m for m in warnings if "ds_1" in m]
        assert len(short) == 1, warnings
        assert "160" in short[0] and "400" in short[0], short[0]
        assert not any("ds_0" in m for m in warnings), warnings

    def test_a_source_that_fills_its_quota_is_not_warned_about(self, caplog):
        with caplog.at_level(logging.WARNING, logger=MAS.logger.name):
            buf = _build([_source(0, rows=200), _source(1, rows=200)], [480, 160])
            buf.refill()
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "R1-B F2, recorded not fixed: the 64-row estimate ranges 0.39x-1.42x on mixed-length rows. "
            "Raising the sample changes quota planning, which the integrator's R1D-1/R1D-2 storage-plan "
            "fix owns; this flips to a failure (strict) once the estimate is fixed, so remove the mark then."
        ),
    )
    def test_the_real_token_estimate_is_close_on_mixed_length_rows(self):
        rng = np.random.default_rng(0)
        lengths = [16 if rng.random() < 0.1 else 1 for _ in range(4_000)]
        actual = sum(lengths)
        src = MAS.TokenRowSource("mixed", _Rows(0, lengths, 16), np.arange(len(lengths)), 16)
        ratios = [MAS.estimate_real_tokens(src, seed=seed, index=0) / actual for seed in range(20)]
        assert all(0.8 <= r <= 1.2 for r in ratios), f"estimate/actual over 20 seeds: {min(ratios):.2f}-{max(ratios):.2f}"


class TestTheBufferStateRoundTrips:
    """The buffer-state contract: serve N, save, load into a fresh source, identical batches after."""

    @staticmethod
    def _saved(state):
        blob = io.BytesIO()
        torch.save(state, blob)
        blob.seek(0)
        return torch.load(blob, weights_only=True)

    @pytest.mark.parametrize("served", [0, 5, 7, 16])
    @pytest.mark.parametrize("prefill", [True, False])
    def test_a_loaded_source_serves_exactly_what_the_original_would(self, served, prefill):
        def sources():
            lengths = [4 + (3 * r) % 12 for r in range(60)]
            return [_source(0, rows=60, lengths=lengths), _source(1, rows=25, width=16)]

        original = _build(sources(), [90, 40], seed=11)
        for _ in range(served):
            original.next_batch(20)
        state = self._saved(original.state_dict())

        resumed = _build(sources(), [90, 40], seed=11, prefill=prefill)
        resumed.load_state_dict(state)

        for step in range(40):  # crosses several refills and a source's pass
            a, b = original.next_batch(20), resumed.next_batch(20)
            for key in KEYS:
                assert torch.equal(a[key], b[key]), f"batch {step} after resuming differs for {key}"
        assert resumed.epochs_completed() == original.epochs_completed()
        assert resumed.tokens_loaded == original.tokens_loaded
        assert resumed.tokens_dropped == original.tokens_dropped
        assert resumed.refills == original.refills

    def test_a_state_replayed_over_changed_tokenized_data_is_refused(self):
        """R1-B (2026-09-15). The replay re-selects the saved rows from the dataset, so a
        tokenization rewritten between the checkpoint and the resume — same row count, same
        held-out split, so the row digest still matches — yields a buffer of another size, and
        `load_state_dict` must refuse rather than continue on other tokens.

        MUTATION CONTROL P7: the size check disabled (`if False:`) SURVIVED the whole of this
        file and test_activation_source_state.py (68 green) before this test existed. Re-run
        with it: red.
        """
        lengths = [4 + (3 * r) % 12 for r in range(60)]
        original = _build([_source(0, rows=60, lengths=lengths)], [96], seed=11)
        original.next_batch(20)
        state = self._saved(original.state_dict())

        rewritten = _build([_source(0, rows=60, length=16)], [96], seed=11, prefill=False)
        probe = _build([_source(0, rows=60, length=16)], [96], seed=11)
        assert probe.size != original.size, "precondition: the rewrite must change the buffer's size"

        with pytest.raises(RuntimeError, match="tokenized data changed"):
            rewritten.load_state_dict(state)

    def test_the_state_is_serialisable_without_pickled_objects(self):
        buf = _build([_source(rows=20)], [48])
        buf.next_batch(10)
        state = self._saved(buf.state_dict())
        assert state["kind"] == MAS.STATE_KIND and state["position"] == 10

    def test_a_different_seed_does_not_reproduce_the_batches(self):
        """NEGATIVE CONTROL for the round trip: without the loaded state the batches differ."""
        a = _build([_source(rows=60)], [90], seed=11)
        b = _build([_source(rows=60)], [90], seed=12)
        assert not torch.equal(a.next_batch(20)[KEYS[0]], b.next_batch(20)[KEYS[0]])

    @pytest.mark.parametrize(
        "change", ["quotas", "rows", "same_count_other_rows", "labels", "keys"],
    )
    def test_a_state_for_another_configuration_is_refused(self, change):
        """`same_count_other_rows` is the case that matters: a state saved under one
        held-out split, loaded under another of the SAME size. The cycle's length
        check cannot see it, and without the row digest the load succeeds and serves
        different tokens while claiming to continue (control D33 survived until this
        case was added — the `rows` case alone changed the count and failed anyway)."""
        held_saved = (4,) if change == "same_count_other_rows" else ()
        buf = _build([_source(rows=20, holdout_rows=held_saved)], [48])
        state = buf.state_dict()
        kwargs = dict(quotas=[48], held=held_saved, keys=KEYS, label=0)
        if change == "quotas":
            kwargs["quotas"] = [64]
        elif change == "rows":
            kwargs["held"] = (3,)
        elif change == "same_count_other_rows":
            kwargs["held"] = (3,)
        elif change == "labels":
            kwargs["label"] = 1
        else:
            kwargs["keys"] = KEYS[:1]
        other = _build(
            [_source(kwargs["label"], rows=20, holdout_rows=kwargs["held"])], kwargs["quotas"],
            prefill=False, keys=kwargs["keys"],
        )
        with pytest.raises(ValueError):
            other.load_state_dict(state)


class TestHeldOutCollection:
    def test_each_source_gives_its_quota_from_its_held_rows_only(self):
        a = _Rows(0, [16] * 50, 16)
        b = _Rows(1, [10] * 50, 16)
        held_a = np.array([1, 5, 9, 13, 17, 21, 25, 29, 33, 37])
        held_b = np.array([0, 2, 4, 6, 8, 10, 12, 14])
        out = MAS.collect_holdout_activations(
            [("a", a, held_a), ("b", b, held_b)], KEYS, [50, 25],
            capture=_Capture(), seed=4, micro_batch_rows=2, pad_token_id=PAD,
        )
        origins = _origins(out[KEYS[0]])
        per = {0: [o for o in origins if o[0] == 0], 1: [o for o in origins if o[0] == 1]}
        assert len(per[0]) == 50 and len(per[1]) == 25
        assert {r for _, r, _ in per[0]} <= set(held_a.tolist())
        assert {r for _, r, _ in per[1]} <= set(held_b.tolist())
        assert torch.equal(out[KEYS[0]][:, 0], out[KEYS[1]][:, 0])
        assert out[KEYS[0]].device.type == "cpu"

    def test_rows_are_taken_in_a_seeded_order_not_lowest_first(self):
        data = _Rows(0, [8] * 200, 8)
        held = np.arange(200)
        firsts = set()
        for seed in range(4):
            out = MAS.collect_holdout_activations(
                [("a", data, held)], KEYS[:1], [24], capture=_Capture(KEYS[:1]), seed=seed,
                micro_batch_rows=4, pad_token_id=PAD,
            )
            firsts.add(frozenset(r for _, r, _ in _origins(out[KEYS[0]])))
        assert frozenset({0, 1, 2}) not in firsts and len(firsts) > 1

    def test_a_source_short_of_its_quota_hands_the_rest_to_the_others(self, caplog):
        """R1-B F3 (2026-09-15). The cached path's rule — a source that cannot fill its share
        hands the remainder on (`holdout_quotas`) — was lost on the fly: quotas were sized in
        PADDED tokens, so a padding-heavy dataset's quota exceeded its real held-out tokens,
        it came up short, and the rest was simply not evaluated. Measured: 2,200 of 4,000
        tokens, 200 from the sparse source, while the dense one had 6,400 unread.

        MUTATION CONTROL R1-B C7: no redistribution round -> this test fails (1,000 of 1,600).
        """
        dense = _Rows(0, [16] * 200, 16)
        sparse = _Rows(1, [2] * 200, 16)
        held = np.arange(100)
        with caplog.at_level(logging.WARNING, logger=MAS.logger.name):
            out = MAS.collect_holdout_activations(
                [("dense", dense, held), ("sparse", sparse, held)], KEYS, [800, 800],
                capture=_Capture(), seed=0, micro_batch_rows=8, pad_token_id=PAD,
            )
        per = {0: 0, 1: 0}
        for s, r, _ in _origins(out[KEYS[0]]):
            assert r in set(held.tolist())
            per[s] += 1
        assert per == {0: 1_400, 1: 200}, per
        assert torch.equal(out[KEYS[0]][:, 0], out[KEYS[1]][:, 0]), "the layers were evaluated on different tokens"
        assert len(set(_origins(out[KEYS[0]]))) == 1_600, "a held-out token was evaluated twice"
        assert any("sparse" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)

    def test_the_redistribution_follows_the_weights(self):
        """Weighted 1:3 over two dense sources after a third runs dry."""
        sources = [("a", _Rows(0, [16] * 300, 16), np.arange(300)), ("b", _Rows(1, [16] * 300, 16), np.arange(300)),
                   ("dry", _Rows(2, [1] * 40, 16), np.arange(40))]
        out = MAS.collect_holdout_activations(
            sources, KEYS[:1], [400, 1_200, 400], capture=_Capture(KEYS[:1]), seed=1, micro_batch_rows=16,
            pad_token_id=PAD, weights=[1.0, 3.0, 1.0],
        )
        per = {0: 0, 1: 0, 2: 0}
        for s, _, _ in _origins(out[KEYS[0]]):
            per[s] += 1
        assert per[2] == 40 and sum(per.values()) == 2_000
        assert per == {0: 490, 1: 1_470, 2: 40}, per


class TestRowsAndMasks:
    def test_split_rows_holds_out_what_split_documents_would(self):
        n, seq = 97, 8
        for seed in (1, 2, 3):
            train, held = activation_mask.split_rows(n, 0.2, seed)
            _, held_flat = activation_mask.split_documents(np.arange(n * seq), seq, 0.2, seed)
            assert np.array_equal(held, np.unique(held_flat // seq))
            assert np.array_equal(np.sort(np.concatenate([train, held])), np.arange(n))

    def test_split_rows_at_zero_keeps_every_row(self):
        train, held = activation_mask.split_rows(10, 0.0, 1)
        assert np.array_equal(train, np.arange(10)) and held.size == 0

    def test_select_real_tokens_keeps_only_real_positions(self):
        acts = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        kept = activation_mask.select_real_tokens(acts, [[1, 1, 0, 0], [1, 0, 0, 0]])
        assert torch.equal(kept, torch.stack([acts[0, 0], acts[0, 1], acts[1, 0]]))

    def test_select_real_tokens_refuses_a_mask_that_does_not_match(self):
        with pytest.raises(ValueError, match="refusing"):
            activation_mask.select_real_tokens(torch.zeros(2, 4, 3), [[1, 1, 1]])

    def test_select_real_tokens_of_all_padding_is_empty(self):
        assert activation_mask.select_real_tokens(torch.zeros(1, 4, 3), [[0, 0, 0, 0]]).shape == (0, 3)

    def test_the_mask_is_built_on_the_cpu(self):
        """It must index activations wherever they are; a mask on the input card cannot."""
        meta = torch.empty(1, 4, 3, device="meta")
        with pytest.raises(NotImplementedError):
            meta[torch.ones(4, dtype=torch.bool, device="meta")].tolist()
        kept = activation_mask.select_real_tokens(torch.ones(1, 4, 3), [[1, 1, 0, 0]])
        assert kept.device.type == "cpu"

    def test_a_row_without_a_mask_is_read_as_all_real_and_counted(self):
        data = _Rows(0, [5] * 6, 8, with_mask=False)
        src = MAS.TokenRowSource("nomask", data, np.arange(6), 8)
        assert MAS.estimate_real_tokens(src, seed=0, index=0) == 6 * 8

    def test_estimate_real_tokens_reads_the_masks(self):
        src = _source(rows=50, length=6, width=16)
        assert MAS.estimate_real_tokens(src, seed=3, index=0) == 50 * 6

    def test_a_real_arrow_dataset_row_is_read(self):
        from datasets import Dataset

        ds = Dataset.from_dict({"input_ids": [[5, 6, 7, 0]], "attention_mask": [[1, 1, 1, 0]]})
        assert MAS.read_row(ds, 0) == ([5, 6, 7, 0], [1, 1, 1, 0])
