"""SAE training must read the whole extraction, not one GPU-sized sample of it.

2026-09-12. Training loaded a single random subsample sized to free GPU memory —
~2.4M tokens per layer for LFM2.5-1.2B on a 24 GB card — and drew all 102M
default samples from it with replacement. A 20M-token extraction was ~90%
unread and every token that was read was seen ~43 times.

2026-09-14 (multi-GPU Phase 4). A 3-layer run on the 3090 stalled 31-33 s at
every refill against ~41 s of training per buffer: 44% of wall time reading.
Refills are now read by a thread pool and the next buffer is prepared on a
background thread. These tests hold that change to the rule it must obey: it
changes WHEN the bytes are read, never WHICH bytes are served or in what order.
Every invariant test below runs in four modes — synchronous with 1 and 4 read
threads, prefetched with 1 and 12 — and `TestServedBuffersMatchTheOracle`
compares every mode against the refill as it was before Phase 4.

These tests build real `.npy` extractions whose every activation ENCODES its own
origin — (source, row, position, layer) — so what the buffer serves can be
checked exactly rather than statistically.

MUTATION CONTROLS (2026-09-14, Phase 4). Each broke one load-bearing line of
src/services/activation_buffer.py or src/workers/training_tasks.py, ran this
whole module, and was restored from the original bytes with the sha256
re-checked. Every one was KILLED; the count is the failures it caused (106 tests).
  M1   the permutation drawn from the global RNG, not the buffer's generator
         -> TestServedBuffersMatchTheOracle, TestDeterminism (16 failed)
  M1b  the prefetch selects twice, skipping a buffer
         -> TestServedBuffersMatchTheOracle and 5 more, prefetch modes (15 failed)
  M2   a different permutation for the second layer
         -> TestEveryLayerSeesTheSameTokens, TestServedBuffersMatchTheOracle (17 failed)
  M3   refill swallows the prefetch exception and re-selects instead
         -> TestPrefetchFailures (4 failed)
  M4   close() shuts the pools down with wait=False
         -> TestClose::test_close_joins_every_thread (2 failed, plus 87 teardown errors
            from the `opened` fixture's leaked-thread guard)
  M5   rows written to their unshuffled offset (dest[lo:hi]) instead of inverse[lo:hi]
         -> TestServedBuffersMatchTheOracle (9 failed)
  M5b  every source's rows written at offset 0 (the concatenation reordered)
         -> TestServedBuffersMatchTheOracle and more (23 failed)
  M6   plan_prefetch ignores the RAM budget
         -> TestTheRamBudget (2 failed)
  M7   _read_row ignores the stop flag (honours only the per-refill abort)
         -> TestClose::test_a_cancelled_training_does_not_wait_for_the_prefetch (1 failed)
  M8   the read pool built with max_workers=1
         -> TestParallelReads::test_reads_really_run_concurrently (1 failed)
  M9a  the prefetched refill does not wait for the pending host-to-GPU transfer
  M9b  the synchronous refill does not wait for it
         -> TestPendingTransfer (1 failed each)
  M10  after_return no longer calls _close_activation_stream
         -> TestTheTrainingLoopUsesIt::test_after_return_closes_the_buffer (1 failed)
  M11  train_sae_task no longer registers the buffer on the task
         -> TestTheTrainingLoopUsesIt::test_the_task_registers_the_buffer_for_cleanup (1 failed)
  M12  the buffer built without host_ram_tokens
         -> TestTheTrainingLoopUsesIt::test_the_buffer_is_given_the_ram_budget (1 failed)
  M13  epochs_completed() reads the live cycles, not the served snapshot
         -> test_a_source_is_reshuffled_only_after_it_is_exhausted, prefetch modes (2 failed)
  M14  tokens_loaded counted in _select, a buffer early
         -> TestBookkeeping, prefetch modes (2 failed)
  M15  wait_s not measured
         -> TestRefillTimings::test_the_wait_for_a_slow_prefetch_is_measured (1 failed)
  M16  the prepared copy reallocated on every refill
         -> TestMemory, prefetch modes (4 failed). FIRST RUN: 2 failed —
            test_the_prepared_copy_reuses_its_memory compared data_ptr, and the
            allocator returned the freed block at the same address, so it passed
            against the reallocation. Now asserts identity; re-run killed by both.
  M17  the next prefetch never submitted
         -> TestPrefetch::test_the_next_buffer_is_prepared_off_the_training_thread (6 failed)
  M18  a prefetch failure not sticky, so a retry skips the unread buffer
         -> TestPrefetchFailures::test_a_retry_after_a_failed_prefetch_fails_too (2 failed)
  M19  read_thread_count ignores the CPU cap
         -> TestParallelReads::test_read_thread_count (1 failed)
  M20  a failed refill raises before rows still being copied have finished
         -> TestAFailedReadLeavesNoStragglers::test_rows_still_being_read_finish_before_the_failure_is_raised (1 failed)
  M21  close() does not set the stop flag
         -> TestClose::test_a_cancelled_training_does_not_wait_for_the_prefetch (1 failed)
  M22  every row treated as contiguous (holey rows read as a slice)
         -> TestServedBuffersMatchTheOracle (9 failed)
  M23  the unserved tail counted before the refill succeeds
         -> test_a_failed_refill_does_not_count_the_unserved_tail_twice (1 failed)
  M24  the failing row no longer sets the abort flag
         -> test_a_failed_read_stops_the_rest_of_the_refill (1 failed)

Two findings came from controls, not reading. M4's first form (shutdown with
cancel_futures) HUNG the suite: wait() on a future cancelled by shutdown never
returns, so the abort became a flag every row checks. And the first run of
test_a_failed_read_stops_the_rest_of_the_refill measured 49 rows still read after
a failure when the waiting thread set the abort — the failing row now sets it.

REVIEW ROUND 1 (2026-09-14, independent reviewer). Same method: one line broken,
this module run, the file restored from its bytes, sha256 re-checked, git diff
clean. Two mutations SURVIVED the 106 tests as committed:
  R-flatpos  flat_positions drops the last trainable token of every masked row
               -> SURVIVED: the oracle called flat_positions itself, so it agreed
                  with the defect by construction, and the every-token test used
                  unmasked sources only. Now the oracle reads the fixture's mask,
                  and test_every_trainable_token_of_a_masked_source_is_served_once_per_pass
                  was added. Re-run: KILLED (17 failed / 120).
  R-pin      the pinned-to-pageable fallback removed
               -> SURVIVED: nothing ever failed a pin. Re-run on the new
                  _allocate_host as R-F1d below: KILLED.
And one control per fix of this round, every one KILLED (of 120 tests):
  R-F3a  refill_capacity returns the quotas (a row larger than its quota ignored)
  R-F3b  _storage_for sizes the storage to sum(quotas)
           -> test_a_row_larger_than_its_quota_never_reallocates (4 failed each)
  R-F3c  the prepared copy sized to sum(quotas)
           -> the same test, prefetch modes (2 failed)
  R-F1a  plan_prefetch budgets the unrounded buffer for a pinned copy
           -> test_the_plan_budgets_the_rounded_pinned_block,
              test_a_gpu_buffer_budgets_its_pinned_copy_rounded (2 failed)
  R-F1b  the buffer does not pass bytes_per_token to the plan
           -> test_a_gpu_buffer_budgets_its_pinned_copy_rounded (1 failed)
  R-F1c  host storage pinned whatever its rounded size
           -> test_host_storage_is_not_pinned_past_the_budget (1 failed)
  R-F1d  the fallback catches the wrong exception, so a failed pin fails the run
           -> test_pinning_that_fails_falls_back_to_pageable_memory (1 failed)
  R-F1e  close() never empties the pinned host cache
  R-F1f  close() empties it while still holding the storage and views
  R-F1g  pinned bytes never counted, so close() never releases
           -> test_close_releases_the_buffer_and_the_pinned_cache (2 failed each)
Author controls re-run against 18de448c, all still KILLED: M3, M7, M9a, M10, M13,
M17, M18, M20, M22, M24.

REVIEW ROUND 2 (2026-09-14, re-review of round 1's fixes). Same method, controls run
against fab1235c, every file restored from its bytes with the sha256 and git diff
re-checked. Proof first, on c8fccbcd: three mutations SURVIVED all 120 tests, and 21
of the new tests failed before their fixes.
  R2-postinit    BufferSource.__post_init__ deletes one trainable position
                   -> SURVIVED: round 1's oracle read src.valid_flat, which that method
                      rewrites. Fixtures now keep their own mask. Re-run: KILLED (14 failed / 141).
  R2-numpyview   close() keeps a numpy view of every host block
                   -> SURVIVED: the close test watched weakrefs to TENSOR objects, and a
                      view holds the storage without them. Storage references are counted
                      now. Re-run: KILLED (4 failed).
  R2-privname    the release looks up torch._C._host_empty_cache (a torch rename)
                   -> SURVIVED: every test replaced the calling function.
                      Re-run: KILLED by TestPinnedHostRelease::test_the_release_calls_torch.
One control per fix, every one KILLED (of 141 tests):
  R2-pending      close() empties the cache while its `pending` future still holds the
                  stopped prefetch's traceback (the defect: measured 1 reference left
                  on every prepared-copy block on the happy path)
                    -> test_close_releases_the_buffer_and_the_pinned_cache[prefetch-12-prepared copy]
  R2-closegc      close() releases without collecting first
                    -> test_close_releases_blocks_a_dropped_failure_still_references (2 failed)
  R2-logrepr      the refill logs the failure object instead of its repr (a retained log
                  record keeps the traceback)
                    -> test_close_releases_blocks_a_dropped_failure_still_references[prefetch-12-prepared copy]
  R2-retry-sync   a synchronous refill re-selects after a failure
                    -> test_a_read_failure_is_retried_on_the_same_rows[sync-*] and
                       test_a_failure_part_way_through_the_writes...[sync-*] (4 failed)
  R2-retry-bg     the prefetch re-selects instead of re-reading the failed selection
                    -> test_a_read_failure_is_retried_on_the_same_rows[prefetch-*] (2 failed)
  R2-retry-copy   a copy that fails drops the prepared selection (round 1 item a)
                    -> test_a_failure_part_way_through_the_writes...[prefetch-*] (2 failed)
  R2-gathered     a prepared selection is never marked read, so every refill re-reads it
                    -> TestPrefetch::test_the_next_buffer_is_prepared_off_the_training_thread (2 failed)
  R2-retire-sync  the served buffer is not retired before a synchronous refill writes into it
                    -> test_a_failure_part_way_through_the_writes...[sync-*],
                       test_a_failed_refill_does_not_count_the_unserved_tail_twice (3 failed)
  R2-retire-pf    ... nor before the prepared copy is copied in
                    -> test_a_failure_part_way_through_the_writes...[prefetch-*] (2 failed)
  R2-pos          next_batch advances its position before the copies
                    -> test_a_draw_whose_copy_fails_serves_the_same_tokens_on_the_retry (4 failed)
  R2-straggler    M20 re-run against the barrier rewrite of the straggler test
                    -> test_rows_still_being_read_finish_before_the_failure_is_raised (1 failed)
  R2-pin-budget   pin_within_budget ignores the allowance
  R2-pin-round    pin_within_budget budgets the unrounded size
                    -> test_a_pool_whose_rounded_block_is_over_the_budget_stays_pageable (1 failed each)
  R2-cpuall-raw   the fixed pool pinned with a bare tensor.pin_memory() again
  R2-cpuall-budget  ... budgeted by the GPU allowance instead of the RAM one
                    -> test_the_fixed_pool_is_pinned_only_within_the_ram_budget (1 failed each)
  R2-afterreturn  after_return never empties the pinned host cache
  R2-afterreturn-order  ... empties it before its gc.collect()
                    -> test_after_return_releases_pinned_host_memory_once_everything_is_collected (1 failed each)
  R2-closeprev    train_sae_task no longer closes a buffer left on the task (round 1 item c)
                    -> test_a_new_training_closes_a_buffer_left_behind_before_measuring_memory (1 failed)
Round 1's controls re-run against fab1235c, all still KILLED: R-flatpos (21 failed),
R-F3a, R-F3b, R-F3c, R-F1a, R-F1b, R-F1c, R-F1d, R-F1e (re-targeted at the release
after the new gc.collect), R-F1f, R-F1g. Author controls re-run, all still KILLED: M7,
M9a, M9b, M10, M13, M17, M22, M24. M3, M18 and M23 are superseded, not re-run: they
pinned a sticky prefetch failure and a tail counted at success, which this round
replaced with same-selection retries and a tail counted when the served buffer is
retired; R2-retry-bg, R2-retry-copy and R2-retire-sync are their successors.
"""

import ast
import gc
import inspect
import logging
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from src.services import activation_buffer as AB
from src.services import activation_mask

SEQ = 8
KEYS = [(3, "residual"), (4, "residual")]
CPU = torch.device("cpu")
META = torch.device("meta")
#: Enough host RAM for any prepared copy these fixtures make.
BIG = 10**9

MODES = {
    "sync-1": {"prefetch": False, "read_threads": 1},
    "sync-4": {"prefetch": False, "read_threads": 4},
    "prefetch-1": {"prefetch": True, "read_threads": 1, "host_ram_tokens": BIG},
    "prefetch-12": {"prefetch": True, "read_threads": 12, "host_ram_tokens": BIG},
}


def _extraction(tmp_path: Path, source: int, rows: int) -> dict:
    """A synthetic extraction: activation[row, pos] = [source, row, pos, layer]."""
    files = {}
    r, p = np.meshgrid(np.arange(rows), np.arange(SEQ), indexing="ij")
    for layer, hook in KEYS:
        arr = np.stack(
            [np.full_like(r, source), r, p, np.full_like(r, layer)], axis=-1
        ).astype(np.float16)
        path = tmp_path / f"s{source}_layer_{layer}_{hook}.npy"
        np.save(path, arr)
        files[(layer, hook)] = path
    return files


def _source(tmp_path, source, rows, valid_flat=None):
    src = AB.BufferSource(
        label=f"src{source}", files=_extraction(tmp_path, source, rows),
        num_rows=rows, seq_len=SEQ, valid_flat=valid_flat,
    )
    # The fixture's own copy of its mask, taken BEFORE BufferSource rewrites
    # `valid_flat` (it sorts and casts it in __post_init__). The oracle reads this,
    # so a BufferSource that loses a position cannot agree with it by construction.
    src.fixture_valid = (
        None if valid_flat is None else np.array(valid_flat, dtype=np.int64, copy=True)
    )
    return src


def _holey_source(tmp_path, source, rows, seed):
    """Positions missing from the MIDDLE of rows, so reads take the fancy-index path."""
    mask = np.random.default_rng(seed).random((rows, SEQ)) > 0.3
    mask[:, 0] = True
    return _source(tmp_path, source, rows, valid_flat=np.flatnonzero(mask.reshape(-1)))


def _origins(batch):
    """(source, row, pos) triples for one layer's batch."""
    return [tuple(int(v) for v in row[:3]) for row in batch.tolist()]


def _storages(tensors):
    """A handle on each tensor's STORAGE, for counting what still references it.

    Not a weakref to the tensor: `tensor.numpy()` keeps the storage alive without
    keeping the tensor object alive, so a weakref reports a block freed while a
    numpy view still pins it (review round 2, control R2-numpyview)."""
    return [t.untyped_storage() for t in tensors if t is not None]


def _holders(storages):
    """References to each storage other than the handle `_storages` took."""
    return [torch._C._storage_Use_Count(s._cdata) - 1 for s in storages]


def _activation_threads(buf):
    threads = set(buf._pool._threads)
    if buf._prefetcher is not None:
        threads |= set(buf._prefetcher._threads)
    return [t for t in threads if t.is_alive()]


def _settle(buf, timeout=10.0):
    """Wait until the background preparation (if any) has finished."""
    if buf._pending is not None:
        buf._pending.exception(timeout=timeout)


class _Faults:
    """Slow or failing row reads, injected where a real read is slow: inside the
    memmap access, AFTER the buffer's own stop/abort check. (A wrapper that
    slept before that check modelled a disk no read ever sees.)"""

    def __init__(self, delay=0.0, fail_first=False, fail_after=0.01):
        self.delay, self.fail_first, self.fail_after = delay, fail_first, fail_after
        self.reads = 0
        self.started = threading.Event()
        self._lock = threading.Lock()

    def install(self, buf):
        for k, mm in list(buf._mmaps.items()):
            buf._mmaps[k] = _FaultyRows(mm, self)
        return self

    def disarm(self):
        self.delay, self.fail_first = 0.0, False

    def on_read(self):
        with self._lock:
            self.reads += 1
            fail = self.fail_first and self.reads == 1
            delay = self.delay
        self.started.set()
        if fail:
            time.sleep(self.fail_after)   # let other threads get well into their reads
            raise OSError("transient read error")
        if delay:
            time.sleep(delay)


class _FaultyRows:
    def __init__(self, array, faults):
        self.array, self.faults = array, faults

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, index):
        self.faults.on_read()
        return self.array[index]


@pytest.fixture
def opened():
    """Every buffer a test builds is closed, and must leave no thread behind."""
    buffers = []
    yield buffers
    for buf in buffers:
        buf.close()
    leaked = [t.name for buf in buffers for t in _activation_threads(buf)]
    assert not leaked, f"threads outlived close(): {leaked}"


def _builder(opened, mode, storage_device=CPU, train_device=CPU):
    def make(sources, quotas, seed=0, **overrides):
        buf = AB.RollingActivationBuffer(
            sources, KEYS, quotas, seed=seed, storage_device=storage_device,
            train_device=train_device, **{**MODES[mode], **overrides},
        )
        opened.append(buf)
        return buf
    return make


@pytest.fixture(params=list(MODES))
def make_buffer(request, opened):
    return _builder(opened, request.param)


@pytest.fixture(params=["prefetch-1", "prefetch-12"])
def make_prefetched(request, opened):
    return _builder(opened, request.param)


def _oracle_rows(src):
    """Row ids, per-row token counts and sorted trainable positions, derived from the
    fixture's own mask — NOT from BufferSource.valid_flat/rows/row_counts/flat_positions.

    The first oracle called `flat_positions` itself, so it agreed with that method
    by construction: dropping the last trainable token of every masked row left all
    106 tests green (review round 1, control R-flatpos). The second read
    `src.valid_flat`, which BufferSource REWRITES in __post_init__, so a position
    lost there agreed with the oracle too (review round 2, control R2-postinit).
    """
    if src.fixture_valid is None:
        valid = np.arange(src.num_rows * SEQ, dtype=np.int64)
    else:
        valid = np.sort(src.fixture_valid)
    row_of = valid // SEQ
    row_ids = sorted(set(row_of.tolist()))
    counts = np.array([int((row_of == r).sum()) for r in row_ids], dtype=np.int64)
    return np.array(row_ids, dtype=np.int64), counts, valid, row_of


def _reference_buffers(sources, quotas, seed, n):
    """The refill exactly as it was before Phase 4 (c955dfc0): the oracle.

    Row selection from per-source cycles seeded the same way, positions taken from
    the fixture's mask, `gather_tokens` per source, concatenation, then one
    `torch.randperm` applied to every layer.
    """
    cycles = [
        AB.RowCycle(len(_oracle_rows(s)[0]), np.random.default_rng([int(seed), i]))
        for i, s in enumerate(sources)
    ]
    gen = torch.Generator().manual_seed(int(seed))
    buffers = []
    for _ in range(n):
        per_source = []
        for s, c, q in zip(sources, cycles, quotas, strict=True):
            row_ids, counts, valid, row_of = _oracle_rows(s)
            chosen_rows = row_ids[c.take(counts, q)]
            per_source.append(valid[np.isin(row_of, chosen_rows)])
        total = int(sum(f.size for f in per_source))
        order = torch.randperm(total, generator=gen)
        buf = {}
        for key in KEYS:
            parts = [
                activation_mask.gather_tokens(np.load(s.files[key], mmap_mode="r"), f)
                for s, f in zip(sources, per_source, strict=True) if f.size
            ]
            buf[key] = torch.from_numpy(np.concatenate(parts, axis=0))[order]
        buffers.append(buf)
    return buffers


class TestServedBuffersMatchTheOracle:
    """Phase 4 may change when rows are read, never what is served."""

    def test_every_buffer_equals_the_pre_phase4_refill(self, tmp_path, make_buffer):
        sources = [_source(tmp_path, 0, 11), _holey_source(tmp_path, 1, 9, seed=3)]
        quotas = [3 * SEQ, 2 * SEQ]
        expected = _reference_buffers(sources, quotas, seed=5, n=7)

        buf = make_buffer(sources, quotas, seed=5)
        for k, want in enumerate(expected):
            if k:
                buf.refill()
            for key in KEYS:
                assert torch.equal(buf.tensors[key], want[key]), (
                    f"buffer #{k + 1} for {key} differs from the refill before Phase 4"
                )

    def test_batches_drawn_through_next_batch_equal_the_oracle(self, tmp_path, make_buffer):
        """The draw the training loop makes, across refills, byte for byte."""
        sources = [_holey_source(tmp_path, 0, 10, seed=1), _source(tmp_path, 1, 6)]
        quotas = [2 * SEQ, 2 * SEQ]
        expected = _reference_buffers(sources, quotas, seed=9, n=5)

        buf = make_buffer(sources, quotas, seed=9)
        for want in expected:
            size = want[KEYS[0]].shape[0]
            got = {key: [] for key in KEYS}
            served = 0
            while served < size:
                n = min(5, size - served)
                batch = buf.next_batch(n)   # the first draw of each buffer refills
                assert buf.size == size
                for key in KEYS:
                    got[key].append(batch[key])
                served += n
            for key in KEYS:
                assert torch.equal(torch.cat(got[key]), want[key])


class TestTheWholeExtractionIsUsed:
    def test_every_token_is_served_exactly_once_before_any_repeats(self, tmp_path, make_buffer):
        """Two sources of 12 and 6 rows (144 tokens), buffer of 6 rows (48 tokens)."""
        sources = [_source(tmp_path, 0, 12), _source(tmp_path, 1, 6)]
        buf = make_buffer(sources, quotas=[4 * SEQ, 2 * SEQ])

        served = []
        for _ in range(144 // 16):  # three buffers, 3 batches of 16 each
            served += _origins(buf.next_batch(16)[KEYS[0]])

        assert len(served) == 144
        assert len(set(served)) == 144, "a token was served twice before the corpus was used"
        expected = {(0, r, p) for r in range(12) for p in range(SEQ)} | {
            (1, r, p) for r in range(6) for p in range(SEQ)
        }
        assert set(served) == expected, "some of the extraction was never read"

    def test_every_trainable_token_of_a_masked_source_is_served_once_per_pass(self, tmp_path, make_buffer):
        """Masked rows take flat_positions' valid_flat branch, which the unmasked test
        above never reaches. Quotas that cover each source make one buffer one whole
        pass, so the tokens served must be exactly the mask, each once."""
        holey = _holey_source(tmp_path, 0, 9, seed=5)
        plain = _source(tmp_path, 1, 3)
        _, _, valid, _ = _oracle_rows(holey)
        expected = {(0, int(f // SEQ), int(f % SEQ)) for f in valid} | {
            (1, r, p) for r in range(3) for p in range(SEQ)
        }
        buf = make_buffer([holey, plain], quotas=[len(valid), 3 * SEQ])
        for _ in range(3):
            served = _origins(buf.next_batch(len(expected))[KEYS[0]])
            assert len(served) == len(set(served)) == len(expected), (
                "a token was served twice, or some trainable tokens were dropped"
            )
            assert set(served) == expected, "the pass did not serve exactly the trainable tokens"

    def test_the_buffer_really_turns_over(self, tmp_path, make_buffer):
        """NEGATIVE CONTROL for the defect: the old loader's pool never changed."""
        sources = [_source(tmp_path, 0, 20)]
        buf = make_buffer(sources, quotas=[5 * SEQ])
        first_rows = {o[1] for o in _origins(buf.tensors[KEYS[0]])}
        for _ in range(3):
            buf.next_batch(40)
        assert buf.refills >= 3
        later_rows = {o[1] for o in _origins(buf.tensors[KEYS[0]])}
        assert first_rows.isdisjoint(later_rows), "the refill served rows already used"

    def test_a_source_is_reshuffled_only_after_it_is_exhausted(self, tmp_path, make_buffer):
        sources = [_source(tmp_path, 0, 6)]
        buf = make_buffer(sources, quotas=[4 * SEQ])
        _settle(buf)  # the next buffer's selection has already started a new pass
        assert buf.epochs_completed() == [0], "reported the prepared buffer, not the served one"
        buf.next_batch(4 * SEQ)
        buf.next_batch(4 * SEQ)  # needs rows 5-6 plus two from a new pass
        _settle(buf)
        assert buf.epochs_completed() == [1]
        rows = [o[1] for o in _origins(buf.tensors[KEYS[0]])]
        assert len(set(rows)) == len(rows) // SEQ, "a row appeared twice in one buffer"


class TestOnlyTrainableTokensAreServed:
    def test_padding_and_held_out_rows_never_appear(self, tmp_path, make_buffer):
        rows = 10
        mask = np.ones((rows, SEQ), dtype=bool)
        mask[:, -3:] = False            # padding at the end of every row
        mask[7:, :] = False             # rows 7-9 held out
        valid = np.flatnonzero(mask.reshape(-1))
        src = _source(tmp_path, 0, rows, valid_flat=valid)
        buf = make_buffer([src], quotas=[3 * 5])

        served = []
        for _ in range(10):
            served += _origins(buf.next_batch(5)[KEYS[0]])
        assert served, "nothing was served"
        assert all(pos < SEQ - 3 for _, _, pos in served), "a padding position was served"
        assert all(row < 7 for _, row, _ in served), "a held-out row was served"


class TestEveryLayerSeesTheSameTokens:
    def test_batches_are_aligned_across_layers(self, tmp_path, make_buffer):
        buf = make_buffer([_source(tmp_path, 0, 9), _source(tmp_path, 1, 9)], quotas=[3 * SEQ, 3 * SEQ])
        for _ in range(12):
            batch = buf.next_batch(10)
            a, b = batch[KEYS[0]], batch[KEYS[1]]
            assert torch.equal(a[:, :3], b[:, :3]), "layers were served different tokens"
            assert set(a[:, 3].tolist()) == {3.0} and set(b[:, 3].tolist()) == {4.0}


class TestTheMixtureFollowsTheQuotas:
    def test_tokens_loaded_per_source_track_the_quotas(self, tmp_path, make_buffer):
        buf = make_buffer([_source(tmp_path, 0, 40), _source(tmp_path, 1, 40)], quotas=[9 * SEQ, 1 * SEQ])
        for _ in range(30):
            buf.next_batch(40)
        a, b = buf.tokens_loaded
        assert a / (a + b) == pytest.approx(0.9, abs=0.02)


class TestBookkeeping:
    def test_tokens_loaded_counts_only_served_buffers(self, tmp_path, make_buffer):
        """The prefetch selects a buffer ahead; the count must not."""
        buf = make_buffer([_source(tmp_path, 0, 9), _source(tmp_path, 1, 7)], quotas=[2 * SEQ, 1 * SEQ])
        served = buf.size
        for _ in range(4):
            _settle(buf)
            assert sum(buf.tokens_loaded) == served
            buf.refill()
            served += buf.size
        _settle(buf)
        assert sum(buf.tokens_loaded) == served


class TestDeterminism:
    def test_the_same_seed_serves_the_same_batches(self, tmp_path, make_buffer):
        srcs = [_source(tmp_path, 0, 10), _source(tmp_path, 1, 10)]
        one = make_buffer(srcs, [3 * SEQ, 2 * SEQ], seed=7)
        two = make_buffer(srcs, [3 * SEQ, 2 * SEQ], seed=7)
        other = make_buffer(srcs, [3 * SEQ, 2 * SEQ], seed=8)
        a = [one.next_batch(12)[KEYS[0]] for _ in range(6)]
        b = [two.next_batch(12)[KEYS[0]] for _ in range(6)]
        c = [other.next_batch(12)[KEYS[0]] for _ in range(6)]
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
        assert not all(torch.equal(x, z) for x, z in zip(a, c, strict=True))

    @pytest.mark.parametrize("threads", [1, 4, 12])
    def test_prefetch_and_thread_count_change_nothing(self, tmp_path, opened, threads):
        srcs = [_holey_source(tmp_path, 0, 13, seed=2), _source(tmp_path, 1, 8)]
        quotas = [3 * SEQ, 2 * SEQ]
        sync = _builder(opened, "sync-1")(srcs, quotas, seed=11)
        pre = _builder(opened, "prefetch-1")(srcs, quotas, seed=11, read_threads=threads)
        for _ in range(20):
            x, y = sync.next_batch(9), pre.next_batch(9)
            for key in KEYS:
                assert torch.equal(x[key], y[key])


class TestMemory:
    """The buffer is sized to free memory, so a refill must never need a second buffer's worth.

    It used to release the old tensors and allocate new ones, which worked only
    if nothing else held the old tensors. The training loop held views of them
    (the b_dec initialisation, the diagnostic batch, the step's `cached`), and on
    2026-09-14 a 3-layer run on the 3090 died at its first refill. A refill now
    copies into memory allocated once — and so does the prepared copy.

    MUTATION CONTROL (2026-09-14): reallocate on every refill -> both tests fail.
    """

    def test_a_refill_writes_into_the_same_memory_even_with_a_stale_view_held(self, tmp_path, make_buffer):
        buf = make_buffer([_source(tmp_path, 0, 12)], quotas=[4 * SEQ])
        stale = {key: buf.tensors[key] for key in KEYS}      # what the training loop kept
        before = {key: t.untyped_storage().data_ptr() for key, t in stale.items()}

        buf.refill()

        after = {key: buf.tensors[key].untyped_storage().data_ptr() for key in KEYS}
        assert after == before, "a refill allocated a second buffer"

    def test_a_refill_allocates_nothing_on_the_storage_device(self, tmp_path, make_buffer, monkeypatch):
        buf = make_buffer([_source(tmp_path, 0, 12)], quotas=[4 * SEQ])
        allocations = []
        real_empty = AB.torch.empty

        def counting_empty(*args, **kwargs):
            allocations.append(args)
            return real_empty(*args, **kwargs)

        monkeypatch.setattr(AB.torch, "empty", counting_empty)
        for _ in range(3):
            buf.refill()
        _settle(buf)

        assert allocations == [], f"refills allocated buffers: {allocations}"

    def test_the_prepared_copy_reuses_its_memory(self, tmp_path, make_prefetched):
        """Identity, not data_ptr: the allocator hands a freed block straight back
        at the same address, so an address comparison passed against a
        reallocation on every prepared refill (M16, first run)."""
        buf = make_prefetched([_source(tmp_path, 0, 12)], quotas=[4 * SEQ])
        before = dict(buf._staging)          # holding them also stops address reuse
        for _ in range(3):
            buf.refill()
        _settle(buf)
        assert all(buf._staging[key] is before[key] for key in KEYS), "the prepared copy was reallocated"


    def test_a_row_larger_than_its_quota_never_reallocates(self, tmp_path, make_buffer, monkeypatch):
        """RowCycle.take always takes one whole row, so a quota below a row's size is
        exceeded by that row. Storage sized to the quotas was reallocated at the first
        refill drawing a longer row than the first buffer held — on the GPU, while the
        training loop's views of the old buffer were alive (review round 1, F3)."""
        mask = np.zeros((10, SEQ), dtype=bool)
        mask[:, :2] = True           # nine rows of 2 trainable tokens...
        mask[9, :] = True            # ...and one of 8
        src = _source(tmp_path, 0, 10, valid_flat=np.flatnonzero(mask.reshape(-1)))
        seed = next(
            s for s in range(100)
            if int(AB.RowCycle(10, np.random.default_rng([s, 0])).order[0]) != 9
        )
        buf = make_buffer([src], quotas=[1], seed=seed)
        assert buf.size == 2, "precondition: the first buffer must hold a SHORT row"
        _settle(buf)

        allocations = []
        real_empty = AB.torch.empty

        def counting_empty(*args, **kwargs):
            allocations.append(args)
            return real_empty(*args, **kwargs)

        monkeypatch.setattr(AB.torch, "empty", counting_empty)
        sizes = []
        for _ in range(10):          # a whole pass, so the long row is drawn
            buf.refill()
            sizes.append(buf.size)
        _settle(buf)

        assert 8 in sizes, "precondition: the long row was never drawn"
        assert allocations == [], f"a refill reallocated the buffer: {allocations}"


class TestPinnedHostMemory:
    """Pinned host memory is rounded up, and outlives its tensor.

    PyTorch's CUDA caching host allocator rounds every pinned allocation up to a
    power of two, and a freed pinned tensor's block stays locked in its cache;
    `torch.cuda.empty_cache()` releases device memory only (torch 2.9.1,
    ATen/core/CachingHostAllocator.h). So the 3-layer run's 15 GiB prepared copy
    pins 24 GiB, and after the training it stayed locked in the solo worker.
    There is no GPU here: pinning is simulated by accepting `pin_memory=True` for
    CPU tensors, and the release is observed where it is called.
    """

    @staticmethod
    def _fake_pinning(monkeypatch, fail=False):
        requests = []
        real_empty = AB.torch.empty

        def empty(*args, pin_memory=False, **kwargs):
            requests.append(bool(pin_memory))
            if pin_memory and fail:
                raise RuntimeError("CUDA error: out of memory (cudaHostAlloc)")
            return real_empty(*args, **kwargs)

        monkeypatch.setattr(AB.torch, "empty", empty)
        return requests

    def test_the_plan_budgets_the_rounded_pinned_block(self):
        assert AB.pinned_block_bytes(384) == 512 and AB.pinned_block_bytes(512) == 512
        assert AB.pinned_block_bytes(657_408 * 2_048 * 4) == 8 * 1024**3
        # 24 tokens x 16 bytes = 384 bytes, pinned as 512 bytes = 32 tokens.
        assert AB.plan_prefetch(True, 31, 24, storage_on_host=False, bytes_per_token=16)[0] is False
        assert AB.plan_prefetch(True, 32, 24, storage_on_host=False, bytes_per_token=16)[0] is True
        # Not pinned: the byte count is the cost, as before.
        assert AB.plan_prefetch(True, 24, 24, storage_on_host=False)[0] is True

    def test_a_gpu_buffer_budgets_its_pinned_copy_rounded(self, tmp_path, opened, monkeypatch):
        self._fake_pinning(monkeypatch)
        # META storage stands in for the GPU, whose prepared copy is pinned.
        monkeypatch.setattr(AB.RollingActivationBuffer, "_staging_pinned", lambda self: True)
        make = _builder(opened, "prefetch-12", storage_device=META, train_device=META)
        src = _source(tmp_path, 0, 12)
        capacity = 3 * SEQ           # 24 tokens x 4 dims x 4 bytes = 384, pinned as 512 = 32 tokens
        tight = make([src], quotas=[capacity], host_ram_tokens=capacity)
        assert tight.prefetch is False, tight.prefetch_reason
        roomy = make([src], quotas=[capacity], host_ram_tokens=32)
        assert roomy.prefetch is True, roomy.prefetch_reason

    def test_host_storage_is_not_pinned_past_the_budget(self, tmp_path, opened, monkeypatch):
        requests = self._fake_pinning(monkeypatch)
        src = _source(tmp_path, 0, 12)
        capacity = 3 * SEQ
        # cpu_rolling: the storage plan sizes the buffer to the whole host budget.
        _builder(opened, "sync-1")([src], quotas=[capacity], pin_memory=True, host_ram_tokens=capacity)
        assert requests == [False, False], f"pinned past the budget: {requests}"
        requests.clear()
        _builder(opened, "sync-1")([src], quotas=[capacity], pin_memory=True, host_ram_tokens=32)
        assert requests == [True, True], f"an affordable buffer was not pinned: {requests}"

    def test_pinning_that_fails_falls_back_to_pageable_memory(self, tmp_path, opened, monkeypatch, caplog):
        requests = self._fake_pinning(monkeypatch, fail=True)
        monkeypatch.setattr(AB.RollingActivationBuffer, "_staging_pinned", lambda self: True)
        sources = [_holey_source(tmp_path, 0, 10, seed=1), _source(tmp_path, 1, 6)]
        quotas = [2 * SEQ, 2 * SEQ]
        expected = _reference_buffers(sources, quotas, seed=4, n=4)
        with caplog.at_level(logging.WARNING, logger=AB.logger.name):
            buf = _builder(opened, "prefetch-12")(sources, quotas, seed=4)
        assert buf.prefetch and buf._staging and True in requests, "precondition: a pinned copy was attempted"
        assert "using pageable memory" in caplog.text, caplog.text
        for k, want in enumerate(expected):
            if k:
                buf.refill()
            for key in KEYS:
                assert torch.equal(buf.tensors[key], want[key])

    @pytest.mark.parametrize("mode, pinned", [("prefetch-12", "prepared copy"), ("sync-1", "buffer")])
    def test_close_releases_the_buffer_and_the_pinned_cache(self, tmp_path, opened, monkeypatch, mode, pinned):
        self._fake_pinning(monkeypatch)
        overrides = {}
        if pinned == "prepared copy":
            monkeypatch.setattr(AB.RollingActivationBuffer, "_staging_pinned", lambda self: True)
        else:
            overrides["pin_memory"] = True
        buf = _builder(opened, mode)([_source(tmp_path, 0, 12)], quotas=[3 * SEQ], **overrides)
        buf.refill()
        assert buf._pinned_bytes > 0, "precondition: something was pinned"
        storages = _storages([*buf._storage.values(), *buf._staging.values()])
        seen = []

        def release(device=None):
            gc.collect()
            seen.append(_holders(storages))

        monkeypatch.setattr(AB, "_empty_pinned_host_cache", release)
        buf.close()

        assert len(seen) == 1, "close() never released PyTorch's pinned host cache"
        assert seen[0] == [0] * len(storages), (
            f"the cache was emptied while {seen[0]} references still held the blocks, "
            "so nothing was released"
        )

    @pytest.mark.parametrize("mode, pinned", [("prefetch-12", "prepared copy"), ("sync-1", "buffer")])
    def test_close_releases_blocks_a_dropped_failure_still_references(self, tmp_path, opened, monkeypatch, mode, pinned):
        """A read that fails leaves its frames in a reference cycle, and those frames hold
        numpy views of the pinned rows (`dest`, `dests`) and the refill's `storage`. A
        view keeps the STORAGE alive but not the tensor object, which is why the test
        above now counts storage references instead of watching tensors. Measured before
        this round: the cache was emptied with [2, 2] references left on the prepared
        copy (the sticky `_failure` kept the traceback reachable) and 3 on the buffer.
        Celery 5.6 clears a failed task's frames before after_return; close() must not
        depend on a collection having run since."""
        self._fake_pinning(monkeypatch)
        overrides = {}
        if pinned == "prepared copy":
            monkeypatch.setattr(AB.RollingActivationBuffer, "_staging_pinned", lambda self: True)
        else:
            overrides["pin_memory"] = True
        buf = _builder(opened, mode)([_source(tmp_path, 0, 20)], quotas=[4 * SEQ], **overrides)
        _settle(buf)
        assert buf._pinned_bytes > 0, "precondition: something was pinned"
        storages = _storages([*buf._storage.values(), *buf._staging.values()])

        def vanished(self, key, dest, inverse, read, abort):
            raise OSError("extraction file vanished")

        monkeypatch.setattr(AB.RollingActivationBuffer, "_read_row", vanished)

        def a_step_that_fails_and_moves_on():
            for _ in range(2):          # prefetch: the buffer already prepared is served first
                try:
                    buf.refill()
                except OSError:
                    return
            raise AssertionError("precondition: no refill failed")

        seen = []
        monkeypatch.setattr(AB, "_empty_pinned_host_cache", lambda device=None: seen.append(_holders(storages)))
        gc.disable()
        try:
            a_step_that_fails_and_moves_on()
            buf.close()
        finally:
            gc.enable()
        assert seen == [[0] * len(storages)], (
            f"close() emptied the cache while a dropped failure still held the blocks: {seen}"
        )


class TestAHeldBatchDoesNotPinTheOldBuffer:
    """2026-09-14, found by the multi-GPU Phase 4a measurement on the node.

    A 3-layer JumpReLU training on five extractions (gpu_rolling, ~15 GB of
    buffer on the 3090) died at step 318 — the first refill — with "Tried to
    allocate 4.97 GiB ... 18.13 GiB allocated by PyTorch". `next_batch` served
    SLICES of the buffer, and the training loop still held the previous step's
    batch when the next draw triggered the refill, so the old buffer could not
    be freed while the new one was loaded. The OOM handler then halved the batch
    size, which cannot help, and the run failed.

    Since refills copy into the same memory, a served VIEW would also have its
    rows overwritten under the training step still using it. A batch is a copy.

    MUTATION CONTROL (2026-09-14, re-run after the in-place refill): `next_batch`
    back to returning the slice when no device transfer is needed -> this test
    fails on the overwritten rows.
    """

    def test_a_batch_held_across_a_refill_keeps_its_rows(self, tmp_path, make_buffer):
        buf = make_buffer([_source(tmp_path, 0, 20)], quotas=[5 * SEQ])  # a 40-token buffer
        held = buf.next_batch(40)             # the whole buffer, still referenced
        rows = {key: held[key].clone() for key in KEYS}

        buf.next_batch(8)                     # forces the refill while `held` is alive

        for key in KEYS:
            assert torch.equal(held[key], rows[key]), "the refill overwrote a batch the step still held"


class TestPendingTransfer:
    """A batch leaves pinned host storage with non_blocking=True, so the GPU may
    not have copied it when the next draw refills that same memory. Latent since
    the in-place refill (c955dfc0) on the cpu_rolling path; found 2026-09-14."""

    @pytest.mark.parametrize("mode", ["sync-1", "prefetch-12"])
    def test_a_refill_waits_for_the_last_batch_to_leave_host_storage(self, tmp_path, opened, monkeypatch, mode):
        buf = _builder(opened, mode, storage_device=CPU, train_device=META)(
            [_source(tmp_path, 0, 12)], quotas=[4 * SEQ]
        )
        old_rows = buf.tensors[KEYS[0]].clone()
        seen = []

        class Marker:
            def synchronize(self):
                # What the queued copy would read if it ran now.
                seen.append(torch.equal(buf.tensors[KEYS[0]], old_rows))

        monkeypatch.setattr(buf, "_transfer_marker", lambda device: Marker())
        buf.next_batch(4 * SEQ)          # a transfer out of host storage, still "queued"
        assert seen == []
        buf.next_batch(4)                # triggers the refill
        assert seen == [True], "the refill overwrote host rows before the pending transfer finished"


class TestNonCpuStorage:
    """The GPU-storage branches (host staging, transient per-layer host rows, the
    copy into device storage) cannot be checked for values without a GPU; meta
    tensors at least drive every branch end to end."""

    def test_device_storage_refills_in_both_modes(self, tmp_path, opened):
        for mode in ("sync-4", "prefetch-12"):
            make = _builder(opened, mode, storage_device=META, train_device=META)
            buf = make([_source(tmp_path, 0, 12), _holey_source(tmp_path, 1, 6, seed=4)],
                       quotas=[3 * SEQ, 1 * SEQ])
            for _ in range(10):
                batch = buf.next_batch(10)
                assert batch[KEYS[0]].shape == (10, 4) and batch[KEYS[0]].device.type == "meta"
            assert buf.refills >= 3


class TestPrefetch:
    def test_the_next_buffer_is_prepared_off_the_training_thread(self, tmp_path, make_prefetched, monkeypatch):
        prepared_on = []
        real = AB.RollingActivationBuffer._prepare_next

        def recording(self):
            prepared_on.append(threading.current_thread().name)
            return real(self)

        monkeypatch.setattr(AB.RollingActivationBuffer, "_prepare_next", recording)
        buf = make_prefetched([_source(tmp_path, 0, 30)], quotas=[4 * SEQ])
        for _ in range(4):
            buf.refill()
            assert buf.last_refill_timings["prefetched"] is True
        _settle(buf)
        assert prepared_on[0] == threading.current_thread().name  # the first has nothing to wait for
        assert prepared_on[1:] and all(n.startswith("activation-prefetch") for n in prepared_on[1:]), prepared_on

    def test_synchronous_mode_never_starts_a_prefetch_thread(self, tmp_path, opened):
        buf = _builder(opened, "sync-4")([_source(tmp_path, 0, 30)], quotas=[4 * SEQ])
        for _ in range(3):
            buf.refill()
            assert buf.last_refill_timings["prefetched"] is False
        assert buf._prefetcher is None and buf._pending is None and not buf._staging


class TestPrefetchFailures:
    @staticmethod
    def _fail_after_first_buffer(monkeypatch):
        real = AB.RollingActivationBuffer._read_row

        def failing(self, key, dest, inverse, read, abort):
            if self.refills >= 1:
                raise OSError("extraction file vanished")
            return real(self, key, dest, inverse, read, abort)

        monkeypatch.setattr(AB.RollingActivationBuffer, "_read_row", failing)

    def test_an_exception_in_the_prefetch_surfaces_at_the_next_refill(self, tmp_path, make_prefetched, monkeypatch):
        self._fail_after_first_buffer(monkeypatch)
        buf = make_prefetched([_source(tmp_path, 0, 20)], quotas=[4 * SEQ])
        _settle(buf)
        buf.next_batch(4 * SEQ)                # the first buffer is intact

        outcome = {}

        def draw():
            try:
                buf.next_batch(1)
                outcome["result"] = "served"
            except BaseException as exc:  # noqa: BLE001 - the test inspects it
                outcome["result"] = exc

        drawer = threading.Thread(target=draw)
        drawer.start()
        drawer.join(timeout=10)
        assert not drawer.is_alive(), "the refill hung waiting on a failed prefetch"
        assert isinstance(outcome["result"], OSError), f"the failure was swallowed: {outcome['result']!r}"
        assert "vanished" in str(outcome["result"])



class TestAFailedRefillIsRetriedNotSkipped:
    """Review round 2. The rows a refill selects are taken from the cycles when it is
    SELECTED, so a refill that fails and is then retried with a new selection has
    skipped those rows for the whole pass. Before this round that happened in two
    places, and read failures behaved differently by mode:
      * a failure after `pending.result()` (the synchronize, the copy) dropped the
        prepared buffer, and the retry prepared the next one;
      * a synchronous read failure re-selected on the retry;
      * a background read failure was sticky, so the run could never recover.
    Now every refill that fails keeps its selection, and the next refill serves
    exactly those rows."""

    @staticmethod
    def _served(buf):
        return {key: buf.tensors[key].clone() for key in KEYS}

    def test_a_read_failure_is_retried_on_the_same_rows(self, tmp_path, make_buffer, monkeypatch):
        sources = [_holey_source(tmp_path, 0, 12, seed=6), _source(tmp_path, 1, 7)]
        quotas = [3 * SEQ, 2 * SEQ]
        expected = _reference_buffers(sources, quotas, seed=2, n=4)
        buf = make_buffer(sources, quotas, seed=2)
        _settle(buf)
        served = [self._served(buf)]

        real = AB.RollingActivationBuffer._read_row
        state = {"armed": True}

        def transient(self, key, dest, inverse, read, abort):
            if state["armed"]:
                raise OSError("transient read error")
            return real(self, key, dest, inverse, read, abort)

        monkeypatch.setattr(AB.RollingActivationBuffer, "_read_row", transient)
        with pytest.raises(OSError):
            for _ in range(2):          # prefetch: the buffer already prepared is served first
                buf.refill()
                served.append(self._served(buf))
        state["armed"] = False
        buf.refill()
        served.append(self._served(buf))

        for k, got in enumerate(served):
            for key in KEYS:
                assert torch.equal(got[key], expected[k][key]), (
                    f"buffer #{k + 1} is not the oracle's #{k + 1}: a failed refill skipped its rows"
                )

    def test_a_failure_part_way_through_the_writes_is_retried_and_never_served(self, tmp_path, make_buffer, monkeypatch):
        """The first layer's rows are written, the second layer's write fails. A draw
        small enough to fit the old buffer's tail must neither serve the layers out of
        step nor skip the selection."""
        sources = [_source(tmp_path, 0, 12)]
        quotas = [4 * SEQ]
        expected = _reference_buffers(sources, quotas, seed=4, n=2)
        buf = make_buffer(sources, quotas, seed=4)
        _settle(buf)
        buf.next_batch(2 * SEQ)             # 16 of 32 tokens left in the buffer

        real = buf._storage_for
        state = {"armed": True}

        def second_layer_fails(key, total):
            if key == KEYS[1] and state["armed"]:
                state["armed"] = False
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 4.97 GiB")
            return real(key, total)

        monkeypatch.setattr(buf, "_storage_for", second_layer_fails)
        with pytest.raises(torch.OutOfMemoryError):
            buf.refill()

        batch = buf.next_batch(8)           # fits the old tail: the OOM handler's retry
        assert torch.equal(batch[KEYS[0]][:, :3], batch[KEYS[1]][:, :3]), "the layers were served different tokens"
        for key in KEYS:
            assert torch.equal(batch[key], expected[1][key][:8]), (
                "the draw after a failed refill served stale rows or skipped the selection"
            )
        assert buf.tokens_dropped == 2 * SEQ, "the unserved tail of the overwritten buffer was not counted once"

    def test_a_draw_whose_copy_fails_serves_the_same_tokens_on_the_retry(self, tmp_path, make_buffer, monkeypatch):
        """`torch.OutOfMemoryError` is a RuntimeError saying "out of memory", so the
        training loop's OOM handler halves the batch and draws again. The failed draw
        had already advanced the position, so its tokens were never served and were not
        counted as dropped either."""
        assert issubclass(torch.OutOfMemoryError, RuntimeError)
        sources = [_source(tmp_path, 0, 12)]
        expected = _reference_buffers(sources, [4 * SEQ], seed=1, n=1)
        buf = make_buffer(sources, [4 * SEQ], seed=1)
        _settle(buf)

        real_clone = torch.Tensor.clone
        state = {"calls": 0}

        def second_layer_oom(self, *args, **kwargs):
            state["calls"] += 1
            if state["calls"] == 2:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 64.00 MiB")
            return real_clone(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "clone", second_layer_oom)
        with pytest.raises(torch.OutOfMemoryError):
            buf.next_batch(16)
        monkeypatch.undo()

        halves = [buf.next_batch(8), buf.next_batch(8)]
        for key in KEYS:
            assert torch.equal(torch.cat([h[key] for h in halves]), expected[0][key][:16]), (
                "the tokens of the draw that failed were skipped"
            )


class TestAFailedReadLeavesNoStragglers:
    def test_rows_still_being_read_finish_before_the_failure_is_raised(self, tmp_path, opened, monkeypatch):
        """A synchronous refill on host storage writes straight into the served memory.

        If one row fails while others are still being copied and the failure is
        raised at once, those rows land AFTER a retried refill has written the
        next buffer — silently mixing two buffers.
        """
        # Review round 2: a retry now re-reads the SAME selection, so a straggler would
        # write the very bytes the retry writes and a comparison of contents could no
        # longer see it. Deterministic instead: all four read threads are held inside
        # their reads, one fails, and the refill must still be waiting on the other
        # three until the test lets them finish.
        buf = _builder(opened, "sync-4")([_source(tmp_path, 0, 40)], quotas=[8 * SEQ], seed=3)
        inside = threading.Barrier(4, timeout=10)
        gate = threading.Event()
        lock = threading.Lock()
        entered = {"n": 0}

        class HeldRows:
            def __init__(self, array):
                self.array = array

            @property
            def shape(self):
                return self.array.shape

            def __getitem__(self, index):
                with lock:
                    entered["n"] += 1
                    first = entered["n"] == 1
                inside.wait()                      # every read thread is mid-read
                if first:
                    raise OSError("transient read error")
                if not gate.wait(10):
                    raise TimeoutError("the test never released the reads")
                return self.array[index]

        for k, mm in list(buf._mmaps.items()):
            buf._mmaps[k] = HeldRows(mm)
        outcome = {}

        def refill():
            try:
                buf.refill()
                outcome["result"] = "returned"
            except BaseException as exc:  # noqa: BLE001 - the test inspects it
                outcome["result"] = exc

        refiller = threading.Thread(target=refill)
        refiller.start()
        refiller.join(timeout=0.5)
        waited_for_the_rows = refiller.is_alive()
        gate.set()
        refiller.join(timeout=10)

        assert not refiller.is_alive(), "the refill hung"
        assert isinstance(outcome["result"], OSError), outcome
        assert waited_for_the_rows, "the failure was raised while three rows were still being copied"

    def test_a_failed_refill_does_not_count_the_unserved_tail_twice(self, tmp_path, opened, monkeypatch):
        real = AB.RollingActivationBuffer._read_row
        state = {"armed": False}

        def failing_once(self, key, dest, inverse, read, abort):
            if state["armed"]:
                state["armed"] = False
                raise OSError("transient read error")
            return real(self, key, dest, inverse, read, abort)

        monkeypatch.setattr(AB.RollingActivationBuffer, "_read_row", failing_once)
        buf = _builder(opened, "sync-1")([_source(tmp_path, 0, 20)], quotas=[4 * SEQ])
        buf.next_batch(10)                   # 22 of 32 tokens left unserved
        state["armed"] = True
        with pytest.raises(OSError):
            buf.refill()
        buf.refill()
        assert buf.tokens_dropped == 22


    def test_a_failed_read_stops_the_rest_of_the_refill(self, tmp_path, opened):
        """A failing refill must not read the rest of a 15 GB buffer before saying so."""
        buf = _builder(opened, "sync-1")([_source(tmp_path, 0, 200)], quotas=[100 * SEQ])
        faults = _Faults(fail_first=True, fail_after=0.0).install(buf)
        with pytest.raises(OSError):
            buf.refill()
        assert faults.reads == 1, f"{faults.reads - 1} rows were still read after the failure"


class TestClose:
    def test_close_joins_every_thread(self, tmp_path, opened):
        buf = _builder(opened, "prefetch-12")([_source(tmp_path, 0, 40)], quotas=[4 * SEQ])
        buf.refill()
        assert _activation_threads(buf), "nothing was running to join"
        buf.close()
        assert _activation_threads(buf) == []
        buf.close()  # idempotent
        with pytest.raises(RuntimeError, match="closed"):
            buf.refill()

    def test_a_cancelled_training_does_not_wait_for_the_prefetch(self, tmp_path, opened):
        """Closing mid-prefetch stops between rows instead of reading the rest."""
        buf = _builder(opened, "prefetch-1")([_source(tmp_path, 0, 400)], quotas=[60 * SEQ])
        _settle(buf)
        faults = _Faults(delay=0.05).install(buf)
        buf.refill()    # serves the prepared buffer; the next one is now read slowly:
        # 60 rows x 2 layers x 50 ms on one thread = 6 s to prepare in full.
        assert faults.started.wait(5), "the prefetch never started"

        t0 = time.monotonic()
        buf.close()
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"close() waited {elapsed:.1f}s for the prefetch to finish"
        assert _activation_threads(buf) == []


class TestTheRamBudget:
    def test_the_plan(self):
        assert AB.plan_prefetch(True, 1_000, 1_000, storage_on_host=False)[0] is True
        assert AB.plan_prefetch(True, 999, 1_000, storage_on_host=False)[0] is False
        # Buffer AND copy in RAM: the storage plan sizes cpu_rolling to the whole budget.
        assert AB.plan_prefetch(True, 1_000, 1_000, storage_on_host=True)[0] is False
        assert AB.plan_prefetch(True, 2_000, 1_000, storage_on_host=True)[0] is True
        assert AB.plan_prefetch(True, None, 1_000, storage_on_host=False)[0] is False
        assert AB.plan_prefetch(False, BIG, 1_000, storage_on_host=False)[0] is False

    def test_a_budget_that_cannot_hold_a_second_copy_keeps_refills_synchronous(self, tmp_path, opened, caplog):
        src = _source(tmp_path, 0, 30)
        capacity = 4 * SEQ
        with caplog.at_level(logging.INFO, logger=AB.logger.name):
            buf = _builder(opened, "prefetch-12")([src], quotas=[capacity], host_ram_tokens=2 * capacity - 1)
        assert buf.prefetch is False
        assert buf._prefetcher is None and not buf._staging
        buf.refill()
        assert buf._pending is None and buf.last_refill_timings["prefetched"] is False
        assert "prefetch OFF" in caplog.text and f"{2 * capacity:,}" in caplog.text, caplog.text

    def test_a_budget_that_holds_it_prefetches(self, tmp_path, opened):
        capacity = 4 * SEQ
        buf = _builder(opened, "prefetch-12")(
            [_source(tmp_path, 0, 30)], quotas=[capacity], host_ram_tokens=2 * capacity
        )
        assert buf.prefetch is True and buf._pending is not None


class TestParallelReads:
    def test_read_thread_count(self):
        env = AB.READ_THREADS_ENV
        assert AB.read_thread_count({env: "4"}, cpu_count=16) == 4
        assert AB.read_thread_count({}, cpu_count=16) == AB.DEFAULT_READ_THREADS
        assert AB.read_thread_count({}, cpu_count=6) == 6
        assert AB.read_thread_count({env: "64"}, cpu_count=16) == 16
        assert AB.read_thread_count({env: "0"}, cpu_count=16) == AB.DEFAULT_READ_THREADS
        assert AB.read_thread_count({env: "many"}, cpu_count=2) == 2

    def test_the_buffer_reads_the_environment(self, tmp_path, opened, monkeypatch):
        monkeypatch.setenv(AB.READ_THREADS_ENV, "2")
        monkeypatch.setenv(AB.PREFETCH_ENV, "0")
        buf = AB.RollingActivationBuffer(
            [_source(tmp_path, 0, 10)], KEYS, [2 * SEQ], seed=0,
            storage_device=CPU, train_device=CPU, host_ram_tokens=BIG,
        )
        opened.append(buf)
        assert buf.read_threads == 2 and buf.prefetch is False

    def test_reads_really_run_concurrently(self, tmp_path, opened, monkeypatch):
        real = AB.RollingActivationBuffer._read_row
        lock = threading.Lock()
        state = {"now": 0, "max": 0}

        def tracking(self, key, dest, inverse, read, abort):
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.01)
            try:
                return real(self, key, dest, inverse, read, abort)
            finally:
                with lock:
                    state["now"] -= 1

        monkeypatch.setattr(AB.RollingActivationBuffer, "_read_row", tracking)
        _builder(opened, "sync-4")([_source(tmp_path, 0, 20)], quotas=[8 * SEQ])
        assert state["max"] >= 3, f"at most {state['max']} rows were read at once with 4 threads"

        state.update(now=0, max=0)
        _builder(opened, "sync-1")([_source(tmp_path, 1, 20)], quotas=[8 * SEQ])
        assert state["max"] == 1

    def test_row_reads_follow_the_gather_order(self):
        flat = np.array([3 * SEQ + 1, 3 * SEQ + 2, 3 * SEQ + 3, 5 * SEQ + 0, 5 * SEQ + 6])
        reads = AB.plan_row_reads(2, flat, SEQ, offset=10)
        assert [(r[0], r[1], r[3], r[4]) for r in reads] == [(2, 3, 10, 13), (2, 5, 13, 15)]
        assert reads[0][2] == slice(1, 4)
        assert np.array_equal(reads[1][2], [0, 6])


class TestRefillTimings:
    FIELDS = ("stall_s", "wait_s", "copy_s", "select_s", "shuffle_s", "gather_s", "prefetched")

    def test_each_refill_logs_where_the_time_went(self, tmp_path, make_buffer, caplog):
        buf = make_buffer([_source(tmp_path, 0, 20)], quotas=[4 * SEQ])
        with caplog.at_level(logging.INFO, logger=AB.logger.name):
            buf.refill()
        assert set(self.FIELDS) <= set(buf.last_refill_timings)
        line = [r.getMessage() for r in caplog.records if "refill #2" in r.getMessage()]
        assert line, caplog.text
        for word in ("in ", "wait ", "copy ", "select ", "shuffle ", "gather ", "read threads"):
            assert word in line[0], f"{word!r} missing from {line[0]!r}"

    def test_the_wait_for_a_slow_prefetch_is_measured(self, tmp_path, opened):
        buf = _builder(opened, "prefetch-1")([_source(tmp_path, 0, 40)], quotas=[8 * SEQ])
        _settle(buf)
        _Faults(delay=0.02).install(buf)
        buf.refill()    # the prepared buffer; the next takes 8 rows x 2 layers x 20 ms ~ 0.3 s
        buf.refill()    # arrives before it is ready, so it waits
        timings = buf.last_refill_timings
        assert timings["prefetched"] is True
        assert timings["wait_s"] >= 0.15, timings
        assert timings["stall_s"] >= timings["wait_s"]
        assert timings["gather_s"] >= 0.15, timings


class TestShapeChecks:
    def test_a_file_whose_width_disagrees_is_refused(self, tmp_path, opened):
        a = _source(tmp_path, 0, 5)
        b = _source(tmp_path, 1, 5)
        np.save(b.files[KEYS[1]], np.zeros((5, SEQ, 7), dtype=np.float16))
        with pytest.raises(ValueError, match="width"):
            _builder(opened, "sync-1")([a, b], quotas=[SEQ, SEQ])

    def test_a_file_whose_sequence_length_disagrees_is_refused(self, tmp_path, opened):
        src = _source(tmp_path, 0, 5)
        np.save(src.files[KEYS[0]], np.zeros((5, SEQ + 1, 4), dtype=np.float16))
        with pytest.raises(ValueError, match="expected"):
            _builder(opened, "sync-1")([src], quotas=[SEQ])


class TestStoragePlan:
    @pytest.mark.parametrize(
        "total, gpu, useful, ram, expected",
        [
            (1_000, 5_000, 100, 0, ("gpu_all", 1_000)),
            (20_000_000, 2_400_000, 81_920, 10**9, ("gpu_rolling", 2_400_000)),
            (1_000_000, 10, 81_920, 5_000_000, ("cpu_all", 1_000_000)),
            (50_000_000, 10, 81_920, 5_000_000, ("cpu_rolling", 5_000_000)),
        ],
    )
    def test_modes(self, total, gpu, useful, ram, expected):
        assert AB.plan_activation_storage(total, gpu, useful, ram) == expected

    def test_refuses_when_nothing_can_hold_a_useful_buffer(self):
        with pytest.raises(ValueError):
            AB.plan_activation_storage(10**8, 10, 81_920, 1_000)


class TestTheTrainingLoopUsesIt:
    """AST, not text: comments in training_tasks describe these calls."""

    def _tree(self):
        from src.workers import training_tasks

        return ast.parse(inspect.getsource(training_tasks))

    def _calls(self):
        return [n for n in ast.walk(self._tree()) if isinstance(n, ast.Call)]

    def _named(self, name):
        return [
            c for c in self._calls()
            if (getattr(c.func, "attr", None) or getattr(c.func, "id", None)) == name
        ]

    def _train_sae_task(self):
        return next(
            n for n in ast.walk(self._tree())
            if isinstance(n, ast.FunctionDef) and n.name == "train_sae_task"
        )

    def test_the_storage_plan_decides_the_mode(self):
        assert self._named("plan_activation_storage"), "the storage decision is not made by the planner"

    def test_the_buffer_is_built_with_the_mixture_quotas(self):
        builds = self._named("RollingActivationBuffer")
        assert builds, "training never builds the rolling buffer"
        args = [a for c in builds for a in c.args] + [kw.value for c in builds for kw in c.keywords]
        assert any(isinstance(a, ast.Name) and a.id == "allocation" for a in args), (
            "the buffer is not given the allocate_tokens quotas, so dataset_weights do nothing"
        )

    def test_the_buffer_is_given_the_ram_budget(self):
        """Without it the prefetch has no bound, so it stays off: a silent 44% slowdown."""
        builds = self._named("RollingActivationBuffer")
        budgets = [kw.value for c in builds for kw in c.keywords if kw.arg == "host_ram_tokens"]
        assert len(budgets) == 1 and isinstance(budgets[0], ast.Name), budgets
        assert budgets[0].id == "max_ram_tokens_per_layer", (
            "the prefetch must be bounded by the storage plan's RAM allowance"
        )

    def test_the_task_registers_the_buffer_for_cleanup(self):
        fn = self._train_sae_task()
        build = next(
            n for n in ast.walk(fn)
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
            and getattr(n.value.func, "attr", None) == "RollingActivationBuffer"
        )
        built_name = build.targets[0].id
        registrations = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "_activation_stream"
                and isinstance(t.value, ast.Name) and t.value.id == "self"
                for t in n.targets
            )
            and isinstance(n.value, ast.Name) and n.value.id == built_name
            and n.lineno > build.lineno
        ]
        assert len(registrations) == 1, "the buffer is never handed to after_return, so its threads outlive the task"

    def test_after_return_closes_the_buffer(self, tmp_path, opened, monkeypatch):
        """Driven, not scraped: every exit of a Celery task runs after_return."""
        from src.workers import training_tasks

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        buf = _builder(opened, "prefetch-12")([_source(tmp_path, 0, 30)], quotas=[4 * SEQ])
        assert _activation_threads(buf)
        task = training_tasks.train_sae_task
        monkeypatch.setattr(task, "_activation_stream", buf, raising=False)

        task.after_return("REVOKED", None, "task-1", (), {}, None)

        assert buf._closed and _activation_threads(buf) == []
        assert task._activation_stream is None, "the task still holds the finished buffer"

    def test_the_step_loop_draws_through_the_helper(self):
        """The AST half. The driven half is TestTheStepDraw below: this guard
        alone survived a mutation that made the branch dead (`if False:`)."""
        assert self._named("draw_cached_batch"), "the training step does not draw through draw_cached_batch"
        assert self._named("next_batch"), "nothing draws from the rolling buffer"

    def test_after_return_releases_pinned_host_memory_once_everything_is_collected(self, monkeypatch):
        """Driven. Review round 2: only RollingActivationBuffer.close() emptied PyTorch's
        pinned host cache, so the fixed pool (cpu_all), which pins every layer with no
        buffer at all, stayed locked in the solo worker after every such training. And
        close() runs BEFORE after_return's collection, so a block held only by garbage
        was not released either. A stand-in block held by a reference cycle is the
        garbage; the release must see it gone."""
        from src.workers import training_tasks

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        task = training_tasks.train_sae_task
        monkeypatch.setattr(task, "_activation_stream", None, raising=False)

        class Cycle:
            pass

        pool = Cycle()
        pool.me = pool
        pool.layer = torch.empty(64, 4)
        storages = _storages([pool.layer])
        del pool

        seen = []
        monkeypatch.setattr(AB, "_empty_pinned_host_cache", lambda device=None: seen.append(_holders(storages)))
        gc.disable()
        try:
            task.after_return("SUCCESS", {"status": "completed"}, "task-1", (), {}, None)
        finally:
            gc.enable()
        assert seen == [[0]], (
            f"after_return released the pinned host cache {len(seen)} time(s), seeing {seen} "
            "references to a block only garbage held"
        )

    def test_a_new_training_closes_a_buffer_left_behind_before_measuring_memory(self):
        """A training that never reached after_return (called outside Celery) leaves its
        buffer on the task; the next one overwrote `_activation_stream` without closing
        it, and measured free GPU memory with the old buffer still allocated."""
        fn = self._train_sae_task()
        closes = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_close_activation_stream"
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"
        ]
        measures = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) in ("mem_get_info", "plan_activation_storage")
        ]
        assert measures, "precondition: the memory measurement moved"
        assert len(closes) == 1, f"train_sae_task closes a previous buffer {len(closes)} times"
        assert closes[0].lineno < min(n.lineno for n in measures), (
            "the previous buffer is closed only after free memory was measured"
        )

    def test_the_fixed_pool_is_pinned_only_within_the_ram_budget(self):
        """cpu_all pinned each layer with `tensor.pin_memory()`, which PyTorch rounds up
        to a power of two: up to TWICE the per-layer allowance the storage plan approved,
        locked, unswappable, for every layer. The rolling buffer was fixed for exactly
        this in round 1; its sibling was not."""
        fn = self._train_sae_task()
        raw = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "pin_memory"
        ]
        assert raw == [], f"train_sae_task pins with no budget at line(s) {[n.lineno for n in raw]}"
        pins = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "pin_within_budget"
        ]
        assert len(pins) == 1, "the fixed pool is not pinned through pin_within_budget"
        budget = pins[0].args[1] if len(pins[0].args) > 1 else next(
            kw.value for kw in pins[0].keywords if kw.arg == "allowed_bytes"
        )
        assert isinstance(budget, ast.BinOp) and isinstance(budget.op, ast.Mult), ast.dump(budget)
        assert {n.id for n in ast.walk(budget) if isinstance(n, ast.Name)} == {
            "max_ram_tokens_per_layer", "bytes_per_token"
        }, "the pin budget is not the storage plan's per-layer RAM allowance"


class TestPinnedHostRelease:
    """`torch._C._host_emptyCache` is private. Every other test replaces the function
    that calls it, so a torch upgrade that renamed it would leave the suite green while
    production only logged a warning and kept 24 GiB locked (control R2-privname)."""

    @pytest.mark.skipif(
        torch.version.cuda is None,
        reason="a CPU-only torch build (CI) has no pinned host allocator; the image's CUDA build is what must expose it",
    )
    def test_this_torch_exposes_it(self):
        assert callable(getattr(torch._C, "_host_emptyCache", None)), (
            f"torch {torch.__version__} has no torch._C._host_emptyCache: pinned host memory "
            "can no longer be released; find its replacement before upgrading"
        )

    def test_the_release_calls_torch(self, monkeypatch):
        calls = []
        # raising=False: a CPU-only torch build (CI) has no such attribute to replace.
        monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: calls.append(1), raising=False)
        AB._empty_pinned_host_cache()
        assert calls == [1]

    def test_a_torch_without_it_says_so(self, monkeypatch, caplog):
        monkeypatch.delattr(torch._C, "_host_emptyCache", raising=False)
        with caplog.at_level(logging.WARNING, logger=AB.logger.name):
            AB._empty_pinned_host_cache()
        assert "stays locked" in caplog.text


class TestPinWithinBudget:
    @staticmethod
    def _fake_pin(monkeypatch, fail=False):
        calls = []

        def pin_memory(self, *args, **kwargs):
            calls.append(self.numel() * self.element_size())
            if fail:
                raise RuntimeError("CUDA error: out of memory (cudaHostAlloc)")
            return self.clone()

        monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
        return calls

    def test_a_pool_whose_rounded_block_fits_is_pinned(self, monkeypatch):
        calls = self._fake_pin(monkeypatch)
        pool = torch.zeros(24, 4)                  # 384 bytes, pinned as a 512-byte block
        out, pinned = AB.pin_within_budget(pool, 512, "pool")
        assert pinned is True and calls == [384] and out is not pool

    def test_a_pool_whose_rounded_block_is_over_the_budget_stays_pageable(self, monkeypatch, caplog):
        calls = self._fake_pin(monkeypatch)
        pool = torch.zeros(24, 4)
        with caplog.at_level(logging.WARNING, logger=AB.logger.name):
            out, pinned = AB.pin_within_budget(pool, 511, "pool")
        assert pinned is False and calls == [] and out is pool
        assert "pageable" in caplog.text

    def test_a_pin_that_fails_stays_pageable(self, monkeypatch, caplog):
        self._fake_pin(monkeypatch, fail=True)
        pool = torch.zeros(24, 4)
        with caplog.at_level(logging.WARNING, logger=AB.logger.name):
            out, pinned = AB.pin_within_budget(pool, 10**6, "pool")
        assert pinned is False and out is pool and "pageable" in caplog.text


class TestTheStepDraw:
    def test_a_rolling_buffer_is_consumed_without_repeats_across_refills(self, tmp_path, make_buffer):
        from src.workers.training_tasks import draw_cached_batch

        buf = make_buffer([_source(tmp_path, 0, 12)], quotas=[4 * SEQ])  # 32-token buffers, 96 tokens
        served = []
        for _ in range(96 // 16):
            batch = draw_cached_batch(buf, buf.tensors, KEYS, 16, buf.size, CPU)
            assert torch.equal(batch[KEYS[0]][:, :3], batch[KEYS[1]][:, :3])
            served += _origins(batch[KEYS[0]])
        assert len(set(served)) == 96, (
            "the step draw did not consume the rolling buffer: tokens repeated, or "
            "it sampled the first buffer instead of cycling"
        )
        assert buf.refills >= 3

    def test_a_fixed_pool_is_still_sampled_directly(self, tmp_path):
        from src.workers.training_tasks import draw_cached_batch

        pool = {key: torch.arange(40, dtype=torch.float32).repeat(4, 1).T.contiguous() for key in KEYS}
        batch = draw_cached_batch(None, pool, KEYS, 25, 40, CPU)
        assert batch[KEYS[0]].shape == (25, 4)
        assert torch.equal(batch[KEYS[0]], batch[KEYS[1]]), "layers drew different rows"


class TestTheFirstPassIsShuffled:
    def test_the_first_buffer_is_not_the_first_rows_of_the_file(self, tmp_path, make_buffer):
        """B1: an unshuffled first pass would start every run on the same documents."""
        src = _source(tmp_path, 0, 50)
        row_sets = []
        for seed in range(5):
            buf = make_buffer([src], quotas=[5 * SEQ], seed=seed)
            row_sets.append(frozenset(o[1] for o in _origins(buf.tensors[KEYS[0]])))
        assert any(rows != frozenset(range(5)) for rows in row_sets), (
            "the first buffer is always rows 0-4: the first pass is not shuffled"
        )
        assert len(set(row_sets)) > 1, "every seed chose the same rows"
