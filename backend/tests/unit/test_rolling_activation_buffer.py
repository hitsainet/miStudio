"""SAE training must read the whole extraction, not one GPU-sized sample of it.

2026-09-12. Training loaded a single random subsample sized to free GPU memory —
~2.4M tokens per layer for LFM2.5-1.2B on a 24 GB card — and drew all 102M
default samples from it with replacement. A 20M-token extraction was ~90%
unread and every token that was read was seen ~43 times.

These tests build real `.npy` extractions whose every activation ENCODES its own
origin — (source, row, position, layer) — so what the buffer serves can be
checked exactly rather than statistically.
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from src.services import activation_buffer as AB

SEQ = 8
KEYS = [(3, "residual"), (4, "residual")]
CPU = torch.device("cpu")


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
    return AB.BufferSource(
        label=f"src{source}", files=_extraction(tmp_path, source, rows),
        num_rows=rows, seq_len=SEQ, valid_flat=valid_flat,
    )


def _buffer(sources, quotas, seed=0):
    return AB.RollingActivationBuffer(
        sources, KEYS, quotas, seed=seed, storage_device=CPU, train_device=CPU
    )


def _origins(batch):
    """(source, row, pos) triples for one layer's batch."""
    return [tuple(int(v) for v in row[:3]) for row in batch.tolist()]


class TestTheWholeExtractionIsUsed:
    def test_every_token_is_served_exactly_once_before_any_repeats(self, tmp_path):
        """Two sources of 12 and 6 rows (144 tokens), buffer of 6 rows (48 tokens)."""
        sources = [_source(tmp_path, 0, 12), _source(tmp_path, 1, 6)]
        buf = _buffer(sources, quotas=[4 * SEQ, 2 * SEQ])

        served = []
        for _ in range(144 // 16):  # three buffers, 3 batches of 16 each
            served += _origins(buf.next_batch(16)[KEYS[0]])

        assert len(served) == 144
        assert len(set(served)) == 144, "a token was served twice before the corpus was used"
        expected = {(0, r, p) for r in range(12) for p in range(SEQ)} | {
            (1, r, p) for r in range(6) for p in range(SEQ)
        }
        assert set(served) == expected, "some of the extraction was never read"

    def test_the_buffer_really_turns_over(self, tmp_path):
        """NEGATIVE CONTROL for the defect: the old loader's pool never changed."""
        sources = [_source(tmp_path, 0, 20)]
        buf = _buffer(sources, quotas=[5 * SEQ])
        first_rows = {o[1] for o in _origins(buf.tensors[KEYS[0]])}
        for _ in range(3):
            buf.next_batch(40)
        assert buf.refills >= 3
        later_rows = {o[1] for o in _origins(buf.tensors[KEYS[0]])}
        assert first_rows.isdisjoint(later_rows), "the refill served rows already used"

    def test_a_source_is_reshuffled_only_after_it_is_exhausted(self, tmp_path):
        sources = [_source(tmp_path, 0, 6)]
        buf = _buffer(sources, quotas=[4 * SEQ])
        assert buf.epochs_completed() == [0]
        buf.next_batch(4 * SEQ)
        buf.next_batch(4 * SEQ)  # needs rows 5-6 plus two from a new pass
        assert buf.epochs_completed() == [1]
        rows = [o[1] for o in _origins(buf.tensors[KEYS[0]])]
        assert len(set(rows)) == len(rows) // SEQ, "a row appeared twice in one buffer"


class TestOnlyTrainableTokensAreServed:
    def test_padding_and_held_out_rows_never_appear(self, tmp_path):
        rows = 10
        mask = np.ones((rows, SEQ), dtype=bool)
        mask[:, -3:] = False            # padding at the end of every row
        mask[7:, :] = False             # rows 7-9 held out
        valid = np.flatnonzero(mask.reshape(-1))
        src = _source(tmp_path, 0, rows, valid_flat=valid)
        buf = _buffer([src], quotas=[3 * 5])

        served = []
        for _ in range(10):
            served += _origins(buf.next_batch(5)[KEYS[0]])
        assert served, "nothing was served"
        assert all(pos < SEQ - 3 for _, _, pos in served), "a padding position was served"
        assert all(row < 7 for _, row, _ in served), "a held-out row was served"


class TestEveryLayerSeesTheSameTokens:
    def test_batches_are_aligned_across_layers(self, tmp_path):
        buf = _buffer([_source(tmp_path, 0, 9), _source(tmp_path, 1, 9)], quotas=[3 * SEQ, 3 * SEQ])
        for _ in range(12):
            batch = buf.next_batch(10)
            a, b = batch[KEYS[0]], batch[KEYS[1]]
            assert torch.equal(a[:, :3], b[:, :3]), "layers were served different tokens"
            assert set(a[:, 3].tolist()) == {3.0} and set(b[:, 3].tolist()) == {4.0}


class TestTheMixtureFollowsTheQuotas:
    def test_tokens_loaded_per_source_track_the_quotas(self, tmp_path):
        buf = _buffer([_source(tmp_path, 0, 40), _source(tmp_path, 1, 40)], quotas=[9 * SEQ, 1 * SEQ])
        for _ in range(30):
            buf.next_batch(40)
        a, b = buf.tokens_loaded
        assert a / (a + b) == pytest.approx(0.9, abs=0.02)


class TestDeterminism:
    def test_the_same_seed_serves_the_same_batches(self, tmp_path):
        srcs = [_source(tmp_path, 0, 10), _source(tmp_path, 1, 10)]
        one = _buffer(srcs, [3 * SEQ, 2 * SEQ], seed=7)
        two = _buffer(srcs, [3 * SEQ, 2 * SEQ], seed=7)
        other = _buffer(srcs, [3 * SEQ, 2 * SEQ], seed=8)
        a = [one.next_batch(12)[KEYS[0]] for _ in range(6)]
        b = [two.next_batch(12)[KEYS[0]] for _ in range(6)]
        c = [other.next_batch(12)[KEYS[0]] for _ in range(6)]
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        assert not all(torch.equal(x, z) for x, z in zip(a, c))


class TestMemory:
    def test_the_previous_buffer_is_released_before_the_next_is_read(self, tmp_path, monkeypatch):
        """The buffer is sized to free memory; holding two would need twice it."""
        buf = _buffer([_source(tmp_path, 0, 12)], quotas=[4 * SEQ])
        # Strong references, compared with `is`. Comparing id() values was flaky:
        # once a tensor is freed, a new one can be allocated with the same id.
        old = list(buf.tensors.values())
        seen = []
        real = AB.activation_mask.gather_tokens

        def spy(mmap, flat, *a, **k):
            # Layers load one at a time, so a NEW tensor for an earlier layer may
            # already exist here; what must not exist is any tensor from BEFORE.
            seen.append(not any(
                t is o for t in buf.tensors.values() if t is not None for o in old
            ))
            return real(mmap, flat, *a, **k)

        monkeypatch.setattr(AB.activation_mask, "gather_tokens", spy)
        buf.refill()
        assert len(seen) == len(KEYS)
        assert all(seen), "the old buffer was still held while the new one loaded"


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

    def _calls(self):
        from src.workers import training_tasks

        tree = ast.parse(inspect.getsource(training_tasks))
        return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    def _named(self, name):
        return [
            c for c in self._calls()
            if (getattr(c.func, "attr", None) or getattr(c.func, "id", None)) == name
        ]

    def test_the_storage_plan_decides_the_mode(self):
        assert self._named("plan_activation_storage"), "the storage decision is not made by the planner"

    def test_the_buffer_is_built_with_the_mixture_quotas(self):
        builds = self._named("RollingActivationBuffer")
        assert builds, "training never builds the rolling buffer"
        args = [a for c in builds for a in c.args] + [kw.value for c in builds for kw in c.keywords]
        assert any(isinstance(a, ast.Name) and a.id == "allocation" for a in args), (
            "the buffer is not given the allocate_tokens quotas, so dataset_weights do nothing"
        )

    def test_the_step_loop_draws_through_the_helper(self):
        """The AST half. The driven half is TestTheStepDraw below: this guard
        alone survived a mutation that made the branch dead (`if False:`)."""
        assert self._named("draw_cached_batch"), "the training step does not draw through draw_cached_batch"
        assert self._named("next_batch"), "nothing draws from the rolling buffer"


class TestTheStepDraw:
    def test_a_rolling_buffer_is_consumed_without_repeats_across_refills(self, tmp_path):
        from src.workers.training_tasks import draw_cached_batch

        buf = _buffer([_source(tmp_path, 0, 12)], quotas=[4 * SEQ])  # 32-token buffers, 96 tokens
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
    def test_the_first_buffer_is_not_the_first_rows_of_the_file(self, tmp_path):
        """B1: an unshuffled first pass would start every run on the same documents."""
        src = _source(tmp_path, 0, 50)
        row_sets = []
        for seed in range(5):
            buf = _buffer([src], quotas=[5 * SEQ], seed=seed)
            row_sets.append(frozenset(o[1] for o in _origins(buf.tensors[KEYS[0]])))
        assert any(rows != frozenset(range(5)) for rows in row_sets), (
            "the first buffer is always rows 0-4: the first pass is not shuffled"
        )
        assert len(set(row_sets)) > 1, "every seed chose the same rows"
