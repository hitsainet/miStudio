"""Padding must not be SAE training data, and the mask must line up exactly.

WHY THIS FILE EXISTS. Tokenization right-pads every document to `max_length`;
extraction builds an attention mask, feeds it to the model, and then saves the
activation tensor whole, dropping the mask. Training flattened
`(N, seq_len, d)` and sampled every position, so the residual stream at PAD
positions was SAE training data. Measured on the real estate: OpenWebText is
87.7% real tokens, the Pile 68.2%, hard-negatives 31.9%, Bloomberg 3.3% — a run
over all four was roughly 48% padding.

The dangerous failure here is not "no mask" — that is loud. It is a mask that
is off by a row or a column, because it would silently train on the wrong
positions while every count still looked right. Hence the ordering and
alignment tests.
"""

import json

import numpy as np
import pytest

from src.services import activation_mask


def _write_extraction(tmp_path, n, s, *, sidecar=None, dataset_path=None):
    d = tmp_path / "ext"
    d.mkdir(exist_ok=True)
    if sidecar is not None:
        np.save(d / activation_mask.MASK_FILENAME, sidecar)
    meta = {"num_samples_processed": n}
    if dataset_path is not None:
        meta["dataset_path"] = str(dataset_path)
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


def _arrow(tmp_path, masks):
    from datasets import Dataset

    ds = Dataset.from_dict({
        "input_ids": [[1] * len(m) for m in masks],
        "attention_mask": [list(map(int, m)) for m in masks],
    })
    p = tmp_path / "tokenized"
    ds.save_to_disk(str(p))
    return p


class TestFlatIndexOrdering:
    """The one property that, if wrong, is silent."""

    def test_valid_flat_indices_match_reshape_order(self):
        """`valid_flat_indices` must index the tensor the way training flattens it.

        Training does `acts.reshape(-1, d)`, which is row-major over
        (sample, position). An index built column-major would select real
        activations — just the WRONG ones — and nothing downstream could tell.
        """
        n, s, d = 4, 5, 3
        acts = np.arange(n * s * d, dtype=np.float32).reshape(n, s, d)
        mask = np.zeros((n, s), dtype=bool)
        mask[0, :2] = True      # sample 0, first two positions
        mask[2, 3] = True       # sample 2, position 3
        mask[3, :] = True       # all of sample 3

        flat = activation_mask.valid_flat_indices(mask)
        got = acts.reshape(-1, d)[flat]

        expected = np.stack(
            [acts[0, 0], acts[0, 1], acts[2, 3], acts[3, 0],
             acts[3, 1], acts[3, 2], acts[3, 3], acts[3, 4]]
        )
        np.testing.assert_array_equal(got, expected)

    def test_an_all_true_mask_is_the_identity(self):
        """NEGATIVE CONTROL: with nothing padded, masking must change nothing."""
        n, s, d = 3, 4, 2
        acts = np.arange(n * s * d, dtype=np.float32).reshape(n, s, d)
        flat = activation_mask.valid_flat_indices(np.ones((n, s), dtype=bool))
        np.testing.assert_array_equal(acts.reshape(-1, d)[flat], acts.reshape(-1, d))

    def test_padding_positions_are_excluded(self):
        n, s = 2, 6
        mask = np.zeros((n, s), dtype=bool)
        mask[0, :3] = True
        mask[1, :5] = True
        flat = activation_mask.valid_flat_indices(mask)
        assert flat.size == 8
        # row 0 positions 3,4,5 are pad -> flat 3,4,5 absent
        assert set(flat.tolist()) == {0, 1, 2, 6, 7, 8, 9, 10}


class TestRecovery:

    def test_sidecar_is_preferred(self, tmp_path):
        m = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
        d = _write_extraction(tmp_path, 2, 3, sidecar=m)
        mask, source = activation_mask.load_valid_mask(d, 2, 3)
        np.testing.assert_array_equal(mask, m)
        assert activation_mask.MASK_FILENAME in source

    def test_falls_back_to_the_tokenized_dataset(self, tmp_path):
        arrow = _arrow(tmp_path, [[1, 1, 0], [1, 0, 0], [1, 1, 1]])
        d = _write_extraction(tmp_path, 2, 3, dataset_path=arrow)
        mask, source = activation_mask.load_valid_mask(d, 2, 3)
        # Only the first 2 rows: extraction takes a PREFIX of the dataset.
        np.testing.assert_array_equal(
            mask, np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
        )
        assert "tokenized dataset" in source

    def test_refuses_when_nothing_is_recoverable(self, tmp_path):
        d = _write_extraction(tmp_path, 2, 3)
        mask, reason = activation_mask.load_valid_mask(d, 2, 3)
        assert mask is None
        assert reason, "a refusal must carry a reason the operator can act on"

    def test_refuses_a_dataset_with_fewer_rows_than_the_extraction(self, tmp_path):
        """Alignment is positional; a shorter dataset is not the source."""
        arrow = _arrow(tmp_path, [[1, 1, 0]])
        d = _write_extraction(tmp_path, 5, 3, dataset_path=arrow)
        mask, reason = activation_mask.load_valid_mask(d, 5, 3)
        assert mask is None


class TestConform:

    def test_a_wider_mask_is_trimmed(self, tmp_path):
        """Extraction truncates before saving, so the tail was never encoded."""
        m = np.ones((2, 8), dtype=bool)
        m[:, 5:] = False
        d = _write_extraction(tmp_path, 2, 8, sidecar=m)
        mask, _ = activation_mask.load_valid_mask(d, 2, 4)
        assert mask.shape == (2, 4)
        assert mask.all()

    def test_a_narrower_mask_is_refused_not_padded(self, tmp_path):
        """Guessing the missing columns would be inventing training data."""
        d = _write_extraction(tmp_path, 2, 3, sidecar=np.ones((2, 3), dtype=bool))
        mask, reason = activation_mask.load_valid_mask(d, 2, 9)
        assert mask is None

    def test_a_row_count_mismatch_is_refused(self, tmp_path):
        d = _write_extraction(tmp_path, 2, 3, sidecar=np.ones((7, 3), dtype=bool))
        mask, reason = activation_mask.load_valid_mask(d, 2, 3)
        assert mask is None


class TestExtractionWritesTheMask:
    """The sidecar half: new extractions must record what the old ones lost."""

    @staticmethod
    def _service():
        from src.services.activation_service import ActivationService

        return ActivationService

    def test_mask_matches_the_recorded_lengths(self, tmp_path):
        svc = self._service()
        acts = {"layer_0_residual": np.zeros((3, 6, 2), dtype=np.float32)}
        svc._write_attention_mask(tmp_path, acts, [6, 2, 0])

        mask = np.load(tmp_path / activation_mask.MASK_FILENAME)
        expected = np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=bool)
        np.testing.assert_array_equal(mask, expected)

    def test_it_round_trips_through_the_reader(self, tmp_path):
        """The writer and the reader must agree — they are one mechanism."""
        svc = self._service()
        acts = {"layer_0_residual": np.zeros((4, 5, 2), dtype=np.float32)}
        svc._write_attention_mask(tmp_path, acts, [5, 3, 1, 4])
        (tmp_path / "metadata.json").write_text("{}")

        mask, source = activation_mask.load_valid_mask(tmp_path, 4, 5)
        assert mask is not None
        assert activation_mask.MASK_FILENAME in source
        assert activation_mask.valid_flat_indices(mask).size == 5 + 3 + 1 + 4

    def test_a_length_over_seq_len_is_clipped_not_wrapped(self, tmp_path):
        """Extraction truncates; a length longer than what was saved is normal."""
        svc = self._service()
        acts = {"layer_0_residual": np.zeros((2, 3, 2), dtype=np.float32)}
        svc._write_attention_mask(tmp_path, acts, [99, 1])

        mask = np.load(tmp_path / activation_mask.MASK_FILENAME)
        np.testing.assert_array_equal(
            mask, np.array([[1, 1, 1], [1, 0, 0]], dtype=bool)
        )

    def test_it_refuses_when_the_counts_disagree(self, tmp_path):
        """A confidently wrong mask is worse than a missing one.

        A missing mask makes the training loader warn loudly; a mask that is
        silently misaligned would train on the wrong positions and look healthy.
        """
        svc = self._service()
        acts = {"layer_0_residual": np.zeros((5, 3, 2), dtype=np.float32)}
        svc._write_attention_mask(tmp_path, acts, [3, 3])  # only 2 lengths for 5 samples

        assert not (tmp_path / activation_mask.MASK_FILENAME).exists()

    def test_it_refuses_with_no_lengths_at_all(self, tmp_path):
        svc = self._service()
        acts = {"layer_0_residual": np.zeros((2, 3, 2), dtype=np.float32)}
        svc._write_attention_mask(tmp_path, acts, [])
        assert not (tmp_path / activation_mask.MASK_FILENAME).exists()


class TestResolveFlatIndices:
    """The single most dangerous line in the loader.

    Using `local_indices` directly when a mask is present still yields in-range,
    real-looking activations — just the WRONG ones. There is no downstream
    symptom, which is why this lives in a helper with a test rather than inline.
    """

    def test_a_selection_is_mapped_through_the_mask(self):
        valid_flat = np.array([0, 1, 5, 9, 10])
        # "the 2nd and 4th trainable tokens"
        got = activation_mask.resolve_flat_indices(valid_flat, np.array([1, 3]))
        np.testing.assert_array_equal(got, np.array([1, 9]))

    def test_using_the_selection_unmapped_would_pick_different_tokens(self):
        """Proves the mapping is not a no-op on a realistic mask."""
        valid_flat = np.array([0, 1, 5, 9, 10])
        sel = np.array([1, 3])
        mapped = activation_mask.resolve_flat_indices(valid_flat, sel)
        assert not np.array_equal(mapped, sel)

    def test_no_selection_means_every_valid_token(self):
        valid_flat = np.array([2, 3, 7])
        np.testing.assert_array_equal(
            activation_mask.resolve_flat_indices(valid_flat, None), valid_flat
        )

    def test_no_mask_passes_the_selection_straight_through(self):
        sel = np.array([4, 8])
        np.testing.assert_array_equal(
            activation_mask.resolve_flat_indices(None, sel), sel
        )

    def test_no_mask_and_no_selection_means_everything(self):
        """None is the signal for the caller's bulk-reshape fast path."""
        assert activation_mask.resolve_flat_indices(None, None) is None


class TestGatherTokens:

    def test_gathering_a_mask_returns_exactly_the_real_tokens(self):
        """THE END-TO-END PROPERTY: mask -> indices -> gather == the real tokens.

        This is the composition the training loader performs, checked against an
        independently computed expectation rather than against itself.
        """
        n, s, d = 5, 4, 3
        acts = np.arange(n * s * d, dtype=np.float32).reshape(n, s, d)
        lengths = [4, 1, 0, 3, 2]
        mask = np.arange(s)[None, :] < np.array(lengths)[:, None]

        flat = activation_mask.valid_flat_indices(mask)
        got = activation_mask.gather_tokens(acts, flat, chunk_size=2)

        expected = np.concatenate(
            [acts[i, :L] for i, L in enumerate(lengths) if L], axis=0
        )
        np.testing.assert_array_equal(got, expected)
        assert got.shape[0] == sum(lengths)

    def test_gather_matches_a_plain_reshape_index(self):
        """A gather must equal indexing the flattened tensor — same order."""
        n, s, d = 6, 5, 2
        acts = np.arange(n * s * d, dtype=np.float32).reshape(n, s, d)
        flat = np.array([0, 3, 7, 12, 25, 29])
        np.testing.assert_array_equal(
            activation_mask.gather_tokens(acts, flat, chunk_size=3),
            acts.reshape(-1, d)[flat],
        )

    def test_an_empty_selection_yields_an_empty_block_not_a_crash(self):
        acts = np.zeros((3, 4, 2), dtype=np.float32)
        out = activation_mask.gather_tokens(acts, np.array([], dtype=np.int64))
        assert out.shape == (0, 2)

    def test_unsorted_indices_still_come_back_in_flat_order(self):
        """The sort is load-bearing, and every other fixture here hides it.

        `gather_tokens` promises rows in ascending flat-index order, because the
        caller concatenates parts and assumes reshape order. Without the sort,
        tokens come back grouped by sample but ordered as the caller happened to
        pass them — a silent permutation. Every other test in this class passes
        already-ascending indices, so they agree with the defect by construction.
        """
        n, s, d = 4, 5, 2
        acts = np.arange(n * s * d, dtype=np.float32).reshape(n, s, d)
        scrambled = np.array([17, 2, 11, 0, 6])

        got = activation_mask.gather_tokens(acts, scrambled, chunk_size=2)

        np.testing.assert_array_equal(got, acts.reshape(-1, d)[np.sort(scrambled)])

    def test_chunking_does_not_change_the_result(self):
        """NEGATIVE CONTROL on the sequential-read optimisation."""
        n, s, d = 9, 4, 2
        acts = np.arange(n * s * d, dtype=np.float32).reshape(n, s, d)
        flat = np.arange(0, n * s, 3)
        a = activation_mask.gather_tokens(acts, flat, chunk_size=1)
        b = activation_mask.gather_tokens(acts, flat, chunk_size=100)
        np.testing.assert_array_equal(a, b)


class TestHeldOutSplit:
    """Every SAE number reported today is in-sample; this is what changes that.

    The split must be by DOCUMENT. Adjacent token positions share a prefix and a
    topic, so a token-level split leaks near-duplicates of the evaluation set
    into training and the held-out number flatters the model for reasons
    unrelated to generalisation.
    """

    def test_no_document_appears_on_both_sides(self):
        flat = np.arange(20 * 8)  # 20 documents of 8 positions
        train, held = activation_mask.split_documents(flat, 8, 0.25, seed=1)

        train_docs = set((train // 8).tolist())
        held_docs = set((held // 8).tolist())
        assert train_docs & held_docs == set(), (
            "a document's tokens are on both sides — adjacent positions are "
            "near-duplicates, so this is leakage, not a split"
        )

    def test_every_token_lands_on_exactly_one_side(self):
        flat = np.arange(20 * 8)
        train, held = activation_mask.split_documents(flat, 8, 0.25, seed=1)
        assert train.size + held.size == flat.size
        assert set(train.tolist()) | set(held.tolist()) == set(flat.tolist())

    def test_the_requested_fraction_is_honoured_in_documents(self):
        flat = np.arange(20 * 8)
        _, held = activation_mask.split_documents(flat, 8, 0.25, seed=1)
        assert len(set((held // 8).tolist())) == 5

    def test_it_is_deterministic_for_a_seed_and_varies_across_seeds(self):
        flat = np.arange(40 * 4)
        a, _ = activation_mask.split_documents(flat, 4, 0.5, seed=7)
        b, _ = activation_mask.split_documents(flat, 4, 0.5, seed=7)
        c, _ = activation_mask.split_documents(flat, 4, 0.5, seed=8)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_zero_means_no_split_at_all(self):
        """The historical behaviour must remain expressible."""
        flat = np.arange(16)
        train, held = activation_mask.split_documents(flat, 4, 0.0, seed=1)
        np.testing.assert_array_equal(train, flat)
        assert held.size == 0

    def test_it_refuses_a_fraction_that_reserves_nothing(self):
        """Silently returning an empty eval set would report in-sample numbers
        as if they were held out."""
        with pytest.raises(ValueError, match="reserves nothing"):
            activation_mask.split_documents(np.arange(8), 4, 0.01, seed=1)

    def test_an_out_of_range_fraction_is_refused(self):
        for bad in (-0.1, 1.0, 1.5):
            with pytest.raises(ValueError):
                activation_mask.split_documents(np.arange(8), 4, bad, seed=1)

    def test_a_sparse_index_set_still_splits_by_document(self):
        """Real input is masked, so the indices are not contiguous."""
        flat = np.array([0, 1, 4, 5, 6, 9, 12, 13])  # docs 0,1,2,3 at seq_len 4
        train, held = activation_mask.split_documents(flat, 4, 0.5, seed=3)
        assert set((train // 4).tolist()) & set((held // 4).tolist()) == set()
        assert train.size + held.size == flat.size


class TestTheSidecarDescribesRealTokensNotRowWidth:
    """Round 1, H1 — the worst defect of this arc, and it was mine.

    `sample_token_lengths` recorded `len(input_ids)` as read from Arrow. But
    tokenization defaults to `padding="max_length"`, so EVERY stored row is
    already `max_length` long including pads — the recorded length was always
    the row width, and the sidecar was ALL-TRUE. Because `load_valid_mask`
    PREFERS the sidecar, a new extraction got a confidently-wrong 100%-real mask
    while the correct one sat one fallback away in the `attention_mask` column.

    `activation_mask.py`'s own docstring says a confidently-wrong mask is worse
    than none. The code defending that standard violated it.

    The earlier "16 of 16 production extractions recovered" check could not
    have caught this: all 16 predate the sidecar and took the working fallback.

    Compounding it, extraction can only COMPLETE under `padding="max_length"` —
    other modes give ragged batch widths that break the memmap copy — so the
    all-ones case was not an edge case, it was every run.
    """

    @staticmethod
    def _padded_rows(real_lengths, width):
        """Rows exactly as `tokenization_service` writes them: padded to width."""
        return [
            {
                "input_ids": [7] * n + [0] * (width - n),
                "attention_mask": [1] * n + [0] * (width - n),
            }
            for n in real_lengths
        ]

    def test_row_length_is_not_the_real_length(self):
        """Establish the trap before asserting the fix avoids it."""
        rows = self._padded_rows([3, 8, 1], width=16)
        assert [len(r["input_ids"]) for r in rows] == [16, 16, 16]
        assert [sum(r["attention_mask"]) for r in rows] == [3, 8, 1]

    def test_the_writer_records_the_mask_sum_not_the_row_width(self, tmp_path):
        from src.services.activation_service import ActivationService

        rows = self._padded_rows([3, 8, 1], width=16)
        lengths = [int(sum(r["attention_mask"])) for r in rows]

        acts = {"layer_0_residual": np.zeros((3, 16, 2), dtype=np.float32)}
        ActivationService._write_attention_mask(tmp_path, acts, lengths)

        mask = np.load(tmp_path / activation_mask.MASK_FILENAME)
        assert mask.sum() == 12, "the sidecar does not describe the real tokens"
        assert mask.mean() != 1.0, (
            "the sidecar is all-True — the defect this test exists for"
        )


class TestTheSharedPaddedBatchBuilder:
    """Behavioural coverage for the logic BOTH extraction paths now share.

    The three tests that used to sit here were `inspect.getsource` scrapes, and
    round 3 proved all three fail open — including one written as
    `assert "..." not in src or True`, a literal tautology. It also proved the
    sidecar could revert to all-True with the full 4777-test suite green.

    The commit that introduced those scrapes said, about this exact mistake,
    "twice is the argument for not writing that kind of guard at all" — and then
    wrote three more. So: behaviour only.
    """

    def test_pre_padded_rows_do_not_yield_an_all_true_mask(self):
        """THE defect, in the shape it actually occurs.

        Rows arrive from Arrow already padded to max_length, so any mask derived
        from `len(input_ids)` is all-ones. This is what made both the sidecar and
        the on-the-fly fix no-ops.
        """
        width = 16
        ids = [[7] * 3 + [0] * 13, [7] * 8 + [0] * 8]
        masks = [[1] * 3 + [0] * 13, [1] * 8 + [0] * 8]

        _, attn, lengths, non_prefix, missing = activation_mask.build_padded_batch(
            ids, masks, width, pad_token_id=0
        )

        assert lengths == [3, 8], "the row width was recorded instead of the real count"
        assert sum(sum(a) for a in attn) == 11
        assert not non_prefix and missing == 0

    def test_without_a_source_mask_everything_is_real_and_it_is_reported(self):
        ids = [[7, 7, 7]]
        _, attn, lengths, _, missing = activation_mask.build_padded_batch(
            ids, [], 5, pad_token_id=0
        )
        assert lengths == [3]
        assert attn[0] == [1, 1, 1, 0, 0]
        assert missing == 1, "a missing mask must be counted so the caller warns"

    def test_padding_is_appended_to_both_ids_and_mask(self):
        padded, attn, lengths, _, _ = activation_mask.build_padded_batch(
            [[7, 7]], [[1, 1]], 5, pad_token_id=99
        )
        assert padded[0] == [7, 7, 99, 99, 99]
        assert attn[0] == [1, 1, 0, 0, 0]
        assert lengths == [2]

    def test_a_non_prefix_row_is_flagged(self):
        """A length cannot describe a left-padded row: right count, wrong places."""
        _, _, lengths, non_prefix, _ = activation_mask.build_padded_batch(
            [[0, 0, 7, 7]], [[0, 0, 1, 1]], 4, pad_token_id=0
        )
        assert lengths == [2]
        assert non_prefix == [0]

    def test_a_short_mask_does_not_silently_drop_real_text(self):
        """Zero-extending would mark real tokens as padding — unrecoverable.
        Over-including is at worst a little padding, which is recoverable."""
        _, attn, lengths, _, _ = activation_mask.build_padded_batch(
            [[7, 7, 7, 7]], [[1, 1]], 4, pad_token_id=0
        )
        assert lengths == [4]
        assert attn[0] == [1, 1, 1, 1]

    def test_a_long_mask_is_trimmed_to_the_row(self):
        _, attn, lengths, _, _ = activation_mask.build_padded_batch(
            [[7, 7]], [[1, 1, 1, 1, 1]], 2, pad_token_id=0
        )
        assert lengths == [2] and attn[0] == [1, 1]

    def test_both_extraction_paths_call_it(self):
        """Reachability — two inline copies is how the second reproduced the
        first's defect."""
        import ast
        import inspect

        from src.services import activation_service
        from src.workers import training_tasks

        for module in (activation_service, training_tasks):
            tree = ast.parse(inspect.getsource(module))
            called = {
                n.func.attr for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            }
            assert "build_padded_batch" in called, (
                f"{module.__name__} builds its own masks again"
            )


class TestTheGatherScalesLinearly:
    """Round 1, M-a: the gather was quadratic and replaced an O(N) reshape.

    `token_indices[sample_indices == s_idx]` inside the per-sample loop is a
    full pass over every index for every sample. Measured before the fix:
    0.003 s at 200 samples, 0.117 s at 2,000, **10.07 s at 20,000** — clean
    quadratic. A real extraction is ~2x10^5 samples and ~3x10^8 indices, i.e.
    hours per (layer, extraction), against the seconds the pre-mask bulk reshape
    took. Masking would have been unusable at production scale.

    A wall-clock assertion is a bad test, so this asserts the SHAPE of the
    growth with a generous constant.
    """

    @staticmethod
    def _time_gather(n_samples, seq_len=64, d=4):
        import time

        acts = np.zeros((n_samples, seq_len, d), dtype=np.float32)
        mask = np.zeros((n_samples, seq_len), dtype=bool)
        mask[:, : seq_len // 2] = True
        flat = activation_mask.valid_flat_indices(mask)

        start = time.perf_counter()
        out = activation_mask.gather_tokens(acts, flat)
        elapsed = time.perf_counter() - start
        assert out.shape[0] == flat.size
        return elapsed

    def test_ten_times_the_data_is_not_a_hundred_times_the_work(self):
        small = self._time_gather(500)
        large = self._time_gather(5000)

        # Quadratic would be ~100x. Linear is ~10x. The threshold sits far from
        # both so this does not become a flaky wall-clock test.
        ratio = large / max(small, 1e-6)
        assert ratio < 40, (
            f"gather_tokens scaled {ratio:.0f}x for 10x the data — it looks "
            f"quadratic again, which makes masking unusable on a real extraction"
        )

    def test_the_growth_holds_across_three_points_not_two(self):
        """Two points can be luck; three show the trend.

        Deliberately NOT a source assertion. Twice in this arc a scrape for the
        removed expression matched the COMMENT describing its removal and failed
        for the wrong reason. Behaviour is the only honest guard here.
        """
        times = [self._time_gather(n) for n in (250, 1000, 4000)]
        # Each 4x step should cost roughly 4x, not 16x.
        for a, b in zip(times, times[1:]):
            ratio = b / max(a, 1e-6)
            assert ratio < 12, (
                f"a 4x data step cost {ratio:.0f}x — that is superlinear growth"
            )


class TestMaskRecoveryIsNotAuthenticatedByShapeAlone:
    """Round 1, M-f: a directory that row-matches is not necessarily the right one.

    `metadata.json` records `dataset_path` and nothing else — no tokenization
    id, no max_length, no tokenizer fingerprint — and the path itself used to
    omit `max_length`, so a 512 tokenization and a 2048 one occupied the SAME
    directory. A pre-existing extraction at 512 whose directory was later
    overwritten by a 2048 tokenization still row-matches; `_conform` then trims
    the wider mask to its first 512 columns and returns it as authoritative.

    That is wrong by CONTENT rather than by row, so nothing downstream notices.
    The recorded width is the cheap check that catches it.
    """

    def test_a_narrower_tokenization_is_refused(self, tmp_path):
        """Extraction can truncate but cannot INVENT columns.

        A source narrower than what was saved therefore cannot be the directory
        that produced these activations.
        """
        arrow = _arrow(tmp_path, [[1, 1] for _ in range(4)])  # 2 wide
        d = tmp_path / "ext"
        d.mkdir(exist_ok=True)
        (d / "metadata.json").write_text(
            json.dumps({"dataset_path": str(arrow), "seq_len": 8})
        )

        mask, reason = activation_mask.load_valid_mask(d, 4, 8)
        assert mask is None

    def test_a_wider_tokenization_is_ACCEPTED_because_truncation_is_legitimate(
        self, tmp_path
    ):
        """Round 3, H2: the first version of this guard refused exactly this.

        A wider source is the signature of a truncated extraction — the normal
        case once `max_seq_length` is set. Refusing it sends training back to
        `PADDING NOT MASKED` and trains on padding, which is the defect the arc
        exists to remove. My original test asserted the refusal, pinning it.
        """
        arrow = _arrow(tmp_path, [[1, 1, 1, 0] for _ in range(4)])  # 4 wide
        d = tmp_path / "ext"
        d.mkdir(exist_ok=True)
        (d / "metadata.json").write_text(
            json.dumps({"dataset_path": str(arrow), "seq_len": 2})
        )

        mask, source = activation_mask.load_valid_mask(d, 4, 2)
        assert mask is not None, (
            "a legitimately truncated extraction was refused; training will now "
            "run on padding"
        )
        assert mask.shape == (4, 2) and mask.all()

    def test_a_matching_width_is_accepted(self, tmp_path):
        """NEGATIVE CONTROL — the guard must not refuse the normal case."""
        arrow = _arrow(tmp_path, [[1, 1, 0, 0] for _ in range(4)])
        d = tmp_path / "ext"
        d.mkdir(exist_ok=True)
        (d / "metadata.json").write_text(
            json.dumps({"dataset_path": str(arrow), "seq_len": 4})
        )

        mask, source = activation_mask.load_valid_mask(d, 4, 4)
        assert mask is not None and mask.shape == (4, 4)

    def test_metadata_without_a_width_still_recovers(self, tmp_path):
        """Every pre-existing extraction predates the field; they must still work."""
        arrow = _arrow(tmp_path, [[1, 1, 0, 0] for _ in range(4)])
        d = tmp_path / "ext"
        d.mkdir(exist_ok=True)
        (d / "metadata.json").write_text(json.dumps({"dataset_path": str(arrow)}))

        mask, source = activation_mask.load_valid_mask(d, 4, 4)
        assert mask is not None, (
            "the width guard broke recovery for the 16 existing extractions"
        )

    def test_extraction_records_the_width(self):
        import inspect

        from src.services import activation_service

        assert '"seq_len"' in inspect.getsource(activation_service), (
            "extraction does not record the width it saved, so the guard above "
            "has nothing to check against"
        )


class TestGatherEquivalenceOnRandomInputs:
    """The gather has been rewritten twice for speed; pin it to a reference.

    First quadratic -> `np.unique(return_index=True)` (round 1, M-a), then ->
    `flatnonzero(diff)` to drop the full argsort and the extra int64 array the
    size of the input (~2.4 GB at the 3x10^8 indices a real extraction reaches).

    Each rewrite is a chance to change the ANSWER while the timing tests stay
    happy, so this checks against a trivially-correct reference across
    randomised shapes, index sets and chunk sizes.
    """

    def test_it_matches_a_plain_reshape_index_on_random_inputs(self):
        rng = np.random.RandomState(1234)
        for _ in range(200):
            n = int(rng.randint(1, 12))
            s = int(rng.randint(1, 9))
            d = 3
            acts = np.arange(n * s * d, dtype=np.float32).reshape(n, s, d)

            k = int(rng.randint(0, n * s + 1))
            flat = (
                rng.choice(n * s, size=k, replace=False)
                if k else np.array([], dtype=np.int64)
            )

            got = activation_mask.gather_tokens(
                acts, flat, chunk_size=int(rng.randint(1, 5))
            )
            want = (
                acts.reshape(-1, d)[np.sort(flat)]
                if k else np.empty((0, d), dtype=np.float32)
            )
            np.testing.assert_array_equal(got, want)

    def test_a_single_sample_and_a_single_index(self):
        acts = np.arange(1 * 4 * 2, dtype=np.float32).reshape(1, 4, 2)
        np.testing.assert_array_equal(
            activation_mask.gather_tokens(acts, np.array([2])),
            acts.reshape(-1, 2)[[2]],
        )

    def test_every_position_of_every_sample(self):
        """The dense case, where the boundary arithmetic is most exercised."""
        n, s, d = 5, 4, 2
        acts = np.arange(n * s * d, dtype=np.float32).reshape(n, s, d)
        flat = np.arange(n * s)
        np.testing.assert_array_equal(
            activation_mask.gather_tokens(acts, flat, chunk_size=2),
            acts.reshape(-1, d),
        )
