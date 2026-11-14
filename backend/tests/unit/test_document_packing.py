"""Packing turns a 3%-occupancy corpus into a dense one.

WHY THIS FILE EXISTS. Tokenization pads every document to `max_length`.
Measured on this estate, real-token occupancy is: OpenWebText 87.7%, the Pile
68.2%, hard-negatives 31.9%, Bloomberg **3.3%** (headlines, mean 17 tokens in a
512 window). Masking stops padding being trained on; packing stops it being
produced, which also stops the model doing forward passes over it.

The dangerous edge is token LOSS — a packer that drops the tail of the buffer,
or double-counts a boundary, changes the corpus without any error.
"""

import pytest

from src.services.tokenization_service import TokenizationService

pack = TokenizationService.pack_token_blocks
EOS = 99


class TestTokensAreConserved:

    def test_every_token_survives_packing(self):
        """THE load-bearing property: packing must not lose text."""
        docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10], [11]]
        blocks, masks = pack(docs, max_length=4, eos_token_id=EOS)

        flat = [t for b, m in zip(blocks, masks) for t, keep in zip(b, m) if keep]
        expected = []
        for d in docs:
            expected.extend(d)
            expected.append(EOS)
        assert flat == expected

    def test_no_separator_when_there_is_no_eos(self):
        docs = [[1, 2], [3, 4]]
        blocks, masks = pack(docs, max_length=2, eos_token_id=None)
        flat = [t for b, m in zip(blocks, masks) for t, keep in zip(b, m) if keep]
        assert flat == [1, 2, 3, 4]

    def test_a_document_longer_than_the_window_is_split_not_truncated(self):
        """Truncation is what packing exists to avoid."""
        blocks, masks = pack([[1, 2, 3, 4, 5, 6, 7]], max_length=3, eos_token_id=None)
        flat = [t for b, m in zip(blocks, masks) for t, keep in zip(b, m) if keep]
        assert flat == [1, 2, 3, 4, 5, 6, 7]


class TestOccupancy:

    def test_blocks_are_full_except_the_last(self):
        docs = [[1, 2, 3]] * 10
        blocks, masks = pack(docs, max_length=4, eos_token_id=EOS)
        for m in masks[:-1]:
            assert all(m), "only the final block may be partially padded"
        assert len(blocks[0]) == 4

    def test_short_documents_reach_near_full_occupancy(self):
        """The Bloomberg case: 17-token documents in a 512 window.

        The corpus must be large enough that the single partial tail block is
        negligible — at 200 documents the tail alone caps occupancy at ~88%,
        which says nothing about the packer. Bloomberg is 446,762 rows.
        """
        docs = [list(range(17))] * 5000
        blocks, masks = pack(docs, max_length=512, eos_token_id=EOS)
        occupancy = sum(sum(m) for m in masks) / (len(masks) * 512)
        assert occupancy > 0.99, f"packing left {1 - occupancy:.1%} padding"

        # And state the win against the status quo it replaces.
        unpacked_occupancy = 17 / 512
        assert occupancy > 25 * unpacked_occupancy

    def test_the_remainder_is_padded_and_masked_not_silently_dropped(self):
        blocks, masks = pack([[1, 2]], max_length=8, eos_token_id=EOS)
        assert len(blocks) == 1
        assert len(blocks[0]) == 8, "blocks must be uniform width"
        assert masks[0] == [1, 1, 1, 0, 0, 0, 0, 0], "1,2,EOS are real; the rest are pad"

    def test_drop_remainder_never_discards_a_full_block(self):
        """The `>=` boundary is load-bearing, and only this case shows it.

        With `>` instead of `>=`, a buffer that reaches exactly `max_length` is
        not emitted inside the loop — it stays in the buffer, and
        `drop_remainder` then throws away a block of entirely real tokens.
        Measured over random inputs, `>=` and `>` differ in 515 of 3000 cases,
        all of this shape. Every other test here uses documents that never land
        on an exact multiple, so they agree with the defect by construction.
        """
        # 2 docs x 2 tokens + 1 EOS each = exactly 6 tokens = 2 blocks of 3.
        docs = [[1, 2], [3, 4]]
        blocks, masks = pack(docs, max_length=3, eos_token_id=EOS, drop_remainder=True)

        flat = [t for b, m in zip(blocks, masks) for t, keep in zip(b, m) if keep]
        assert flat == [1, 2, EOS, 3, 4, EOS], (
            "a block that exactly fills the window was dropped as if it were a "
            "partial tail"
        )
        assert len(blocks) == 2

    def test_drop_remainder_discards_only_the_tail(self):
        docs = [[1, 2, 3, 4], [5]]
        blocks, masks = pack(docs, max_length=5, eos_token_id=EOS, drop_remainder=True)
        assert all(all(m) for m in masks), "dropping the remainder leaves no padding"
        assert len(blocks) == 1


class TestDegenerateInputs:

    def test_no_documents_yields_no_blocks(self):
        assert pack([], max_length=4, eos_token_id=EOS) == ([], [])

    def test_a_nonpositive_window_is_refused(self):
        """Silently returning nothing here would empty the corpus."""
        with pytest.raises(ValueError):
            pack([[1, 2]], max_length=0, eos_token_id=EOS)

    def test_an_exact_fit_produces_no_partial_block(self):
        blocks, masks = pack([[1, 2, 3]], max_length=4, eos_token_id=EOS)
        assert len(blocks) == 1
        assert all(masks[0])


class TestTheStreamingPackerAgrees:
    """Two packers must not drift apart.

    `pack_token_blocks` returns lists, which suits a test and not a corpus:
    OpenWebText is 1M documents and ~450M tokens, so materialising the input
    column and the output blocks as Python lists is tens of gigabytes.
    `iter_packed_blocks` streams for production. Having two implementations of
    one boundary rule is exactly how they come to disagree, so this pins them
    together.
    """

    @pytest.mark.parametrize(
        "docs,length",
        [
            ([[1, 2, 3], [4, 5], [6, 7, 8, 9, 10], [11]], 4),
            ([[1, 2]], 8),
            ([list(range(17))] * 30, 16),
            ([[1, 2, 3, 4]], 3),
            ([], 5),
        ],
    )
    def test_the_two_packers_produce_identical_output(self, docs, length):
        blocks, masks = pack(docs, max_length=length, eos_token_id=EOS)
        streamed = list(TokenizationService.iter_packed_blocks(iter(docs), max_length=length, eos_token_id=EOS))

        assert [b["input_ids"] for b in streamed] == blocks
        assert [b["attention_mask"] for b in streamed] == masks

    def test_the_streaming_packer_consumes_an_iterator_once(self):
        """It must not require a re-iterable sequence; the source is a dataset."""
        docs = iter([[1, 2, 3], [4, 5, 6]])
        out = list(TokenizationService.iter_packed_blocks(docs, max_length=4, eos_token_id=EOS))
        assert out, "the generator did not consume the iterator"

    def test_it_refuses_a_nonpositive_window_like_its_twin(self):
        with pytest.raises(ValueError):
            list(TokenizationService.iter_packed_blocks(iter([[1, 2]]), max_length=0, eos_token_id=EOS))


class TestPackedRowsAreNotDocuments:
    """Round 4, M-3: packing changes what a row MEANS, and the consumers that
    assume row == document were never checked.

    `pack_token_blocks`' own docstring says exactly that must happen before
    packing is enabled. The clearest case is `max_samples` at extraction time:
    10,000 means 10,000 BLOCKS, each several concatenated documents, so against
    a packed Bloomberg corpus it is ~30x the text and compute of 10,000 unpacked
    rows — under an unchanged label.
    """

    @staticmethod
    def _ds(masks):
        from datasets import Dataset

        return Dataset.from_dict({
            "input_ids": [[1] * len(m) for m in masks],
            "attention_mask": masks,
        })

    def test_packed_rows_are_detected(self):
        from src.services.activation_service import _dataset_looks_packed

        assert _dataset_looks_packed(self._ds([[1, 1, 1, 1]] * 8))

    def test_padded_rows_are_not_flagged(self):
        """NEGATIVE CONTROL — warning on every normal tokenization is noise."""
        from src.services.activation_service import _dataset_looks_packed

        assert not _dataset_looks_packed(self._ds([[1, 1, 0, 0]] * 8))

    def test_a_dataset_without_the_column_is_not_flagged(self):
        from datasets import Dataset

        from src.services.activation_service import _dataset_looks_packed

        assert not _dataset_looks_packed(Dataset.from_dict({"input_ids": [[1, 2]]}))

    def test_an_empty_dataset_does_not_crash_the_heuristic(self):
        from src.services.activation_service import _dataset_looks_packed

        assert not _dataset_looks_packed(self._ds([]))

    def test_extraction_warns_about_the_units(self):
        import inspect

        from src.services import activation_service

        src = inspect.getsource(activation_service)
        assert "BLOCKS, not documents" in src, (
            "max_samples silently means something different for a packed "
            "tokenization and nothing says so"
        )
