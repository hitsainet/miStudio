"""Unit tests for token normalization (Feature 010)."""

import pytest

from src.utils.token_normalization import normalize_bag, normalize_token


class TestNormalizeToken:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("▁Love", "love"),          # SentencePiece marker + case
            ("Ġthe", "the"),            # GPT-2 BPE marker
            ("##ing", "ing"),           # WordPiece continuation
            ("love", "love"),           # already clean
            ("LOVE", "love"),           # case folding
            ("...", None),              # pure punctuation
            ("▁", None),                # marker only
            ("don't", "don't"),         # interior punctuation kept
            ('"quoted"', "quoted"),     # edge punctuation stripped
            ("▁(Hello)", "hello"),      # marker + edge punctuation + case
            ("ﬁle", "file"),            # NFKC compatibility fold (ﬁ ligature)
            ("", None),
            (None, None),
        ],
    )
    def test_normalization(self, raw, expected):
        assert normalize_token(raw) == expected

    def test_single_leading_marker_stripped(self):
        assert normalize_token("▁word") == "word"
        # A doubled marker is unusual; only the first is treated as a marker,
        # and the remainder survives edge-punctuation stripping untouched.
        assert normalize_token("▁▁word") == "▁word"


class TestNormalizeBag:
    def test_drops_empty_and_preserves_order(self):
        assert normalize_bag(["▁The", "...", "Ġcat", "##s"]) == ["the", "cat", "s"]

    def test_none_and_empty(self):
        assert normalize_bag(None) == []
        assert normalize_bag([]) == []
