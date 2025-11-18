"""
Unit tests for token filtering utilities.
"""

import pytest
from src.utils.token_filters import (
    is_junk_token,
    clean_token_display,
    analyze_feature_tokens
)


class TestIsJunkToken:
    """Tests for is_junk_token function."""

    def test_special_tokens(self):
        """Special tokens should be filtered."""
        assert is_junk_token('<s>') is True
        assert is_junk_token('</s>') is True
        assert is_junk_token('<pad>') is True
        assert is_junk_token('<unk>') is True
        assert is_junk_token('\ufeff') is True

    def test_single_characters(self):
        """Single characters should be filtered."""
        assert is_junk_token('a') is True
        assert is_junk_token('Z') is True
        assert is_junk_token('1') is True
        assert is_junk_token(',') is True
        assert is_junk_token(' ') is True

    def test_pure_punctuation(self):
        """Pure punctuation should be filtered."""
        assert is_junk_token(',') is True
        assert is_junk_token('.,!') is True
        assert is_junk_token('---') is True
        assert is_junk_token('▁') is True

    def test_short_fragments_without_vowels(self):
        """Short fragments without vowels should be filtered."""
        assert is_junk_token('th') is True
        assert is_junk_token('str') is True
        assert is_junk_token('by') is True
        assert is_junk_token('ps') is True

    def test_pure_numbers(self):
        """Pure numbers should be filtered."""
        assert is_junk_token('123') is True
        assert is_junk_token('2024') is True

    def test_valid_tokens(self):
        """Valid tokens should not be filtered."""
        assert is_junk_token('The') is False
        assert is_junk_token('about') is False
        assert is_junk_token('news') is False
        assert is_junk_token('Pennsylvania') is False
        # Word fragment filtering can be disabled
        assert is_junk_token('ious', filter_fragments=False) is False
        assert is_junk_token('ious', filter_fragments=True) is True

    def test_short_tokens_with_vowels(self):
        """Short tokens with vowels should not be filtered."""
        assert is_junk_token('he') is False
        assert is_junk_token('in') is False
        assert is_junk_token('on') is False
        assert is_junk_token('are') is False


class TestCleanTokenDisplay:
    """Tests for clean_token_display function."""

    def test_removes_space_marker(self):
        """Should remove leading space marker."""
        assert clean_token_display('▁Hello') == 'Hello'
        assert clean_token_display('▁The') == 'The'
        assert clean_token_display('▁news') == 'news'

    def test_preserves_token_if_empty(self):
        """Should preserve original if empty after cleaning."""
        assert clean_token_display('▁') == '▁'

    def test_no_space_marker(self):
        """Should preserve tokens without space marker."""
        assert clean_token_display('world') == 'world'
        assert clean_token_display('123') == '123'

    def test_multiple_space_markers(self):
        """Should remove all space markers."""
        assert clean_token_display('▁▁test') == 'test'


class TestAnalyzeFeatureTokens:
    """Tests for analyze_feature_tokens function."""

    def test_basic_analysis(self):
        """Should count tokens correctly."""
        tokens_list = [
            ['The', 'cat'],
            ['The', 'dog'],
            ['A', 'bird']
        ]
        result = analyze_feature_tokens(tokens_list, apply_filters=False)

        assert result['summary']['total_examples'] == 3
        assert result['summary']['original_token_count'] == 5  # The, cat, dog, A, bird
        assert result['summary']['total_token_occurrences'] == 6

    def test_filtering(self):
        """Should filter junk tokens."""
        tokens_list = [
            ['<s>', 'The', 'cat', ','],
            ['</s>', 'The', 'dog', '1'],
        ]
        result = analyze_feature_tokens(tokens_list, apply_filters=True)

        # Should remove: <s>, </s>, ,, 1 (4 junk tokens)
        assert result['summary']['junk_removed'] == 4
        # Should keep: The (2x), cat, dog (3 unique)
        assert result['summary']['filtered_token_count'] == 3

    def test_sorting(self):
        """Should sort by count descending, then alphabetically."""
        tokens_list = [
            ['The', 'cat'],
            ['The', 'dog'],
            ['The', 'bird'],
            ['cat', 'bird']
        ]
        result = analyze_feature_tokens(tokens_list, apply_filters=False)

        # The: 3, bird: 2, cat: 2, dog: 1
        # bird and cat have same count, should be sorted alphabetically
        assert result['tokens'][0]['token'] == 'The'
        assert result['tokens'][0]['count'] == 3
        assert result['tokens'][1]['token'] == 'bird'
        assert result['tokens'][1]['count'] == 2
        assert result['tokens'][2]['token'] == 'cat'
        assert result['tokens'][2]['count'] == 2

    def test_percentage_calculation(self):
        """Should calculate percentages correctly."""
        tokens_list = [
            ['The', 'The'],  # The: 50%
            ['cat', 'dog']   # cat: 25%, dog: 25%
        ]
        result = analyze_feature_tokens(tokens_list, apply_filters=False)

        assert result['tokens'][0]['percentage'] == 50.0
        assert result['tokens'][1]['percentage'] == 25.0
        assert result['tokens'][2]['percentage'] == 25.0

    def test_rank_assignment(self):
        """Should assign ranks correctly."""
        tokens_list = [
            ['A', 'B', 'C']
        ]
        result = analyze_feature_tokens(tokens_list, apply_filters=False)

        assert result['tokens'][0]['rank'] == 1
        assert result['tokens'][1]['rank'] == 2
        assert result['tokens'][2]['rank'] == 3

    def test_empty_input(self):
        """Should handle empty input."""
        result = analyze_feature_tokens([], apply_filters=True)

        assert result['summary']['total_examples'] == 0
        assert result['summary']['original_token_count'] == 0
        assert result['tokens'] == []

    def test_diversity_calculation(self):
        """Should calculate diversity percentage."""
        # 3 unique tokens out of 6 total = 50% diversity
        tokens_list = [
            ['The', 'The', 'cat'],
            ['The', 'dog', 'bird']
        ]
        result = analyze_feature_tokens(tokens_list, apply_filters=False)

        # 4 unique / 6 total = 66.67%
        assert result['summary']['diversity_percent'] == 66.67
