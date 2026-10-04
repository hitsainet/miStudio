"""
Unit tests for auto-labeling heuristics.

Tests pattern matching logic for automatic feature label generation.
"""

import pytest
from src.utils.auto_labeling import (
    auto_label_feature,
    extract_high_activation_tokens,
)


class TestExtractHighActivationTokens:
    """Test extract_high_activation_tokens function."""

    def test_extract_tokens_above_threshold(self):
        """Test extracting tokens with activation > 0.7."""
        top_examples = [
            {
                "tokens": ["the", "cat", "sat"],
                "activations": [0.9, 0.8, 0.3]  # max=0.9, intensities: 1.0, 0.89, 0.33
            }
        ]

        tokens = extract_high_activation_tokens(top_examples, intensity_threshold=0.7)

        # Should extract "the" (1.0) and "cat" (0.89), but not "sat" (0.33)
        assert "the" in tokens
        assert "cat" in tokens
        assert "sat" not in tokens

    def test_extract_tokens_from_multiple_examples(self):
        """Test extracting tokens from multiple examples."""
        top_examples = [
            {"tokens": ["the"], "activations": [0.9]},
            {"tokens": ["a"], "activations": [0.85]},
            {"tokens": ["an"], "activations": [0.8]},
        ]

        tokens = extract_high_activation_tokens(top_examples, intensity_threshold=0.7, max_examples=3)

        assert len(tokens) == 3
        assert "the" in tokens
        assert "a" in tokens
        assert "an" in tokens

    def test_respects_max_examples_limit(self):
        """Test that only first N examples are analyzed."""
        top_examples = [
            {"tokens": ["token1"], "activations": [1.0]},
            {"tokens": ["token2"], "activations": [1.0]},
            {"tokens": ["token3"], "activations": [1.0]},
            {"tokens": ["token4"], "activations": [1.0]},
            {"tokens": ["token5"], "activations": [1.0]},
            {"tokens": ["token6"], "activations": [1.0]},  # Should be ignored
        ]

        tokens = extract_high_activation_tokens(top_examples, max_examples=5)

        assert len(tokens) == 5
        assert "token6" not in tokens

    def test_handles_empty_examples(self):
        """Test handling empty examples list."""
        tokens = extract_high_activation_tokens([])

        assert tokens == []

    def test_handles_missing_fields(self):
        """Test handling examples with missing tokens or activations."""
        top_examples = [
            {"tokens": ["test"]},  # Missing activations
            {"activations": [0.9]},  # Missing tokens
            {},  # Missing both
        ]

        tokens = extract_high_activation_tokens(top_examples)

        assert tokens == []


class TestAutoLabelPunctuation:
    """Test punctuation pattern matching."""

    def test_all_punctuation_tokens(self):
        """Test feature with all punctuation tokens."""
        top_examples = [
            {"tokens": [".", ".", "."], "activations": [0.9, 0.85, 0.8]},
            {"tokens": [",", ","], "activations": [0.9, 0.85]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Punctuation"

    def test_mixed_punctuation(self):
        """Test feature with various punctuation marks."""
        top_examples = [
            {"tokens": ["!", "?", ";", ":"], "activations": [1.0, 0.9, 0.85, 0.8]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Punctuation"


class TestAutoLabelQuestions:
    """Test question pattern matching."""

    def test_question_word_what(self):
        """Test feature activating on 'what'."""
        top_examples = [
            {"tokens": ["what", "is", "this"], "activations": [0.9, 0.3, 0.2]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Question Pattern"

    def test_question_word_how(self):
        """Test feature activating on 'how'."""
        top_examples = [
            {"tokens": ["how", "to", "do"], "activations": [0.95, 0.2, 0.2]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Question Pattern"

    def test_multiple_question_words(self):
        """Test feature with multiple question words."""
        top_examples = [
            {"tokens": ["why"], "activations": [0.9]},
            {"tokens": ["when"], "activations": [0.85]},
            {"tokens": ["where"], "activations": [0.8]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Question Pattern"


class TestAutoLabelCode:
    """Test code syntax pattern matching."""

    def test_code_token_def(self):
        """Test feature activating on 'def'."""
        top_examples = [
            {"tokens": ["def", "function_name"], "activations": [0.95, 0.3]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Code Syntax"

    def test_code_token_function(self):
        """Test feature activating on 'function'."""
        top_examples = [
            {"tokens": ["function", "test"], "activations": [0.9, 0.2]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Code Syntax"

    def test_multiple_code_tokens(self):
        """Test feature with multiple code tokens."""
        top_examples = [
            {"tokens": ["class"], "activations": [0.9]},
            {"tokens": ["import"], "activations": [0.85]},
            {"tokens": ["return"], "activations": [0.8]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Code Syntax"


class TestAutoLabelSentiment:
    """Test sentiment pattern matching."""

    def test_positive_sentiment(self):
        """Test feature with positive sentiment words."""
        top_examples = [
            {"tokens": ["good", "job"], "activations": [0.9, 0.3]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Sentiment Positive"

    def test_multiple_positive_words(self):
        """Test feature with multiple positive words."""
        top_examples = [
            {"tokens": ["great"], "activations": [0.9]},
            {"tokens": ["excellent"], "activations": [0.85]},
            {"tokens": ["amazing"], "activations": [0.8]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Sentiment Positive"

    def test_negative_sentiment(self):
        """Test feature with negative sentiment words."""
        top_examples = [
            {"tokens": ["bad", "experience"], "activations": [0.9, 0.3]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Sentiment Negative"

    def test_multiple_negative_words(self):
        """Test feature with multiple negative words."""
        top_examples = [
            {"tokens": ["terrible"], "activations": [0.9]},
            {"tokens": ["awful"], "activations": [0.85]},
            {"tokens": ["horrible"], "activations": [0.8]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Sentiment Negative"


class TestAutoLabelNegation:
    """Test negation pattern matching."""

    def test_negation_not(self):
        """Test feature activating on 'not'."""
        top_examples = [
            {"tokens": ["not", "good"], "activations": [0.9, 0.3]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Negation Logic"

    def test_negation_no(self):
        """Test feature activating on 'no'."""
        top_examples = [
            {"tokens": ["no", "way"], "activations": [0.95, 0.2]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Negation Logic"

    def test_negation_contraction(self):
        """Test feature activating on negation contractions."""
        top_examples = [
            {"tokens": ["don't"], "activations": [0.9]},
            {"tokens": ["won't"], "activations": [0.85]},
            {"tokens": ["can't"], "activations": [0.8]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Negation Logic"


class TestAutoLabelPronouns:
    """Test pronoun pattern matching."""

    def test_first_person_pronoun_I(self):
        """Test feature activating on 'I'."""
        top_examples = [
            {"tokens": ["I", "think"], "activations": [0.9, 0.3]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Pronouns First Person"

    def test_multiple_first_person_pronouns(self):
        """Test feature with multiple first-person pronouns."""
        top_examples = [
            {"tokens": ["we"], "activations": [0.9]},
            {"tokens": ["my"], "activations": [0.85]},
            {"tokens": ["our"], "activations": [0.8]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        assert label == "Pronouns First Person"


class TestAutoLabelFallback:
    """Test fallback behavior."""

    def test_fallback_with_random_tokens(self):
        """Test fallback label when no pattern matches."""
        top_examples = [
            {"tokens": ["xyzabc", "qwerty"], "activations": [0.9, 0.85]},
        ]

        label = auto_label_feature(top_examples, neuron_index=123)

        assert label == "Feature 123"

    def test_fallback_with_low_activation_tokens(self):
        """Test fallback when all tokens have low activation."""
        top_examples = [
            {"tokens": ["the", "cat", "sat"], "activations": [0.3, 0.2, 0.1]},
        ]

        label = auto_label_feature(top_examples, neuron_index=456)

        assert label == "Feature 456"

    def test_fallback_with_empty_examples(self):
        """Test fallback with no examples."""
        label = auto_label_feature([], neuron_index=789)

        assert label == "Feature 789"


class TestPatternPriority:
    """Test pattern matching priority order."""

    def test_question_takes_priority_over_sentiment(self):
        """Test that question pattern has higher priority."""
        top_examples = [
            {"tokens": ["what", "good"], "activations": [0.9, 0.85]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        # Question pattern should match first
        assert label == "Question Pattern"

    def test_code_takes_priority_over_sentiment(self):
        """Test that code pattern has higher priority."""
        top_examples = [
            {"tokens": ["class", "Good"], "activations": [0.9, 0.85]},
        ]

        label = auto_label_feature(top_examples, neuron_index=42)

        # Code pattern should match first
        assert label == "Code Syntax"


class TestCaseSensitivity:
    """Test case sensitivity handling."""

    def test_question_words_case_insensitive(self):
        """Test question words work with different cases."""
        examples_lower = [{"tokens": ["what"], "activations": [0.9]}]
        examples_upper = [{"tokens": ["What"], "activations": [0.9]}]

        label_lower = auto_label_feature(examples_lower, neuron_index=1)
        label_upper = auto_label_feature(examples_upper, neuron_index=2)

        assert label_lower == "Question Pattern"
        assert label_upper == "Question Pattern"

    def test_sentiment_words_case_insensitive(self):
        """Test sentiment words work with different cases."""
        examples_lower = [{"tokens": ["good"], "activations": [0.9]}]
        examples_upper = [{"tokens": ["Good"], "activations": [0.9]}]

        label_lower = auto_label_feature(examples_lower, neuron_index=1)
        label_upper = auto_label_feature(examples_upper, neuron_index=2)

        assert label_lower == "Sentiment Positive"
        assert label_upper == "Sentiment Positive"
