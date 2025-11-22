"""
Unit tests for LabelingContextFormatter.

Tests the context-based example formatting for different template types:
- miStudio Internal (K=10, basic context)
- Anthropic Style (K=50, with logit effects)
- EleutherAI Detection (K=20, for scoring)
"""

import pytest
from src.services.labeling_context_formatter import LabelingContextFormatter


@pytest.fixture
def sample_examples():
    """Create sample activation examples for testing."""
    return [
        {
            "prefix_tokens": ["The", "dog", "was"],
            "prime_token": "running",
            "suffix_tokens": ["in", "the", "park"],
            "max_activation": 8.5,
        },
        {
            "prefix_tokens": ["She", "started"],
            "prime_token": "running",
            "suffix_tokens": ["every", "morning"],
            "max_activation": 7.2,
        },
        {
            "prefix_tokens": ["Keep"],
            "prime_token": "running",
            "suffix_tokens": ["the", "server", "24/7"],
            "max_activation": 6.8,
        },
    ]


@pytest.fixture
def sample_logit_effects():
    """Create sample logit effects for testing."""
    return {
        "promoted": ["running", "jogging", "sprinting", "racing", "moving"],
        "suppressed": ["stopped", "still", "paused", "halted", "stationary"],
    }


@pytest.fixture
def basic_template_config():
    """Create basic template configuration for testing."""
    return {
        "prime_token_marker": "<< >>",  # Note: space in middle for proper split
        "include_prefix": True,
        "include_suffix": True,
        "template_type": "mistudio_context",
        "max_examples": 10,
        "include_logit_effects": False,
    }


@pytest.fixture
def anthropic_template_config():
    """Create Anthropic-style template configuration for testing."""
    return {
        "prime_token_marker": "<< >>",  # Note: space in middle for proper split
        "include_prefix": True,
        "include_suffix": True,
        "template_type": "anthropic_logit",
        "max_examples": 50,
        "include_logit_effects": True,
        "top_promoted_tokens_count": 10,
        "top_suppressed_tokens_count": 10,
    }


class TestFormatMiStudioInternal:
    """Tests for miStudio Internal template formatting."""

    def test_format_basic(self, sample_examples, basic_template_config):
        """Test basic formatting with default settings."""
        result = LabelingContextFormatter.format_mistudio_context(
            examples=sample_examples,
            template_config=basic_template_config,
            feature_id="feat_123"
        )

        assert "Example 1" in result
        assert "Example 2" in result
        assert "Example 3" in result
        assert "<<running>>" in result
        assert "(activation: 8.5" in result
        assert "The dog was" in result
        assert "in the park" in result

    def test_format_with_custom_marker(self, sample_examples, basic_template_config):
        """Test formatting with custom prime token marker."""
        custom_config = basic_template_config.copy()
        custom_config["prime_token_marker"] = "**"  # Splits to * and *

        result = LabelingContextFormatter.format_mistudio_context(
            examples=sample_examples,
            template_config=custom_config,
            feature_id="feat_123"
        )

        # ** is symmetric and splits to * and *, giving us *running*
        assert "*running*" in result
        assert "<<" not in result

    def test_format_without_prefix(self, sample_examples, basic_template_config):
        """Test formatting without prefix tokens."""
        no_prefix_config = basic_template_config.copy()
        no_prefix_config["include_prefix"] = False

        result = LabelingContextFormatter.format_mistudio_context(
            examples=sample_examples,
            template_config=no_prefix_config,
            feature_id="feat_123"
        )

        assert "<<running>>" in result
        assert "The dog was" not in result  # Prefix should be excluded
        assert "in the park" in result  # Suffix should still be included

    def test_format_without_suffix(self, sample_examples, basic_template_config):
        """Test formatting without suffix tokens."""
        no_suffix_config = basic_template_config.copy()
        no_suffix_config["include_suffix"] = False

        result = LabelingContextFormatter.format_mistudio_context(
            examples=sample_examples,
            template_config=no_suffix_config,
            feature_id="feat_123"
        )

        assert "<<running>>" in result
        assert "The dog was" in result  # Prefix should still be included
        assert "in the park" not in result  # Suffix should be excluded

    def test_format_prime_only(self, sample_examples, basic_template_config):
        """Test formatting with only prime tokens."""
        prime_only_config = basic_template_config.copy()
        prime_only_config["include_prefix"] = False
        prime_only_config["include_suffix"] = False

        result = LabelingContextFormatter.format_mistudio_context(
            examples=sample_examples,
            template_config=prime_only_config,
            feature_id="feat_123"
        )

        assert "<<running>>" in result
        assert "The dog was" not in result
        assert "in the park" not in result
        # Should still show example structure
        assert "Example 1" in result

    def test_format_empty_examples(self, basic_template_config):
        """Test formatting with empty examples list."""
        result = LabelingContextFormatter.format_mistudio_context(
            examples=[],
            template_config=basic_template_config,
            feature_id="feat_123"
        )

        # Empty list returns empty string
        assert result == ""

    def test_format_truncates_long_context(self, basic_template_config):
        """Test that very long contexts are handled properly."""
        long_examples = [
            {
                "prefix_tokens": ["word"] * 100,  # Very long prefix
                "prime_token": "test",
                "suffix_tokens": ["word"] * 100,  # Very long suffix
                "max_activation": 5.0,
            }
        ]

        result = LabelingContextFormatter.format_mistudio_context(
            examples=long_examples,
            template_config=basic_template_config,
            feature_id="feat_123"
        )

        # Should handle long contexts (may or may not truncate at formatter level)
        assert "<<test>>" in result
        assert "Example 1" in result


class TestFormatAnthropicStyle:
    """Tests for Anthropic Style template formatting with logit effects."""

    def test_format_with_logit_effects(self, sample_examples, sample_logit_effects, anthropic_template_config):
        """Test formatting with logit effects included."""
        result = LabelingContextFormatter.format_anthropic_logit(
            examples=sample_examples,
            logit_effects=sample_logit_effects,
            template_config=anthropic_template_config,
            feature_id="feat_456"
        )

        assert "<<running>>" in result
        assert "TOP ACTIVATING EXAMPLES" in result
        assert "LOGIT EFFECTS" in result
        assert "Top promoted tokens" in result
        assert "Top suppressed tokens" in result
        assert "running" in result
        assert "jogging" in result
        assert "stopped" in result

    def test_format_without_logit_effects(self, sample_examples, anthropic_template_config):
        """Test formatting without logit effects."""
        result = LabelingContextFormatter.format_anthropic_logit(
            examples=sample_examples,
            logit_effects=None,
            template_config=anthropic_template_config,
            feature_id="feat_456"
        )

        assert "<<running>>" in result
        assert "TOP ACTIVATING EXAMPLES" in result
        assert "LOGIT EFFECTS" in result
        # Should show "No logit effects available" message
        assert "No logit effects available" in result

    def test_format_with_partial_logit_effects(self, sample_examples, anthropic_template_config):
        """Test formatting with only some logit effects."""
        partial_effects = {
            "promoted": ["running", "jogging"],
            "suppressed": []
        }

        result = LabelingContextFormatter.format_anthropic_logit(
            examples=sample_examples,
            logit_effects=partial_effects,
            template_config=anthropic_template_config,
            feature_id="feat_456"
        )

        assert "running" in result
        assert "jogging" in result
        assert "Top promoted tokens" in result


class TestFormatEleutherAIDetection:
    """Tests for EleutherAI Detection template formatting."""

    def test_format_detection_basic(self):
        """Test basic detection template formatting."""
        feature_explanation = {
            "name": "continuous_actions",
            "category": "semantic",
            "description": "This feature detects continuous actions or processes."
        }
        test_examples = [
            "The dog was running in the park",
            "She started running every morning",
            "Keep running the server 24/7"
        ]

        result = LabelingContextFormatter.format_eleutherai_detection(
            feature_explanation=feature_explanation,
            test_examples=test_examples
        )

        assert "1. The dog was running in the park" in result
        assert "2. She started running every morning" in result
        assert "3. Keep running the server 24/7" in result

    def test_format_detection_empty_examples(self):
        """Test detection format with empty examples list."""
        feature_explanation = {
            "name": "test_feature",
            "category": "semantic",
            "description": "Test"
        }

        result = LabelingContextFormatter.format_eleutherai_detection(
            feature_explanation=feature_explanation,
            test_examples=[]
        )

        # Empty list returns empty string
        assert result == ""

    def test_format_detection_single_example(self):
        """Test detection format with single example."""
        feature_explanation = {
            "name": "test_feature",
            "category": "semantic",
            "description": "Test"
        }
        test_examples = ["Single example text"]

        result = LabelingContextFormatter.format_eleutherai_detection(
            feature_explanation=feature_explanation,
            test_examples=test_examples
        )

        assert "1. Single example text" in result
        assert "2." not in result


class TestFormatEdgeCases:
    """Tests for edge cases and error handling."""

    def test_format_with_special_characters(self, basic_template_config):
        """Test formatting with special characters in tokens."""
        examples = [
            {
                "prefix_tokens": ["Test", "with", "\"quotes\""],
                "prime_token": "<special>",
                "suffix_tokens": ["and", "&", "symbols"],
                "max_activation": 5.0,
            }
        ]

        result = LabelingContextFormatter.format_mistudio_context(
            examples=examples,
            template_config=basic_template_config,
            feature_id="feat_special"
        )

        # Should handle special characters gracefully
        assert "<<" in result
        assert "&" in result or "symbols" in result

    def test_format_with_missing_token_fields(self, basic_template_config):
        """Test formatting when some token fields are missing."""
        examples = [
            {
                "prime_token": "test",
                "max_activation": 5.0,
                # Missing prefix_tokens and suffix_tokens
            }
        ]

        result = LabelingContextFormatter.format_mistudio_context(
            examples=examples,
            template_config=basic_template_config,
            feature_id="feat_partial"
        )

        # Should still format with available data
        assert "<<test>>" in result

    def test_format_with_zero_activation(self, basic_template_config):
        """Test formatting with zero activation value."""
        examples = [
            {
                "prefix_tokens": ["no"],
                "prime_token": "activation",
                "suffix_tokens": ["here"],
                "max_activation": 0.0,
            }
        ]

        result = LabelingContextFormatter.format_mistudio_context(
            examples=examples,
            template_config=basic_template_config,
            feature_id="feat_zero"
        )

        assert "(activation: 0.0" in result
        assert "<<activation>>" in result
