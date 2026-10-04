"""
Unit tests for OpenAILabelingService context-based methods.

Tests the new context-based labeling functionality:
- _build_user_prompt() method
- generate_label_from_examples() method
- Template dispatcher integration
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.services.openai_labeling_service import OpenAILabelingService


@pytest.fixture
def sample_examples():
    """Create sample activation examples with BPE-style tokens.

    Uses GPT-2 style BPE tokens where 'Ġ' prefix indicates start of a new word.
    This matches how transformer tokenizers actually encode text.
    """
    return [
        {
            "prefix_tokens": ["The", "Ġcat"],
            "prime_token": "Ġjumped",
            "suffix_tokens": ["Ġover", "Ġfence"],
            "max_activation": 9.2,
        },
        {
            "prefix_tokens": ["She"],
            "prime_token": "Ġjumped",
            "suffix_tokens": ["Ġhigh"],
            "max_activation": 8.5,
        },
    ]


@pytest.fixture
def mistudio_template_config():
    """Create miStudio Internal template configuration."""
    return {
        "template_type": "mistudio_context",
        "max_examples": 10,
        "include_prefix": True,
        "include_suffix": True,
        "prime_token_marker": "<< >>",  # Note: space in middle for proper split
        "include_logit_effects": False,
        "top_promoted_tokens_count": None,
        "top_suppressed_tokens_count": None,
        "is_detection_template": False,
    }


@pytest.fixture
def anthropic_template_config():
    """Create Anthropic Style template configuration."""
    return {
        "template_type": "anthropic_logit",
        "max_examples": 50,
        "include_prefix": True,
        "include_suffix": True,
        "prime_token_marker": "<< >>",  # Note: space in middle for proper split
        "include_logit_effects": True,
        "top_promoted_tokens_count": 10,
        "top_suppressed_tokens_count": 10,
        "is_detection_template": False,
    }


@pytest.fixture
def detection_template_config():
    """Create EleutherAI Detection template configuration."""
    return {
        "template_type": "eleutherai_detection",
        "max_examples": 20,
        "include_prefix": False,
        "include_suffix": False,
        "prime_token_marker": "<<>>",
        "include_logit_effects": False,
        "top_promoted_tokens_count": None,
        "top_suppressed_tokens_count": None,
        "is_detection_template": True,
    }


@pytest.fixture
def openai_service():
    """Create OpenAILabelingService instance."""
    return OpenAILabelingService(
        api_key="test-key",
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=256,
        top_p=0.9,
    )


class TestBuildUserPrompt:
    """Tests for _build_user_prompt method."""

    def test_build_prompt_mistudio_template(
        self, openai_service, sample_examples, mistudio_template_config
    ):
        """Test building user prompt with miStudio Internal template."""
        user_prompt_template = "Analyze feature {feature_id}.\n\n{examples_block}"

        result = openai_service._build_user_prompt(
            examples=sample_examples,
            template_config=mistudio_template_config,
            user_prompt_template=user_prompt_template,
            feature_id="feat_123",
            logit_effects=None,
        )

        assert "feat_123" in result
        assert "<<jumped>>" in result
        assert "The cat" in result
        assert "over fence" in result
        assert "Example 1" in result
        assert "Example 2" in result

    def test_build_prompt_anthropic_template(
        self, openai_service, sample_examples, anthropic_template_config
    ):
        """Test building user prompt with Anthropic Style template."""
        user_prompt_template = "Analyze feature {feature_id}.\n\n{examples_block}\n\nLogit effects:\n{top_promoted_tokens}\n{top_suppressed_tokens}"

        logit_effects = {
            "top_promoted": ["jumping", "leaping", "hopping"],
            "top_suppressed": ["falling", "dropping"],
        }

        result = openai_service._build_user_prompt(
            examples=sample_examples,
            template_config=anthropic_template_config,
            user_prompt_template=user_prompt_template,
            feature_id="feat_456",
            logit_effects=logit_effects,
        )

        assert "feat_456" in result
        assert "<<jumped>>" in result
        assert "jumping" in result
        assert "leaping" in result
        assert "falling" in result

    def test_build_prompt_with_simple_template(
        self, openai_service, sample_examples, mistudio_template_config
    ):
        """Test building user prompt with simple template."""
        user_prompt_template = "Analyze these examples:\n{examples_block}"

        result = openai_service._build_user_prompt(
            examples=sample_examples,
            template_config=mistudio_template_config,
            user_prompt_template=user_prompt_template,
            feature_id="feat_simple",
            logit_effects=None,
        )

        # Should contain formatted examples
        assert "Analyze these examples:" in result
        assert "<<jumped>>" in result
        assert "Example 1" in result

    def test_build_prompt_empty_examples(
        self, openai_service, mistudio_template_config
    ):
        """Test building prompt with empty examples list."""
        user_prompt_template = "Analyze feature {feature_id}.\n\n{examples_block}"

        result = openai_service._build_user_prompt(
            examples=[],
            template_config=mistudio_template_config,
            user_prompt_template=user_prompt_template,
            feature_id="feat_empty",
            logit_effects=None,
        )

        assert "feat_empty" in result
        # Should not crash, even with no examples


class TestGenerateLabelFromExamples:
    """Tests for generate_label_from_examples method."""

    @pytest.mark.asyncio
    async def test_generate_label_success(
        self, openai_service, sample_examples, mistudio_template_config
    ):
        """Test successful label generation from examples."""
        # Mock OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '''
        {
            "specific": "jumping_action",
            "category": "semantic",
            "description": "Represents jumping or leaping actions."
        }
        '''

        with patch.object(
            openai_service.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await openai_service.generate_label_from_examples(
                examples=sample_examples,
                template_config=mistudio_template_config,
                user_prompt_template="Analyze feature {feature_id}.\n\n{examples_block}",
                system_message="You are an expert.",
                feature_id="feat_test",
                logit_effects=None,
            )

        assert result["specific"] == "jumping_action"
        assert result["category"] == "semantic"
        assert "jumping" in result["description"].lower()

    @pytest.mark.asyncio
    async def test_generate_label_empty_examples(
        self, openai_service, mistudio_template_config
    ):
        """Test label generation with empty examples."""
        result = await openai_service.generate_label_from_examples(
            examples=[],
            template_config=mistudio_template_config,
            user_prompt_template="Test",
            system_message="Test",
            feature_id="feat_empty",
            logit_effects=None,
        )

        assert result["category"] == "empty_features"
        assert "feat_empty" in result["specific"]

    @pytest.mark.asyncio
    async def test_generate_label_api_error(
        self, openai_service, sample_examples, mistudio_template_config
    ):
        """Test error handling when API call fails."""
        with patch.object(
            openai_service.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("API Error"),
        ):
            result = await openai_service.generate_label_from_examples(
                examples=sample_examples,
                template_config=mistudio_template_config,
                user_prompt_template="Test",
                system_message="Test",
                feature_id="feat_error",
                logit_effects=None,
            )

        assert result["category"] == "error_feature"
        assert "feat_error" in result["specific"]

    @pytest.mark.asyncio
    async def test_generate_label_invalid_json_response(
        self, openai_service, sample_examples, mistudio_template_config
    ):
        """Test handling of invalid JSON in API response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This is not JSON"

        with patch.object(
            openai_service.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await openai_service.generate_label_from_examples(
                examples=sample_examples,
                template_config=mistudio_template_config,
                user_prompt_template="Test",
                system_message="Test",
                feature_id="feat_invalid",
                logit_effects=None,
            )

        # Should fallback gracefully
        assert "specific" in result
        assert "category" in result
        assert "feat_invalid" in result["specific"]


class TestTemplatePlaceholders:
    """Tests for template placeholder replacement."""

    def test_logit_effects_placeholders_with_data(
        self, openai_service, sample_examples, anthropic_template_config
    ):
        """Test that logit effect placeholders are replaced with actual data."""
        user_prompt_template = "Promoted: {top_promoted_tokens}\nSuppressed: {top_suppressed_tokens}"

        logit_effects = {
            "top_promoted": ["jump", "leap"],
            "top_suppressed": ["fall", "drop"],
        }

        result = openai_service._build_user_prompt(
            examples=sample_examples,
            template_config=anthropic_template_config,
            user_prompt_template=user_prompt_template,
            feature_id="feat_placeholder",
            logit_effects=logit_effects,
        )

        assert "jump" in result
        assert "leap" in result
        assert "fall" in result
        assert "drop" in result
        assert "{top_promoted_tokens}" not in result
        assert "{top_suppressed_tokens}" not in result

    def test_logit_effects_placeholders_without_data(
        self, openai_service, sample_examples, anthropic_template_config
    ):
        """Test that logit effect placeholders remain when no data provided."""
        user_prompt_template = "Promoted: {top_promoted_tokens}\nSuppressed: {top_suppressed_tokens}"

        result = openai_service._build_user_prompt(
            examples=sample_examples,
            template_config=anthropic_template_config,
            user_prompt_template=user_prompt_template,
            feature_id="feat_no_logit",
            logit_effects=None,
        )

        # Placeholders should remain unreplaced when logit_effects is None
        assert "{top_promoted_tokens}" in result
        assert "{top_suppressed_tokens}" in result
