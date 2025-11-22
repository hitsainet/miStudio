"""
Unit tests for LocalLabelingService context-based methods.

Tests the refactored context-based labeling functionality:
- generate_label() with examples
- _build_prompt_from_examples() method
- _parse_dual_label() method
"""

import pytest
from unittest.mock import Mock, patch
from src.services.local_labeling_service import LocalLabelingService


@pytest.fixture
def sample_examples():
    """Create sample activation examples."""
    return [
        {
            "prefix_tokens": ["In", "the"],
            "prime_token": "garden",
            "suffix_tokens": ["there", "were", "flowers"],
            "max_activation": 7.8,
        },
        {
            "prefix_tokens": ["She", "planted", "a"],
            "prime_token": "garden",
            "suffix_tokens": ["last", "spring"],
            "max_activation": 6.5,
        },
    ]


@pytest.fixture
def local_service():
    """Create LocalLabelingService instance."""
    return LocalLabelingService(model_name="phi3")


class TestBuildPromptFromExamples:
    """Tests for _build_prompt_from_examples method."""

    def test_build_prompt_basic(self, local_service, sample_examples):
        """Test basic prompt building with examples."""
        result = local_service._build_prompt_from_examples(
            examples=sample_examples, feature_id="feat_123"
        )

        assert "feat_123" in result
        assert "Example 1" in result
        assert "Example 2" in result
        assert "<<garden>>" in result
        assert "(activation: 7.80)" in result
        assert "In the" in result
        assert "there were flowers" in result

    def test_build_prompt_truncates_long_context(self, local_service):
        """Test that long contexts are truncated."""
        long_examples = [
            {
                "prefix_tokens": ["word"] * 50,
                "prime_token": "test",
                "suffix_tokens": ["word"] * 50,
                "max_activation": 5.0,
            }
        ]

        result = local_service._build_prompt_from_examples(
            examples=long_examples, feature_id="feat_long"
        )

        # Check for truncation markers
        assert "..." in result
        assert "<<test>>" in result

    def test_build_prompt_empty_examples(self, local_service):
        """Test prompt building with empty examples."""
        result = local_service._build_prompt_from_examples(
            examples=[], feature_id="feat_empty"
        )

        # Should still contain structure
        assert "feat_empty" in result or "feature" in result.lower()

    def test_build_prompt_missing_context(self, local_service):
        """Test handling when prefix/suffix are missing."""
        minimal_examples = [
            {"prime_token": "test", "max_activation": 3.0}
        ]

        result = local_service._build_prompt_from_examples(
            examples=minimal_examples, feature_id="feat_minimal"
        )

        assert "<<test>>" in result
        assert "feat_minimal" in result


class TestParseDualLabel:
    """Tests for _parse_dual_label method."""

    def test_parse_valid_json(self, local_service):
        """Test parsing valid JSON response."""
        response = '''
        {
            "specific": "botanical_spaces",
            "category": "semantic",
            "description": "Refers to outdoor gardening areas."
        }
        '''

        result = local_service._parse_dual_label(response, "fallback_label")

        assert result["specific"] == "botanical_spaces"
        assert result["category"] == "semantic"
        assert "garden" in result["description"].lower()

    def test_parse_json_with_markdown_fences(self, local_service):
        """Test parsing JSON wrapped in markdown code fences."""
        response = '''
        ```json
        {
            "specific": "test_feature",
            "category": "semantic",
            "description": "Test description"
        }
        ```
        '''

        result = local_service._parse_dual_label(response, "fallback")

        assert result["specific"] == "test_feature"
        assert result["category"] == "semantic"

    def test_parse_json_with_extra_text(self, local_service):
        """Test parsing JSON with extra text before/after."""
        response = '''
        Here is the JSON:
        {
            "specific": "found_it",
            "category": "semantic",
            "description": "Description text"
        }
        And some text after.
        '''

        result = local_service._parse_dual_label(response, "fallback")

        assert result["specific"] == "found_it"
        assert result["category"] == "semantic"

    def test_parse_invalid_json_fallback(self, local_service):
        """Test fallback when JSON is invalid."""
        response = "This is not JSON at all!"

        result = local_service._parse_dual_label(response, "my_fallback")

        assert result["specific"] == "my_fallback"
        assert result["category"] == "semantic"
        assert result["description"] == ""

    def test_parse_json_missing_fields(self, local_service):
        """Test handling when required fields are missing."""
        response = '''
        {
            "specific": "only_this_field"
        }
        '''

        result = local_service._parse_dual_label(response, "fallback")

        # Should fallback when category is missing
        assert "specific" in result
        assert "category" in result

    def test_parse_normalizes_labels(self, local_service):
        """Test that labels are normalized (lowercase, underscores)."""
        response = '''
        {
            "specific": "Test Feature With Spaces",
            "category": "Semantic Category",
            "description": "Test"
        }
        '''

        result = local_service._parse_dual_label(response, "fallback")

        assert result["specific"] == "test_feature_with_spaces"
        assert result["category"] == "semantic_category"
        assert "_" not in result["description"]  # Description not normalized

    def test_parse_removes_special_characters(self, local_service):
        """Test that special characters are removed from labels."""
        response = '''
        {
            "specific": "test@feature#123",
            "category": "semantic!",
            "description": "Test"
        }
        '''

        result = local_service._parse_dual_label(response, "fallback")

        # Special characters should be removed
        assert "@" not in result["specific"]
        assert "#" not in result["specific"]
        assert "!" not in result["category"]
        # But alphanumeric and underscores should remain
        assert "test" in result["specific"]
        assert "feature" in result["specific"]


class TestGenerateLabel:
    """Tests for generate_label method (integration-style)."""

    def test_generate_label_requires_loaded_model(
        self, local_service, sample_examples
    ):
        """Test that generate_label loads model if not loaded."""
        # Model not loaded initially
        assert not local_service.is_loaded

        # Mock the tokenizer and model objects
        mock_tokenizer = Mock()
        mock_model = Mock()

        # Setup side effect for load_model to set the mocked objects
        def mock_load():
            local_service.tokenizer = mock_tokenizer
            local_service.model = mock_model
            local_service.is_loaded = True

        with patch.object(local_service, "load_model", side_effect=mock_load) as mock_load_method:
            with patch.object(
                local_service, "_build_prompt_from_examples", return_value="test prompt"
            ):
                mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
                mock_tokenizer.return_value = {"input_ids": Mock()}
                mock_model.generate.return_value = [Mock()]
                mock_tokenizer.decode.return_value = '{"specific": "test", "category": "semantic", "description": "Test description"}'

                with patch.object(
                    local_service,
                    "_parse_dual_label",
                    return_value={
                        "category": "semantic",
                        "specific": "test",
                        "description": "",
                    },
                ):
                    # Generate label
                    result = local_service.generate_label(
                        examples=sample_examples,
                        neuron_index=0,
                        feature_id="feat_test",
                    )

                    # Verify load_model was called
                    mock_load_method.assert_called_once()
                    assert result["specific"] == "test"

    def test_generate_label_empty_examples_fallback(self, local_service):
        """Test that empty examples return fallback label."""
        result = local_service.generate_label(
            examples=[], neuron_index=42, feature_id="feat_empty"
        )

        assert result["category"] == "empty_features"
        assert "feat_empty" in result["specific"] or "feature_42" in result["specific"]


class TestBatchGenerateLabels:
    """Tests for batch_generate_labels method."""

    def test_batch_loads_and_unloads_model_once(self, local_service):
        """Test that batch processing loads model once and unloads after."""
        examples_list = [
            [{"prime_token": "test1", "max_activation": 5.0}],
            [{"prime_token": "test2", "max_activation": 4.0}],
        ]

        with patch.object(local_service, "load_model") as mock_load:
            with patch.object(local_service, "unload_model") as mock_unload:
                with patch.object(
                    local_service,
                    "generate_label",
                    return_value={
                        "category": "semantic",
                        "specific": "test",
                        "description": "",
                    },
                ):
                    local_service.batch_generate_labels(
                        features_examples=examples_list,
                        neuron_indices=[0, 1],
                        feature_ids=["feat_1", "feat_2"],
                    )

                    # Verify model lifecycle
                    mock_load.assert_called_once()
                    mock_unload.assert_called_once()

    def test_batch_returns_correct_number_of_labels(self, local_service):
        """Test that batch processing returns one label per feature."""
        examples_list = [[{"prime_token": f"test{i}", "max_activation": 5.0}] for i in range(5)]

        with patch.object(local_service, "load_model"):
            with patch.object(local_service, "unload_model"):
                with patch.object(
                    local_service,
                    "generate_label",
                    return_value={
                        "category": "semantic",
                        "specific": "test",
                        "description": "",
                    },
                ):
                    results = local_service.batch_generate_labels(
                        features_examples=examples_list,
                        neuron_indices=list(range(5)),
                        feature_ids=[f"feat_{i}" for i in range(5)],
                    )

                    assert len(results) == 5
                    assert all("specific" in r for r in results)
                    assert all("category" in r for r in results)
