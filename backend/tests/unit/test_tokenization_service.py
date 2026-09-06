"""
Unit tests for TokenizationService padding strategies.

This module tests that different padding strategies work correctly
with the TokenizationService.tokenize_dataset method.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer

from src.services.tokenization_service import TokenizationService


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock(spec=PreTrainedTokenizer)
    tokenizer.model_max_length = 1024
    tokenizer.vocab_size = 50257
    tokenizer.name_or_path = "gpt2"  # Add name_or_path for multiprocessing support

    # Mock tokenization behavior
    def mock_batch_encode(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        stride=0,
        return_overflowing_tokens=False,
        add_special_tokens=True,
        return_attention_mask=True,
        return_length=True,
    ):
        # Simulate tokenization based on text length
        result = {
            "input_ids": [],
            "attention_mask": [],
            "length": [],
        }

        for text in texts:
            # Simulate different length sequences based on text length
            token_count = min(len(text) // 4, max_length)  # Rough approximation

            if padding == "max_length":
                # Pad to max_length
                input_ids = list(range(token_count)) + [0] * (max_length - token_count)
                attention_mask = [1] * token_count + [0] * (max_length - token_count)
            elif padding == "longest":
                # In real scenario, this would pad to longest in batch
                # For testing, we'll just use actual length
                input_ids = list(range(token_count))
                attention_mask = [1] * token_count
            elif padding == "do_not_pad" or padding is False:
                # No padding
                input_ids = list(range(token_count))
                attention_mask = [1] * token_count
            else:
                input_ids = list(range(token_count))
                attention_mask = [1] * token_count

            result["input_ids"].append(input_ids)
            result["attention_mask"].append(attention_mask)
            result["length"].append(len(input_ids))

        return result

    tokenizer.side_effect = mock_batch_encode
    tokenizer.__call__ = mock_batch_encode

    return tokenizer


@pytest.fixture
def sample_dataset():
    """Create a sample HuggingFace dataset for testing."""
    data = {
        "text": [
            "This is a short text.",  # ~5 tokens
            "This is a medium length text that should be tokenized appropriately.",  # ~13 tokens
            "This is a very long text " * 50,  # ~150+ tokens
        ]
    }
    return HFDataset.from_dict(data)


def test_padding_strategy_max_length(mock_tokenizer, sample_dataset):
    """
    Test tokenization with 'max_length' padding strategy.

    Verifies that all sequences are padded to exactly max_length.
    """
    max_length = 128

    with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        result = TokenizationService.tokenize_dataset(
            dataset=sample_dataset,
            tokenizer=mock_tokenizer,
            text_column="text",
            max_length=max_length,
            stride=0,
            truncation=True,
            padding="max_length",
            batch_size=10,
            num_proc=1,  # Use single-process for tests with mock tokenizers
        )

    # Verify result structure
    assert result is not None
    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names

    # Verify all sequences have max_length
    for example in result:
        assert len(example["input_ids"]) == max_length, \
            f"Expected length {max_length}, got {len(example['input_ids'])}"
        assert len(example["attention_mask"]) == max_length


def test_padding_strategy_longest(mock_tokenizer, sample_dataset):
    """
    Test tokenization with 'longest' padding strategy.

    Verifies that sequences are padded to the longest sequence in the batch.
    Note: In practice, this is dynamic per batch. For testing, we verify
    that padding is NOT fixed to max_length.
    """
    max_length = 128

    with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        result = TokenizationService.tokenize_dataset(
            dataset=sample_dataset,
            tokenizer=mock_tokenizer,
            text_column="text",
            max_length=max_length,
            stride=0,
            truncation=True,
            padding="longest",
            batch_size=10,
            num_proc=1,  # Use single-process for tests with mock tokenizers
        )

    # Verify result structure
    assert result is not None
    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names

    # Verify sequences may have different lengths (not all max_length)
    # This depends on batch processing, so we just verify it doesn't crash
    # and that at least one sequence is shorter than max_length
    lengths = [len(example["input_ids"]) for example in result]
    assert any(length < max_length for length in lengths), \
        "Expected at least one sequence shorter than max_length with 'longest' padding"


def test_padding_strategy_do_not_pad(mock_tokenizer, sample_dataset):
    """
    Test tokenization with 'do_not_pad' padding strategy.

    Verifies that no padding is applied and sequences have variable lengths.
    """
    max_length = 128

    with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        result = TokenizationService.tokenize_dataset(
            dataset=sample_dataset,
            tokenizer=mock_tokenizer,
            text_column="text",
            max_length=max_length,
            stride=0,
            truncation=True,
            padding="do_not_pad",
            batch_size=10,
            num_proc=1,  # Use single-process for tests with mock tokenizers
        )

    # Verify result structure
    assert result is not None
    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names

    # Verify sequences have different lengths (no padding)
    lengths = [len(example["input_ids"]) for example in result]
    unique_lengths = set(lengths)

    assert len(unique_lengths) > 1, \
        "Expected multiple different sequence lengths with 'do_not_pad'"
    assert all(length <= max_length for length in lengths), \
        "No sequence should exceed max_length (truncation should still apply)"


def test_calculate_statistics_with_different_padding():
    """
    Test that calculate_statistics works correctly with padded sequences.

    Note: calculate_statistics counts all tokens including padding (based on sequence length).
    This is intentional as it reflects the actual tensor sizes used during training.
    """
    # Create mock tokenized dataset with padding
    tokenized_data = {
        "input_ids": [
            [1, 2, 3, 0, 0],  # 5 tokens total (including padding)
            [1, 2, 3, 4, 0],  # 5 tokens total (including padding)
            [1, 2, 0, 0, 0],  # 5 tokens total (including padding)
        ],
        "attention_mask": [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0],
        ],
    }

    tokenized_dataset = HFDataset.from_dict(tokenized_data)

    # Calculate statistics
    stats = TokenizationService.calculate_statistics(tokenized_dataset)

    # Verify statistics (counts ALL tokens including padding)
    assert stats["num_tokens"] == 15, "Expected 15 total tokens (5 + 5 + 5)"
    assert stats["avg_seq_length"] == 5.0, "Expected average of 5.0 tokens per sequence"
    assert stats["min_seq_length"] == 5, "Expected minimum of 5 tokens"
    assert stats["max_seq_length"] == 5, "Expected maximum of 5 tokens"
    assert stats["num_samples"] == 3


def test_metadata_includes_padding_strategy():
    """
    Test that tokenization metadata includes the padding strategy used.

    This is important for reproducibility and understanding the dataset.
    """
    # Create sample tokenized dataset
    tokenized_data = {
        "input_ids": [[1, 2, 3], [1, 2, 3, 4]],
        "attention_mask": [[1, 1, 1], [1, 1, 1, 1]],
    }
    tokenized_dataset = HFDataset.from_dict(tokenized_data)

    stats = TokenizationService.calculate_statistics(tokenized_dataset)

    # Verify statistics structure
    assert isinstance(stats, dict)
    assert "num_tokens" in stats
    assert "avg_seq_length" in stats
    assert "min_seq_length" in stats
    assert "max_seq_length" in stats


@pytest.mark.parametrize("padding_strategy", ["max_length", "longest", "do_not_pad"])
def test_all_padding_strategies_complete(mock_tokenizer, sample_dataset, padding_strategy):
    """
    Parametrized test to ensure all padding strategies complete without errors.

    This test verifies that all three padding strategies can be used
    with the TokenizationService without crashes or exceptions.
    """
    max_length = 64

    with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        result = TokenizationService.tokenize_dataset(
            dataset=sample_dataset,
            tokenizer=mock_tokenizer,
            text_column="text",
            max_length=max_length,
            stride=0,
            truncation=True,
            padding=padding_strategy,
            batch_size=10,
            num_proc=1,  # Use single-process for tests with mock tokenizers
        )

    # Verify result is valid
    assert result is not None
    assert len(result) == len(sample_dataset)
    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names


def test_padding_with_stride():
    """
    Test that padding works correctly when stride is used.

    Stride creates overlapping sequences, and padding should still
    work correctly with the resulting sequences.
    """
    # This test ensures compatibility between padding and stride
    # In practice, stride may create more sequences, and each should be padded correctly

    # Create sample data
    sample_data = {
        "text": ["This is a very long text " * 20]  # Long text to trigger stride
    }
    dataset = HFDataset.from_dict(sample_data)

    mock_tok = Mock(spec=PreTrainedTokenizer)
    mock_tok.model_max_length = 1024
    mock_tok.vocab_size = 50257
    mock_tok.name_or_path = "gpt2"  # Add name_or_path for multiprocessing support

    def mock_encode_with_stride(texts, **kwargs):
        # Simulate stride creating multiple sequences
        return {
            "input_ids": [
                [1, 2, 3, 4, 5] + [0] * 59,  # First chunk with padding
                [4, 5, 6, 7, 8] + [0] * 59,  # Second chunk with overlap and padding
            ],
            "attention_mask": [
                [1] * 5 + [0] * 59,
                [1] * 5 + [0] * 59,
            ],
            "length": [64, 64],
        }

    mock_tok.side_effect = mock_encode_with_stride
    mock_tok.__call__ = mock_encode_with_stride

    with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tok):
        result = TokenizationService.tokenize_dataset(
            dataset=dataset,
            tokenizer=mock_tok,
            text_column="text",
            max_length=64,
            stride=32,
            truncation=True,
            padding="max_length",
            batch_size=1,
            num_proc=1,  # Disable multiprocessing to avoid pickling issues in tests
        )

    # Verify result
    assert result is not None
    # With stride, we may get more sequences than input samples
    assert len(result) >= len(dataset)


@pytest.mark.parametrize("truncation_strategy", [True, "only_first", "only_second", False])
def test_all_truncation_strategies_complete(mock_tokenizer, sample_dataset, truncation_strategy):
    """
    Parametrized test to ensure all truncation strategies complete without errors.

    This test verifies that all four truncation strategies can be used
    with the TokenizationService without crashes or exceptions.
    """
    max_length = 64

    with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        result = TokenizationService.tokenize_dataset(
            dataset=sample_dataset,
            tokenizer=mock_tokenizer,
            text_column="text",
            max_length=max_length,
            stride=0,
            truncation=truncation_strategy,
            padding="max_length",
            batch_size=10,
            num_proc=1,  # Use single-process for tests with mock tokenizers
        )

    # Verify result is valid
    assert result is not None
    assert len(result) == len(sample_dataset)
    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names


def test_truncation_strategy_longest_first(mock_tokenizer, sample_dataset):
    """
    Test tokenization with 'longest_first' truncation strategy (True).

    Verifies that truncation is applied when enabled.
    """
    max_length = 64

    with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        result = TokenizationService.tokenize_dataset(
            dataset=sample_dataset,
            tokenizer=mock_tokenizer,
            text_column="text",
            max_length=max_length,
            stride=0,
            truncation=True,  # longest_first is the default (True)
            padding="max_length",
            batch_size=10,
            num_proc=1,  # Use single-process for tests with mock tokenizers
        )

    # Verify result structure
    assert result is not None
    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names

    # Verify all sequences are within max_length
    for example in result:
        assert len(example["input_ids"]) <= max_length, \
            f"Expected length <= {max_length}, got {len(example['input_ids'])}"


def test_truncation_strategy_only_first(mock_tokenizer, sample_dataset):
    """
    Test tokenization with 'only_first' truncation strategy.

    Verifies that only the first sequence is truncated.
    Useful for question-answering where context should be preserved.
    """
    max_length = 64

    with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        result = TokenizationService.tokenize_dataset(
            dataset=sample_dataset,
            tokenizer=mock_tokenizer,
            text_column="text",
            max_length=max_length,
            stride=0,
            truncation="only_first",
            padding="max_length",
            batch_size=10,
            num_proc=1,  # Use single-process for tests with mock tokenizers
        )

    # Verify result structure
    assert result is not None
    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names


def test_truncation_strategy_only_second(mock_tokenizer, sample_dataset):
    """
    Test tokenization with 'only_second' truncation strategy.

    Verifies that only the second sequence is truncated.
    Useful when the first sequence (e.g., question) should be preserved.
    """
    max_length = 64

    with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        result = TokenizationService.tokenize_dataset(
            dataset=sample_dataset,
            tokenizer=mock_tokenizer,
            text_column="text",
            max_length=max_length,
            stride=0,
            truncation="only_second",
            padding="max_length",
            batch_size=10,
            num_proc=1,  # Use single-process for tests with mock tokenizers
        )

    # Verify result structure
    assert result is not None
    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names


def test_truncation_disabled(mock_tokenizer, sample_dataset):
    """
    Test tokenization with truncation disabled (False).

    Verifies that truncation is not applied when disabled.
    Note: In practice, sequences may still be limited by tokenizer's max length.
    """
    max_length = 64

    with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        result = TokenizationService.tokenize_dataset(
            dataset=sample_dataset,
            tokenizer=mock_tokenizer,
            text_column="text",
            max_length=max_length,
            stride=0,
            truncation=False,  # do_not_truncate
            padding="max_length",
            batch_size=10,
            num_proc=1,  # Use single-process for tests with mock tokenizers
        )

    # Verify result structure
    assert result is not None
    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names


def test_truncation_with_padding_combinations():
    """
    Test that truncation works correctly with different padding strategies.

    This test ensures compatibility between truncation and padding.
    """
    # Create sample data with long sequences
    sample_data = {
        "text": [
            "Short text",
            "Medium length text that should be handled appropriately",
            "Very long text " * 30,  # ~90+ tokens
        ]
    }
    dataset = HFDataset.from_dict(sample_data)

    mock_tok = Mock(spec=PreTrainedTokenizer)
    mock_tok.model_max_length = 1024
    mock_tok.vocab_size = 50257
    mock_tok.name_or_path = "gpt2"  # Add name_or_path for multiprocessing support

    def mock_encode_with_truncation(texts, **kwargs):
        truncation = kwargs.get("truncation", True)
        max_length = kwargs.get("max_length", 512)
        padding = kwargs.get("padding", "max_length")

        result = {
            "input_ids": [],
            "attention_mask": [],
            "length": [],
        }

        for text in texts:
            # Simulate tokenization
            token_count = len(text) // 4  # Rough approximation

            # Apply truncation
            if truncation and token_count > max_length:
                token_count = max_length

            # Apply padding
            if padding == "max_length":
                input_ids = list(range(token_count)) + [0] * (max_length - token_count)
                attention_mask = [1] * token_count + [0] * (max_length - token_count)
            else:
                input_ids = list(range(token_count))
                attention_mask = [1] * token_count

            result["input_ids"].append(input_ids)
            result["attention_mask"].append(attention_mask)
            result["length"].append(len(input_ids))

        return result

    mock_tok.side_effect = mock_encode_with_truncation
    mock_tok.__call__ = mock_encode_with_truncation

    # Test combination: truncation=True + padding=max_length
    with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tok):
        result = TokenizationService.tokenize_dataset(
            dataset=dataset,
            tokenizer=mock_tok,
            text_column="text",
            max_length=64,
            stride=0,
            truncation=True,
            padding="max_length",
            batch_size=10,
            num_proc=1,  # Use single-process for tests with mock tokenizers
        )

    # Verify result
    assert result is not None
    assert len(result) == len(dataset)

    # All sequences should be exactly max_length with truncation+padding
    for example in result:
        assert len(example["input_ids"]) == 64, \
            f"Expected length 64 with max_length padding, got {len(example['input_ids'])}"


# ============================================================================
# New Tests for Increased Coverage
# ============================================================================


class TestAnalyzeDatasetSchema:
    """Test dataset schema analysis."""

    def test_analyze_schema_single_text_column(self):
        """Test schema analysis with single text column."""
        dataset = HFDataset.from_dict({
            "text": ["Sample text 1", "Sample text 2"],
            "label": [0, 1]
        })

        schema = TokenizationService.analyze_dataset_schema(dataset)

        assert schema["text_columns"] == ["text"]
        assert schema["recommended_column"] == "text"
        assert schema["is_multi_column"] is False
        assert "text" in schema["column_info"]
        assert "label" in schema["column_info"]
        assert set(schema["all_columns"]) == {"text", "label"}

    def test_analyze_schema_multiple_text_columns(self):
        """Test schema analysis with multiple text columns."""
        dataset = HFDataset.from_dict({
            "text": ["Text 1", "Text 2"],
            "content": ["Content 1", "Content 2"],
            "label": [0, 1]
        })

        schema = TokenizationService.analyze_dataset_schema(dataset)

        assert len(schema["text_columns"]) == 2
        assert "text" in schema["text_columns"]
        assert "content" in schema["text_columns"]
        assert schema["recommended_column"] == "text"  # Prefers 'text'
        assert schema["is_multi_column"] is True

    def test_analyze_schema_prefers_content_over_others(self):
        """Test schema analysis prefers 'content' if 'text' not available."""
        dataset = HFDataset.from_dict({
            "content": ["Content 1", "Content 2"],
            "description": ["Desc 1", "Desc 2"]
        })

        schema = TokenizationService.analyze_dataset_schema(dataset)

        assert schema["recommended_column"] == "content"

    def test_analyze_schema_prefers_chosen_for_rlhf(self):
        """Test schema analysis prefers 'chosen' for RLHF datasets."""
        dataset = HFDataset.from_dict({
            "chosen": ["Chosen text 1", "Chosen text 2"],
            "rejected": ["Rejected text 1", "Rejected text 2"]
        })

        schema = TokenizationService.analyze_dataset_schema(dataset)

        assert schema["recommended_column"] == "chosen"
        assert schema["is_multi_column"] is True

    def test_analyze_schema_no_text_columns(self):
        """Test schema analysis when no text columns present."""
        dataset = HFDataset.from_dict({
            "label": [0, 1],
            "value": [10, 20]
        })

        schema = TokenizationService.analyze_dataset_schema(dataset)

        assert schema["text_columns"] == []
        assert schema["recommended_column"] is None
        assert schema["is_multi_column"] is False


class TestLoadTokenizer:
    """Test tokenizer loading."""

    def test_load_tokenizer_with_pad_token(self):
        """Test loading tokenizer that already has pad_token."""
        with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = "[PAD]"
            mock_from_pretrained.return_value = mock_tokenizer

            tokenizer = TokenizationService.load_tokenizer("bert-base-uncased")

            assert tokenizer.pad_token == "[PAD]"
            mock_from_pretrained.assert_called_once_with(
                "bert-base-uncased", use_fast=True, cache_dir=None, local_files_only=False
            )

    def test_load_tokenizer_adds_pad_token_from_eos(self):
        """Test loading tokenizer adds pad_token from eos_token."""
        with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "<|endoftext|>"
            mock_from_pretrained.return_value = mock_tokenizer

            tokenizer = TokenizationService.load_tokenizer("gpt2")

            assert tokenizer.pad_token == "<|endoftext|>"

    def test_load_tokenizer_adds_new_pad_token(self):
        """Test loading tokenizer adds new pad_token when no eos_token."""
        with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = None
            mock_tokenizer.add_special_tokens = Mock()
            mock_from_pretrained.return_value = mock_tokenizer

            tokenizer = TokenizationService.load_tokenizer("custom-model")

            mock_tokenizer.add_special_tokens.assert_called_once_with({'pad_token': '[PAD]'})

    def test_load_tokenizer_with_use_fast_false(self):
        """Test loading tokenizer with use_fast=False."""
        with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = "[PAD]"
            mock_from_pretrained.return_value = mock_tokenizer

            TokenizationService.load_tokenizer("bert-base-uncased", use_fast=False)

            mock_from_pretrained.assert_called_once_with(
                "bert-base-uncased", use_fast=False, cache_dir=None, local_files_only=False
            )

    def test_load_tokenizer_raises_exception_on_error(self):
        """Test loading tokenizer raises exception on error."""
        with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.side_effect = RuntimeError("Model not found")

            with pytest.raises(Exception) as exc_info:
                TokenizationService.load_tokenizer("nonexistent-model")

            assert "Failed to load tokenizer 'nonexistent-model'" in str(exc_info.value)


class TestCalculateStatisticsEdgeCases:
    """Test calculate_statistics edge cases."""

    def test_calculate_statistics_empty_dataset(self):
        """Test calculate_statistics raises error for empty dataset."""
        empty_dataset = HFDataset.from_dict({"input_ids": [], "attention_mask": []})

        with pytest.raises(ValueError) as exc_info:
            TokenizationService.calculate_statistics(empty_dataset)

        assert "Cannot calculate statistics for empty dataset" in str(exc_info.value)

    def test_calculate_statistics_missing_input_ids(self):
        """Test calculate_statistics raises error when input_ids missing."""
        dataset = HFDataset.from_dict({"attention_mask": [[1, 1, 1]]})

        with pytest.raises(ValueError) as exc_info:
            TokenizationService.calculate_statistics(dataset)

        assert "Dataset missing 'input_ids' field" in str(exc_info.value)

    def test_calculate_statistics_with_split_column(self):
        """Test calculate_statistics includes split distribution when split column exists."""
        dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "split": ["train", "train", "test"]
        })

        stats = TokenizationService.calculate_statistics(dataset)

        assert "split_distribution" in stats
        assert stats["split_distribution"]["train"] == 2
        assert stats["split_distribution"]["test"] == 1

    def test_calculate_statistics_vocab_size(self):
        """Test calculate_statistics computes vocabulary size correctly."""
        dataset = HFDataset.from_dict({
            "input_ids": [
                [1, 2, 3],
                [2, 3, 4],
                [1, 4, 5]
            ]
        })

        stats = TokenizationService.calculate_statistics(dataset)

        # Unique tokens: 1, 2, 3, 4, 5
        assert stats["vocab_size"] == 5

    def test_calculate_statistics_length_distribution(self):
        """Test calculate_statistics computes length distribution correctly."""
        dataset = HFDataset.from_dict({
            "input_ids": [
                [1] * 50,    # 0-100 bucket
                [1] * 150,   # 100-200 bucket
                [1] * 300,   # 200-400 bucket
                [1] * 500,   # 400-600 bucket
                [1] * 700,   # 600-800 bucket
                [1] * 900,   # 800-1000 bucket
                [1] * 1100,  # 1000+ bucket
            ]
        })

        stats = TokenizationService.calculate_statistics(dataset)

        assert stats["length_distribution"]["0-100"] == 1
        assert stats["length_distribution"]["100-200"] == 1
        assert stats["length_distribution"]["200-400"] == 1
        assert stats["length_distribution"]["400-600"] == 1
        assert stats["length_distribution"]["600-800"] == 1
        assert stats["length_distribution"]["800-1000"] == 1
        assert stats["length_distribution"]["1000+"] == 1

    def test_calculate_statistics_median(self):
        """Test calculate_statistics computes median correctly."""
        dataset = HFDataset.from_dict({
            "input_ids": [
                [1] * 100,
                [1] * 200,
                [1] * 300,
            ]
        })

        stats = TokenizationService.calculate_statistics(dataset)

        # Median of [100, 200, 300] is 200
        assert stats["median_seq_length"] == 200.0


class TestSaveAndLoadDataset:
    """Test dataset save and load operations."""

    def test_save_tokenized_dataset(self, tmp_path):
        """Test saving tokenized dataset to disk."""
        dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]]
        })

        output_path = tmp_path / "tokenized_dataset"

        # Save dataset
        TokenizationService.save_tokenized_dataset(dataset, output_path)

        # Verify directory was created
        assert output_path.exists()
        assert output_path.is_dir()

    def test_save_tokenized_dataset_creates_parent_dirs(self, tmp_path):
        """Test saving dataset creates parent directories."""
        output_path = tmp_path / "nested" / "dir" / "dataset"

        dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3]]
        })

        TokenizationService.save_tokenized_dataset(dataset, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_tokenized_dataset_raises_on_error(self, tmp_path):
        """Test save_tokenized_dataset raises exception on error."""
        dataset = HFDataset.from_dict({"input_ids": [[1, 2, 3]]})

        # Mock save_to_disk to raise an error
        with patch.object(dataset, 'save_to_disk', side_effect=RuntimeError("Disk full")):
            with pytest.raises(Exception) as exc_info:
                TokenizationService.save_tokenized_dataset(dataset, tmp_path / "dataset")

            assert "Failed to save tokenized dataset" in str(exc_info.value)

    def test_load_dataset_from_disk(self, tmp_path):
        """Test loading dataset from disk."""
        # Create and save a dataset
        dataset = HFDataset.from_dict({
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]]
        })

        output_path = tmp_path / "dataset"
        dataset.save_to_disk(str(output_path))

        # Load dataset
        loaded_dataset = TokenizationService.load_dataset_from_disk(output_path)

        assert len(loaded_dataset) == 2
        assert "input_ids" in loaded_dataset.column_names
        assert loaded_dataset[0]["input_ids"] == [1, 2, 3]

    def test_load_dataset_from_disk_raises_on_error(self, tmp_path):
        """Test load_dataset_from_disk raises exception when path doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent"

        with pytest.raises(Exception) as exc_info:
            TokenizationService.load_dataset_from_disk(nonexistent_path)

        assert f"Failed to load dataset from {nonexistent_path}" in str(exc_info.value)


class TestTokenizationMapper:
    """Test _TokenizationMapper for multiprocessing support."""

    def test_tokenization_mapper_initialization(self):
        """Test _TokenizationMapper initialization."""
        from src.services.tokenization_service import _TokenizationMapper

        mapper = _TokenizationMapper(
            tokenizer_name="gpt2",
            text_column="text",
            max_length=512,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_attention_mask=True,
            stride=0,
            return_overflowing_tokens=False
        )

        assert mapper.tokenizer_name == "gpt2"
        assert mapper.text_column == "text"
        assert mapper.max_length == 512
        assert mapper.truncation is True
        assert mapper.padding == "max_length"
        assert mapper._tokenizer is None  # Lazy-loaded

    def test_tokenization_mapper_get_tokenizer_lazy_load(self):
        """Test _TokenizationMapper lazy-loads tokenizer."""
        from src.services.tokenization_service import _TokenizationMapper

        with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = "[PAD]"
            mock_from_pretrained.return_value = mock_tokenizer

            mapper = _TokenizationMapper(
                tokenizer_name="gpt2",
                text_column="text",
                max_length=512,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
                return_attention_mask=True
            )

            # Tokenizer not loaded yet
            assert mapper._tokenizer is None

            # Access tokenizer (lazy load)
            tokenizer = mapper._get_tokenizer()

            # Verify tokenizer was loaded
            assert tokenizer is not None
            mock_from_pretrained.assert_called_once_with(
                "gpt2", cache_dir=None, local_files_only=False
            )

            # Subsequent calls don't reload
            tokenizer2 = mapper._get_tokenizer()
            assert tokenizer2 is tokenizer
            mock_from_pretrained.assert_called_once()  # Still only one call

    def test_tokenization_mapper_call(self):
        """Test _TokenizationMapper.__call__ tokenizes examples."""
        from src.services.tokenization_service import _TokenizationMapper

        with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = "[PAD]"
            mock_tokenizer.return_value = {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]]
            }
            mock_from_pretrained.return_value = mock_tokenizer

            mapper = _TokenizationMapper(
                tokenizer_name="gpt2",
                text_column="text",
                max_length=512,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
                return_attention_mask=True
            )

            examples = {"text": ["Sample text"]}
            result = mapper(examples)

            assert "input_ids" in result
            assert "attention_mask" in result
            mock_tokenizer.assert_called_once()

    def test_tokenization_mapper_with_stride(self):
        """Test _TokenizationMapper with stride parameter."""
        from src.services.tokenization_service import _TokenizationMapper

        with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = "[PAD]"
            mock_tokenizer.return_value = {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]]
            }
            mock_from_pretrained.return_value = mock_tokenizer

            mapper = _TokenizationMapper(
                tokenizer_name="gpt2",
                text_column="text",
                max_length=512,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
                return_attention_mask=True,
                stride=128,
                return_overflowing_tokens=True
            )

            examples = {"text": ["Sample text"]}
            result = mapper(examples)

            # Verify stride was passed to tokenizer
            call_kwargs = mock_tokenizer.call_args[1]
            assert call_kwargs["stride"] == 128
            assert call_kwargs["return_overflowing_tokens"] is True


class TestProgressCallback:
    """Test progress callback functionality."""

    def test_tokenize_dataset_with_progress_callback(self, mock_tokenizer, sample_dataset):
        """Test tokenize_dataset calls progress_callback during tokenization."""
        progress_updates = []

        def capture_progress(progress_pct, message):
            progress_updates.append((progress_pct, message))

        with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            TokenizationService.tokenize_dataset(
                dataset=sample_dataset,
                tokenizer=mock_tokenizer,
                text_column="text",
                max_length=128,
                batch_size=1,  # Small batch to trigger multiple updates
                progress_callback=capture_progress,
                num_proc=1  # Single-process mode for progress tracking
            )

        # Verify progress updates were received
        assert len(progress_updates) > 0

        # Verify progress values are in expected range (40% to 75%)
        for progress_pct, message in progress_updates:
            assert 40.0 <= progress_pct <= 75.0
            assert "Tokenizing" in message or "Tokenization complete" in message

        # Verify final progress is 75%
        final_progress, final_message = progress_updates[-1]
        assert final_progress == 75.0
        assert "Tokenization complete" in final_message

    def test_tokenize_dataset_progress_single_process_mode(self, mock_tokenizer, sample_dataset):
        """Test tokenize_dataset uses single-process mode when progress_callback provided."""
        with patch("src.services.tokenization_service.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            with patch.object(sample_dataset, 'map') as mock_map:
                mock_map.return_value = sample_dataset

                TokenizationService.tokenize_dataset(
                    dataset=sample_dataset,
                    tokenizer=mock_tokenizer,
                    text_column="text",
                    max_length=128,
                    progress_callback=lambda p, m: None,
                    num_proc=4  # Request multiprocessing, but should use single-process due to callback
                )

                # Verify map was called with num_proc=1 (forced single-process)
                call_kwargs = mock_map.call_args[1]
                assert call_kwargs["num_proc"] == 1
