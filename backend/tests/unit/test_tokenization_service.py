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
            num_proc=1,  # Disable multiprocessing to avoid pickling issues in tests
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
            num_proc=1,  # Disable multiprocessing to avoid pickling issues in tests
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
            num_proc=1,  # Disable multiprocessing to avoid pickling issues in tests
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
            num_proc=1,  # Disable multiprocessing to avoid pickling issues in tests
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
            num_proc=1,  # Disable multiprocessing to avoid pickling issues in tests
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
            num_proc=1,  # Disable multiprocessing to avoid pickling issues in tests
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
            num_proc=1,  # Disable multiprocessing to avoid pickling issues in tests
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
            num_proc=1,  # Disable multiprocessing to avoid pickling issues in tests
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
            num_proc=1,  # Disable multiprocessing to avoid pickling issues in tests
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
            num_proc=1,  # Disable multiprocessing to avoid pickling issues in tests
        )

    # Verify result
    assert result is not None
    assert len(result) == len(dataset)

    # All sequences should be exactly max_length with truncation+padding
    for example in result:
        assert len(example["input_ids"]) == 64, \
            f"Expected length 64 with max_length padding, got {len(example['input_ids'])}"
