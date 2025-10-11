"""
Tokenization service for dataset processing.

This module provides services for tokenizing datasets using HuggingFace tokenizers.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from datasets import load_from_disk, Dataset as HFDataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TokenizationService:
    """Service for tokenizing datasets with schema-aware column detection."""

    @staticmethod
    def analyze_dataset_schema(dataset: HFDataset) -> Dict[str, Any]:
        """
        Analyze the schema of a dataset to identify text columns and structure.

        Args:
            dataset: HuggingFace dataset to analyze

        Returns:
            Dictionary with schema information:
                - text_columns: List of string-type columns
                - column_info: Dict mapping column names to their types
                - recommended_column: Best column to use for tokenization
                - is_multi_column: Whether dataset has multiple text columns
        """
        text_columns = []
        column_info = {}

        # Analyze each column's feature type
        for col_name, feature in dataset.features.items():
            dtype = feature.dtype if hasattr(feature, 'dtype') else 'unknown'
            column_info[col_name] = str(dtype)

            # Identify string/text columns
            if dtype == 'string':
                text_columns.append(col_name)

        # Determine recommended column based on common patterns
        recommended_column = None
        if 'text' in text_columns:
            # Prefer 'text' column if it exists (most common)
            recommended_column = 'text'
        elif 'content' in text_columns:
            # Second preference: 'content'
            recommended_column = 'content'
        elif 'chosen' in text_columns:
            # For RLHF datasets, prefer 'chosen' over 'rejected'
            recommended_column = 'chosen'
        elif text_columns:
            # Otherwise, use first available text column
            recommended_column = text_columns[0]

        return {
            'text_columns': text_columns,
            'column_info': column_info,
            'recommended_column': recommended_column,
            'is_multi_column': len(text_columns) > 1,
            'all_columns': list(dataset.column_names),
        }

    @staticmethod
    def load_tokenizer(tokenizer_name: str, use_fast: bool = True):
        """
        Load a HuggingFace tokenizer with proper configuration.

        Args:
            tokenizer_name: Name or path of tokenizer (e.g., 'gpt2', 'bert-base-uncased')
            use_fast: Whether to use fast tokenizer implementation

        Returns:
            Loaded tokenizer instance

        Raises:
            Exception: If tokenizer cannot be loaded
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=use_fast
            )

            # Ensure tokenizer has padding token (required for batched tokenization)
            if tokenizer.pad_token is None:
                # For GPT-2 and similar models, use eos_token as pad_token
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
                else:
                    # If no eos_token, add a new pad token
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    print(f"Added new pad_token: [PAD]")

            return tokenizer
        except Exception as e:
            raise Exception(f"Failed to load tokenizer '{tokenizer_name}': {str(e)}")

    @staticmethod
    def tokenize_dataset(
        dataset: HFDataset,
        tokenizer,
        text_column: str = "text",
        max_length: int = 512,
        stride: int = 0,
        truncation: bool = True,
        padding: str = "max_length",
        return_overflowing_tokens: bool = False,
        batch_size: int = 1000,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> HFDataset:
        """
        Tokenize a dataset using the provided tokenizer.

        Args:
            dataset: HuggingFace dataset to tokenize
            tokenizer: Tokenizer instance
            text_column: Name of column containing text
            max_length: Maximum sequence length
            stride: Sliding window stride for long sequences
            truncation: Whether to truncate sequences
            padding: Padding strategy ('max_length', 'longest', or False)
            return_overflowing_tokens: Whether to return overflow from sliding window
            batch_size: Batch size for tokenization
            progress_callback: Optional callback function(progress_pct, message) for progress updates

        Returns:
            Tokenized dataset with 'input_ids', 'attention_mask', etc.
        """
        # Track progress manually since dataset.map() doesn't provide callbacks
        total_samples = len(dataset)
        processed_samples = 0

        # Calculate total batches for more accurate progress reporting
        total_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
        current_batch = 0

        # Report progress every N batches (adjust based on dataset size)
        # For small datasets: report every batch
        # For medium datasets (10k+): report every 5 batches
        # For large datasets (100k+): report every 10 batches
        if total_samples < 10000:
            report_interval = max(1, total_batches // 20)  # Report ~20 times total
        elif total_samples < 100000:
            report_interval = max(5, total_batches // 15)  # Report ~15 times
        else:
            report_interval = max(10, total_batches // 10)  # Report ~10 times

        def tokenize_function(examples, indices):
            """Tokenize a batch of examples and report progress."""
            nonlocal processed_samples, current_batch

            kwargs = {
                "max_length": max_length,
                "truncation": truncation,
                "padding": padding,
            }

            if stride > 0:
                kwargs["stride"] = stride
                kwargs["return_overflowing_tokens"] = return_overflowing_tokens

            result = tokenizer(
                examples[text_column],
                **kwargs
            )

            # Update progress tracking
            current_batch += 1
            batch_size_actual = len(examples[text_column])
            processed_samples = min(processed_samples + batch_size_actual, total_samples)

            # Calculate progress percentage (40% to 75% range for tokenization)
            progress_pct = 40.0 + (processed_samples / total_samples) * 35.0  # 40% to 75%

            # Report progress at calculated intervals
            if progress_callback and (current_batch % report_interval == 0 or current_batch == total_batches):
                progress_callback(
                    progress_pct,
                    f"Tokenizing... {processed_samples:,}/{total_samples:,} samples (Batch {current_batch}/{total_batches})"
                )

            return result

        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            with_indices=True,  # Pass indices to track progress
            remove_columns=dataset.column_names,  # Remove original columns
            desc="Tokenizing dataset",
        )

        # Final progress update
        if progress_callback:
            progress_callback(75.0, f"Tokenization complete ({total_samples} samples)")

        return tokenized_dataset

    @staticmethod
    def calculate_statistics(tokenized_dataset: HFDataset) -> Dict[str, Any]:
        """
        Calculate statistics for a tokenized dataset.

        Args:
            tokenized_dataset: Tokenized HuggingFace dataset

        Returns:
            Dictionary with statistics:
                - num_tokens: Total number of tokens
                - num_samples: Number of samples
                - avg_seq_length: Average sequence length
                - min_seq_length: Minimum sequence length
                - max_seq_length: Maximum sequence length

        Raises:
            ValueError: If dataset is empty or no samples have input_ids
        """
        # Validate dataset is not empty
        if len(tokenized_dataset) == 0:
            raise ValueError(
                "Cannot calculate statistics for empty dataset. "
                "Dataset must contain at least one sample."
            )

        # Calculate sequence lengths
        seq_lengths = []
        total_tokens = 0
        samples_without_input_ids = 0
        total_samples = len(tokenized_dataset)

        for idx, example in enumerate(tokenized_dataset):
            if "input_ids" in example:
                seq_len = len(example["input_ids"])
                seq_lengths.append(seq_len)
                total_tokens += seq_len
            else:
                samples_without_input_ids += 1
                # Log first few missing samples for debugging
                if samples_without_input_ids <= 3:
                    logger.warning(
                        f"Sample {idx} missing 'input_ids' field. "
                        f"Available keys: {list(example.keys())}"
                    )

        # Check if all samples are missing input_ids
        if not seq_lengths:
            raise ValueError(
                f"Cannot calculate statistics: All {total_samples} samples are missing 'input_ids' field. "
                f"This indicates the tokenization process failed. "
                f"Sample keys: {list(tokenized_dataset[0].keys()) if total_samples > 0 else 'N/A'}"
            )

        # Warn about partial failures
        if samples_without_input_ids > 0:
            logger.warning(
                f"Found {samples_without_input_ids}/{total_samples} samples "
                f"({samples_without_input_ids/total_samples*100:.1f}%) without 'input_ids'. "
                f"Statistics calculated from {len(seq_lengths)} valid samples only."
            )

        return {
            "num_tokens": total_tokens,
            "num_samples": len(seq_lengths),  # Only count valid samples
            "avg_seq_length": total_tokens / len(seq_lengths),
            "min_seq_length": min(seq_lengths),
            "max_seq_length": max(seq_lengths),
        }

    @staticmethod
    def save_tokenized_dataset(
        tokenized_dataset: HFDataset,
        output_path: str | Path,
    ) -> None:
        """
        Save tokenized dataset to disk in Arrow format.

        Args:
            tokenized_dataset: Tokenized dataset to save
            output_path: Path to save dataset

        Raises:
            Exception: If saving fails
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tokenized_dataset.save_to_disk(str(output_path))
        except Exception as e:
            raise Exception(f"Failed to save tokenized dataset: {str(e)}")

    @staticmethod
    def load_dataset_from_disk(dataset_path: str | Path) -> HFDataset:
        """
        Load a dataset from disk.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Loaded HuggingFace dataset

        Raises:
            Exception: If loading fails
        """
        try:
            return load_from_disk(str(dataset_path))
        except Exception as e:
            raise Exception(f"Failed to load dataset from {dataset_path}: {str(e)}")
