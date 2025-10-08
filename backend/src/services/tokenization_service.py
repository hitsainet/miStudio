"""
Tokenization service for dataset processing.

This module provides services for tokenizing datasets using HuggingFace tokenizers.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_from_disk, Dataset as HFDataset
from transformers import AutoTokenizer


class TokenizationService:
    """Service for tokenizing datasets."""

    @staticmethod
    def load_tokenizer(tokenizer_name: str, use_fast: bool = True):
        """
        Load a HuggingFace tokenizer.

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

        Returns:
            Tokenized dataset with 'input_ids', 'attention_mask', etc.
        """
        def tokenize_function(examples):
            """Tokenize a batch of examples."""
            kwargs = {
                "max_length": max_length,
                "truncation": truncation,
                "padding": padding,
            }

            if stride > 0:
                kwargs["stride"] = stride
                kwargs["return_overflowing_tokens"] = return_overflowing_tokens

            return tokenizer(
                examples[text_column],
                **kwargs
            )

        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,  # Remove original columns
            desc="Tokenizing dataset",
        )

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
        """
        if len(tokenized_dataset) == 0:
            return {
                "num_tokens": 0,
                "num_samples": 0,
                "avg_seq_length": 0.0,
                "min_seq_length": 0,
                "max_seq_length": 0,
            }

        # Calculate sequence lengths
        seq_lengths = []
        total_tokens = 0

        for example in tokenized_dataset:
            if "input_ids" in example:
                seq_len = len(example["input_ids"])
                seq_lengths.append(seq_len)
                total_tokens += seq_len

        if not seq_lengths:
            return {
                "num_tokens": 0,
                "num_samples": len(tokenized_dataset),
                "avg_seq_length": 0.0,
                "min_seq_length": 0,
                "max_seq_length": 0,
            }

        return {
            "num_tokens": total_tokens,
            "num_samples": len(tokenized_dataset),
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
