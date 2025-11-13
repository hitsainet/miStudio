"""
Tokenization service for dataset processing.

This module provides services for tokenizing datasets using HuggingFace tokenizers.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
from datasets import load_from_disk, Dataset as HFDataset
from transformers import AutoTokenizer

from ..core.config import settings
from ..utils.text_cleaning import TextCleaner, get_standard_cleaner

logger = logging.getLogger(__name__)


class _TokenizationMapper:
    """
    Picklable tokenization mapper for multiprocessing support.

    This class encapsulates the tokenization logic in a way that can be
    pickled for use with HuggingFace's multiprocessing. The tokenizer and
    text cleaner are loaded lazily in each worker process to avoid pickling issues.
    """

    def __init__(
        self,
        tokenizer_name: str,
        text_column: str,
        max_length: int,
        truncation: bool,
        padding: str,
        add_special_tokens: bool,
        return_attention_mask: bool,
        stride: int = 0,
        return_overflowing_tokens: bool = False,
        enable_cleaning: bool = True,
        enable_filtering: bool = False,
        filter_mode: str = "conservative",
        junk_ratio_threshold: float = 0.7,
        remove_all_punctuation: bool = False,
        custom_filter_chars: Optional[str] = None,
    ):
        """Initialize the tokenization mapper with parameters."""
        self.tokenizer_name = tokenizer_name
        self.text_column = text_column
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.return_attention_mask = return_attention_mask
        self.stride = stride
        self.return_overflowing_tokens = return_overflowing_tokens
        self.enable_cleaning = enable_cleaning
        self.enable_filtering = enable_filtering
        self.filter_mode = filter_mode
        self.junk_ratio_threshold = junk_ratio_threshold
        self.remove_all_punctuation = remove_all_punctuation
        self.custom_filter_chars = custom_filter_chars
        self._tokenizer = None  # Lazy-loaded in worker process
        self._text_cleaner = None  # Lazy-loaded in worker process
        self._token_filter = None  # Lazy-loaded in worker process

    def _get_tokenizer(self):
        """Lazy-load tokenizer in worker process (avoids pickling issues)."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            # Ensure tokenizer has padding token
            if self._tokenizer.pad_token is None:
                if self._tokenizer.eos_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                else:
                    self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return self._tokenizer

    def _get_text_cleaner(self):
        """Lazy-load text cleaner in worker process (avoids pickling issues)."""
        if self._text_cleaner is None:
            self._text_cleaner = get_standard_cleaner()
        return self._text_cleaner

    def _get_token_filter(self):
        """Lazy-load token filter in worker process (avoids pickling issues)."""
        if self._token_filter is None:
            from ..utils.token_filter import TokenFilter, FilterMode
            mode = FilterMode[self.filter_mode.upper()]
            self._token_filter = TokenFilter(
                mode=mode,
                remove_all_punctuation=self.remove_all_punctuation,
                custom_filter_chars=self.custom_filter_chars
            )
        return self._token_filter

    def __call__(self, examples):
        """
        Tokenize a batch of examples with optional text cleaning.

        Note: Progress tracking is not supported in multiprocessing mode
        because shared state between processes is complex. Progress tracking
        only works when num_proc=1 (single process mode).
        """
        tokenizer = self._get_tokenizer()

        # Apply text cleaning if enabled
        if self.enable_cleaning:
            text_cleaner = self._get_text_cleaner()
            texts = examples[self.text_column]
            cleaned_texts = []
            for text in texts:
                cleaned = text_cleaner.clean(text)
                # Keep empty string if cleaning returns None (better than dropping samples)
                cleaned_texts.append(cleaned if cleaned is not None else "")
            texts_to_tokenize = cleaned_texts
        else:
            texts_to_tokenize = examples[self.text_column]

        kwargs = {
            "max_length": self.max_length,
            "truncation": self.truncation,
            "padding": self.padding,
            "add_special_tokens": self.add_special_tokens,
            "return_attention_mask": self.return_attention_mask,
        }

        if self.stride > 0:
            kwargs["stride"] = self.stride
            kwargs["return_overflowing_tokens"] = self.return_overflowing_tokens

        result = tokenizer(
            texts_to_tokenize,
            **kwargs
        )

        # Filter junk samples if enabled (Stage 1: Tokenization Filter)
        if self.enable_filtering:
            token_filter = self._get_token_filter()
            tokenizer_obj = self._get_tokenizer()

            original_count = len(result['input_ids'])

            # Filter out samples with too many junk tokens
            filtered_indices = []
            for idx, input_ids in enumerate(result['input_ids']):
                if not token_filter.is_junk_sequence(
                    input_ids,
                    tokenizer_obj,
                    self.junk_ratio_threshold
                ):
                    filtered_indices.append(idx)

            kept_count = len(filtered_indices)
            filtered_count = original_count - kept_count
            filter_pct = (filtered_count / original_count * 100) if original_count > 0 else 0

            # Log filtering statistics
            logger.info(
                f"Tokenization filter: {kept_count} samples kept, "
                f"{filtered_count} junk samples filtered ({filter_pct:.1f}% filtered)"
            )

            # Keep only non-junk samples
            if filtered_indices:
                result = {
                    key: [value[i] for i in filtered_indices]
                    for key, value in result.items()
                }
            else:
                # All samples filtered - return empty result
                result = {key: [] for key in result.keys()}

        return result


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
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        batch_size: int = 1000,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        num_proc: Optional[int] = None,
        text_cleaner: Optional[TextCleaner] = None,
        enable_cleaning: bool = True,
        enable_filtering: bool = False,
        filter_mode: str = "conservative",
        junk_ratio_threshold: float = 0.7,
        remove_all_punctuation: bool = False,
        custom_filter_chars: Optional[str] = None,
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
            add_special_tokens: Add special tokens (BOS, EOS, PAD, etc.)
            return_attention_mask: Return attention mask
            batch_size: Batch size for tokenization
            progress_callback: Optional callback function(progress_pct, message) for progress updates
                              Note: Progress tracking only works in single-process mode (num_proc=1)
            num_proc: Number of processes for parallel processing (None = auto, 1 = single-process)
            text_cleaner: Optional TextCleaner instance for preprocessing text
            enable_cleaning: Whether to enable text cleaning (default: True)

        Returns:
            Tokenized dataset with 'input_ids', 'attention_mask', etc.
        """
        total_samples = len(dataset)

        # Initialize text cleaner if enabled
        if enable_cleaning and text_cleaner is None:
            text_cleaner = get_standard_cleaner()
            logger.info("Using standard text cleaner for preprocessing")
        elif enable_cleaning:
            logger.info(f"Using provided text cleaner for preprocessing")
        else:
            logger.info("Text cleaning disabled")

        # Determine which columns to remove (keep 'split' if it exists)
        columns_to_remove = [col for col in dataset.column_names if col != "split"]

        # Auto-detect number of processes if not specified
        if num_proc is None:
            import os
            num_proc = max(1, os.cpu_count() // 2)  # Use half of available CPU cores

        # Choose between single-process and multi-process modes
        # Single-process when: num_proc==1 OR progress_callback requested
        # Multi-process when: num_proc>1 AND no progress_callback
        if num_proc == 1 or progress_callback:
            # Single-process mode: Use closure with progress tracking
            processed_samples = 0
            total_batches = (total_samples + batch_size - 1) // batch_size
            current_batch = 0

            # Calculate progress reporting interval
            if total_samples < 10000:
                report_interval = max(1, total_batches // 20)
            elif total_samples < 100000:
                report_interval = max(5, total_batches // 15)
            else:
                report_interval = max(10, total_batches // 10)

            def tokenize_function(examples, indices):
                """Tokenize a batch of examples and report progress."""
                nonlocal processed_samples, current_batch

                # Clean text if enabled
                if enable_cleaning and text_cleaner:
                    texts = examples[text_column]
                    cleaned_texts = []
                    for text in texts:
                        cleaned = text_cleaner.clean(text)
                        # If text is filtered out (too short), keep original to maintain batch size
                        # The short texts will just produce fewer meaningful tokens
                        cleaned_texts.append(cleaned if cleaned is not None else "")
                    texts_to_tokenize = cleaned_texts
                else:
                    texts_to_tokenize = examples[text_column]

                kwargs = {
                    "max_length": max_length,
                    "truncation": truncation,
                    "padding": padding,
                    "add_special_tokens": add_special_tokens,
                    "return_attention_mask": return_attention_mask,
                }

                if stride > 0:
                    kwargs["stride"] = stride
                    kwargs["return_overflowing_tokens"] = return_overflowing_tokens

                result = tokenizer(
                    texts_to_tokenize,
                    **kwargs
                )

                # Update progress tracking
                current_batch += 1
                batch_size_actual = len(examples[text_column])
                processed_samples = min(processed_samples + batch_size_actual, total_samples)

                # Calculate progress percentage (40% to 75% range for tokenization)
                progress_pct = 40.0 + (processed_samples / total_samples) * 35.0

                # Report progress at calculated intervals
                if progress_callback and (current_batch % report_interval == 0 or current_batch == total_batches):
                    progress_callback(
                        progress_pct,
                        f"Tokenizing... {processed_samples:,}/{total_samples:,} samples (Batch {current_batch}/{total_batches})"
                    )

                return result

            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=batch_size,
                with_indices=True,
                remove_columns=columns_to_remove,
                num_proc=1,
                desc="Tokenizing dataset",
            )

            # Final progress update
            if progress_callback:
                progress_callback(75.0, f"Tokenization complete ({total_samples} samples)")

        else:
            # Multi-process mode: Use picklable mapper (no progress tracking)
            # Extract tokenizer name from tokenizer object (for lazy loading in workers)
            tokenizer_name = getattr(tokenizer, 'name_or_path', None)
            if not tokenizer_name:
                # Fallback: Try to get from init_kwargs or raise error
                tokenizer_name = getattr(tokenizer, 'init_kwargs', {}).get('name_or_path')
                if not tokenizer_name:
                    raise ValueError(
                        "Cannot determine tokenizer name for multiprocessing. "
                        "Please pass num_proc=1 to use single-process mode."
                    )

            mapper = _TokenizationMapper(
                tokenizer_name=tokenizer_name,
                text_column=text_column,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                add_special_tokens=add_special_tokens,
                return_attention_mask=return_attention_mask,
                stride=stride,
                return_overflowing_tokens=return_overflowing_tokens,
                enable_cleaning=enable_cleaning,
                enable_filtering=enable_filtering,
                filter_mode=filter_mode,
                junk_ratio_threshold=junk_ratio_threshold,
                remove_all_punctuation=remove_all_punctuation,
                custom_filter_chars=custom_filter_chars,
            )

            logger.info(
                f"Using multiprocessing mode with {num_proc} processes. "
                f"Text cleaning: {'enabled' if enable_cleaning else 'disabled'}, "
                f"Token filtering: {'enabled' if enable_filtering else 'disabled'}"
                + (f" (mode={filter_mode}, threshold={junk_ratio_threshold})" if enable_filtering else "")
            )

            tokenized_dataset = dataset.map(
                mapper,
                batched=True,
                batch_size=batch_size,
                remove_columns=columns_to_remove,
                num_proc=num_proc,
                desc="Tokenizing dataset",
            )

            # Single progress update at the end (for multi-process mode)
            if progress_callback:
                progress_callback(75.0, f"Tokenization complete ({total_samples} samples)")

        return tokenized_dataset

    @staticmethod
    def calculate_statistics(
        tokenized_dataset: HFDataset,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Calculate statistics for a tokenized dataset using batched processing.

        Processes dataset in 10K sample batches to avoid OOM on large datasets.
        Memory-efficient: uses ~200MB per batch instead of loading all samples.

        Args:
            tokenized_dataset: Tokenized HuggingFace dataset
            progress_callback: Optional callback function called with progress percent (0-100)

        Returns:
            Dictionary with statistics:
                - num_tokens: Total number of tokens
                - num_samples: Number of samples
                - avg_seq_length: Average sequence length
                - min_seq_length: Minimum sequence length
                - max_seq_length: Maximum sequence length
                - median_seq_length: Median sequence length
                - vocab_size: Number of unique tokens (vocabulary size)
                - length_distribution: Dictionary mapping length ranges to counts

        Raises:
            ValueError: If dataset is empty or no samples have input_ids
        """
        # Validate dataset is not empty
        if len(tokenized_dataset) == 0:
            raise ValueError(
                "Cannot calculate statistics for empty dataset. "
                "Dataset must contain at least one sample."
            )

        # Process dataset in batches to avoid loading all into memory
        # Critical for large datasets (8M+ samples) to prevent OOM
        try:
            batch_size = 10000  # Process 10K samples at a time (~200MB per batch)
            total_samples = len(tokenized_dataset)

            # Initialize accumulators
            seq_lengths_list = []
            unique_tokens = set()

            # Process in batches
            print(f"Calculating statistics for {total_samples:,} samples in batches of {batch_size:,}...")
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                batch = tokenized_dataset[start_idx:end_idx]["input_ids"]

                # Calculate lengths for this batch
                batch_lengths = [len(ids) for ids in batch]
                seq_lengths_list.extend(batch_lengths)

                # Update unique tokens for this batch
                for ids in batch:
                    unique_tokens.update(ids)

                # Progress callback and indicator
                pct = (end_idx / total_samples) * 100
                if progress_callback:
                    progress_callback(pct)

                # Log every 10th batch
                if (start_idx // batch_size) % 10 == 0:
                    print(f"  Statistics progress: {pct:.1f}% ({end_idx:,}/{total_samples:,})")

            # Convert to numpy array for statistical calculations
            seq_lengths = np.array(seq_lengths_list)

            # Calculate median sequence length
            median_seq_length = float(np.median(seq_lengths))

            # Vocabulary size from accumulated unique tokens
            vocab_size = len(unique_tokens)
            print(f"Statistics complete: {len(seq_lengths):,} samples, vocab size: {vocab_size:,}")

            # Calculate length distribution with bucketing
            # Buckets: 0-100, 100-200, 200-400, 400-600, 600-800, 800-1000, 1000+
            length_distribution = {
                "0-100": 0,
                "100-200": 0,
                "200-400": 0,
                "400-600": 0,
                "600-800": 0,
                "800-1000": 0,
                "1000+": 0,
            }

            for length in seq_lengths:
                if length < 100:
                    length_distribution["0-100"] += 1
                elif length < 200:
                    length_distribution["100-200"] += 1
                elif length < 400:
                    length_distribution["200-400"] += 1
                elif length < 600:
                    length_distribution["400-600"] += 1
                elif length < 800:
                    length_distribution["600-800"] += 1
                elif length < 1000:
                    length_distribution["800-1000"] += 1
                else:
                    length_distribution["1000+"] += 1

            # Calculate split distribution if 'split' column exists
            split_distribution = None
            try:
                if "split" in tokenized_dataset.column_names:
                    splits = tokenized_dataset["split"]
                    split_counts = {}
                    for split_name in splits:
                        split_counts[split_name] = split_counts.get(split_name, 0) + 1
                    split_distribution = split_counts
            except (KeyError, AttributeError):
                # If split column doesn't exist or can't be accessed, skip it
                pass

            stats = {
                "num_tokens": int(seq_lengths.sum()),
                "num_samples": len(tokenized_dataset),
                "avg_seq_length": float(seq_lengths.mean()),
                "min_seq_length": int(seq_lengths.min()),
                "max_seq_length": int(seq_lengths.max()),
                "median_seq_length": median_seq_length,
                "vocab_size": vocab_size,
                "length_distribution": length_distribution,
            }

            # Only add split_distribution if it was calculated
            if split_distribution is not None:
                stats["split_distribution"] = split_distribution

            return stats
        except KeyError:
            raise ValueError(
                f"Cannot calculate statistics: Dataset missing 'input_ids' field. "
                f"This indicates the tokenization process failed. "
                f"Available keys: {list(tokenized_dataset.features.keys())}"
            )

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
        Load a dataset from disk with intelligent format detection.

        Handles three formats:
        1. save_to_disk format (preferred): Flat structure with dataset_info.json at root
        2. HuggingFace cache format: Triple underscore path name
        3. HuggingFace nested cache: default/0.0.0/hash/ structure with Arrow files

        Args:
            dataset_path: Path to dataset directory (relative or absolute)

        Returns:
            Loaded HuggingFace dataset

        Raises:
            Exception: If loading fails
        """
        import json
        import pyarrow as pa
        from datasets import Dataset

        path = Path(dataset_path)

        # Convert relative paths to absolute using settings.data_dir as base
        if not path.is_absolute():
            # If path starts with "data/", replace it with settings.data_dir
            path_str = str(path)
            if path_str.startswith("data/"):
                # Strip "data/" prefix and join with settings.data_dir
                relative_part = path_str[5:]  # Remove "data/" prefix
                path = settings.data_dir / relative_part
            else:
                # Resolve relative to settings.data_dir
                path = settings.data_dir / path

        logger.info(f"Resolving dataset path: {path}")

        # Strategy 1: Try direct load_from_disk (save_to_disk format)
        if path.exists():
            try:
                logger.info(f"Attempting load_from_disk: {path}")
                return load_from_disk(str(path))
            except Exception as e:
                logger.warning(f"load_from_disk failed on {path}: {e}")

        # Strategy 2: Try HuggingFace cache path (single → triple underscore)
        # Example: vietgpt_openwebtext_en → vietgpt___openwebtext_en
        hf_cache_name = path.name.replace('_', '___', 1)
        hf_cache_path = path.parent / hf_cache_name

        if hf_cache_path.exists():
            try:
                logger.info(f"Attempting load_from_disk on HF cache path: {hf_cache_path}")
                return load_from_disk(str(hf_cache_path))
            except Exception as e:
                logger.warning(f"load_from_disk failed on HF cache path: {e}")

                # Strategy 3: HF nested cache format - use Dataset.from_file for memory efficiency
                # Structure: vietgpt___openwebtext_en/default/0.0.0/hash/*.arrow
                logger.info(f"Attempting to load from HF nested cache structure (memory-efficient)")

                try:
                    # Find dataset_info.json in nested structure
                    nested_pattern = list(hf_cache_path.glob("*/*/*/dataset_info.json"))

                    if not nested_pattern:
                        raise Exception(f"No dataset_info.json found in nested structure: {hf_cache_path}")

                    nested_dir = nested_pattern[0].parent
                    logger.info(f"Found HF cache nested directory: {nested_dir}")

                    # Get all Arrow files
                    arrow_files = sorted(nested_dir.glob("*.arrow"))

                    if not arrow_files:
                        raise Exception(f"No Arrow files found in {nested_dir}")

                    logger.info(f"Loading dataset from {len(arrow_files)} Arrow files (using lazy loading)")

                    # Use Dataset.from_file which is memory-efficient
                    # It uses memory mapping instead of loading everything into RAM
                    if len(arrow_files) == 1:
                        # Single file - easy
                        dataset = Dataset.from_file(str(arrow_files[0]))
                    else:
                        # Multiple files - concatenate using IterableDataset approach
                        # This is MUCH more memory efficient than loading all into RAM
                        from datasets import concatenate_datasets

                        datasets_list = []
                        for i, arrow_file in enumerate(arrow_files):
                            if i % 10 == 0:
                                logger.info(f"Loading Arrow file {i+1}/{len(arrow_files)}")
                            ds = Dataset.from_file(str(arrow_file))
                            datasets_list.append(ds)

                        logger.info(f"Concatenating {len(datasets_list)} dataset shards...")
                        dataset = concatenate_datasets(datasets_list)

                    logger.info(f"Successfully loaded dataset from HF nested cache: {len(dataset)} samples")
                    return dataset

                except Exception as nested_error:
                    logger.error(f"Failed to load from HF nested cache: {nested_error}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Continue to final error

        # If all strategies failed
        raise Exception(
            f"Failed to load dataset from {dataset_path}. "
            f"Tried: {path}, {hf_cache_path}, and nested HF cache format. "
            f"None of these paths exist or contain valid dataset format."
        )
