"""
Tokenization service for dataset processing.

This module provides services for tokenizing datasets using HuggingFace tokenizers.
It includes intelligent schema detection for various dataset formats including
conversation datasets (OpenAI, ShareGPT, LMSYS, etc.).
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
from datasets import load_from_disk, Dataset as HFDataset, Sequence, Value
from transformers import AutoTokenizer

from ..core.config import settings
from ..utils.text_cleaning import TextCleaner, get_standard_cleaner
from ..utils.conversation_formats import (
    ConversationFormat,
    ConversationColumnInfo,
    SchemaAnalysisResult,
    detect_conversation_format,
    analyze_column_for_conversation,
    create_conversation_preprocessor,
    get_format_recommendations,
)

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
        enable_cleaning: bool = False,
        enable_filtering: bool = False,
        filter_mode: str = "conservative",
        junk_ratio_threshold: float = 0.7,
        remove_all_punctuation: bool = False,
        custom_filter_chars: Optional[str] = None,
        cache_dir: Optional[str] = None,
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
        self.cache_dir = cache_dir  # Local cache for tokenizer files
        self._tokenizer = None  # Lazy-loaded in worker process
        self._text_cleaner = None  # Lazy-loaded in worker process
        self._token_filter = None  # Lazy-loaded in worker process

    def _get_tokenizer(self):
        """Lazy-load tokenizer in worker process (avoids pickling issues)."""
        if self._tokenizer is None:
            # Use local cache if provided (for gated models that are already downloaded)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                cache_dir=self.cache_dir,
                local_files_only=self.cache_dir is not None,
            )
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

        This method detects both simple text columns and complex conversation
        formats (OpenAI, ShareGPT, LMSYS, etc.).

        Args:
            dataset: HuggingFace dataset to analyze

        Returns:
            Dictionary with schema information:
                - text_columns: List of string-type columns
                - column_info: Dict mapping column names to their types
                - recommended_column: Best column to use for tokenization
                - is_multi_column: Whether dataset has multiple text columns
                - conversation_columns: List of detected conversation column info
                - requires_preprocessing: Whether preprocessing is needed
                - preprocessing_config: Suggested preprocessing configuration
                - warnings: List of warning messages
                - suggestions: List of user-facing suggestions
        """
        text_columns = []
        list_columns = []
        column_info = {}
        conversation_columns = []
        warnings = []
        suggestions = []

        # Analyze each column's feature type
        for col_name, feature in dataset.features.items():
            # Get the dtype - handle different feature types
            if hasattr(feature, 'dtype'):
                dtype = feature.dtype
                column_info[col_name] = str(dtype)

                # Identify string/text columns
                if dtype == 'string':
                    text_columns.append(col_name)

            elif isinstance(feature, Sequence):
                # This is a list/sequence column - could be conversation data
                column_info[col_name] = f"Sequence({feature.feature})"
                list_columns.append(col_name)

            elif str(type(feature).__name__) == 'list':
                # Alternative way lists are represented
                column_info[col_name] = "list"
                list_columns.append(col_name)

            else:
                # Unknown type - record it
                column_info[col_name] = str(type(feature).__name__)

        # Analyze list columns for conversation formats
        logger.info(f"Found {len(list_columns)} list/sequence columns to analyze for conversation formats")
        for col_name in list_columns:
            try:
                conv_info = analyze_column_for_conversation(dataset, col_name, num_samples=10)
                if conv_info.format != ConversationFormat.NOT_CONVERSATION:
                    conversation_columns.append(conv_info)
                    logger.info(
                        f"Detected conversation format in '{col_name}': {conv_info.format.value} "
                        f"(confidence: {conv_info.confidence:.0%})"
                    )
            except Exception as e:
                logger.warning(f"Error analyzing column {col_name} for conversation format: {e}")

        # Determine recommended column based on priority:
        # 1. High-confidence conversation columns (need preprocessing)
        # 2. Standard text columns with common names
        # 3. Any text column
        # 4. Any conversation column (even low confidence)

        recommended_column = None
        requires_preprocessing = False
        preprocessing_config = None

        # Priority 1: High-confidence conversation columns
        high_confidence_convs = [c for c in conversation_columns if c.confidence >= 0.7]
        if high_confidence_convs:
            # Prefer columns named 'conversation', 'messages', 'chat'
            priority_names = ['conversation', 'messages', 'chat', 'dialog', 'dialogue']
            best_conv = None
            for name in priority_names:
                for conv in high_confidence_convs:
                    if conv.column_name.lower() == name:
                        best_conv = conv
                        break
                if best_conv:
                    break

            if not best_conv:
                # Use highest confidence conversation column
                best_conv = max(high_confidence_convs, key=lambda x: x.confidence)

            recommended_column = best_conv.column_name
            requires_preprocessing = True
            preprocessing_config = {
                "type": "conversation",
                "source_column": best_conv.column_name,
                "format": best_conv.format.value,
                "text_key": best_conv.text_key,
                "role_key": best_conv.role_key,
                "include_roles": True,
                "output_column": "text",
            }

            suggestions.append(
                f"Detected {best_conv.format.value} conversation format in '{best_conv.column_name}'. "
                f"Text will be automatically extracted from conversations."
            )
            if best_conv.sample_roles:
                suggestions.append(f"Detected roles: {', '.join(best_conv.sample_roles)}")
            if best_conv.avg_turns > 0:
                suggestions.append(f"Average conversation length: {best_conv.avg_turns:.1f} turns")

        # Priority 2 & 3: Standard text columns
        elif text_columns:
            if 'text' in text_columns:
                recommended_column = 'text'
            elif 'content' in text_columns:
                recommended_column = 'content'
            elif 'chosen' in text_columns:
                recommended_column = 'chosen'
                suggestions.append(
                    "Detected RLHF-style dataset with 'chosen' column. "
                    "Using 'chosen' responses for tokenization."
                )
            else:
                recommended_column = text_columns[0]

        # Priority 4: Any conversation column (fallback)
        elif conversation_columns:
            best_conv = max(conversation_columns, key=lambda x: x.confidence)
            recommended_column = best_conv.column_name
            requires_preprocessing = True
            preprocessing_config = {
                "type": "conversation",
                "source_column": best_conv.column_name,
                "format": best_conv.format.value,
                "text_key": best_conv.text_key,
                "role_key": best_conv.role_key,
                "include_roles": True,
                "output_column": "text",
            }
            warnings.append(
                f"Using conversation column '{best_conv.column_name}' with low confidence "
                f"({best_conv.confidence:.0%}). Please verify the extraction results."
            )

        # Generate warnings if no suitable columns found
        if not recommended_column:
            warnings.append(
                "No suitable text or conversation columns detected! "
                f"Available columns: {', '.join(dataset.column_names)}"
            )
            # Check if there are list columns that weren't recognized
            if list_columns:
                warnings.append(
                    f"Found list columns {list_columns} but couldn't detect conversation format. "
                    "Please manually inspect the data structure."
                )

        # Warn about potential ID columns being selected
        if recommended_column and not requires_preprocessing:
            # Check if the recommended column might be an ID column
            id_indicators = ['id', 'uuid', 'guid', 'key', 'idx', 'index']
            if any(ind in recommended_column.lower() for ind in id_indicators):
                warnings.append(
                    f"WARNING: Selected column '{recommended_column}' may contain IDs rather than text content. "
                    "Please verify this is the correct column for tokenization."
                )

        # Convert conversation column info to serializable format
        conversation_columns_serialized = []
        for conv in conversation_columns:
            conversation_columns_serialized.append({
                "column_name": conv.column_name,
                "format": conv.format.value,
                "confidence": conv.confidence,
                "sample_roles": conv.sample_roles,
                "avg_turns": conv.avg_turns,
                "text_key": conv.text_key,
                "role_key": conv.role_key,
                "description": conv.description,
                "extraction_hint": conv.extraction_hint,
            })

        return {
            'text_columns': text_columns,
            'list_columns': list_columns,
            'column_info': column_info,
            'recommended_column': recommended_column,
            'is_multi_column': len(text_columns) > 1,
            'all_columns': list(dataset.column_names),
            'conversation_columns': conversation_columns_serialized,
            'requires_preprocessing': requires_preprocessing,
            'preprocessing_config': preprocessing_config,
            'warnings': warnings,
            'suggestions': suggestions,
        }

    @staticmethod
    def preprocess_conversation_dataset(
        dataset: HFDataset,
        preprocessing_config: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None,
        batch_size: int = 1000,
        num_proc: Optional[int] = None,
        tokenizer: Any = None,
        chat_format: str = "auto",
    ) -> HFDataset:
        """
        Preprocess a conversation dataset by extracting text from conversations.

        This method flattens conversation data into a single text column that can
        then be tokenized normally.

        Args:
            dataset: HuggingFace dataset with conversation column
            preprocessing_config: Configuration from analyze_dataset_schema
            progress_callback: Optional callback for progress updates
            batch_size: Batch size for processing
            num_proc: Number of processes (None = auto)

        Returns:
            Preprocessed dataset with 'text' column containing flattened conversations
        """
        source_column = preprocessing_config.get("source_column")
        format_str = preprocessing_config.get("format", "openai")
        text_key = preprocessing_config.get("text_key", "content")
        role_key = preprocessing_config.get("role_key", "role")
        include_roles = preprocessing_config.get("include_roles", True)
        output_column = preprocessing_config.get("output_column", "text")

        # Map format string to enum
        format_mapping = {
            "openai": ConversationFormat.OPENAI,
            "sharegpt": ConversationFormat.SHAREGPT,
            "simple_list": ConversationFormat.SIMPLE_LIST,
        }
        conv_format = format_mapping.get(format_str, ConversationFormat.OPENAI)

        logger.info(
            f"Preprocessing conversation dataset: column='{source_column}', "
            f"format={conv_format.value}, include_roles={include_roles}"
        )

        if progress_callback:
            progress_callback(5.0, f"Preprocessing conversations from '{source_column}'...")

        # Default role mappings for normalization
        role_mapping = {
            "human": "user",
            "gpt": "assistant",
            "bot": "assistant",
            "ai": "assistant",
            "model": "assistant",
        }

        # THE MODEL'S OWN TURN DELIMITERS, when we have a tokenizer to ask.
        #
        # This used to render `<|user|>` / `<|assistant|>` unconditionally. Those
        # strings are in no vocabulary here, so they tokenize as literal
        # characters and the SAE learns features for `<`, `|`, `user` rather than
        # for turn structure — on an estate where every model is an INSTRUCT
        # model, i.e. where the chat scaffolding is the point.
        use_real_template = (
            tokenizer is not None
            and chat_format in ("auto", "chat_template")
            and bool(getattr(tokenizer, "chat_template", None))
        )
        if chat_format == "chat_template" and tokenizer is not None and not use_real_template:
            raise ValueError(
                "chat_format='chat_template' was requested but this tokenizer "
                "defines no chat_template. Use 'auto' to fall back to plain text."
            )
        logger.info(
            "Conversation rendering: %s",
            "tokenizer chat_template" if use_real_template
            else f"plain text (chat_format={chat_format}, tokenizer={'yes' if tokenizer else 'none'})",
        )

        # Legacy pseudo-marker template, kept ONLY for the no-tokenizer path so
        # existing callers do not change shape. It is not the default any more.
        role_template = "<|{role}|>\n{content}\n"

        def extract_text_batch(examples):
            """Extract text from a batch of conversations."""
            from ..utils.conversation_formats import (
                extract_messages_from_conversation,
                extract_text_from_conversation,
            )

            texts = []
            conversations = examples[source_column]

            for conv in conversations:
                if tokenizer is not None:
                    # ONE implementation, shared with the unit tests. Rendering
                    # this inline as well would be two copies of the same
                    # decision that can drift apart — the shape of defect this
                    # repo has recorded as "fixed one representative".
                    text = TokenizationService.render_conversation(
                        conv, conv_format, tokenizer, chat_format
                    )
                else:
                    # No tokenizer to ask: preserve the legacy shape rather than
                    # change behaviour for callers that never pass one.
                    text = extract_text_from_conversation(
                        conversation=conv,
                        format=conv_format,
                        include_roles=include_roles,
                        role_template=role_template,
                        join_separator="\n",
                        role_mapping=role_mapping,
                    )
                texts.append(text)

            return {output_column: texts}

        # Auto-detect number of processes
        if num_proc is None:
            import os
            num_proc = max(1, os.cpu_count() // 2)

        # Process the dataset
        total_samples = len(dataset)
        logger.info(f"Extracting text from {total_samples:,} conversations using {num_proc} process(es)")

        preprocessed_dataset = dataset.map(
            extract_text_batch,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="Extracting conversation text",
        )

        if progress_callback:
            progress_callback(35.0, f"Preprocessed {total_samples:,} conversations")

        # Log sample of extracted text for verification
        if len(preprocessed_dataset) > 0:
            sample_text = preprocessed_dataset[0][output_column][:500]
            logger.info(f"Sample extracted text (first 500 chars): {sample_text}")

        return preprocessed_dataset

    @staticmethod
    def load_tokenizer(tokenizer_name: str, use_fast: bool = True, cache_dir: str = None):
        """
        Load a HuggingFace tokenizer with proper configuration.

        Args:
            tokenizer_name: Name or path of tokenizer (e.g., 'gpt2', 'bert-base-uncased')
            use_fast: Whether to use fast tokenizer implementation
            cache_dir: Local cache directory containing downloaded tokenizer files.
                       If provided, loads from local cache without needing HuggingFace auth.

        Returns:
            Loaded tokenizer instance

        Raises:
            Exception: If tokenizer cannot be loaded
        """
        try:
            logger.debug(f"Loading tokenizer: {tokenizer_name}")
            logger.debug(f"cache_dir: {cache_dir}, local_files_only: {cache_dir is not None}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=use_fast,
                cache_dir=cache_dir,
                local_files_only=cache_dir is not None,  # Use local files if cache provided
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
        enable_cleaning: bool = False,
        enable_filtering: bool = False,
        filter_mode: str = "conservative",
        junk_ratio_threshold: float = 0.7,
        remove_all_punctuation: bool = False,
        custom_filter_chars: Optional[str] = None,
        cache_dir: Optional[str] = None,
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
                cache_dir=cache_dir,
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
    def render_conversation(
        conversation,
        conversation_format,
        tokenizer,
        chat_format: str = "auto",
    ) -> str:
        """Turn one conversation into the text the model is actually served.

        WHY. The previous renderer emitted ``<|user|>`` / ``<|assistant|>``
        pseudo-markers from a hardcoded template. Those strings are in no
        model's vocabulary here, so they tokenize as ordinary characters and the
        SAE learns features for the literal text `<`, `|`, `user`, `|`, `>`
        rather than for the turn structure. Every model trained on in this
        project is an INSTRUCT model, so the chat scaffolding is exactly the
        part of the deployment distribution that was missing.

        `chat_format`:
          * ``chat_template`` - require the tokenizer's template; raise if absent
          * ``plain``         - concatenate turn content, no role markers at all
          * ``auto``          - template when the tokenizer has one, else plain
          * ``none``          - same as plain

        Deliberately NOT falling back to the pseudo-marker template: it was
        never right, and keeping it as a default would preserve the defect for
        every tokenizer that happens to lack a template.
        """
        from ..utils.conversation_formats import extract_messages_from_conversation

        messages = extract_messages_from_conversation(conversation, conversation_format)
        if not messages:
            return ""

        has_template = bool(getattr(tokenizer, "chat_template", None))

        if chat_format == "chat_template" and not has_template:
            raise ValueError(
                "chat_format='chat_template' was requested but this tokenizer "
                "defines no chat_template. Use 'auto' to fall back to plain text, "
                "or pick a model whose tokenizer carries one."
            )

        if chat_format in ("chat_template", "auto") and has_template:
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    # The SAE reads text the model produced turns over; it is not
                    # being asked to continue one, so no generation prompt.
                    add_generation_prompt=False,
                )
            except Exception as exc:  # noqa: BLE001
                # Real templates reject shapes they were not written for: two
                # user turns in a row, a trailing system turn, an empty content.
                # `lmsys-chat-1m` contains all three. Losing the document is
                # worse than losing its scaffolding — but say so, never silently.
                logger.warning(
                    "chat_template rejected a conversation (%s); falling back to "
                    "plain content for it", exc,
                )
            else:
                # A TEMPLATE CAN ALSO SWALLOW A CONVERSATION WITHOUT RAISING.
                #
                # Zephyr-family templates are an if/elif over user|system|
                # assistant with NO else, so a role the mapping did not
                # canonicalise — "User", "tool", "function", or the literal
                # "unknown" emitted for a row with no role key — renders to the
                # EMPTY STRING and raises nothing. The document then tokenizes
                # as an all-pad row and disappears from the corpus silently.
                # This is the "a silent fallback is worse than a crash" class
                # this project has already paid for once.
                if rendered.strip():
                    return rendered
                logger.warning(
                    "chat_template produced empty output for a %d-turn "
                    "conversation (roles=%s); falling back to plain content. A "
                    "role the template does not recognise renders to nothing.",
                    len(messages), sorted({m["role"] for m in messages}),
                )

        # Plain: the content only. No invented markers.
        plain = TokenizationService._plain_text(messages)
        if not plain.strip():
            # The empty-output guard above covers the TEMPLATE branch only, and
            # this is the branch it falls back INTO. Round 2's null-content fix
            # turned visible junk ('None\nNone') into an empty string, which
            # tokenizes as an all-pad row and removes the document silently —
            # the exact failure class the template guard exists to stop.
            logger.warning(
                "A %d-turn conversation rendered to empty text (roles=%s); it "
                "will become an all-pad row. Every turn's content was empty.",
                len(messages), sorted({m["role"] for m in messages}),
            )
        return plain

    @staticmethod
    def resolve_padding_and_truncation(pack_sequences: bool, padding, truncation):
        """What to pass the tokenizer, given whether we are packing.

        Packing needs the UNPADDED, UNTRUNCATED stream: padding each document is
        exactly what packing undoes, and truncating discards the tail that
        should have started the next block.

        Inverting this ternary is one character and catastrophic — measured on a
        1000-document, 17-token corpus at max_length 512: the intended path
        gives 36 blocks that are genuinely 100% real, the inverted one gives
        1,002 blocks whose mask CLAIMS 100% real over data that is 3.5% real.
        A mask that confidently describes padding as text is what this whole arc
        exists to remove. So it is a function with a test, not an inline ternary.
        """
        if pack_sequences:
            return "do_not_pad", False
        return padding, truncation

    @staticmethod
    def resolve_text_column(requested, detected, available, rendered_from=None):
        """Which column to actually read.

        An explicit request beats auto-detection — the arc's single most-cited
        defect is Bloomberg being tokenized on `Headline` because detection fell
        through to the first text-like column while `Article` (41x the content)
        sat unused. Refuses a column the dataset does not have rather than
        silently falling back, since falling back IS the defect.

        ⚠ `rendered_from` NAMES THE COLUMN PREPROCESSING CONSUMED, and pointing
        an override back at it is the trap this parameter exists to close.
        When a conversation column is rendered through the chat template, the
        text lands in a NEW column and `detected` becomes that new column. An
        operator choosing the conversation column by name — the obviously
        correct-looking choice for a chat dataset, and the one the UI's dropdown
        invites, since it lists the RAW columns — then overrode the rendered
        text with the raw list-of-dicts.

        Measured, not hypothetical: `teknium/OpenHermes-2.5` with
        text_column="conversations" produced 490 blocks whose every token was
        `<|im_end|>` — `unique_tokens_used: 1` across 1,001,551 documents — and
        was reported READY at "99.8% real tokens", because EOS separators are
        real by the attention mask. Naming the source column now resolves to the
        rendered output, which is what the operator meant by it.
        """
        if not requested:
            return detected
        if rendered_from and requested == rendered_from:
            return detected
        if requested not in available:
            raise ValueError(
                f"text_column={requested!r} is not a column of this dataset. "
                f"Available: {list(available)}"
            )
        return requested

    @staticmethod
    def resolve_enable_cleaning(requested: bool, rendered_chat: bool) -> bool:
        """Should the text cleaner run over this corpus?

        Never on chat-template-rendered text. That string IS the model's input
        format, and the cleaner's job is to rewrite text. Measured 2026-09-12:
        with cleaning on, a stored OpenHermes tokenization held 0
        `<|startoftext|>`, 0 `<|im_start|>` and 0 newline tokens in 614,400
        sampled, and decoded as "user Every day ... assistant Here's the
        logic ..." -- the chat-template fix undone by the very next step.

        A function rather than an inline branch so the decision is testable and
        the worker's use of it can be asserted by AST.
        """
        return bool(requested) and not rendered_chat

    @staticmethod
    def resolve_add_special_tokens(requested: bool, tokenizer, used_template: bool) -> bool:
        """Should the tokenizer add BOS/EOS on top of the rendered text?

        No when the chat template already put BOS in the string: tokenizers do
        not de-duplicate, so position 0 of every conversation becomes a doubled,
        maximally-correlated artefact.

        A function rather than an inline branch because the inline version was
        guarded only by a source scrape, and round 3 proved deleting it left the
        suite green with the doubled BOS restored.
        """
        if not requested or not used_template:
            return requested
        return not TokenizationService.chat_template_emits_bos(tokenizer)

    @staticmethod
    def resolve_recorded_text_column(
        requires_preprocessing: bool, preprocessing_config, text_column
    ):
        """Which column name to RECORD on the tokenization row.

        For a conversation dataset `text_column` has by this point been
        reassigned to the synthetic output column ('text'), which records
        nothing about where the text came from. The source column is the answer,
        and recording the synthetic one reintroduces the Bloomberg failure in a
        new place: a row that says nothing useful about its own provenance.
        """
        if requires_preprocessing:
            source = (preprocessing_config or {}).get("source_column")
            if source:
                return source
        return text_column

    @staticmethod
    def chat_template_emits_bos(tokenizer) -> bool:
        """Does this tokenizer's chat template already put BOS in the string?

        Llama-2, Mistral and Gemma templates do. `apply_chat_template(...,
        tokenize=False)` therefore returns text that STARTS with the BOS token,
        and tokenizing that with `add_special_tokens=True` prepends a second
        one — tokenizers do not de-duplicate. Position 0 of every chat document
        becomes a doubled, maximally-correlated artefact, which is exactly the
        kind of high-magnitude outlier an SAE will happily spend capacity on.

        Detected by rendering, not assumed from the model name: templates vary
        within a family and this is cheap to just measure.
        """
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is None or not getattr(tokenizer, "chat_template", None):
            return False
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}],
                tokenize=False,
                add_generation_prompt=False,
            )
            ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        except Exception:  # noqa: BLE001 - a template that cannot render a
            # trivial turn tells us nothing; assume no BOS and let the normal
            # path handle it.
            return False
        return bool(ids) and ids[0] == bos

    @staticmethod
    def _plain_text(messages) -> str:
        return "\n".join(m["content"] for m in messages if m["content"])

    @staticmethod
    def pack_token_blocks(
        sequences,
        max_length: int,
        eos_token_id,
        drop_remainder: bool = False,
    ):
        """Concatenate documents into dense `max_length` blocks.

        WHY. With `padding="max_length"`, a corpus of short documents spends most
        of its window on PAD. Measured on this estate: Bloomberg headlines are
        mean 17 tokens in a 512 window — **3.3% real**. Those pad positions were
        being sampled as SAE training data, and even once masked out they are
        wasted extraction compute: the model still runs a forward pass over them.

        Packing concatenates documents, separated by EOS so the boundary stays
        visible to the model, and cuts the stream into full blocks. A 17-token
        corpus becomes ~97% occupancy instead of 3%.

        Returns `(blocks, attention_masks)`. Only the final block can be short;
        it is padded and masked, or dropped when `drop_remainder` is set.

        NOTE this changes what a "row" means — one row is no longer one
        document. Any consumer that assumes row == document must be checked
        before this is turned on by default.
        """
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        blocks = []
        masks = []
        buffer = []
        for seq in sequences:
            buffer.extend(seq)
            if eos_token_id is not None:
                buffer.append(eos_token_id)
            while len(buffer) >= max_length:
                blocks.append(buffer[:max_length])
                masks.append([1] * max_length)
                buffer = buffer[max_length:]

        if buffer and not drop_remainder:
            pad = max_length - len(buffer)
            masks.append([1] * len(buffer) + [0] * pad)
            # Pad with EOS when there is one; it is always a valid id, and the
            # mask marks these positions as not-real regardless.
            filler = eos_token_id if eos_token_id is not None else 0
            blocks.append(buffer + [filler] * pad)

        return blocks, masks

    @staticmethod
    def packing_heartbeat_progress(blocks: int) -> float:
        """A progress value for the packing pass that the janitor can see move.

        Strictly increasing in `blocks` and confined to [80.0, 80.9): after the
        tokenize stage's 80 and before the statistics stage's 80-90 band, since
        the total block count is unknown until packing ends. The janitor
        compares the counter between sweeps, so it only has to CHANGE.
        """
        b = max(0, int(blocks))
        return 80.0 + 0.9 * (1.0 - 1.0 / (1.0 + b / 1_000_000))

    @staticmethod
    def iter_packed_blocks(sequences, max_length: int, eos_token_id, drop_remainder: bool = False):
        """Stream `pack_token_blocks` so a large corpus does not have to fit in RAM.

        `pack_token_blocks` returns lists, which is right for a test and wrong
        for OpenWebText: 1M documents is ~450M tokens, and materialising both the
        input column and the output blocks as Python lists is tens of gigabytes.
        This yields one block at a time so the caller can stream into Arrow.

        Same semantics as `pack_token_blocks` — they share the boundary rule, and
        a test asserts the two agree.
        """
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        buffer = []
        for seq in sequences:
            buffer.extend(seq)
            if eos_token_id is not None:
                buffer.append(eos_token_id)
            while len(buffer) >= max_length:
                yield {
                    "input_ids": buffer[:max_length],
                    "attention_mask": [1] * max_length,
                }
                buffer = buffer[max_length:]

        if buffer and not drop_remainder:
            pad = max_length - len(buffer)
            filler = eos_token_id if eos_token_id is not None else 0
            yield {
                "input_ids": buffer + [filler] * pad,
                "attention_mask": [1] * len(buffer) + [0] * pad,
            }

    @staticmethod
    def tokenized_dataset_path(raw_path, model_id: str, max_length: int) -> Path:
        """Where one tokenization's Arrow directory lives.

        `max_length` IS part of the identity. The DB row's id and its uniqueness
        constraint are `(dataset, model, max_length)`, but the directory name used
        to be `{dataset}_tokenized_{model}` only — so tokenizing the same dataset
        for the same model at 512 and at 2048 produced two distinct rows pointing
        at ONE directory, and the second silently overwrote the first. Nothing
        errored; the 512 row simply started describing 2048 data.

        Rows written before this carry the old path and keep working — they are
        only re-pointed when that tokenization is rebuilt.
        """
        raw = Path(raw_path)
        return raw.parent / f"{raw.name}_tokenized_{model_id}_{max_length}"

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
