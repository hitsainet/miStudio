"""
Pydantic schemas for dataset metadata validation.

This module defines the structure and validation rules for dataset metadata
including schema information and tokenization statistics.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class SchemaMetadata(BaseModel):
    """
    Dataset schema information.

    Captures the structure of the dataset including available columns
    and their types.
    """

    text_columns: List[str] = Field(
        ...,
        description="List of text/string columns in the dataset",
        min_length=0
    )
    column_info: Dict[str, str] = Field(
        ...,
        description="Mapping of column names to their data types"
    )
    all_columns: List[str] = Field(
        ...,
        description="Complete list of all columns in the dataset",
        min_length=1
    )
    is_multi_column: bool = Field(
        ...,
        description="Whether the dataset has multiple text columns"
    )

    @model_validator(mode='after')
    def validate_schema_consistency(self) -> 'SchemaMetadata':
        """
        Validate that schema metadata is internally consistent.

        Ensures:
        - text_columns are a subset of all_columns
        - is_multi_column is consistent with text_columns length
        """
        # Validate text_columns are subset of all_columns
        text_cols_set = set(self.text_columns)
        all_cols_set = set(self.all_columns)

        if not text_cols_set.issubset(all_cols_set):
            extra_cols = text_cols_set - all_cols_set
            raise ValueError(
                f"text_columns contains columns not in all_columns: {extra_cols}"
            )

        # Validate is_multi_column consistency
        actual_multi = len(self.text_columns) > 1
        if self.is_multi_column != actual_multi:
            raise ValueError(
                f"is_multi_column={self.is_multi_column} inconsistent with "
                f"text_columns length ({len(self.text_columns)})"
            )

        return self


class TokenizationMetadata(BaseModel):
    """
    Tokenization statistics and configuration.

    Records the tokenization settings used and resulting statistics
    for the tokenized dataset.
    """

    tokenizer_name: str = Field(
        ...,
        min_length=1,
        description="HuggingFace tokenizer identifier (e.g., 'gpt2', 'bert-base-uncased')"
    )
    text_column_used: str = Field(
        ...,
        min_length=1,
        description="Name of the text column that was tokenized"
    )
    max_length: int = Field(
        ...,
        ge=1,
        le=8192,
        description="Maximum sequence length in tokens"
    )
    stride: int = Field(
        ...,
        ge=0,
        description="Sliding window stride (0 = no overlap)"
    )
    num_tokens: int = Field(
        ...,
        ge=0,
        description="Total number of tokens across all samples"
    )
    avg_seq_length: float = Field(
        ...,
        ge=0,
        description="Average sequence length in tokens"
    )
    min_seq_length: int = Field(
        ...,
        ge=0,
        description="Minimum sequence length in tokens"
    )
    max_seq_length: int = Field(
        ...,
        ge=0,
        description="Maximum sequence length in tokens"
    )

    @model_validator(mode='after')
    def validate_sequence_length_consistency(self) -> 'TokenizationMetadata':
        """
        Validate that sequence length statistics are consistent.

        Ensures:
        - min_seq_length <= avg_seq_length <= max_seq_length
        - max_seq_length <= max_length (can't exceed tokenizer limit)
        """
        if self.min_seq_length > self.max_seq_length:
            raise ValueError(
                f"min_seq_length ({self.min_seq_length}) cannot exceed "
                f"max_seq_length ({self.max_seq_length})"
            )

        if self.avg_seq_length < self.min_seq_length:
            raise ValueError(
                f"avg_seq_length ({self.avg_seq_length:.2f}) cannot be less than "
                f"min_seq_length ({self.min_seq_length})"
            )

        if self.avg_seq_length > self.max_seq_length:
            raise ValueError(
                f"avg_seq_length ({self.avg_seq_length:.2f}) cannot exceed "
                f"max_seq_length ({self.max_seq_length})"
            )

        if self.max_seq_length > self.max_length:
            raise ValueError(
                f"max_seq_length ({self.max_seq_length}) cannot exceed "
                f"max_length ({self.max_length})"
            )

        return self

    @field_validator('num_tokens')
    @classmethod
    def validate_num_tokens_nonzero(cls, v: int) -> int:
        """Validate that tokenized dataset has at least some tokens."""
        if v == 0:
            raise ValueError(
                "num_tokens cannot be 0 - tokenized dataset must contain at least one token"
            )
        return v


class DownloadMetadata(BaseModel):
    """
    Dataset download metadata.

    Records information about the dataset download process including
    configuration and split information.
    """

    split: Optional[str] = Field(
        None,
        description="Dataset split downloaded (e.g., 'train', 'validation', 'test')"
    )
    config: Optional[str] = Field(
        None,
        description="Dataset configuration name (e.g., 'en', 'zh') for multi-config datasets"
    )
    access_token_provided: Optional[bool] = Field(
        None,
        description="Whether an access token was used for download"
    )


class DatasetMetadata(BaseModel):
    """
    Complete dataset metadata structure.

    Top-level metadata container that can hold schema information,
    tokenization statistics, and download metadata.
    """

    dataset_schema: Optional[SchemaMetadata] = Field(
        None,
        description="Dataset schema information",
        alias="schema"  # Accept "schema" in input, but use "dataset_schema" internally
    )
    tokenization: Optional[TokenizationMetadata] = Field(
        None,
        description="Tokenization statistics and configuration"
    )
    download: Optional[DownloadMetadata] = Field(
        None,
        description="Download metadata (split, config, etc.)"
    )

    def has_complete_tokenization(self) -> bool:
        """
        Check if complete tokenization metadata is present.

        Returns:
            True if tokenization metadata exists and is complete, False otherwise.
        """
        return self.tokenization is not None

    def has_schema_info(self) -> bool:
        """
        Check if schema metadata is present.

        Returns:
            True if schema metadata exists, False otherwise.
        """
        return self.dataset_schema is not None

    def has_download_info(self) -> bool:
        """
        Check if download metadata is present.

        Returns:
            True if download metadata exists, False otherwise.
        """
        return self.download is not None

    model_config = {
        "populate_by_name": True,  # Allow using both 'dataset_schema' and 'schema'
        "json_schema_extra": {
            "examples": [
                {
                    "schema": {
                        "text_columns": ["text", "content"],
                        "column_info": {"text": "string", "label": "int64"},
                        "all_columns": ["text", "label"],
                        "is_multi_column": True
                    },
                    "tokenization": {
                        "tokenizer_name": "gpt2",
                        "text_column_used": "text",
                        "max_length": 512,
                        "stride": 0,
                        "num_tokens": 1000000,
                        "avg_seq_length": 245.5,
                        "min_seq_length": 10,
                        "max_seq_length": 512
                    },
                    "download": {
                        "split": "train",
                        "config": None,
                        "access_token_provided": False
                    }
                }
            ]
        }
    }
