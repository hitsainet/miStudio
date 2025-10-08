"""
SQLAlchemy model for Dataset entity.

This module defines the Dataset table schema for storing dataset metadata
including download status, tokenization state, and statistics.
"""

import enum
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    BigInteger,
    Float,
    DateTime,
    Enum as SQLEnum,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from ..core.database import Base


class DatasetStatus(str, enum.Enum):
    """Dataset processing status enumeration."""

    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class Dataset(Base):
    """
    Dataset model representing a training dataset.

    A dataset can be downloaded from HuggingFace or uploaded locally.
    It goes through stages: downloading → processing → ready.
    Statistics are computed during processing and stored for quick access.

    Attributes:
        id: Unique identifier (UUID)
        name: Human-readable dataset name
        source: Source type ('HuggingFace', 'Local', 'Custom')
        hf_repo_id: HuggingFace repository ID (e.g., 'roneneldan/TinyStories')
        status: Current processing status
        progress: Download/processing progress (0-100)
        error_message: Error details if status is ERROR
        raw_path: Path to raw downloaded data
        tokenized_path: Path to tokenized data (Arrow format)
        num_samples: Total number of samples in dataset
        num_tokens: Total number of tokens across all samples
        avg_seq_length: Average sequence length in tokens
        vocab_size: Vocabulary size (unique tokens)
        size_bytes: Total size in bytes
        metadata: Flexible JSONB field for additional metadata
        created_at: Record creation timestamp
        updated_at: Record last update timestamp
    """

    __tablename__ = "datasets"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        comment="Unique dataset identifier",
    )

    # Basic information
    name = Column(String(255), nullable=False, comment="Dataset name")
    source = Column(
        String(50),
        nullable=False,
        comment="Source type: HuggingFace, Local, or Custom",
    )
    hf_repo_id = Column(
        String(255),
        nullable=True,
        comment="HuggingFace repository ID",
    )

    # Status and progress
    status = Column(
        SQLEnum(DatasetStatus, name="dataset_status_enum"),
        nullable=False,
        default=DatasetStatus.DOWNLOADING,
        comment="Current processing status",
    )
    progress = Column(
        Float,
        nullable=True,
        comment="Download/processing progress (0-100)",
    )
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if status is ERROR",
    )

    # File paths
    raw_path = Column(
        String(512),
        nullable=True,
        comment="Path to raw dataset files",
    )
    tokenized_path = Column(
        String(512),
        nullable=True,
        comment="Path to tokenized dataset (Arrow format)",
    )

    # Statistics
    num_samples = Column(
        Integer,
        nullable=True,
        comment="Total number of samples",
    )
    num_tokens = Column(
        BigInteger,
        nullable=True,
        comment="Total number of tokens",
    )
    avg_seq_length = Column(
        Float,
        nullable=True,
        comment="Average sequence length in tokens",
    )
    vocab_size = Column(
        Integer,
        nullable=True,
        comment="Vocabulary size (unique tokens)",
    )
    size_bytes = Column(
        BigInteger,
        nullable=True,
        comment="Total size in bytes",
    )

    # Flexible metadata storage (JSONB for efficient querying)
    # Note: Using 'extra_metadata' instead of 'metadata' (reserved by SQLAlchemy)
    extra_metadata = Column(
        "metadata",  # Column name in database
        JSONB,
        nullable=True,
        default=dict,
        comment="Additional metadata (splits, features, etc.)",
    )

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Record creation timestamp",
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Record last update timestamp",
    )

    # Indexes for query optimization
    __table_args__ = (
        Index("idx_datasets_status", "status"),
        Index("idx_datasets_source", "source"),
        Index("idx_datasets_created_at", "created_at"),
        # GIN index for JSONB metadata queries (created in migration)
        # Index("idx_datasets_metadata_gin", "metadata", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        """String representation of Dataset."""
        return (
            f"<Dataset(id={self.id}, name='{self.name}', "
            f"status={self.status.value}, source='{self.source}')>"
        )

    def to_dict(self) -> dict:
        """
        Convert model instance to dictionary for API responses.

        Returns:
            dict: Dictionary representation of dataset
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "source": self.source,
            "hf_repo_id": self.hf_repo_id,
            "status": self.status.value,
            "progress": self.progress,
            "error_message": self.error_message,
            "raw_path": self.raw_path,
            "tokenized_path": self.tokenized_path,
            "num_samples": self.num_samples,
            "num_tokens": self.num_tokens,
            "avg_seq_length": self.avg_seq_length,
            "vocab_size": self.vocab_size,
            "size_bytes": self.size_bytes,
            "metadata": self.extra_metadata,  # Map to API name
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
