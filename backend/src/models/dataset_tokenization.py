"""
SQLAlchemy model for DatasetTokenization entity.

This module defines the DatasetTokenization table schema for storing
multiple tokenizations of the same dataset. Each tokenization is specific
to a model's tokenizer, allowing one dataset to be used with multiple models.
"""

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    BigInteger,
    Float,
    Boolean,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..core.database import Base


class TokenizationStatus(str, enum.Enum):
    """Tokenization processing status enumeration."""

    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class DatasetTokenization(Base):
    """
    DatasetTokenization model representing a specific tokenization of a dataset.

    Each dataset can have multiple tokenizations - one for each model/tokenizer.
    This enables reusing the same raw dataset with different models without
    re-downloading or re-processing the raw data.

    Attributes:
        id: Unique identifier (format: tok_{dataset_id}_{model_id})
        dataset_id: Foreign key to parent dataset
        model_id: Foreign key to model whose tokenizer was used
        tokenized_path: Path to tokenized data (Arrow format)
        tokenizer_repo_id: HuggingFace tokenizer repo ID
        vocab_size: Vocabulary size for this tokenization
        num_tokens: Total number of tokens in tokenized dataset
        avg_seq_length: Average sequence length in tokens
        status: Current tokenization status
        progress: Tokenization progress (0-100)
        error_message: Error details if status is ERROR
        celery_task_id: ID of Celery task performing tokenization
        created_at: Record creation timestamp
        updated_at: Record last update timestamp
        completed_at: Timestamp when tokenization completed
    """

    __tablename__ = "dataset_tokenizations"

    # Primary key
    id = Column(
        String(255),
        primary_key=True,
        nullable=False,
        comment="Unique tokenization identifier (format: tok_{dataset_id}_{model_id})",
    )

    # Foreign keys
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent dataset ID",
    )
    model_id = Column(
        String(255),
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Model whose tokenizer was used",
    )

    # Tokenization metadata
    tokenized_path = Column(
        String(512),
        nullable=True,
        comment="Path to tokenized dataset (Arrow format)",
    )
    tokenizer_repo_id = Column(
        String(255),
        nullable=False,
        comment="HuggingFace tokenizer repository ID",
    )

    # Statistics (specific to this tokenization)
    vocab_size = Column(
        Integer,
        nullable=True,
        comment="Vocabulary size for this tokenization",
    )
    num_tokens = Column(
        BigInteger,
        nullable=True,
        comment="Total number of tokens in tokenized dataset",
    )
    avg_seq_length = Column(
        Float,
        nullable=True,
        comment="Average sequence length in tokens",
    )

    # Status and progress
    status = Column(
        SQLEnum(TokenizationStatus, name="tokenization_status_enum"),
        nullable=False,
        default=TokenizationStatus.QUEUED,
        comment="Current tokenization status",
    )
    progress = Column(
        Float,
        nullable=True,
        default=0.0,
        comment="Tokenization progress (0-100)",
    )
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if status is ERROR",
    )

    # Task tracking
    celery_task_id = Column(
        String(255),
        nullable=True,
        comment="Celery task ID for async tokenization",
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
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when tokenization completed",
    )

    # Token filtering configuration
    remove_all_punctuation = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="If true, removes ALL punctuation characters from tokens",
    )
    custom_filter_chars = Column(
        String(255),
        nullable=True,
        comment="Custom characters to filter (e.g., '~@#$%')",
    )

    # Relationships
    dataset = relationship("Dataset", back_populates="tokenizations")
    model = relationship("Model")

    # Constraints and indexes
    __table_args__ = (
        # Unique constraint: one tokenization per dataset-model pair
        UniqueConstraint("dataset_id", "model_id", name="uq_dataset_model_tokenization"),
        # Indexes for query optimization
        Index("idx_tokenizations_dataset_id", "dataset_id"),
        Index("idx_tokenizations_model_id", "model_id"),
        Index("idx_tokenizations_status", "status"),
        Index("idx_tokenizations_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """String representation of DatasetTokenization."""
        return (
            f"<DatasetTokenization(id={self.id}, "
            f"dataset_id={self.dataset_id}, model_id={self.model_id}, "
            f"status={self.status.value})>"
        )

    def to_dict(self) -> dict:
        """
        Convert model instance to dictionary for API responses.

        Returns:
            dict: Dictionary representation of tokenization
        """
        return {
            "id": self.id,
            "dataset_id": str(self.dataset_id),
            "model_id": self.model_id,
            "tokenized_path": self.tokenized_path,
            "tokenizer_repo_id": self.tokenizer_repo_id,
            "vocab_size": self.vocab_size,
            "num_tokens": self.num_tokens,
            "avg_seq_length": self.avg_seq_length,
            "status": self.status.value,
            "progress": self.progress,
            "error_message": self.error_message,
            "celery_task_id": self.celery_task_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
