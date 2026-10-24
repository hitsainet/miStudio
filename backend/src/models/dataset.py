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
    Boolean,
    DateTime,
    Enum as SQLEnum,
    Index,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
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
    Tokenization is now handled separately via DatasetTokenization model.

    Attributes:
        id: Unique identifier (UUID)
        name: Human-readable dataset name
        source: Source type ('HuggingFace', 'Local', 'Custom')
        hf_repo_id: HuggingFace repository ID (e.g., 'roneneldan/TinyStories')
        status: Current processing status
        progress: Download/processing progress (0-100)
        error_message: Error details if status is ERROR
        raw_path: Path to raw downloaded data
        num_samples: Total number of samples in raw dataset
        size_bytes: Total size in bytes
        metadata: Flexible JSONB field for additional metadata
        tokenizations: Relationship to DatasetTokenization records
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

    # Statistics (dataset-level only)
    num_samples = Column(
        Integer,
        nullable=True,
        comment="Total number of samples in raw dataset",
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

    # Token filtering configuration (per-job)
    tokenization_filter_enabled = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default='false',
        comment="Enable sample filtering during tokenization",
    )
    tokenization_filter_mode = Column(
        String(20),
        nullable=False,
        default='conservative',
        server_default='conservative',
        comment="Filter mode: minimal or conservative",
    )
    tokenization_junk_ratio_threshold = Column(
        Float,
        nullable=False,
        default=0.7,
        server_default='0.7',
        comment="Junk ratio threshold (0.0-1.0)",
    )


    #: Feature 035 (XR-3): the Hugging Face COMMIT this row's files were downloaded at — always
    #: the 40-character sha the request's ref resolved to, never a branch or tag name. NULL means
    #: "revision not recorded": every row downloaded before 035 was fetched unpinned, so nothing
    #: is backfilled and NULL is never displayed as "latest" (FR-5, D1). Part of the download
    #: identity (repo, config, split, revision) and of the raw directory name.
    revision = Column(
        String(40),
        nullable=True,
        comment="Hugging Face commit the files were downloaded at; NULL = not recorded",
    )

    #: Set by the cancel endpoint; polled by the running job. A TIMESTAMP
    #: rather than a status value because this table's status is a native
    #: Postgres enum with no CANCELLED member — and because "the operator
    #: asked" and "the job stopped" are two different facts, previously
    #: conflated into `status = ERROR` + "Cancelled by user": a deliberate
    #: stop recorded as a crash, indistinguishable afterwards from one.
    cancel_requested_at = Column(DateTime(timezone=True), nullable=True)

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

    # Relationships
    tokenizations = relationship(
        "DatasetTokenization",
        back_populates="dataset",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # Indexes for query optimization
    __table_args__ = (
        Index("idx_datasets_status", "status"),
        Index("idx_datasets_source", "source"),
        Index("idx_datasets_created_at", "created_at"),
        # The value is only ever written from a resolved commit; the constraint makes a
        # branch name ("main") or a short sha unstorable, so a row cannot claim a pin it
        # does not have (035 FTDD §4.1).
        CheckConstraint(
            "revision IS NULL OR revision ~ '^[0-9a-f]{40}$'",
            name="ck_datasets_revision_is_commit",
        ),
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
            "num_samples": self.num_samples,
            "size_bytes": self.size_bytes,
            "metadata": self.extra_metadata,  # Map to API name
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
