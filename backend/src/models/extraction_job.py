"""
Extraction Job database model.

This module defines the SQLAlchemy model for feature extraction jobs.
Feature extraction jobs extract interpretable features from trained SAE models.
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey, Enum as SQLEnum, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class ExtractionStatus(str, Enum):
    """Feature extraction job status."""
    QUEUED = "queued"
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExtractionJob(Base):
    """
    Extraction Job database model.

    Stores metadata about feature extraction jobs from trained SAE models.
    Each extraction analyzes a completed training to discover interpretable features.
    """

    __tablename__ = "extraction_jobs"

    # Primary identifiers
    id = Column(String(255), primary_key=True)  # Format: ext_{training_id}_{timestamp} or ext_sae_{sae_id}_{timestamp}
    training_id = Column(
        String(255),
        ForeignKey("trainings.id", ondelete="CASCADE"),
        nullable=True,  # Nullable: either training_id OR external_sae_id must be set
        index=True
    )
    external_sae_id = Column(
        String(255),
        ForeignKey("external_saes.id", ondelete="CASCADE"),
        nullable=True,  # Nullable: either training_id OR external_sae_id must be set
        index=True
    )
    celery_task_id = Column(String(255), nullable=True)

    # Extraction configuration
    config = Column(JSONB, nullable=False)
    # Contains: {evaluation_samples: int, top_k_examples: int}

    # Layer selection for multi-layer trainings
    layer_index = Column(Integer, nullable=True)  # Layer index (e.g., 10, 17, 24)
    # Hook type selection for multi-hook trainings
    hook_type = Column(String(50), nullable=True)  # Hook type (e.g., "residual", "mlp", "attention")

    # Token filtering configuration (per-job) - matches labeling filter structure
    # These filters control which tokens are stored in FeatureActivation records during extraction
    filter_special = Column(Boolean, nullable=False, default=True, server_default='true')  # Filter special tokens (<s>, </s>, etc.)
    filter_single_char = Column(Boolean, nullable=False, default=True, server_default='true')  # Filter single character tokens
    filter_punctuation = Column(Boolean, nullable=False, default=True, server_default='true')  # Filter pure punctuation
    filter_numbers = Column(Boolean, nullable=False, default=True, server_default='true')  # Filter pure numeric tokens
    filter_fragments = Column(Boolean, nullable=False, default=True, server_default='true')  # Filter word fragments (BPE subwords)
    filter_stop_words = Column(Boolean, nullable=False, default=False, server_default='false')  # Filter common stop words

    # Context window configuration (per-job)
    # Captures tokens before and after the prime token (max activation) for better interpretability
    # Based on Anthropic/OpenAI research showing asymmetric windows improve feature understanding
    context_prefix_tokens = Column(Integer, nullable=False, default=25, server_default='25')  # Tokens before prime token
    context_suffix_tokens = Column(Integer, nullable=False, default=25, server_default='25')  # Tokens after prime token

    # Processing status
    status = Column(
        SQLEnum(
            ExtractionStatus,
            name='extraction_status_enum',
            create_type=False,  # Don't create the enum (it already exists in DB)
            values_callable=lambda x: [e.value for e in x]  # Use enum values, not names
        ),
        nullable=False,
        default=ExtractionStatus.QUEUED
    )
    progress = Column(Float, nullable=True, default=0.0)  # 0-100
    features_extracted = Column(Integer, nullable=True, default=0)
    total_features = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    # NLP Processing status (separate from feature extraction)
    # NLP analysis is performed AFTER extraction completes
    nlp_status = Column(String(50), nullable=True)  # null, "pending", "processing", "completed", "failed"
    nlp_progress = Column(Float, nullable=True, default=0.0)  # 0.0-1.0
    nlp_processed_count = Column(Integer, nullable=True, default=0)  # Features with NLP completed
    nlp_error_message = Column(Text, nullable=True)

    # Output statistics
    statistics = Column(JSONB, nullable=True)
    # Contains: {total_features: int, avg_interpretability: float,
    #            avg_activation_freq: float, interpretable_count: int}

    # Batch extraction support
    # Allows grouping multiple extraction jobs that were started together
    batch_id = Column(String(255), nullable=True, index=True)  # Group ID for batch extractions
    batch_position = Column(Integer, nullable=True)  # Position in batch (1-indexed, e.g., 1 of 5)
    batch_total = Column(Integer, nullable=True)  # Total jobs in batch

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    training = relationship("Training", back_populates="extraction_jobs")
    external_sae = relationship("ExternalSAE", back_populates="extraction_jobs")
    features = relationship("Feature", back_populates="extraction_job", cascade="all, delete-orphan")
    labeling_jobs = relationship("LabelingJob", back_populates="extraction_job", cascade="all, delete-orphan")

    @property
    def source_type(self) -> str:
        """Return the source type: 'training' or 'external_sae'."""
        return "external_sae" if self.external_sae_id else "training"

    @property
    def source_id(self) -> str:
        """Return the source ID (either training_id or external_sae_id)."""
        return self.external_sae_id if self.external_sae_id else self.training_id

    def __repr__(self) -> str:
        return f"<ExtractionJob(id={self.id}, training_id={self.training_id}, status={self.status}, progress={self.progress})>"
