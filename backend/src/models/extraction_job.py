"""
Extraction Job database model.

This module defines the SQLAlchemy model for feature extraction jobs.
Feature extraction jobs extract interpretable features from trained SAE models.
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey
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
    id = Column(String(255), primary_key=True)  # Format: ext_{training_id}_{timestamp}
    training_id = Column(
        String(255),
        ForeignKey("trainings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    celery_task_id = Column(String(255), nullable=True)

    # Extraction configuration
    config = Column(JSONB, nullable=False)
    # Contains: {evaluation_samples: int, top_k_examples: int}

    # Processing status
    status = Column(String(50), nullable=False, default=ExtractionStatus.QUEUED.value)
    progress = Column(Float, nullable=True, default=0.0)  # 0-100
    features_extracted = Column(Integer, nullable=True, default=0)
    total_features = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    # Output statistics
    statistics = Column(JSONB, nullable=True)
    # Contains: {total_features: int, avg_interpretability: float,
    #            avg_activation_freq: float, interpretable_count: int}

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    training = relationship("Training", back_populates="extraction_jobs")
    features = relationship("Feature", back_populates="extraction_job", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ExtractionJob(id={self.id}, training_id={self.training_id}, status={self.status}, progress={self.progress})>"
