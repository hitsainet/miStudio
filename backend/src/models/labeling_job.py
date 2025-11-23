"""
Labeling Job database model.

This module defines the SQLAlchemy model for feature labeling jobs.
Labeling jobs apply semantic labels to features extracted from SAE models.
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class LabelingStatus(str, Enum):
    """Feature labeling job status."""
    QUEUED = "queued"
    LABELING = "labeling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LabelingMethod(str, Enum):
    """Feature labeling method."""
    OPENAI = "openai"           # OpenAI API
    OPENAI_COMPATIBLE = "openai_compatible"  # OpenAI-compatible endpoint (Ollama, vLLM, etc.)
    LOCAL = "local"             # Local HuggingFace model
    MANUAL = "manual"           # Manual labeling


class ExportFormat(str, Enum):
    """API request export format for debugging."""
    POSTMAN = "postman"         # Postman collection only
    CURL = "curl"               # cURL command only
    BOTH = "both"               # Both Postman and cURL


class LabelingJob(Base):
    """
    Labeling Job database model.

    Stores metadata about feature labeling jobs.
    Each labeling job applies semantic labels to features from an extraction.
    Labeling is independent from extraction, allowing re-labeling without re-extraction.
    """

    __tablename__ = "labeling_jobs"

    # Primary identifiers
    id = Column(String(255), primary_key=True)  # Format: label_{extraction_id}_{timestamp}
    extraction_job_id = Column(
        String(255),
        ForeignKey("extraction_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    celery_task_id = Column(String(255), nullable=True)

    # Labeling configuration
    labeling_method = Column(String(50), nullable=False)  # openai, openai_compatible, local, manual
    openai_model = Column(String(100), nullable=True)  # e.g., "gpt-4o-mini"
    openai_api_key = Column(String(500), nullable=True)  # Encrypted API key
    openai_compatible_endpoint = Column(String(500), nullable=True)  # e.g., "http://ollama.mcslab.io"
    openai_compatible_model = Column(String(100), nullable=True)  # e.g., "llama3.2"
    local_model = Column(String(100), nullable=True)  # e.g., "meta-llama/Llama-3.2-1B"
    prompt_template_id = Column(
        String(255),
        ForeignKey("labeling_prompt_templates.id", ondelete="RESTRICT"),
        nullable=True,
        index=True
    )

    # Token filtering configuration
    filter_special = Column(Boolean, nullable=False, default=True)  # Filter special tokens (<s>, </s>, etc.)
    filter_single_char = Column(Boolean, nullable=False, default=True)  # Filter single character tokens
    filter_punctuation = Column(Boolean, nullable=False, default=True)  # Filter pure punctuation
    filter_numbers = Column(Boolean, nullable=False, default=True)  # Filter pure numeric tokens
    filter_fragments = Column(Boolean, nullable=False, default=True)  # Filter word fragments (BPE subwords)
    filter_stop_words = Column(Boolean, nullable=False, default=False)  # Filter common stop words

    # Debugging configuration
    save_requests_for_testing = Column(Boolean, nullable=False, default=False)  # Save API requests to /tmp/ for testing
    export_format = Column(String(20), nullable=False, default=ExportFormat.BOTH.value)  # Format: 'postman', 'curl', 'both'
    save_poor_quality_labels = Column(Boolean, nullable=False, default=False)  # Save poor quality labels for debugging
    poor_quality_sample_rate = Column(Float, nullable=False, default=1.0)  # Sample rate for poor quality labels (0.0-1.0)

    # Processing status
    status = Column(String(50), nullable=False, default=LabelingStatus.QUEUED.value)
    progress = Column(Float, nullable=False, default=0.0)  # 0.0-1.0
    features_labeled = Column(Integer, nullable=False, default=0)
    total_features = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    # Output statistics
    statistics = Column(JSONB, nullable=True)
    # Contains: {total_features: int, successfully_labeled: int,
    #            failed_labels: int, avg_label_length: float,
    #            labeling_duration_seconds: float}

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    extraction_job = relationship("ExtractionJob", back_populates="labeling_jobs")
    features = relationship("Feature", back_populates="labeling_job")
    prompt_template = relationship("LabelingPromptTemplate", back_populates="labeling_jobs")

    def __repr__(self) -> str:
        return f"<LabelingJob(id={self.id}, extraction_job_id={self.extraction_job_id}, status={self.status}, progress={self.progress})>"
