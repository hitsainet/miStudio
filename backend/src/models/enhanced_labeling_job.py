"""
EnhancedLabelingJob database model.

Tracks single-feature two-pass LLM labeling jobs kicked off from the
Feature Detail modal. Unlike LabelingJob (which operates on a full extraction),
each record here covers exactly one feature.
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class EnhancedLabelingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    #: Added 2026-09-05 with the cancel route. SAFE AS A PLAIN ADDITION: the
    #: column is String(50), not a native Postgres enum, so no ALTER TYPE and
    #: no migration. Without it an operator's stop had to be written as FAILED,
    #: which is the conflation this arc exists to remove.
    CANCELLED = "cancelled"


class EnhancedLabelingPhase(str, Enum):
    PASS1 = "pass1"
    PASS2 = "pass2"


class EnhancedLabelingJob(Base):
    """
    Single-feature enhanced labeling job.

    Two-pass strategy:
      Pass 1 — parallel per-example summarization (workers concurrent HTTP calls)
      Pass 2 — synthesis of all summaries into a structured label

    The result is written directly to features.name / category / description /
    notes / label_source='enhanced_llm'.
    """

    __tablename__ = "enhanced_labeling_jobs"

    id = Column(String(255), primary_key=True)  # elj_{neuron_index}_{timestamp_ms}
    feature_id = Column(
        String(255),
        ForeignKey("features.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Job state
    status = Column(String(50), nullable=False, default=EnhancedLabelingStatus.QUEUED.value)
    phase = Column(String(50), nullable=True)  # pass1 | pass2 | None
    celery_task_id = Column(String(255), nullable=True)

    # Configuration (copied from settings at job creation time)
    method = Column(String(50), nullable=False, default="openai_compatible")
    endpoint = Column(String(500), nullable=False)
    model = Column(String(255), nullable=False)
    workers = Column(Integer, nullable=False, default=8)
    examples_total = Column(Integer, nullable=False, default=20)
    examples_completed = Column(Integer, nullable=False, default=0)

    # Results
    pass1_summaries = Column(JSONB, nullable=True)  # [{n, prime, activation, summary}, ...]
    raw_synthesis = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationship
    feature = relationship("Feature", backref="enhanced_labeling_jobs")

    def __repr__(self) -> str:
        return (
            f"<EnhancedLabelingJob(id={self.id}, feature_id={self.feature_id}, "
            f"status={self.status}, phase={self.phase})>"
        )
