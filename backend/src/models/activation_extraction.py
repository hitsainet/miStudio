"""
Activation Extraction database model.

This module defines the SQLAlchemy model for activation extractions.
"""

from datetime import datetime
from enum import Enum
from uuid import uuid4

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, Enum as SQLEnum, ForeignKey, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from ..core.database import Base


class ExtractionStatus(str, Enum):
    """Extraction processing status."""
    QUEUED = "queued"
    LOADING = "loading"
    EXTRACTING = "extracting"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExtractionPhase(str, Enum):
    """Which step of a running extraction is under way.

    `status` stays EXTRACTING through all three, so no guard that reads status
    had to change. The phase is stored so a page reload shows the step the live
    WebSocket message showed, instead of falling back to "Extracting N/N".
    """
    EXTRACTING = "extracting"   # the GPU pass
    MERGING = "merging"         # per-batch shards copied into one file per layer
    STATISTICS = "statistics"   # per-layer statistics over the merged files


#: Every value `activation_extractions.phase` may hold, besides NULL.
EXTRACTION_PHASES = frozenset(p.value for p in ExtractionPhase)


class ActivationExtraction(Base):
    """
    Activation Extraction database model.

    Stores metadata about activation extraction jobs including progress,
    configuration, and results. Provides persistence so extractions can
    be tracked across page refreshes.
    """

    __tablename__ = "activation_extractions"

    # Primary identifiers
    id = Column(String(255), primary_key=True)  # Format: ext_{model_id}_{timestamp}
    model_id = Column(String(255), ForeignKey("models.id"), nullable=False)
    dataset_id = Column(String(255), nullable=False)  # UUID string
    celery_task_id = Column(String(255), nullable=True)  # Celery task ID for tracking

    # Extraction configuration
    layer_indices = Column(ARRAY(Integer), nullable=False)  # [0, 5, 10, 15]
    hook_types = Column(ARRAY(String), nullable=False)  # ["residual", "mlp", "attention"]
    max_samples = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False, default=8)
    micro_batch_size = Column(Integer, nullable=True)  # GPU micro-batch size (defaults to batch_size)
    # Index-era field, kept for existing rows and no longer read: an NVML index
    # names a different card whenever one is added (the 3090 moved from 0 to 1).
    gpu_id = Column(Integer, nullable=False, default=0)
    # What the job asked for ("auto", "all" or a GPU UUID) and the card it ran on.
    gpu_request = Column(String(64), nullable=True)
    gpu_uuid = Column(String(64), nullable=True)
    # Every card of a split, most free first (the first is gpu_uuid; layers fill
    # them in CUDA index order); NULL for a single-card job.
    gpu_uuids = Column(JSONB, nullable=True)

    # Processing status
    status = Column(SQLEnum(ExtractionStatus), nullable=False, default=ExtractionStatus.QUEUED)
    # One of EXTRACTION_PHASES while the job is past loading. NULL before the
    # GPU pass starts and after COMPLETED; a FAILED or CANCELLED row keeps the
    # phase it stopped in, which says where it stopped.
    phase = Column(String(32), nullable=True)
    progress =Column(Float, nullable=True, default=0.0)  # 0-100
    samples_processed = Column(Integer, nullable=True, default=0)
    error_message = Column(Text, nullable=True)
    error_type = Column(String(50), nullable=True)  # OOM, VALIDATION, TIMEOUT, EXTRACTION, UNKNOWN

    # Retry tracking
    retry_count = Column(Integer, nullable=False, default=0)  # Number of times this extraction has been retried
    original_extraction_id = Column(String(255), nullable=True)  # Points to original extraction if this is a retry
    retry_reason = Column(Text, nullable=True)  # Human-readable reason for retry
    auto_retried = Column(Boolean, nullable=False, default=False)  # True if automatic retry, False if manual

    # Output information
    output_path = Column(String(1000), nullable=True)  # /data/activations/{extraction_id}/
    metadata_path = Column(String(1000), nullable=True)  # /data/activations/{extraction_id}/metadata.json

    # Statistics (updated incrementally)
    statistics = Column(JSONB, nullable=True, default=dict)
    # Contains per-layer stats: shape, mean_magnitude, max_activation, sparsity_percent, size_mb

    saved_files = Column(ARRAY(String), nullable=True)  # ["layer_0_residual.npy", "layer_5_mlp.npy"]

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<ActivationExtraction(id={self.id}, model_id={self.model_id}, status={self.status}, progress={self.progress})>"
