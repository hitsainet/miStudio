"""
Neuronpedia Export Job database model.

This module defines the SQLAlchemy model for tracking Neuronpedia export jobs.
Export jobs generate Neuronpedia-compatible archives containing SAE weights,
feature dashboard data (logit lens, histograms, top tokens), and metadata.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import uuid

from sqlalchemy import Column, String, Integer, BigInteger, Float, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID, ENUM as PGEnum
from sqlalchemy.sql import func

from ..core.database import Base


class ExportStatus(str, Enum):
    """Status of Neuronpedia export job."""
    PENDING = "pending"          # Job created, waiting to start
    COMPUTING = "computing"      # Computing dashboard data (logit lens, histograms)
    PACKAGING = "packaging"      # Packaging files into ZIP archive
    COMPLETED = "completed"      # Export finished successfully
    FAILED = "failed"           # Export failed with error
    CANCELLED = "cancelled"     # Export cancelled by user


class NeuronpediaExportJob(Base):
    """
    Neuronpedia Export Job database model.

    Tracks the status and progress of exports to Neuronpedia-compatible format.
    Each job exports an SAE with its feature data to a ZIP archive that can
    be uploaded to Neuronpedia.
    """

    __tablename__ = "neuronpedia_export_jobs"

    # Primary identifier - UUID for external referencing
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # SAE reference - can be either external_sae.id or training.id
    sae_id = Column(String(255), nullable=False, index=True)
    source_type = Column(String(50), nullable=False)  # 'external_sae' or 'training'

    # Job configuration as JSONB
    # Example: {
    #   "format": "neuronpedia_v1",
    #   "feature_selection": "all" | {"indices": [0, 1, 2]} | {"top_n": 1000},
    #   "include_logit_lens": true,
    #   "include_histograms": true,
    #   "include_top_tokens": true,
    #   "histogram_bins": 50,
    #   "top_tokens_k": 20,
    #   "logit_lens_k": 10
    # }
    config = Column(JSONB, nullable=False, default=dict)

    # Status tracking
    status = Column(
        PGEnum(
            'pending', 'computing', 'packaging', 'completed', 'failed', 'cancelled',
            name='export_status',
            create_type=False
        ),
        nullable=False,
        default=ExportStatus.PENDING.value
    )
    progress = Column(Float, nullable=False, default=0.0)  # 0-100
    current_stage = Column(String(100), nullable=True)  # Human-readable stage description

    # Results
    output_path = Column(Text, nullable=True)  # Path to generated ZIP archive
    file_size_bytes = Column(BigInteger, nullable=True)
    feature_count = Column(Integer, nullable=True)

    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    error_details = Column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<NeuronpediaExportJob(id={self.id}, sae_id={self.sae_id}, status={self.status})>"

    @property
    def is_pending(self) -> bool:
        """Check if job is pending."""
        return self.status == ExportStatus.PENDING.value

    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status in (ExportStatus.COMPUTING.value, ExportStatus.PACKAGING.value)

    @property
    def is_completed(self) -> bool:
        """Check if job completed successfully."""
        return self.status == ExportStatus.COMPLETED.value

    @property
    def is_failed(self) -> bool:
        """Check if job failed."""
        return self.status == ExportStatus.FAILED.value

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get duration of job in seconds, or None if not started/completed."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now(self.started_at.tzinfo)
        return (end_time - self.started_at).total_seconds()

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self.config.get(key, default) if self.config else default
