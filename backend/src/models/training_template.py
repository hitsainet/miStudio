"""
TrainingTemplate database model.

This module defines the SQLAlchemy model for training templates,
which allow users to save and reuse SAE training configurations.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from ..core.database import Base


class TrainingTemplate(Base):
    """
    TrainingTemplate database model for managing SAE training configurations.

    Stores named templates with hyperparameters, architecture settings, and other
    training parameters that users can save, load, favorite, and reuse across
    different training jobs.
    """

    __tablename__ = "training_templates"

    # Primary identifiers
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Optional model and dataset references (templates can be generic or specific)
    model_id = Column(
        String(255),
        ForeignKey("models.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    dataset_id = Column(String(255), nullable=True)  # Kept for backward compat
    dataset_ids = Column(JSONB, nullable=False, default=list)  # Array of dataset IDs

    # SAE Architecture type
    encoder_type = Column(String(20), nullable=False, index=True)
    # Valid values: 'standard', 'skip', 'transcoder'

    # Training hyperparameters (flexible JSONB storage)
    hyperparameters = Column(JSONB, nullable=False, default=dict)
    # Contains: hidden_dim, latent_dim, l1_alpha, learning_rate, batch_size,
    #           num_steps, warmup_steps, save_every, log_every, etc.

    # User preferences
    is_favorite = Column(Boolean, nullable=False, default=False, server_default="false", index=True)

    # Additional metadata (flexible JSONB for future extensions)
    # Note: Using 'extra_metadata' instead of 'metadata' as 'metadata' is reserved by SQLAlchemy
    extra_metadata = Column(JSONB, nullable=True, default=dict)
    # Contains: tags, author, version, export_source, use_count, last_used_at, etc.

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<TrainingTemplate(id={self.id}, name={self.name}, encoder_type={self.encoder_type}, is_favorite={self.is_favorite})>"
