"""
Checkpoint database model.

This module defines the SQLAlchemy model for training checkpoints.
"""

from datetime import datetime

from sqlalchemy import Column, String, Integer, BigInteger, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class Checkpoint(Base):
    """
    Checkpoint database model for SAE training checkpoints.

    Stores checkpoint metadata and file paths for model state snapshots
    saved during training. Enables resuming training and model recovery.
    """

    __tablename__ = "checkpoints"

    # Primary identifier
    id = Column(String(255), primary_key=True)  # Format: ckpt_{uuid}

    # Foreign key
    training_id = Column(
        String(255),
        ForeignKey("trainings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Checkpoint information
    step = Column(Integer, nullable=False, index=True)

    # Metrics at checkpoint
    loss = Column(Float, nullable=False)
    l0_sparsity = Column(Float, nullable=True)

    # File storage
    storage_path = Column(String(1000), nullable=False)  # Path to .safetensors file
    file_size_bytes = Column(BigInteger, nullable=True)

    # Checkpoint metadata
    is_best = Column(Boolean, nullable=False, default=False)  # Best checkpoint by loss
    extra_metadata = Column(JSONB, nullable=True, default=dict)
    # Contains: optimizer_state, scheduler_state, dead_neurons_mask,
    #           feature_statistics, training_config, etc.

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationship
    training = relationship("Training", back_populates="checkpoints")

    def __repr__(self) -> str:
        best_str = " [BEST]" if self.is_best else ""
        return (
            f"<Checkpoint(id={self.id}, training_id={self.training_id}, "
            f"step={self.step}, loss={self.loss:.4f}{best_str})>"
        )
