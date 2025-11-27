"""
Training Metric database model.

This module defines the SQLAlchemy model for time-series training metrics.
"""

from datetime import datetime

from sqlalchemy import Column, String, Integer, BigInteger, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class TrainingMetric(Base):
    """
    Training Metric database model for time-series metrics data.

    Stores detailed metrics at each logging step during training,
    enabling real-time monitoring and post-training analysis.
    """

    __tablename__ = "training_metrics"

    # Primary key
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Foreign key
    training_id = Column(
        String(255),
        ForeignKey("trainings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Step information
    step = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Layer information (NULL for aggregated metrics across all layers)
    layer_idx = Column(Integer, nullable=True, index=True, comment="Layer index (NULL for aggregated metrics)")

    # Loss metrics
    loss = Column(Float, nullable=False)  # Total reconstruction loss
    loss_reconstructed = Column(Float, nullable=True)  # Reconstruction component
    loss_zero = Column(Float, nullable=True)  # Zero ablation loss

    # Sparsity metrics
    l0_sparsity = Column(Float, nullable=True)  # Fraction of active features
    l1_sparsity = Column(Float, nullable=True)  # L1 sparsity penalty
    dead_neurons = Column(Integer, nullable=True)  # Count of dead neurons

    # Reconstruction quality metrics
    fvu = Column(Float, nullable=True)  # Fraction of Variance Unexplained (var_residuals / var_original)

    # Training dynamics
    learning_rate = Column(Float, nullable=True)  # Current learning rate
    grad_norm = Column(Float, nullable=True)  # Gradient norm

    # Resource metrics
    gpu_memory_used_mb = Column(Float, nullable=True)  # GPU memory usage in MB
    samples_per_second = Column(Float, nullable=True)  # Training throughput

    # Relationship
    training = relationship("Training", back_populates="metrics")

    def __repr__(self) -> str:
        return (
            f"<TrainingMetric(id={self.id}, training_id={self.training_id}, "
            f"step={self.step}, loss={self.loss:.4f})>"
        )
