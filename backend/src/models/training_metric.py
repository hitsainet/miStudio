"""
Training Metric database model.

This module defines the SQLAlchemy model for time-series training metrics.
"""

from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, BigInteger, Float, DateTime, ForeignKey,
    Index, text,
)
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

    # Which SAE of the layer (review R1-A, A5). A training trains one SAE per
    # (layer, hook type); the per-SAE and held-out rows carry the hook, so two hook
    # types on one layer are two rows, not one key written twice. NULL on aggregated
    # rows (they describe every SAE) and on rows written before 2026-09-15.
    hook_type = Column(
        String(50),
        nullable=True,
        comment=(
            "The SAE's hook type (residual, mlp, attention) on per-SAE and held-out rows. "
            "NULL on aggregated rows and on rows written before 2026-09-15."
        ),
    )

    # Loss metrics
    loss = Column(Float, nullable=False)  # Total reconstruction loss
    loss_reconstructed = Column(Float, nullable=True)  # Reconstruction component
    loss_zero = Column(Float, nullable=True)  # Zero ablation loss

    # Sparsity metrics
    l0_mean = Column(
        Float,
        nullable=True,
        comment=(
            "Active features PER TOKEN (a count). `l0_sparsity` beside it is the "
            "fraction of d_sae — at d_sae=8192 a fraction of 0.0094 is ~77 "
            "features. The count is the interpretable number and was computed "
            "every step and discarded."
        ),
    )
    l0_sparsity = Column(Float, nullable=True)  # Fraction of active features
    l1_sparsity = Column(Float, nullable=True)  # L1 sparsity penalty
    dead_neurons = Column(Integer, nullable=True)  # Count of dead neurons

    # Reconstruction quality metrics.
    #
    # TWO FVUs, NOT ONE (SAE training remediation item 5). `fvu` is the LEGACY
    # value, var(x - x_hat) / var(x) over every element with one global mean, and
    # it keeps that meaning in every row, old and new. It reads LOW on activations
    # with large constant offset dimensions (0.26 vs 0.32 at LFM2.5-1.2B L11).
    # `fvu_centred` is the standard per-dimension-centred value and the headline;
    # NULL on every row written before it existed. See src/ml/sae_metrics.py.
    fvu = Column(Float, nullable=True)
    fvu_centred = Column(
        Float,
        nullable=True,
        comment=(
            "Standard FVU: sum|x - x_hat|^2 / sum|x - mu|^2 with mu the per-dimension "
            "mean, raw activation space. `fvu` beside it is the legacy global-mean value. "
            "NULL on rows written before 2026-09-15."
        ),
    )

    # Training dynamics
    learning_rate = Column(Float, nullable=True)  # Current learning rate
    grad_norm = Column(Float, nullable=True)  # Gradient norm

    # Resource metrics
    gpu_memory_used_mb = Column(Float, nullable=True)  # GPU memory usage in MB
    samples_per_second = Column(Float, nullable=True)  # Training throughput

    # Relationship
    training = relationship("Training", back_populates="metrics")

    __table_args__ = (
        # One metric row per (training, step, layer, hook): review R1-A finding A5.
        # The key was (training, step, layer), so a training over two hook types
        # wrote the same key twice at its first log step and failed.
        #
        # COALESCE(hook_type, '') and NOT layer_idx, deliberately (migration
        # d5a1f3c7e9b2 has the full reasoning):
        # * rows written before hook_type existed (NULL hook) keep exactly the old
        #   (training, step, layer) uniqueness;
        # * a per-SAE row logged WITHOUT its hook still collides on a multi-hook
        #   run, loudly, instead of being accepted beside its sibling;
        # * aggregated rows (layer_idx NULL) stay unconstrained, as before: Postgres
        #   treats NULLs as distinct. `NULLS NOT DISTINCT` is Postgres 15+, and the
        #   compose files and the dev database run 14.
        Index(
            "uq_training_metrics_tid_step_layer_hook",
            "training_id", "step", "layer_idx", text("COALESCE(hook_type, '')"),
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingMetric(id={self.id}, training_id={self.training_id}, "
            f"step={self.step}, loss={self.loss:.4f})>"
        )
