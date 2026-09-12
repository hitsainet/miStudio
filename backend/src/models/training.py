"""
Training database model.

This module defines the SQLAlchemy model for SAE training jobs.
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Training(Base):
    """
    Training database model for SAE training jobs.

    Stores training configuration, hyperparameters, progress tracking,
    and real-time metrics for Sparse Autoencoder training.
    """

    __tablename__ = "trainings"

    # Primary identifiers
    id = Column(String(255), primary_key=True)  # Format: train_{uuid}

    # Foreign keys
    model_id = Column(String(255), ForeignKey("models.id", ondelete="RESTRICT"), nullable=False)
    dataset_id = Column(String(255), nullable=False)  # Kept for backward compat, use dataset_ids instead
    dataset_ids = Column(JSONB, nullable=False, default=list)  # Array of dataset IDs for multi-dataset training
    extraction_id = Column(
        String(255),
        ForeignKey("activation_extractions.id", ondelete="RESTRICT"),
        nullable=True
    )  # Backward compat: first extraction in list
    extraction_ids = Column(JSONB, nullable=True, default=list)  # Array of extraction IDs for multi-dataset cached activations

    # Status and progress
    status = Column(String(50), nullable=False, default=TrainingStatus.PENDING.value)
    progress = Column(Float, nullable=False, default=0.0)  # 0-100
    current_step = Column(Integer, nullable=False, default=0)
    total_steps = Column(Integer, nullable=False)
    # Set when a stopped run was finalized from a checkpoint. status becomes
    # COMPLETED so the SAE import path unlocks, but progress/current_step stay
    # truthful — this column is what lets the UI say "finalized early at step N"
    # instead of presenting a partial run as a finished one.
    finalized_from_step = Column(
        Integer,
        nullable=True,
        comment=(
            "Checkpoint step this training was finalized from when stopped "
            "early; NULL if it ran to completion"
        ),
    )

    # Hyperparameters (flexible JSONB storage)
    hyperparameters = Column(JSONB, nullable=False, default=dict)
    # Contains: hidden_dim, latent_dim, l1_alpha, learning_rate, batch_size,
    #           architecture_type (standard/skip/transcoder), warmup_steps,
    #           checkpoint_interval, log_interval, etc.

    # Current metrics (latest values for quick access)
    current_loss = Column(Float, nullable=True)
    current_l0_sparsity = Column(Float, nullable=True)
    #: Fraction of Variance Unexplained: var(x - x_hat) / var(x).
    #: 0.0 = perfect reconstruction, 1.0 = no better than predicting the mean.
    #: Scale-free, unlike raw MSE, so it is comparable across layers and models
    #: — which is why it is the metric to watch for convergence. Nullable
    #: because only architectures that compute it (JumpReLU) report one.
    current_fvu = Column(Float, nullable=True)
    current_dead_neurons = Column(Integer, nullable=True)
    current_learning_rate = Column(Float, nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)

    # Celery task tracking
    celery_task_id = Column(String(255), nullable=True)

    # File paths
    checkpoint_dir = Column(String(1000), nullable=True)  # /data/trainings/{id}/checkpoints/
    logs_path = Column(String(1000), nullable=True)  # /data/trainings/{id}/logs.txt

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    metrics = relationship(
        "TrainingMetric",
        back_populates="training",
        cascade="all, delete-orphan",
        order_by="TrainingMetric.step"
    )
    checkpoints = relationship(
        "Checkpoint",
        back_populates="training",
        cascade="all, delete-orphan",
        order_by="Checkpoint.step"
    )
    extraction_jobs = relationship(
        "ExtractionJob",
        back_populates="training",
        cascade="all, delete-orphan"
    )
    features = relationship(
        "Feature",
        back_populates="training",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Training(id={self.id}, status={self.status}, progress={self.progress:.1f}%)>"
