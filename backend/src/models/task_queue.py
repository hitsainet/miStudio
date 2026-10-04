"""
TaskQueue model for tracking background operations and retry state.

This model provides visibility into all background tasks (downloads, training, extraction)
and enables user-controlled retry of failed operations.
"""

from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Text, DateTime, JSON
from sqlalchemy.sql import func
from ..core.database import Base


class TaskQueue(Base):
    """
    Background task tracking and retry management.

    Tracks all Celery tasks to provide visibility and enable manual retry control.
    """
    __tablename__ = "task_queue"

    # Primary identification
    id = Column(String, primary_key=True)  # Format: tq_{uuid}
    task_id = Column(String, nullable=True, index=True)  # Celery task ID
    # The GPU card (UUID) the task ran on; its request travels with the task.
    gpu_uuid = Column(String(64), nullable=True)
    # Every card of a split, most free first (the first is gpu_uuid; layers fill
    # them in CUDA index order); NULL for a single-card task. Generic
    # JSON like `retry_params`, not JSONB: tests build this table on SQLite, which
    # cannot render JSONB.
    gpu_uuids = Column(JSON, nullable=True)

    # Task classification
    task_type = Column(String, nullable=False)  # download, training, extraction
    entity_id = Column(String, nullable=False, index=True)  # model_id, training_id, etc.
    entity_type = Column(String, nullable=False, index=True)  # model, training, extraction

    # Task state
    status = Column(String, nullable=False, index=True)  # queued, running, failed, completed, cancelled
    progress = Column(Float, nullable=True)  # 0-100
    error_message = Column(Text, nullable=True)

    # Retry management
    retry_params = Column(JSON, nullable=True)  # Parameters for retry (repo_id, quantization, etc.)
    retry_count = Column(Integer, nullable=False, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<TaskQueue(id={self.id}, type={self.task_type}, entity={self.entity_id}, status={self.status})>"
