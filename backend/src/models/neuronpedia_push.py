"""
Neuronpedia Push Job database model.

Tracks direct pushes of an SAE's features/weights/dashboard data to a local
Neuronpedia instance. This model wraps the pre-existing ``neuronpedia_pushes``
table (created by migration ``m9n0o1p2q3r4``), which was previously accessed
only via raw SQL. The column set here mirrors that table exactly.
"""

from enum import Enum

from sqlalchemy import Column, String, Integer, Float, DateTime, Text
from sqlalchemy.sql import func

from ..core.database import Base


class NeuronpediaPushStatus(str, Enum):
    """Status of a Neuronpedia local push job.

    Values match the status strings written by
    ``workers/neuronpedia_push_tasks.py`` and read by the task_queue
    federation in ``api/v1/endpoints/task_queue.py``.
    """
    QUEUED = "queued"          # Job created, waiting for the worker
    PREPARING = "preparing"    # Creating the Neuronpedia model/source
    PUSHING = "pushing"        # Streaming features to Neuronpedia
    COMPLETED = "completed"    # Push finished successfully
    FAILED = "failed"          # Push failed with error


class NeuronpediaPushJob(Base):
    """
    Neuronpedia local push job.

    Tracks a direct push to a local Neuronpedia instance so it appears in the
    Active/Failed Operations monitor. Progress is streamed via WebSocket on
    channel ``neuronpedia/push/{id}``.
    """

    __tablename__ = "neuronpedia_pushes"

    # Primary identifier — format: push_{sae_id}_{timestamp}
    id = Column(String(255), primary_key=True)

    # SAE being pushed (external_sae.id)
    sae_id = Column(String(255), nullable=False, index=True)

    # Job state
    status = Column(String(50), nullable=False, server_default="queued")
    progress = Column(Float, nullable=False, server_default="0")
    features_pushed = Column(Integer, nullable=False, server_default="0")
    total_features = Column(Integer, nullable=False, server_default="0")
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<NeuronpediaPushJob(id={self.id}, sae_id={self.sae_id}, "
            f"status={self.status}, progress={self.progress})>"
        )
