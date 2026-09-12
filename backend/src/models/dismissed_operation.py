"""Dismissed-operation markers.

A row here hides one federated failure from the Monitor's Failed Operations
list. It does NOT delete the underlying job record: the training, extraction,
labeling job or Neuronpedia push keeps its status and error message, and the
dismissal can be undone by removing the marker.

See the ``c7e2a4f18b93`` migration for why this is a separate table rather than
a column on each federated source.
"""

from sqlalchemy import Column, DateTime, Index, String
from sqlalchemy.sql import func

from ..core.database import Base


class DismissedOperation(Base):
    """Marks one (source_type, source_id) pair as cleared from the Monitor."""

    __tablename__ = "dismissed_operations"

    source_type = Column(String(50), primary_key=True)
    source_id = Column(String(255), primary_key=True)
    dismissed_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_dismissed_operations_source_type", "source_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<DismissedOperation({self.source_type}:{self.source_id} "
            f"at {self.dismissed_at})>"
        )
