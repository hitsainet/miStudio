"""
SQLAlchemy model for Steering Experiments.

Stores saved steering comparison experiments for later viewing and analysis.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy import String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class SteeringExperiment(Base):
    """
    Database model for saved steering experiments.

    Stores the complete configuration and results of a steering comparison
    for later viewing, sharing, and analysis.
    """

    __tablename__ = "steering_experiments"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Experiment metadata
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Source identifiers
    sae_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    comparison_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)

    # Input configuration
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    selected_features: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )
    generation_params: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # Complete results (SteeringComparisonResponse as JSONB)
    results: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )

    # Tags for categorization
    tags: Mapped[List[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<SteeringExperiment(id={self.id}, name='{self.name}', sae_id='{self.sae_id}')>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "sae_id": self.sae_id,
            "model_id": self.model_id,
            "prompt": self.prompt,
            "selected_features": self.selected_features,
            "generation_params": self.generation_params,
            "results": self.results,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
