"""
Feature Activation database model.

This module defines the SQLAlchemy model for feature activation examples.
Stores top-K max-activating examples for each feature.
"""

from datetime import datetime

from sqlalchemy import Column, BigInteger, String, Float, Integer, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSON, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class FeatureActivation(Base):
    """
    Feature Activation database model.

    Stores max-activating examples for each discovered feature.
    Each record contains a dataset sample that strongly activates the feature,
    along with token-level activation values for interpretability.

    Note: This table is range-partitioned by feature_id for efficient querying.
    """

    __tablename__ = "feature_activations"

    # Primary identifiers
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    feature_id = Column(
        String(255),
        ForeignKey("features.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True  # Required for partitioning
    )
    sample_index = Column(Integer, nullable=False)

    # Activation data
    max_activation = Column(Float, nullable=False)
    tokens = Column(JSONB, nullable=False)  # Array of token strings (legacy format)
    activations = Column(JSONB, nullable=False)  # Array of activation values

    # Context window data (enhanced format for better interpretability)
    # Based on Anthropic/OpenAI research showing asymmetric context windows improve feature understanding
    prefix_tokens = Column(JSON, nullable=True)  # Array of tokens before the prime token
    prime_token = Column(Text, nullable=True)  # The prime token (token with max activation)
    suffix_tokens = Column(JSON, nullable=True)  # Array of tokens after the prime token
    prime_activation_index = Column(Integer, nullable=True)  # Index of prime token in original activation sequence

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    feature = relationship("Feature", back_populates="activations")

    def __repr__(self) -> str:
        return f"<FeatureActivation(id={self.id}, feature_id={self.feature_id}, max_activation={self.max_activation:.3f})>"
