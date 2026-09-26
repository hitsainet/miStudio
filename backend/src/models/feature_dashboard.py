"""
Feature Dashboard Data database model.

This module defines the SQLAlchemy model for storing computed dashboard data
for SAE features. This includes logit lens data, activation histograms, and
aggregated top tokens - data needed for Neuronpedia feature visualization.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, BigInteger, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class FeatureDashboardData(Base):
    """
    Feature Dashboard Data database model.

    Stores computed visualization data for a feature including:
    - Logit lens: Top tokens influenced by this feature (positive and negative)
    - Histogram: Distribution of activation values
    - Top tokens: Aggregated ranking of tokens by total activation across examples

    This data is computed during extraction or on-demand for Neuronpedia export.
    """

    __tablename__ = "feature_dashboard_data"

    # Primary identifier - auto-incrementing BIGINT
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Feature reference - one-to-one with features table
    feature_id = Column(
        String(255),
        ForeignKey("features.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )

    # Logit lens data - top tokens influenced by this feature
    # Format: {
    #   "top_positive": [{"token": "dog", "token_id": 123, "logit": 2.34}, ...],
    #   "top_negative": [{"token": "cat", "token_id": 456, "logit": -1.23}, ...]
    # }
    logit_lens_data = Column(JSONB, nullable=True)

    # Histogram data - activation value distribution
    # Format: {
    #   "bin_edges": [0.0, 0.1, 0.2, ...],
    #   "counts": [100, 50, 25, ...],
    #   "total_count": 10000,
    #   "nonzero_count": 500,
    #   "mean": 0.05,
    #   "std": 0.1,
    #   "max": 5.2
    # }
    histogram_data = Column(JSONB, nullable=True)

    # Top tokens aggregated across all examples
    # Format: [
    #   {"token": "the", "token_id": 1, "total_activation": 123.4, "count": 50,
    #    "mean_activation": 2.47, "max_activation": 8.5},
    #   ...
    # ]
    top_tokens = Column(JSONB, nullable=True)

    # Computation metadata
    computed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    computation_version = Column(String(50), nullable=True)  # Algorithm version for cache invalidation

    # Relationship to Feature (backref)
    feature = relationship("Feature", backref="dashboard_data", uselist=False)

    def __repr__(self) -> str:
        return f"<FeatureDashboardData(id={self.id}, feature_id={self.feature_id})>"

    @property
    def has_logit_lens(self) -> bool:
        """Check if logit lens data is computed."""
        return self.logit_lens_data is not None and len(self.logit_lens_data) > 0

    @property
    def has_histogram(self) -> bool:
        """Check if histogram data is computed."""
        return self.histogram_data is not None and len(self.histogram_data) > 0

    @property
    def has_top_tokens(self) -> bool:
        """Check if top tokens data is computed."""
        return self.top_tokens is not None and len(self.top_tokens) > 0

    @property
    def is_complete(self) -> bool:
        """Check if all dashboard data is computed."""
        return self.has_logit_lens and self.has_histogram and self.has_top_tokens

    def get_top_positive_tokens(self, k: int = 10) -> List[Dict[str, Any]]:
        """Get top K tokens with positive logit influence."""
        if not self.logit_lens_data:
            return []
        return self.logit_lens_data.get("top_positive", [])[:k]

    def get_top_negative_tokens(self, k: int = 10) -> List[Dict[str, Any]]:
        """Get top K tokens with negative logit influence."""
        if not self.logit_lens_data:
            return []
        return self.logit_lens_data.get("top_negative", [])[:k]

    def get_histogram_stats(self) -> Optional[Dict[str, Any]]:
        """Get summary statistics from histogram data."""
        if not self.histogram_data:
            return None
        return {
            "total_count": self.histogram_data.get("total_count"),
            "nonzero_count": self.histogram_data.get("nonzero_count"),
            "mean": self.histogram_data.get("mean"),
            "std": self.histogram_data.get("std"),
            "max": self.histogram_data.get("max"),
        }
