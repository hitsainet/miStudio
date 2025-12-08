"""
Feature Analysis Cache database model.

This module defines the SQLAlchemy model for caching expensive analysis results.
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import Column, BigInteger, String, DateTime, ForeignKey, Enum as SQLAEnum
from sqlalchemy.dialects.postgresql import JSONB, ENUM as PostgreSQLEnum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class AnalysisType(str, Enum):
    """Feature analysis type."""
    LOGIT_LENS = "logit_lens"  # Logit lens analysis: top predicted tokens
    CORRELATIONS = "correlations"  # Correlation with other features
    ABLATION = "ablation"  # Ablation impact on model performance
    NLP_ANALYSIS = "nlp_analysis"  # Enhanced NLP analysis: POS, NER, patterns, clusters


# Create the PostgreSQL ENUM type
analysis_type_enum = PostgreSQLEnum(
    'logit_lens', 'correlations', 'ablation', 'nlp_analysis',
    name='analysis_type_enum',
    create_type=False  # Type already exists from migration
)


class FeatureAnalysisCache(Base):
    """
    Feature Analysis Cache database model.

    Caches expensive analysis computations (logit lens, correlations, ablation)
    to avoid recomputing them on every request. Cache expires after 7 days.
    """

    __tablename__ = "feature_analysis_cache"

    # Primary identifiers
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    feature_id = Column(
        String(255),
        ForeignKey("features.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    analysis_type = Column(analysis_type_enum, nullable=False, index=True)

    # Analysis results
    result = Column(JSONB, nullable=False)
    # Structure depends on analysis_type:
    # - logit_lens: {top_tokens: [str], probabilities: [float], interpretation: str}
    # - correlations: {correlated_features: [{feature_id, correlation}]}
    # - ablation: {perplexity_delta: float, impact_score: float, baseline_perplexity: float, ablated_perplexity: float}

    # Cache metadata
    computed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)

    # Relationships
    feature = relationship("Feature", back_populates="analysis_cache")

    def __repr__(self) -> str:
        return f"<FeatureAnalysisCache(id={self.id}, feature_id={self.feature_id}, analysis_type={self.analysis_type})>"
