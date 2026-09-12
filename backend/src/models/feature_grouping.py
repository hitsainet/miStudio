"""
Feature grouping database models.

Cross-feature grouping (Feature 010): a precompute job builds a token→feature
inverted index for an extraction, then splits each shared-token bucket into
context-similarity subgroups. Labels/stars are NEVER copied into these tables —
they are joined live from ``features`` at query time (one source of truth).

Extractions are immutable once complete, so an index never goes stale; a
recompute (``force`` or changed params) replaces the previous run's rows via
the ``run_id`` cascade.
"""

import uuid
from enum import Enum

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    ForeignKey,
    BigInteger,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from ..core.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class GroupingRunStatus(str, Enum):
    """Lifecycle of a feature-grouping precompute run."""
    PENDING = "pending"
    COMPUTING = "computing"
    COMPLETED = "completed"
    FAILED = "failed"
    #: See EnhancedLabelingStatus.CANCELLED — String(20) column, plain addition.
    CANCELLED = "cancelled"


class FeatureGroupingRun(Base):
    """One precompute run of the grouping index for an extraction."""

    __tablename__ = "feature_grouping_runs"

    id = Column(String(36), primary_key=True, default=_uuid)
    extraction_id = Column(String(255), nullable=False, index=True)
    status = Column(String(20), nullable=False, server_default="pending")
    # {context_window, similarity_threshold, top_examples, min_group_size}
    params = Column(JSONB, nullable=False)
    params_hash = Column(String(64), nullable=False)
    feature_count = Column(Integer, nullable=True)
    group_count = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<FeatureGroupingRun(id={self.id}, extraction={self.extraction_id}, status={self.status})>"


class FeatureTokenIndex(Base):
    """Inverted index row: one (feature, normalized token) pair with context bag."""

    __tablename__ = "feature_token_index"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(
        String(36),
        ForeignKey("feature_grouping_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    extraction_id = Column(String(255), nullable=False)
    feature_id = Column(
        String(255),
        ForeignKey("features.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    neuron_index = Column(Integer, nullable=False)
    raw_token = Column(Text, nullable=False)
    normalized_token = Column(Text, nullable=False)
    token_rank = Column(Integer, nullable=False)  # 1 = feature's top token
    weight = Column(Float, nullable=False)  # activation-weighted share of examples
    context_tokens = Column(JSONB, nullable=True)  # normalized ±window token bag

    __table_args__ = (
        Index("ix_fti_ext_token", "extraction_id", "normalized_token"),
    )


class FeatureGroup(Base):
    """A context-similarity subgroup of features sharing a normalized top token."""

    __tablename__ = "feature_groups"

    id = Column(String(36), primary_key=True, default=_uuid)
    run_id = Column(
        String(36),
        ForeignKey("feature_grouping_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    extraction_id = Column(String(255), nullable=False, index=True)
    normalized_token = Column(Text, nullable=False)
    display_token = Column(Text, nullable=False)  # most common raw surface form
    member_count = Column(Integer, nullable=False)
    cohesion = Column(Float, nullable=False)  # mean pairwise cosine within group

    __table_args__ = (
        Index("ix_fg_ext_token", "extraction_id", "normalized_token"),
    )

    def __repr__(self) -> str:
        return f"<FeatureGroup(token={self.normalized_token!r}, members={self.member_count}, cohesion={self.cohesion:.2f})>"


class FeatureGroupMember(Base):
    """Membership row linking a group to a feature."""

    __tablename__ = "feature_group_members"

    group_id = Column(
        String(36),
        ForeignKey("feature_groups.id", ondelete="CASCADE"),
        primary_key=True,
    )
    feature_id = Column(
        String(255),
        ForeignKey("features.id", ondelete="CASCADE"),
        primary_key=True,
    )
    similarity = Column(Float, nullable=False)  # cosine to group centroid
    context_snippet = Column(Text, nullable=True)  # "prefix *prime* suffix", ≤160 chars
