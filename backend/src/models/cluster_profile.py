"""
Cluster profile model (Feature 014, IDL-30).

A cluster profile is the durable, user-authored capture of a tuned cluster:
name + narrative + members with their tuned strengths + the budget/allocation
snapshot from Feature 013. Profiles are deliberately DECOUPLED from the
recomputable grouping tables (`feature_groups` et al.): recomputing the cluster
index must never destroy tuned human work, so `source_group_id` is a soft
reference (no FK). Members/budget are JSONB snapshots by design — the profile
records what the user tuned against, not a live join.

The profile is also the unit of portability: it serializes to the
`mistudio.cluster-definition/v1` interchange JSON (consumer-neutral; the future
MILLM / unified-MCP / Open WebUI arc consumes the same format).
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from ..core.database import Base


def _profile_id() -> str:
    return f"clp_{uuid.uuid4().hex[:12]}"


class ClusterProfile(Base):
    """A named, narrated, strength-tuned cluster — durable and exportable."""

    __tablename__ = "cluster_profiles"

    id = Column(String(36), primary_key=True, default=_profile_id)

    # Binding: profiles belong to an SAE (indices are meaningless without it).
    # RESTRICT: deleting an SAE with profiles must be an explicit, surfaced act.
    sae_id = Column(
        String(255),
        ForeignKey("external_saes.id", ondelete="RESTRICT"),
        nullable=True,  # imported-unbound profiles have no local SAE yet
        index=True,
    )
    model_id = Column(String(255), nullable=True)  # miStudio model id at save time
    extraction_id = Column(String(255), nullable=True)  # soft context
    source_group_id = Column(String(36), nullable=True)  # SOFT ref — groups are ephemeral

    name = Column(String(120), nullable=False)
    narrative = Column(Text, nullable=True)  # markdown
    display_token = Column(String(255), nullable=True)

    # [{feature_idx, label, similarity, activation_frequency, max_activation,
    #   strength, sign, pinned}]
    members = Column(JSONB, nullable=False)
    # {B, B_dir, G, f_eff, formula_id, constants, intensity, intensity_range}
    budget = Column(JSONB, nullable=True)

    schema_version = Column(String(8), nullable=False, default="1")
    # Import provenance: {kind, schema_version, exported_at, source_note}
    imported_from = Column(JSONB, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (Index("ix_clp_sae_name", "sae_id", "name"),)

    def __repr__(self) -> str:
        return f"<ClusterProfile(id={self.id}, name={self.name!r}, sae={self.sae_id})>"
