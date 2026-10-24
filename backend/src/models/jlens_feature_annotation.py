"""Persisted J-space annotations of SAE features (024 tasks 4.3/4.4, BR-012..015).

One row per (SAE, feature, SURFACE, lens type). `surface` separates a feature's DECODER
direction (what it writes) from its ENCODER direction (what it reads) — BR-015 requires
them as separate readouts, because one readout of both hides the transformation the
feature performs, which is the thing worth seeing.

THE DISAGREEMENT QUEUE IS A QUERY OVER THESE ROWS, not a flag on the feature. A flag is
a warning and warnings are not read (024 FPRD §3); a sortable score joined to the feature
list is a queue. Nothing here edits a label: the lens is rung 0 and does not overrule one.

Optional fields are ABSENT (NULL) when not computable, never zero — the arc's rule.
"""

import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from ..core.clock import utc_now
from ..core.database import Base


def _jfa_id() -> str:
    return f"jfa_{uuid.uuid4().hex[:12]}"


class JLensFeatureAnnotation(Base):
    __tablename__ = "jlens_feature_annotations"
    __table_args__ = (
        UniqueConstraint(
            "sae_id", "feature_index", "surface", "lens_type",
            name="uq_jlens_feature_annotation",
        ),
    )

    id = Column(String(36), primary_key=True, default=_jfa_id)
    sae_id = Column(String(255), nullable=False, index=True)
    feature_index = Column(Integer, nullable=False)
    #: "decoder" | "encoder"
    surface = Column(String(16), nullable=False)
    #: "LOGIT_LENS" | "JACOBIAN_LENS"
    lens_type = Column(String(32), nullable=False)
    layer = Column(Integer, nullable=False)
    lens_kurtosis = Column(Float, nullable=True)
    workspace_class = Column(String(16), nullable=False)
    top_tokens = Column(JSONB, nullable=False)
    #: The label compared against, as it was when compared — a later relabel must not
    #: make an old score describe a label it was never computed on.
    compared_label = Column(String(500), nullable=True)
    disagreement_score = Column(Float, nullable=True, index=True)
    has_disagreement = Column(Boolean, nullable=False, default=False)
    #: The sweep task that wrote it, or NULL for a single annotate call.
    sweep_task_id = Column(String(155), nullable=True)
    evidence_rung = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)
