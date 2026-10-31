"""
Validation manifest model (Feature 017, IDL-34 — FTDD §4).

A manifest is a SELF-CONTAINED record of a validation run: everything needed
to REPRODUCE it (intervention config, baseline, prompts, seeds, null summary,
per-edge/per-member values). No live refs that can drift — reproduction is
the correctness test. Manifest ids travel into 018's contract as
`validation_manifest_ref`, so an exported circuit's causal claims point at a
reproducible record.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Index, String
from sqlalchemy.dialects.postgresql import JSONB

from ..core.database import Base


def _vman_id() -> str:
    return f"vman_{uuid.uuid4().hex[:12]}"


class ValidationManifest(Base):
    """A reproducible validation record (edge batch, faithfulness, or a
    reproduction of one)."""

    __tablename__ = "validation_manifests"

    id = Column(String(36), primary_key=True, default=_vman_id)
    kind = Column(String(24), nullable=False)  # edge_batch | faithfulness | reproduction | calibration | steering_samples

    # Soft parent refs (prunable working data — the manifest is the record).
    discovery_run_id = Column(String(36), nullable=True)
    circuit_id = Column(String(36), nullable=True)
    parent_manifest_id = Column(String(36), nullable=True)  # reproduction → its source

    # Self-contained payload — everything §2/§3 lists (intervention, baseline,
    # prompts, seeds, cfg, null summary, per-edge/member values, metric_id).
    payload = Column(JSONB, nullable=False, default=dict)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_vman_discovery", "discovery_run_id"),
        Index("idx_vman_circuit", "circuit_id"),
    )
