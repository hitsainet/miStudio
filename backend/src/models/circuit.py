"""
Circuit model (Feature 018, IDL-33).

A circuit is the durable record of a discovered (or hand-assembled)
cross-layer structure: members keyed to layers (feature refs or cluster
refs), typed edges carrying their full evidence trail (rung, statistics,
attribution, validation-manifest refs), per-layer budgets under one global
intensity, and optional faithfulness scores. A PROMOTED circuit IS the
loadable multi-layer steering profile — there is deliberately no dual
entity (018 FTDD §3).

Members/edges/budget/faithfulness are JSONB snapshots mirroring the
mistudio.circuit-definition/v1 contract shapes so export is a projection,
not a join. `discovery_run_id` is a soft reference — discovery runs are
prunable working data; circuits are curated artifacts.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from ..core.database import Base


def _circuit_id() -> str:
    return f"crc_{uuid.uuid4().hex[:12]}"


class Circuit(Base):
    """A cross-layer circuit — reviewable, promotable, portable."""

    __tablename__ = "circuits"

    id = Column(String(36), primary_key=True, default=_circuit_id)

    name = Column(String(120), nullable=False)
    narrative = Column(Text, nullable=True)  # markdown
    granularity = Column(String(16), nullable=False, default="feature")  # feature | cluster

    # Contract-shaped JSONB snapshots (circuit_definition.py shapes).
    saes = Column(JSONB, nullable=False, default=list)      # [DefinitionSAERef]
    members = Column(JSONB, nullable=False, default=list)   # [CircuitMember]
    edges = Column(JSONB, nullable=False, default=list)     # [CircuitEdge]
    budget = Column(JSONB, nullable=True)                    # CircuitBudget
    faithfulness = Column(JSONB, nullable=True)              # CircuitFaithfulness
    calibration = Column(JSONB, nullable=True)               # CircuitCalibration (IDL-37)
    discovery = Column(JSONB, nullable=True)                 # CircuitDiscoveryProvenance

    # Denormalized display rung (min over edges; recomputed on every write).
    rung = Column(Integer, nullable=False, default=0)

    promoted = Column(Boolean, nullable=False, default=False)
    # Optimistic-concurrency version (017 Task 3.0): increments on every
    # update() so a validation writer and a user editing in the panel can't
    # silently clobber each other — a stale write 409s.
    version = Column(Integer, nullable=False, default=1)
    discovery_run_id = Column(String(36), nullable=True)  # SOFT ref — runs are prunable
    model_id = Column(String(255), nullable=True)
    # HF repo id — the CROSS-INSTANCE-stable model identifier (model_id is
    # instance-local); 015's model-mismatch check compares against this.
    model_hf_id = Column(String(500), nullable=True)
    # Faithfulness (rung 3) runs on the CIRCUIT (not a discovery run), so its
    # in-flight lifecycle lives here (R2 B-5): the single-GPU guard checks it,
    # cleanup reclaims a stuck one, and two runs can't race.
    faithfulness_status = Column(String(16), nullable=True)  # pending|running|completed|failed
    faithfulness_task_id = Column(String(155), nullable=True)
    # Calibration (IDL-37) runs on the CIRCUIT and holds the GPU like
    # faithfulness — same in-flight lifecycle so the single-GPU guard sees it,
    # cleanup can reclaim a stuck one, and two calibrations can't race.
    calibration_status = Column(String(16), nullable=True)  # pending|running|completed|failed
    calibration_task_id = Column(String(155), nullable=True)
    schema_version = Column(String(8), nullable=False, default="1")

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        Index("idx_circuits_promoted_rung", "promoted", "rung"),
    )
