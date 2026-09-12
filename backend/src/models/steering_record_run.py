"""In-flight marker for steered-transcript recording jobs.

A record job loads the model on the single GPU exactly like calibration, so it
must be visible to the single-GPU guard (`assert_no_active_gpu_run`) and
reclaimable by the stuck-run cleanup. Circuit calibration/faithfulness track that
lifecycle on the Circuit row, but a record job may target a CLUSTER or an ad-hoc
FEATURE set — which have no circuit row — so the marker lives in its own table,
uniform across all three artifact types.

The record's OUTPUT is a `steering_samples` manifest; this row is only the
lifecycle/guard marker (status + task id + which artifact), not the transcripts.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, String

from ..core.database import Base


def _srr_id() -> str:
    return f"srr_{uuid.uuid4().hex[:12]}"


class SteeringRecordRun(Base):
    """A steered-transcript recording job's in-flight marker."""

    __tablename__ = "steering_record_runs"

    id = Column(String(36), primary_key=True, default=_srr_id)
    status = Column(String(16), nullable=False, default="pending")  # pending|running|completed|failed
    task_id = Column(String(155), nullable=True)
    artifact_kind = Column(String(16), nullable=False)  # circuit | cluster | features
    artifact_ref = Column(String(64), nullable=True)    # circuit_id / cluster_profile_id (features: null)
    manifest_ref = Column(String(36), nullable=True)    # the steering_samples manifest it produced
    error = Column(String(500), nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
