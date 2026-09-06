"""Results of one labeling prompt-template trial over a fixed feature panel.

A trial runs ONE prompt template over a panel of features and records the labels
it produced WITHOUT writing them to the Feature rows. Two trials over the same
panel are then directly comparable, and running five template variants cannot
stomp the user's real labels five times.

Why a dedicated table rather than a `ValidationManifest`:

* `manifest_service._assert_no_paths` raises on any string starting with `/data/`
  or `/home/`, exempting only a small `_TEXT_KEYS` set and only for plain strings.
  A trial payload is corpus passages, labels and descriptions — a feature ABOUT
  filesystem paths would discard a completed, paid run at write time, and
  `validate_payload` runs on write only so there is no recovery. Widening the
  exemption would weaken a guard that exists to protect circuit evidence.
* `ManifestService.list_by_parent` filters only `discovery_run_id`/`circuit_id`,
  both NULL here, so trials would be reachable only through the unbounded
  `GET /validation-manifests` — no limit, no kind filter, and no index on `kind`.
* Comparison needs indexed lookups on `panel_id` and `prompt_template_id`; on a
  JSONB payload those are scans.

The row OUTLIVES its LabelingJob on purpose: `delete_labeling_job` is a
user-reachable endpoint, and deleting the job that produced a measurement must
not delete the measurement.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB

from ..core.database import Base


def _ltr_id() -> str:
    return f"ltr_{uuid.uuid4().hex[:12]}"


class LabelingTrialRun(Base):
    """One prompt template's labels over one fixed feature panel."""

    __tablename__ = "labeling_trial_runs"

    id = Column(String(36), primary_key=True, default=_ltr_id)

    # Content-addressed panel identity: sha256(extraction_job_id | sorted feature ids).
    # Equal panel_id PROVES an identical, order-independent, extraction-bound
    # feature set, so `compare` can refuse a mismatch instead of trusting a join.
    panel_id = Column(String(68), nullable=False)
    extraction_job_id = Column(
        String(255),
        ForeignKey("extraction_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    # SET NULL, not CASCADE — see the module docstring.
    labeling_job_id = Column(
        String(255),
        ForeignKey("labeling_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )
    # The A/B variable. RESTRICT would block template deletion forever, and
    # CASCADE would delete the measurement, so the payload carries a FROZEN COPY
    # of the template body and this ref is provenance only.
    prompt_template_id = Column(String(255), nullable=True)
    name = Column(String(200), nullable=True)

    status = Column(String(16), nullable=False, default="queued")  # queued|running|completed|failed|cancelled
    payload = Column(JSONB, nullable=False, default=dict)
    error = Column(String(500), nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_ltr_panel", "panel_id", "created_at"),
        Index("idx_ltr_extraction", "extraction_job_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<LabelingTrialRun(id={self.id}, panel_id={self.panel_id[:16]}…, "
            f"template={self.prompt_template_id}, status={self.status})>"
        )
