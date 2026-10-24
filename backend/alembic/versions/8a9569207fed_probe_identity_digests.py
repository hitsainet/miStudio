"""probe identity digests: decision_snapshot, decision_digest, built_detector_digest

Revision ID: 8a9569207fed
Revises: f3a9c2e5b8d1
Create Date: 2026-10-23

Feature 039 (BRD-MIS-BEHAVIOUR-DETECTORS-001 BR-025; PADR IDL-60). Three nullable columns and two
b-tree indexes on `probe_monitors`; no data statement and no backfill. NULL means NOT RECORDED: a probe
cut before 039 forms its decision snapshot lazily on first need, and its identity is recomputed once
from its cached definition (FR-17). Adding nullable columns without a default rewrites no row; the two
index builds are non-concurrent, acceptable for a table of a few hundred rows (row count checked before
the push, FTASKS 3.2).
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "8a9569207fed"
down_revision = "f3a9c2e5b8d1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("probe_monitors", sa.Column("decision_snapshot", JSONB(), nullable=True))
    op.add_column("probe_monitors", sa.Column("decision_digest", sa.String(length=64), nullable=True))
    op.add_column(
        "probe_monitors", sa.Column("built_detector_digest", sa.String(length=64), nullable=True)
    )
    op.create_index("ix_probe_monitors_decision_digest", "probe_monitors", ["decision_digest"])
    op.create_index(
        "ix_probe_monitors_built_detector_digest", "probe_monitors", ["built_detector_digest"]
    )


def downgrade() -> None:
    op.drop_index("ix_probe_monitors_built_detector_digest", table_name="probe_monitors")
    op.drop_index("ix_probe_monitors_decision_digest", table_name="probe_monitors")
    op.drop_column("probe_monitors", "built_detector_digest")
    op.drop_column("probe_monitors", "decision_digest")
    op.drop_column("probe_monitors", "decision_snapshot")
