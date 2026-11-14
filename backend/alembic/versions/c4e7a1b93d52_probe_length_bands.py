"""probe_monitors.length_bands — a threshold that varies with input length

Revision ID: c4e7a1b93d52
Revises: b8d3f1a92c47
Create Date: 2026-10-01

⚠ NULLABLE WITH NO SERVER DEFAULT AND NO BACKFILL, DELIBERATELY.

NULL means "this probe was calibrated before length bands existed, or without a calibration
view". That is a true statement about every existing row and it must stay distinguishable from
`[]`, which would mean "bands were attempted and came back empty".

This estate has shipped the other choice and paid for it: `chat_format` was added
`NOT NULL DEFAULT 'auto'`, which made every historical row claim it had used the tokenizer's
real chat template — a column added to stop a silent misattribution introducing one. A default
here would make every probe ever trained claim a length-varying operating point it has not got.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "c4e7a1b93d52"
down_revision = "b8d3f1a92c47"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "probe_monitors",
        sa.Column("length_bands", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("probe_monitors", "length_bands")
