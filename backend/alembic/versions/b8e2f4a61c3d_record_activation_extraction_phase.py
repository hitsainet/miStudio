"""record which step a running activation extraction is in

Revision ID: b8e2f4a61c3d
Revises: 7c1e9a4d2b60
Create Date: 2026-09-13

After the GPU pass reaches N/N an activation extraction merges its per-batch
shards into one file per layer, then computes statistics. Both steps can take
longer than the GPU pass. The steps were reported only in WebSocket messages, so
after a page reload the card fell back to the row's `status` and read
"Extracting 20000/20000" for the rest of the job.

`phase` is one nullable String column: "extracting", "merging" or "statistics".
`status` is unchanged and stays EXTRACTING through all three, so every guard
that reads it (active-extraction queries, cancel, delete, the janitor) keeps its
meaning. Existing rows get NULL, which the UI reads as "no step recorded".
Additive; the downgrade drops the column.
"""
from alembic import op
import sqlalchemy as sa

revision = "b8e2f4a61c3d"
down_revision = "7c1e9a4d2b60"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "activation_extractions",
        sa.Column("phase", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("activation_extractions", "phase")
