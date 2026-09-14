"""gpu leases: which job holds each GPU

Revision ID: e6a1c4f8b2d9
Revises: d3f7b1c9e5a2
Create Date: 2026-09-14

Multi-GPU Phase 3 runs one job per card at once. A lease records which job holds
a card from the moment it is chosen, because NVML free memory cannot: a job that
has just been placed has not allocated yet. One row per card (the UUID is the
primary key, so a second holder is refused by the key itself); a split job holds
one row per card. `expires_at` frees a card whose holder stopped renewing.
New table; the downgrade drops it.
"""
from alembic import op
import sqlalchemy as sa

revision = "e6a1c4f8b2d9"
down_revision = "d3f7b1c9e5a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gpu_leases",
        sa.Column("gpu_uuid", sa.String(length=64), primary_key=True),
        sa.Column("holder", sa.String(length=128), nullable=False),
        sa.Column("task_id", sa.String(length=155), nullable=True),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("gpu_leases")
