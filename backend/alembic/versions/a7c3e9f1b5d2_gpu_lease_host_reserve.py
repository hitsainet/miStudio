"""gpu leases: the host RAM a holder reserved and has not yet allocated

Revision ID: a7c3e9f1b5d2
Revises: e6a1c4f8b2d9
Create Date: 2026-09-14

Multi-GPU Phase 3, review round 1 (R1-6). The host RAM guard subtracts, for every
other live job, the host memory its kind reserves (a cached-activation training's
pinned rolling buffer) because the buffer may not be allocated yet. Once it IS
allocated, the kernel's MemAvailable already excludes it, and subtracting the
reserve again counted it twice. A holder now records its reserve on its lease rows
and zeroes it once allocated. Nullable: a row written before this column falls back
to its kind's reserve, as before. Additive; the downgrade drops the column.
"""
from alembic import op
import sqlalchemy as sa

revision = "a7c3e9f1b5d2"
down_revision = "e6a1c4f8b2d9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("gpu_leases", sa.Column("host_reserve_mb", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("gpu_leases", "host_reserve_mb")
