"""record every card a split job uses

Revision ID: d3f7b1c9e5a2
Revises: b8e2f4a61c3d
Create Date: 2026-09-14

Multi-GPU Phase 2 loads a model that fits no single card across several. Each
job row records its card in `gpu_uuid` (String(64)), which holds one UUID.
`gpu_uuids` is a nullable JSONB list of every card a split job used, most free
first as placement chose them (transformers fills them in CUDA index order). `gpu_uuid` keeps its meaning — the job's first card — so every reader
written for one card is unchanged, and a single-card job leaves `gpu_uuids`
NULL. Additive; the downgrade drops the columns.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "d3f7b1c9e5a2"
down_revision = "b8e2f4a61c3d"
branch_labels = None
depends_on = None

#: The job tables that gained `gpu_uuid` in 7c1e9a4d2b60, all JSONB-using already.
JSONB_TABLES = (
    "activation_extractions",
    "trainings",
    "extraction_jobs",
    "circuit_capture_runs",
    "steering_record_runs",
    "labeling_jobs",
)


def upgrade() -> None:
    for table in JSONB_TABLES:
        op.add_column(table, sa.Column("gpu_uuids", postgresql.JSONB(), nullable=True))
    # Generic JSON, matching `task_queue.retry_params` and the model: tests build
    # this table on SQLite, which cannot render JSONB.
    op.add_column("task_queue", sa.Column("gpu_uuids", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("task_queue", "gpu_uuids")
    for table in JSONB_TABLES:
        op.drop_column(table, "gpu_uuids")
