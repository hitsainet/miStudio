"""record which GPU each job asked for and ran on

Revision ID: 7c1e9a4d2b60
Revises: 04361e846b22
Create Date: 2026-09-13

The node gained a second GPU on 2026-09-13 (an RTX 3080 Ti at NVML index 0; the
RTX 3090 moved to index 1). Jobs had no record of which card they used, and the
one table that stored a choice, `activation_extractions.gpu_id`, stored an
INDEX — which the new card silently re-pointed at a different GPU.

Two nullable columns per row-driven GPU job table:

* `gpu_request` — what the job asked for: "auto", or a GPU UUID (an index given
  at submit time is resolved to the UUID then). A retry honours it.
* `gpu_uuid` — the card the job actually ran on, written when it starts.

`task_queue` gets `gpu_uuid` only: it carries J-lens runs, whose request travels
with the task.

`activation_extractions.gpu_id` is left in place for existing rows; new code
records the UUID. Additive and nullable; the downgrade drops what it added.
Plan: 0xcc/plans/Multi-GPU-Plan.md, Phase 1.
"""
from alembic import op
import sqlalchemy as sa

revision = "7c1e9a4d2b60"
down_revision = "04361e846b22"
branch_labels = None
depends_on = None

REQUEST_AND_RUN_TABLES = (
    "activation_extractions",
    "trainings",
    "extraction_jobs",
    "circuit_capture_runs",
    "steering_record_runs",
    "labeling_jobs",
)


def upgrade() -> None:
    for table in REQUEST_AND_RUN_TABLES:
        op.add_column(table, sa.Column("gpu_request", sa.String(length=64), nullable=True))
        op.add_column(table, sa.Column("gpu_uuid", sa.String(length=64), nullable=True))
    op.add_column("task_queue", sa.Column("gpu_uuid", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("task_queue", "gpu_uuid")
    for table in reversed(REQUEST_AND_RUN_TABLES):
        op.drop_column(table, "gpu_uuid")
        op.drop_column(table, "gpu_request")
