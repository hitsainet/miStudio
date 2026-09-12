"""a resume sweep: many batches, one row, one batch per task

Revision ID: e5b91d4a2c73
Revises: d2a7c8e103b6
Create Date: 2026-09-08

Finishing the L46 extraction is 27 batches of 2000 at the measured ~8 s/feature
— about 59 GPU-hours. Celery's soft limit here is 10 h, so a task that looped
over batches would be killed part-way and, on an `acks_late` queue, strand its
message for the full 12 h visibility timeout. A sweep is therefore a ROW plus a
task that does ONE batch and re-enqueues itself.

`max_batches` is NOT NULL with no default: an open-ended sweep is a request to
spend an unknown number of GPU-hours, and it must not be startable by omitting a
parameter.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "e5b91d4a2c73"
down_revision = "d2a7c8e103b6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "labeling_resume_sweeps",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "extraction_job_id",
            sa.String(255),
            sa.ForeignKey("extraction_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("config", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("batches_done", sa.Integer, nullable=False, server_default="0"),
        sa.Column("features_labeled", sa.Integer, nullable=False, server_default="0"),
        sa.Column("features_failed", sa.Integer, nullable=False, server_default="0"),
        # No server_default: a sweep without an explicit ceiling must fail to
        # insert rather than quietly become unbounded.
        sa.Column("max_batches", sa.Integer, nullable=False),
        sa.Column("batch_size", sa.Integer, nullable=False, server_default="2000"),
        # NAMED FOR core.cancellation, not invented beside it: the scope is
        # registered there and `request_cancel` writes this column.
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        # So the registry's progress_field names a real column.
        sa.Column("progress", sa.Float, nullable=False, server_default="0"),
        sa.Column("last_labeling_job_id", sa.String(255), nullable=True),
        sa.Column("error_message", sa.String(1000), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_labeling_resume_sweeps_extraction_job_id",
        "labeling_resume_sweeps",
        ["extraction_job_id"],
    )
    # The janitor's query: find sweeps still claiming to run.
    op.create_index(
        "ix_labeling_resume_sweeps_status", "labeling_resume_sweeps", ["status"]
    )


def downgrade() -> None:
    op.drop_index("ix_labeling_resume_sweeps_status", table_name="labeling_resume_sweeps")
    op.drop_index(
        "ix_labeling_resume_sweeps_extraction_job_id", table_name="labeling_resume_sweeps"
    )
    op.drop_table("labeling_resume_sweeps")
