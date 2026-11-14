"""Steered-transcript record-run marker table (Steered Transcript Recorder).

A record job holds the single GPU like calibration but may target a cluster or an
ad-hoc feature set (no circuit row), so its in-flight lifecycle lives in a
dedicated table the single-GPU guard checks and cleanup reclaims.

Idempotent create. Additive — no existing table changes.

Revision ID: 5cede2a1b3f7
Revises: ca11b7a7e020
Create Date: 2026-07-22
"""
from typing import Sequence, Union

from alembic import op

revision: str = "5cede2a1b3f7"
down_revision: Union[str, None] = "ca11b7a7e020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS steering_record_runs (
            id            VARCHAR(36) PRIMARY KEY,
            status        VARCHAR(16) NOT NULL DEFAULT 'pending',
            task_id       VARCHAR(155),
            artifact_kind VARCHAR(16) NOT NULL,
            artifact_ref  VARCHAR(64),
            manifest_ref  VARCHAR(36),
            error         VARCHAR(500),
            created_at    TIMESTAMP NOT NULL DEFAULT now(),
            updated_at    TIMESTAMP NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_srr_status ON steering_record_runs (status)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS steering_record_runs")
