"""Labeling trial mode: scoped, non-persisting prompt-template A/B runs.

Adds two things:

1. Three nullable columns on `labeling_jobs` — `mode`, `feature_ids`,
   `trial_run_id`. NULL on every existing row and read as APPLY, so behaviour is
   unchanged for jobs created before this.

   `feature_ids` is a real column rather than a key inside `statistics` because
   the completion write replaces `statistics` wholesale; a scope stashed there
   would be destroyed at the moment the run finished, taking reproducibility
   with it.

2. `labeling_trial_runs` — the results of one template over one fixed panel.
   `labeling_job_id` is ON DELETE SET NULL, not CASCADE: `delete_labeling_job` is
   a user-reachable endpoint and deleting the job that produced a measurement
   must not delete the measurement.

Idempotent create. Additive — no existing column is altered or dropped.

Revision ID: d7e3a91c04b8
Revises: b1f0c7a34d55
Create Date: 2026-08-29
"""
from typing import Sequence, Union

from alembic import op

revision: str = "d7e3a91c04b8"
down_revision: Union[str, None] = "b1f0c7a34d55"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE labeling_jobs
            ADD COLUMN IF NOT EXISTS mode VARCHAR(16),
            ADD COLUMN IF NOT EXISTS feature_ids JSONB,
            ADD COLUMN IF NOT EXISTS trial_run_id VARCHAR(36)
        """
    )
    op.execute(
        "COMMENT ON COLUMN labeling_jobs.mode IS "
        "'NULL/''apply'' writes labels to features; ''trial'' writes none'"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_labeling_jobs_trial_run_id "
        "ON labeling_jobs (trial_run_id)"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS labeling_trial_runs (
            id                 VARCHAR(36)  PRIMARY KEY,
            panel_id           VARCHAR(68)  NOT NULL,
            extraction_job_id  VARCHAR(255) NOT NULL
                REFERENCES extraction_jobs(id) ON DELETE CASCADE,
            labeling_job_id    VARCHAR(255)
                REFERENCES labeling_jobs(id) ON DELETE SET NULL,
            prompt_template_id VARCHAR(255),
            name               VARCHAR(200),
            status             VARCHAR(16)  NOT NULL DEFAULT 'queued',
            payload            JSONB        NOT NULL DEFAULT '{}'::jsonb,
            error              VARCHAR(500),
            created_at         TIMESTAMP    NOT NULL DEFAULT NOW(),
            updated_at         TIMESTAMP    NOT NULL DEFAULT NOW(),
            completed_at       TIMESTAMP
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ltr_panel "
        "ON labeling_trial_runs (panel_id, created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ltr_extraction "
        "ON labeling_trial_runs (extraction_job_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS labeling_trial_runs")
    op.execute("DROP INDEX IF EXISTS idx_labeling_jobs_trial_run_id")
    op.execute(
        """
        ALTER TABLE labeling_jobs
            DROP COLUMN IF EXISTS trial_run_id,
            DROP COLUMN IF EXISTS feature_ids,
            DROP COLUMN IF EXISTS mode
        """
    )
