"""add celery_task_id to models

Revision ID: e1a2b3c4d5e6
Revises: cd6c46abac48
Create Date: 2026-07-11

Adds a nullable celery_task_id column to the models table so an in-flight
download can be revoked when the user cancels (mirrors the pattern already on
trainings / dataset_tokenizations / activation_extractions).
"""
from alembic import op
import sqlalchemy as sa

revision = "e1a2b3c4d5e6"
down_revision = "cd6c46abac48"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "models",
        sa.Column("celery_task_id", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("models", "celery_task_id")
