"""add finalized_from_step to trainings

Records the checkpoint step a training was finalized from when it was stopped
early. NULL for runs that completed normally.

Why a column instead of reusing status: finalizing sets status=COMPLETED so the
downstream import path unlocks, but a run stopped at step 10k of 50k did NOT
complete. Keeping progress/current_step truthful and recording the finalize
step lets the UI badge "Finalized early at step N" instead of silently
presenting a partial run as a full one.

Revision ID: b4d19f0c73ae
Revises: 5cede2a1b3f7
Create Date: 2026-07-26

"""
from alembic import op
import sqlalchemy as sa


revision = "b4d19f0c73ae"
down_revision = "5cede2a1b3f7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "trainings",
        sa.Column(
            "finalized_from_step",
            sa.Integer(),
            nullable=True,
            comment=(
                "Checkpoint step this training was finalized from when stopped "
                "early; NULL if it ran to completion"
            ),
        ),
    )


def downgrade() -> None:
    op.drop_column("trainings", "finalized_from_step")
