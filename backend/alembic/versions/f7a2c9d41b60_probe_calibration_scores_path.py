"""probe_monitors.calibration_scores_path

Keeps the negative scores a threshold was cut from, so the operating point can be re-derived at
another target FPR without re-running the GPU pipeline. Purely additive and nullable: every
existing probe keeps NULL, which correctly reads as "this predates the persistence and its
threshold cannot be recomputed without a new run".

Revision ID: f7a2c9d41b60
Revises: d3b8e05c1a74
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f7a2c9d41b60"
down_revision: Union[str, None] = "d3b8e05c1a74"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "probe_monitors",
        sa.Column("calibration_scores_path", sa.String(length=1000), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("probe_monitors", "calibration_scores_path")
