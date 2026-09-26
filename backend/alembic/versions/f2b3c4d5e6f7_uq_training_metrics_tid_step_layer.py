"""add unique constraint on training_metrics (training_id, step, layer_idx)

Revision ID: f2b3c4d5e6f7
Revises: e1a2b3c4d5e6
Create Date: 2026-07-11

Prevents silent duplicate metric rows under multi-hook training / resume.
First de-dups any existing duplicates defensively (keeps the highest id per
group — metrics are display-only time-series, so the latest write wins), then
adds the unique constraint. layer_idx is nullable (aggregated rows); Postgres
treats NULLs as distinct, which is the desired behaviour.
"""
from alembic import op
import sqlalchemy as sa

revision = "f2b3c4d5e6f7"
down_revision = "e1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Defensive de-dup: delete all but the max(id) per (training_id, step, layer_idx).
    # NULL layer_idx groups are handled by IS NOT DISTINCT FROM so aggregated
    # duplicates collapse too (while aggregated vs per-layer stay separate).
    op.execute(
        """
        DELETE FROM training_metrics tm
        USING training_metrics keep
        WHERE tm.training_id = keep.training_id
          AND tm.step = keep.step
          AND tm.layer_idx IS NOT DISTINCT FROM keep.layer_idx
          AND tm.id < keep.id
        """
    )
    op.create_unique_constraint(
        "uq_training_metrics_tid_step_layer",
        "training_metrics",
        ["training_id", "step", "layer_idx"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_training_metrics_tid_step_layer",
        "training_metrics",
        type_="unique",
    )
