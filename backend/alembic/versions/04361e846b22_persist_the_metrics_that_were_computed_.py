"""persist the metrics that were computed and discarded

Revision ID: 04361e846b22
Revises: 2c7bfd0f3f99
Create Date: 2026-09-11

Two things.

1. `training_metrics.l0_mean` — the number of active features PER TOKEN. It was
   computed on every forward pass (`sparse_autoencoder`: `l0_mean`) and had no
   consumer and no column. The column that did exist, `l0_sparsity`, holds a
   FRACTION of d_sae, so a healthy-looking 0.0094 is ~77 features at d_sae=8192
   — while the UI's stated target of "10-100" is a count. The two were being
   compared directly.

2. Backfill `trainings.current_fvu`. Every completed run has it NULL while
   `training_metrics.fvu` is fully populated (20,916/20,916 rows measured
   2026-09-11), so the summary column contradicted the series beside it. Filled
   from the last recorded step per training — a read of data that already
   exists, not a recomputation.

Additive and nullable; nothing is destroyed and the downgrade is symmetric
(the backfill is not reverted, because NULL carried no information to restore).
"""
from alembic import op
import sqlalchemy as sa

revision = '04361e846b22'
down_revision = '2c7bfd0f3f99'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('training_metrics', sa.Column('l0_mean', sa.Float(), nullable=True))

    # DISTINCT ON takes the newest row per training in one pass; the table is
    # ~21k rows, so no batching is needed.
    op.execute(
        """
        UPDATE trainings t
        SET current_fvu = latest.fvu
        FROM (
            SELECT DISTINCT ON (training_id) training_id, fvu
            FROM training_metrics
            WHERE fvu IS NOT NULL
            -- `(training_id, step)` carries a unique constraint, so a tie is
            -- unreachable in practice; ordering on id too makes the choice
            -- deterministic rather than relying on that constraint holding.
            ORDER BY training_id, step DESC, id DESC
        ) AS latest
        WHERE t.id = latest.training_id
          AND t.current_fvu IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column('training_metrics', 'l0_mean')
