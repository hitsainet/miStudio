"""trainings: centred FVU beside the legacy one, and a dedicated evaluation record

Revision ID: c4e8b2d6f1a9
Revises: a7c3e9f1b5d2
Create Date: 2026-09-15

SAE training remediation, items 5 and 6.

ITEM 5. The FVU miStudio stored, var(x - x_hat) / var(x) over every element with
one global mean, reads about 0.05 below the standard per-dimension-centred FVU on
activations with large constant offset dimensions. The stored value keeps its
column and its meaning; the centred value gets NEW columns, so no historical row
silently changes what it says:

* ``training_metrics.fvu_centred`` — per-step (and held-out) rows;
* ``trainings.current_fvu_centred`` — the training's latest value.

ITEM 6. The spliced-CE evaluation wrote its results into ``training_metrics`` by
overloading columns (``loss`` = spliced CE, ``loss_reconstructed`` = baseline CE,
``loss_zero`` = ablated CE, ``l0_mean`` = loss recovered) on rows told apart only
by a negative ``layer_idx``. ``trainings.evaluation`` is the dedicated place: one
JSONB document holding the post-run evaluation's status, configuration, sources and
per-layer results.

All additive and nullable, with no backfill: NULL is the honest value for a run
recorded before these existed. The downgrade drops the three columns.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "c4e8b2d6f1a9"
down_revision = "a7c3e9f1b5d2"
branch_labels = None
depends_on = None

METRIC_FVU_CENTRED_COMMENT = (
    "Standard FVU: sum|x - x_hat|^2 / sum|x - mu|^2 with mu the per-dimension "
    "mean, raw activation space. `fvu` beside it is the legacy global-mean value. "
    "NULL on rows written before 2026-09-15."
)
CURRENT_FVU_CENTRED_COMMENT = (
    "Latest per-dimension-centred FVU (the headline). current_fvu is the legacy "
    "global-mean value. NULL for runs recorded before 2026-09-15."
)
EVALUATION_COMMENT = (
    "Post-run evaluation on blocks the training never read: status, config, sources, "
    "per-layer spliced/mean-ablated/zero-ablated CE, KL, L0 and centred FVU. NULL "
    "when never evaluated."
)


def upgrade() -> None:
    op.add_column(
        "training_metrics",
        sa.Column("fvu_centred", sa.Float(), nullable=True, comment=METRIC_FVU_CENTRED_COMMENT),
    )
    op.add_column(
        "trainings",
        sa.Column(
            "current_fvu_centred", sa.Float(), nullable=True, comment=CURRENT_FVU_CENTRED_COMMENT
        ),
    )
    op.add_column(
        "trainings",
        sa.Column(
            "evaluation",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment=EVALUATION_COMMENT,
        ),
    )


def downgrade() -> None:
    op.drop_column("trainings", "evaluation")
    op.drop_column("trainings", "current_fvu_centred")
    op.drop_column("training_metrics", "fvu_centred")
