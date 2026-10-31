"""let a whole-extraction labeling job skip features that already have a verdict

Revision ID: d2a7c8e103b6
Revises: c1f5b3e9a204
Create Date: 2026-09-08

`label_features_for_extraction` selects the whole extraction with no predicate on
label state, so pressing Label relabels everything — including the 266,144
features across the estate that already carry a verdict, and the refusals the
judge reached deliberately.

Making that skip adjudicated work BY DEFAULT was the obvious change and is the
wrong one: an operator pressing an existing button expects the behaviour it has
always had, and would quietly stop getting it with nothing on screen to say so.
So the choice becomes visible instead, and DEFAULTS TO FALSE — today's behaviour,
exactly.

Additive and defaulted, so every existing row reads as false without a rewrite.
"""
from alembic import op

revision = "d2a7c8e103b6"
down_revision = "c1f5b3e9a204"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE labeling_jobs ADD COLUMN IF NOT EXISTS "
        "skip_adjudicated BOOLEAN NOT NULL DEFAULT false"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE labeling_jobs DROP COLUMN IF EXISTS skip_adjudicated")
