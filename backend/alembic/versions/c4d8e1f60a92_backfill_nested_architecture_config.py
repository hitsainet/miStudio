"""backfill architecture_config for composite (multi-tower) models

Models downloaded before this stored only TOP-LEVEL fields from config.json.
Composite configs -- vision-language, audio and "omni" models -- keep the
decoder's dimensions in a sub-config, so google/gemma-4-12B-it recorded three
keys and no num_hidden_layers, and the Training page offered no layers at all.

The logic lives in src/db/architecture_backfill so a later revision can re-run
it without a second copy. See b1f0c7a34d55, which had to.

Revision ID: c4d8e1f60a92
Revises: e2a4c81b9d17
Create Date: 2026-08-25
"""

from alembic import op

revision = "c4d8e1f60a92"
down_revision = "e2a4c81b9d17"
branch_labels = None
depends_on = None


def upgrade() -> None:
    from src.db.architecture_backfill import backfill

    backfill(op.get_bind())


def downgrade() -> None:
    """Not reversed: the previous values were incomplete readings, not data."""
    pass
