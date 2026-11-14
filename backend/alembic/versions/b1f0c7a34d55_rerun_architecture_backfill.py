"""re-run the architecture backfill after teaching it heterogeneous configs

c4d8e1f60a92 ran and repaired NOTHING. It called
`extract_architecture_config`, which read fields with
`getattr(obj, field, None)` -- and that swallows AttributeError only.
google/gemma-4-12B-it is heterogeneous: its 48 layers differ, so asking its
text config for `num_key_value_heads` raises
AmbiguousGlobalPerLayerAttributeError by design, because no single global value
exists. One refusing field discarded the whole description:

    architecture_config backfill: skipped m_b55c6926
      ('num_key_value_heads' is a per-layer attribute and may vary across
       layers ...)
    architecture_config backfill: repaired 0 model(s)

The extractor now records the fields that DO have a global answer and marks the
stack heterogeneous. An applied migration does not run again, so the repair
needs a new revision. Same function, so there is one implementation.

Idempotent: only rows missing a layer count are selected. On a database where
c4d8e1f60a92 already succeeded this finds nothing and does nothing.

Revision ID: b1f0c7a34d55
Revises: c4d8e1f60a92
Create Date: 2026-08-26
"""

from alembic import op

revision = "b1f0c7a34d55"
down_revision = "c4d8e1f60a92"
branch_labels = None
depends_on = None


def upgrade() -> None:
    from src.db.architecture_backfill import backfill

    backfill(op.get_bind())


def downgrade() -> None:
    """Not reversed: the previous values were incomplete readings, not data."""
    pass
