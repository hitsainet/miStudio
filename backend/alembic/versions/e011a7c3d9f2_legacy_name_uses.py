"""legacy name uses: evidence that an old environment name, host or MCP category key is still used

Revision ID: e011a7c3d9f2
Revises: e010d84a7c31
Create Date: 2026-11-08

ENH-009 / 042 FR-042.9 (FTDD §4.1; CF-4, CF-5). Two additive tables; no existing table is touched, so the
revision is safe on the production database before any rebrand window (P0).

* ``legacy_name_uses`` — key (kind, old); ``last_seen`` only ever moves forward (the writers upsert with
  GREATEST). Bounded by the hand-kept tables of ``src/core/legacy_names.py`` (23 pairs at P0).
* ``legacy_name_reporters`` — key (reporter); per-process freshness for ``legacy_names_meta.reporters``.

Downgrade drops both.
"""

from alembic import op
import sqlalchemy as sa

revision = "e011a7c3d9f2"
down_revision = "e010d84a7c31"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "legacy_name_uses",
        sa.Column("kind", sa.String(16), primary_key=True),
        sa.Column("old", sa.String(255), primary_key=True),
        sa.Column("new", sa.String(255), nullable=False),
        sa.Column("first_seen", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("kind IN ('env','host','category')", name="ck_legacy_name_uses_kind"),
    )
    op.create_table(
        "legacy_name_reporters",
        sa.Column("reporter", sa.String(64), primary_key=True),
        sa.Column("first_report", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_report", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("legacy_name_reporters")
    op.drop_table("legacy_name_uses")
