"""feature_activations.dataset_id — which corpus a sample_index belongs to

Revision ID: c3f1a9d27b64
Revises: b9d4e7a2c815
Create Date: 2026-09-18

WHY. Multi-corpus SAE feature extraction concatenates several tokenized corpora
into one dataset before sampling, so `sample_index` is a GLOBAL row offset into
that concatenation. Two corpora therefore contribute overlapping index ranges,
and `sample_index` alone stops being a usable identity key for a feature's
examples — which is exactly how four services currently address these rows.
This column records which corpus a row's `sample_index` refers to.

NULLABLE, NO DEFAULT, NO server_default, NO BACKFILL — the convention that
b9d4e7a2c815 settled on for provenance columns. NULL means "not recorded",
which is the honest value for every row written before multi-corpus extraction
existed. A default would make every historical row claim a corpus nothing ever
verified, converting an honest absence into a silent lie.

CATALOGUE-ONLY. Adding a nullable column with no default does not rewrite the
table. That matters here: `feature_activations` is partitioned and large, and a
rewrite would be an outage rather than a migration.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "c3f1a9d27b64"
down_revision = "b9d4e7a2c815"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "feature_activations",
        sa.Column("dataset_id", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("feature_activations", "dataset_id")
