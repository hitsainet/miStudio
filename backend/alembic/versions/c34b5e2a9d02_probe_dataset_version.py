"""record the miDataworks dataset version a probe dataset was verified against (Feature 034)

Revision ID: c34b5e2a9d02
Revises: c34a0d1e7f01
Create Date: 2026-10-16

One nullable JSONB column, no default, no backfill: `probe_monitor_datasets.dataset_version`.

A probe dataset may now be registered with a miDataworks `midataworks.dataset-version` manifest
(FR-26). miStudio validates it against the vendored schema, checks the manifest's row count for the
split equals the downloaded split's, and records exactly what it verified here: version identity,
split, both counts, repository, commit, and the manifest's canonical SHA-256.

One column, not five: the fields are read together and never filtered on, and the record can grow
additively without another migration. NULL is truthful for every existing row — none was
registered with a manifest.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "c34b5e2a9d02"
down_revision = "c34a0d1e7f01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "probe_monitor_datasets",
        sa.Column("dataset_version", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("probe_monitor_datasets", "dataset_version")
