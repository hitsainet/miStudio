"""probe monitor definitions: the cached export and its publication history

Revision ID: c9f2a71e4b08
Revises: b8e1c04a7f52
Create Date: 2026-09-26

033 FR-7 / task 3.1. Five additive columns on `probe_monitors`:

  definition_path      the cached `mistudio.probe-definition/v1` document on disk
  definition_built_at  when it was built
  definition_sha256    of the serialised bytes, so a download is verifiable
  definition_build     what the build was asked for and what it resolved
  published            every HuggingFace publication, appended never replaced

⚠ `published` IS NOT NULL DEFAULT '[]', AND `definition_*` ARE NULLABLE, DELIBERATELY. The
distinction is "no definition has been built" versus "a definition was built and is empty", and
those are different facts. Backfilling `definition_path` with anything would assert the second.

The estate has paid for the opposite mistake: `chat_format` shipped as NOT NULL DEFAULT 'auto',
which made every historical row claim it had used the tokenizer's real template — a column added
to stop a silent misattribution introducing one. `published` gets a default because an empty
publication list IS the truth for every existing row: none of them have been published, and the
column is a list whose emptiness is meaningful rather than unknown.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "c9f2a71e4b08"
down_revision = "b8e1c04a7f52"
branch_labels = None
depends_on = None

COLUMNS = (
    ("definition_path", sa.String(length=1000), True, None),
    ("definition_built_at", sa.DateTime(timezone=True), True, None),
    ("definition_sha256", sa.String(length=64), True, None),
    ("definition_build", postgresql.JSONB(astext_type=sa.Text()), True, None),
    ("published", postgresql.JSONB(astext_type=sa.Text()), False, sa.text("'[]'::jsonb")),
)


def upgrade() -> None:
    for name, type_, nullable, server_default in COLUMNS:
        op.add_column(
            "probe_monitors",
            sa.Column(name, type_, nullable=nullable, server_default=server_default),
        )
    # An index on the one column a listing filters by: "which probes have been published".
    op.create_index(
        "ix_probe_monitors_definition_built_at",
        "probe_monitors",
        ["definition_built_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_probe_monitors_definition_built_at", table_name="probe_monitors")
    for name, *_ in reversed(COLUMNS):
        op.drop_column("probe_monitors", name)
