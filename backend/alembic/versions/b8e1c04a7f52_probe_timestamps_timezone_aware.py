"""probe monitor timestamps: TIMESTAMP WITH TIME ZONE

Revision ID: b8e1c04a7f52
Revises: a7f3c8e90d21
Create Date: 2026-09-26

WHY. `a7f3c8e90d21` created these nine columns as `TIMESTAMP WITHOUT TIME ZONE` while
their ORM defaults are `utc_now()`, which returns an AWARE datetime. asyncpg refuses an
aware value for a naive column outright:

    invalid input for query argument $14: datetime.datetime(2026, 9, 26, ... tzinfo=utc)
    (can't subtract offset-naive and offset-aware datetimes)

FOUND IN PRODUCTION ON THE FIRST REAL ROW, and the shape of the failure is the point:
creating the first probe dataset parsed all 8,000 training rows correctly — 4,000
positive, 4,000 negative, 0 unparseable — and then the INSERT raised. The expensive work
succeeded and was discarded at the commit, which is exactly the class of failure this
repo keeps paying for (see the `RenderedExample`-in-JSONB defect in the same feature: a
whole run's stages completing and dying at the final write).

HOW IT GOT IN. `test_clock_is_timezone_aware` refuses `datetime.utcnow()` on the grounds
that "every DateTime column in this schema is `timezone=True`". I satisfied it by
changing the VALUE to aware and left the COLUMN naive — half a fix, and worse than
neither, because the original naive default at least matched its column. The guard
checks the value; nothing checked the column, which is now pinned by
`test_probe_monitor_schema.py`.

The estate's convention is `DateTime(timezone=True)` (`app_setting`, `checkpoint`,
`feature_dashboard`), so this aligns the five probe tables with it rather than inventing
a local rule.

SAFE ON EXISTING DATA. These tables are new and, at the time of writing, empty in
production — the failure above is precisely why no row was ever inserted. Postgres
interprets an existing naive value as being in the session TimeZone when widening to
`timestamptz`, and the pods run UTC, so even a populated table converts correctly.
`USING <col> AT TIME ZONE 'UTC'` makes that explicit rather than relying on the session.
"""
from alembic import op
import sqlalchemy as sa

revision = "b8e1c04a7f52"
down_revision = "a7f3c8e90d21"
branch_labels = None
depends_on = None

#: (table, column) for every timestamp the five probe tables carry.
COLUMNS = [
    ("probe_monitor_datasets", "created_at"),
    ("probe_monitor_runs", "created_at"),
    ("probe_monitor_runs", "updated_at"),
    ("probe_monitor_runs", "completed_at"),
    ("probe_monitors", "created_at"),
    ("probe_monitor_evaluations", "created_at"),
    ("probe_monitor_judge_runs", "created_at"),
    ("probe_monitor_judge_runs", "completed_at"),
]


def upgrade() -> None:
    for table, column in COLUMNS:
        op.execute(
            f'ALTER TABLE {table} ALTER COLUMN {column} '
            f"TYPE TIMESTAMP WITH TIME ZONE USING {column} AT TIME ZONE 'UTC'"
        )


def downgrade() -> None:
    # Back to naive UTC. Real, not a stub: the values are preserved as their UTC
    # wall-clock reading, which is what the naive columns meant.
    for table, column in COLUMNS:
        op.execute(
            f'ALTER TABLE {table} ALTER COLUMN {column} '
            f"TYPE TIMESTAMP WITHOUT TIME ZONE USING {column} AT TIME ZONE 'UTC'"
        )
