"""Repair circuit_discovery_runs.attribution_task_id drift.

WHY THIS EXISTS
---------------
`e5f6a7b8c9da` adds four columns to `circuit_discovery_runs`:
attribution_status, attribution_progress, attribution_error and
attribution_task_id.

The production database has the first three and NOT the fourth, because the
migration was EDITED AFTER IT HAD ALREADY BEEN APPLIED:

    963a6a2  2026-07-19 23:54  migration created (3 columns)
    e6b6cdc  2026-07-20 00:11  review round 2 ADDED attribution_task_id to it

Any database that ran the migration in that 17-minute window records
`e5f6a7b8c9da` as applied and will never re-run it, so the fourth column never
arrives. Alembic tracks revisions, not their contents.

The symptom is severe and remote from the cause: every `circuit_discovery_runs`
SELECT emits the full column list, so `UndefinedColumn` 500s the CAPTURE
endpoint — a route that does not touch attribution at all. The whole circuit
pipeline is unusable, and nothing points at a migration.

Idempotent (`IF NOT EXISTS`), so it is a no-op on databases that got the
complete version. Kept as a separate revision rather than another edit to
`e5f6a7b8c9da` — editing an applied migration is what caused this.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "f9a1c2d3e4b5"
down_revision: Union[str, None] = "b4046f2741dd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Matches e5f6a7b8c9da's definition exactly: VARCHAR(155), nullable.
    op.execute(
        "ALTER TABLE circuit_discovery_runs "
        "ADD COLUMN IF NOT EXISTS attribution_task_id VARCHAR(155)"
    )


def downgrade() -> None:
    # Deliberately a no-op. Dropping the column would re-break every database
    # this repairs, and `e5f6a7b8c9da`'s own downgrade already removes it for
    # databases that legitimately have it.
    pass
