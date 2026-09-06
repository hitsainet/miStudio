"""Widen circuit-run byte and event counters to BIGINT.

MIS-E2E-029. `bytes_total` counts a capture store in BYTES and was a 32-bit
INTEGER, capping at 2,147,483,647 — about 2 GiB, which per-token multi-layer
SAE activations over a real corpus reach readily.

The failure mode is what makes this P1 rather than cosmetic: the capture
COMPLETES, then the final commit raises on numeric overflow. That poisons the
session, so the task's own error handler — which needs the same session to mark
the run failed — fails too. The row stays `running` forever, `store_path` is
never set, and the multi-gigabyte store is leaked on disk with nothing pointing
at it. `assert_no_active_gpu_run` then counts that row and refuses every later
capture with a 409.

Widening is safe in both directions here: no existing value can fail to fit in
BIGINT, and the downgrade is only safe because nothing has yet been able to
STORE a value above the 32-bit ceiling — the overflow raised instead. The
downgrade therefore checks before narrowing rather than truncating silently.

Revision ID: e2a4c81b9d17
Revises: d7f3a91c2e08
"""

from alembic import op
import sqlalchemy as sa

revision = "e2a4c81b9d17"
down_revision = "d7f3a91c2e08"
branch_labels = None
depends_on = None


#: (table, column) pairs to widen. Both counters on both run tables — the
#: finding named `circuit_capture_runs`, and its sibling has the identical
#: declaration. Fixing one and not the other is this audit's most repeated
#: anti-pattern.
_COLUMNS = [
    ("circuit_capture_runs", "bytes_total"),
    ("circuit_capture_runs", "events_total"),
]


def _existing_columns(table: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table not in inspector.get_table_names():
        return set()
    return {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    for table, column in _COLUMNS:
        if column in _existing_columns(table):
            op.alter_column(
                table,
                column,
                existing_type=sa.Integer(),
                type_=sa.BigInteger(),
                existing_nullable=True,
            )


def downgrade() -> None:
    bind = op.get_bind()
    for table, column in _COLUMNS:
        if column not in _existing_columns(table):
            continue
        # Refuse rather than truncate. A value above the 32-bit ceiling cannot
        # exist today, but a downgrade run after this has been live for a while
        # could meet one, and silently wrapping a byte count is worse than a
        # failed migration.
        too_big = bind.execute(
            sa.text(
                f"SELECT COUNT(*) FROM {table} WHERE {column} > 2147483647"  # noqa: S608
            )
        ).scalar()
        if too_big:
            raise RuntimeError(
                f"{too_big} row(s) in {table}.{column} exceed the 32-bit range; "
                f"refusing to narrow the column and corrupt them"
            )
        op.alter_column(
            table,
            column,
            existing_type=sa.BigInteger(),
            type_=sa.Integer(),
            existing_nullable=True,
        )
