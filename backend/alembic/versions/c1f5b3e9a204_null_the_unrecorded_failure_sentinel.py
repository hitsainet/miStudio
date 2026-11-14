"""a failure with no known reason must read as NO reason, not as a reason

Revision ID: c1f5b3e9a204
Revises: b8e4a2d0c517
Create Date: 2026-09-08

CAUGHT ON LIVE DATA, MINUTES AFTER DEPLOY. The coverage endpoint reported
`failures_without_a_recorded_reason: 0` for an extraction whose 14,560 failures
every one predate error capture. The honest answer was 14,560.

`b8e4a2d0c517` wrote the literal string
"(reason not recorded: this failure predates per-feature error capture)" into
`label_error` for every historical failure, and
`unreported_failure_count_query` tests `label_error IS NULL OR = ''`. So the
placeholder READ AS A GENUINE REASON, the caveat could never fire, and the field
built to say "these are not diagnosable" reported the opposite. Every test passed
because they used synthetic error strings and never the backfilled sentinel.

That is the exact failure mode the whole arc exists to remove: a value that looks
informative and is not. It was introduced by the cleanup for it, three files
away.

THE REASONING IN `b8e4a2d0c517` WAS WRONG, and this reverses it. It argued that
NULL "would read as 'this failure was never explained' — true, but
indistinguishable from a bug in the current code". There is nothing to
distinguish: the current code writes a reason on EVERY failure path, all twelve
of them, and `test_every_failure_return_carries_an_error` holds that. So on a
`failed` row, NULL can only mean historical. One rule, no string matching in
queries, and the query is correct by construction rather than by remembering a
sentence.

The prose is not lost — it moves to the coverage response's `caveat`, which is
where an operator reads it anyway.

`b8e4a2d0c517` is deliberately NOT edited: it has already run in production, and
rewriting an applied migration makes the history a lie. On a fresh database it
writes the sentinel and this immediately clears it — same end state, honest
record of how we got here.
"""
from alembic import op

revision = "c1f5b3e9a204"
down_revision = "b8e4a2d0c517"
branch_labels = None
depends_on = None

# Matched on the distinctive prefix rather than the whole sentence: the tail was
# reflowed once already in review, and a migration that silently matches nothing
# is worse than one that over-matches a string only we ever wrote.
_SENTINEL_PREFIX = "(reason not recorded"


def upgrade() -> None:
    op.execute(
        f"""
        UPDATE features
           SET label_error = NULL
         WHERE label_status = 'failed'
           AND label_error LIKE '{_SENTINEL_PREFIX}%'
        """
    )


def downgrade() -> None:
    # The exact inverse: restore the sentinel on failures that carry no reason.
    # Safe because the only rows this can touch are ones upgrade() emptied —
    # a failure written by current code always has a real reason.
    op.execute(
        """
        UPDATE features
           SET label_error = '(reason not recorded: this failure predates '
                             'per-feature error capture)'
         WHERE label_status = 'failed'
           AND label_error IS NULL
        """
    )
