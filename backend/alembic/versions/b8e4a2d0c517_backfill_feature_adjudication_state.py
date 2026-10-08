"""derive label_status for features labelled before it existed

Revision ID: b8e4a2d0c517
Revises: a7d3f1c9b2e8
Create Date: 2026-09-08

Separate from `a7d3f1c9b2e8`, which only added the columns. Additive schema and
data migration are never mixed here: a backfill over a large table is the part
that can be slow, be interrupted, or need re-running, and it must be possible to
retry it without re-litigating the schema.

WHAT IS RECOVERABLE, AND WHAT IS NOT.

The old encoding wrote a failure as a fake label and discarded the exception
text at a log line. The REASON is therefore gone for every historical failure —
permanently, for all 16,824 of them. What survives is the fact that they failed,
in the sentinel categories the fake labels used, and that is what this recovers.

Three rules, applied in order:

  label_source = 'auto'        -> pending    the judge never ran
  category IN (the sentinels)  -> failed     it ran and crashed
  anything else                -> succeeded  it ran and produced a verdict

WHY EVERY BACKFILLED VERDICT GETS A NULL FINGERPRINT.

NULL means "we do not know which judge produced this", which is exactly true:
these rows predate provenance. That encoding is deliberate and load-bearing —
`eligibility_filter` treats a NULL fingerprint as STALE, so a backfilled verdict
is re-adjudicable the moment an operator names a judge to compare against, and
is never touched by a routine resume. Writing a fabricated fingerprint would
claim knowledge this migration does not have, and would make the rows look
current forever.

WHY THE NAME HEURISTIC IS NOT USED.

One historical failure path is unrecoverable even in principle: a local parse
failure wrote `category='semantic'` with a placeholder `name='feature_{n}'`, so
it is indistinguishable from a real semantic verdict by any column. Testing
`name LIKE 'feature_%'` would catch some of those — and would also condemn every
feature a judge legitimately named `feature_store_checkout`. That heuristic is
the defect this whole change removes, and reintroducing it inside the migration
that cleans up after it would be self-defeating. Those rows land in `succeeded`
with a NULL fingerprint, which makes them stale and therefore re-adjudicable —
the honest outcome.

`label_attempts` is set to 1 for anything that was attempted, so the retry cap
means something on day one rather than granting historical failures a fresh full
allowance.

EVERY RULE IS GUARDED ON THE UNTOUCHED DEFAULT STATE, so re-running this is a
no-op rather than a corruption. It matters more than it looks: a failure written
by the CURRENT code correctly leaves `label_source='auto'` (the feature is
truthfully still unlabelled), so an unguarded rule 1 would match it on a second
run and reset a recorded failure back to "never attempted" — destroying exactly
the information this whole change exists to keep. Alembic will not re-run a
revision on its own, but operators re-run data migrations by hand.
"""
from alembic import op

revision = "b8e4a2d0c517"
down_revision = "a7d3f1c9b2e8"
branch_labels = None
depends_on = None


# The categories the old code used as failure sentinels. `uninterpretable` is
# deliberately ABSENT: it is a judge's honest verdict that a feature has no
# coherent pattern, produced on purpose by `_enforce_refusal` below a 0.5 fit
# ratio. Listing it here would mark 196 adjudicated features on the L46
# extraction alone as outstanding work and re-derive, at ~8 s each, answers the
# judge already gave.
_FAILURE_CATEGORIES = ("error_feature", "rate_limited", "empty_features", "uncategorized")


def upgrade() -> None:
    failures = ", ".join(f"'{c}'" for c in _FAILURE_CATEGORIES)

    # Never attempted. `label_source='auto'` is the auto-generated placeholder
    # every feature starts with.
    op.execute(
        """
        UPDATE features
           SET label_status = 'pending',
               label_attempts = 0
         WHERE label_source = 'auto'
           AND label_status = 'pending'
           AND label_attempts = 0
           AND label_error IS NULL
        """
    )

    # Attempted and failed. The reason is gone; say so rather than leaving the
    # column NULL, which would read as "this failure was never explained" —
    # true, but indistinguishable from a bug in the current code.
    op.execute(
        f"""
        UPDATE features
           SET label_status = 'failed',
               label_attempts = GREATEST(label_attempts, 1),
               label_error = '(reason not recorded: this failure predates '
                             'per-feature error capture)',
               label_error_at = COALESCE(labeled_at, updated_at)
         WHERE label_source <> 'auto'
           AND label_status = 'pending'
           AND label_attempts = 0
           AND label_error IS NULL
           AND category IN ({failures})
        """
    )

    # Attempted and adjudicated. Fingerprint and model stay NULL on purpose.
    op.execute(
        f"""
        UPDATE features
           SET label_status = 'succeeded',
               label_attempts = GREATEST(label_attempts, 1)
         WHERE label_source <> 'auto'
           AND label_status = 'pending'
           AND label_attempts = 0
           AND label_error IS NULL
           AND (category IS NULL OR category NOT IN ({failures}))
        """
    )


def downgrade() -> None:
    # A derivation has no exact inverse: the state it read (category,
    # label_source) is still there, but any outcome written AFTER this migration
    # is indistinguishable from one it wrote. Resetting to the column defaults
    # is the honest inverse — it returns every feature to "never attempted",
    # which is what the schema said before this ran.
    #
    # It is destructive, and loudly so, because the alternative is a downgrade
    # that leaves a half-derived table nobody can reason about.
    print(
        "WARNING: downgrading b8e4a2d0c517 resets label_status/label_attempts/"
        "label_error for EVERY feature. Outcomes recorded since this migration "
        "ran will be lost; the labels themselves are untouched."
    )
    op.execute(
        """
        UPDATE features
           SET label_status = 'pending',
               label_attempts = 0,
               label_error = NULL,
               label_error_at = NULL
        """
    )
