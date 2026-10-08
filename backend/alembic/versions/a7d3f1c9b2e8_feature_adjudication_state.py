"""per-feature labeling adjudication state on `features`

Revision ID: a7d3f1c9b2e8
Revises: f3c8a92b1e07
Create Date: 2026-09-08

WHY THESE COLUMNS EXIST.

The OUTCOME of a labeling attempt is recorded nowhere. It is inferred from
`name`, `description`, `category` and `label_source`, and that inference is
wrong in both directions. A failure is written as a FAKE LABEL —
`category='error_feature'`, `name='feature_{n}'` — with `label_source` and
`labeled_at` both set, so every "is it labeled?" test says yes. Measured on the
live database: 16,824 features across the estate have failed and look finished.
One extraction has carried 2,161 silent failures since April. The one predicate
that looks reusable, `has_label` = `label_source != 'auto'`, counts every one of
them as labeled.

The distinction the schema must make is not two-way but three-way:

    judge ran, produced a verdict            -> never redo
    judge ran, honestly said "no pattern"    -> never redo; this is a RESULT
    judge never ran, or crashed              -> always retryable

`uninterpretable` is the sharp edge. `_enforce_refusal` produces it
deliberately when the fit ratio is below 0.5 — it is an adjudication, not an
absence. Treating it as a gap turns "resume" into "relabel everything" and
re-derives, at ~8 s/feature, verdicts the judge already reached.

`label_status` therefore carries the OUTCOME, and `category` goes back to
carrying only the semantic VERDICT.

WHY NOT A POSTGRES ENUM.

`features.label_source` beside it is a native enum, and adding a value to one
costs a dedicated migration because `ALTER TYPE ... ADD VALUE` cannot run in
the transaction that then uses the value (see `v9w0x1y2z3a4`). A status set
this young will gain members; VARCHAR makes that free.

WHY A PROMPT FINGERPRINT AND NOT A TEMPLATE VERSION.

Content-addressed, so it cannot drift and needs no discipline to maintain —
the same reasoning as `LabelingTrialRun.panel_id` and PADR IDL-48's pinned
scoring prompt, whose rationale is that an editable ruler silently invalidates
every prior score. It is what makes "already adjudicated" mean "by this same
judge": edit the template or swap the model and the affected features become
eligible again on their own, with no hand-written SQL.

ADDITIVE ONLY. Nothing here reads or rewrites an existing row — deriving
`label_status` from what is already in the table is a separate revision, per
house style. This one is safe to run against a live database: every column is
either nullable or carries a server default, so PostgreSQL adds them as
catalogue-only changes without rewriting the table.
"""
from alembic import op

revision = "a7d3f1c9b2e8"
down_revision = "f3c8a92b1e07"
branch_labels = None
depends_on = None


# (column, DDL type and constraints) in creation order.
_COLUMNS = (
    # pending | in_progress | succeeded | failed | skipped
    ("label_status", "VARCHAR(20) NOT NULL DEFAULT 'pending'"),
    # Attempts spent. Caps retries so a permanently-broken feature cannot be
    # re-queued forever by successive resumes.
    ("label_attempts", "INTEGER NOT NULL DEFAULT 0"),
    # Why the last attempt failed, truncated by the writer to 2000 chars.
    ("label_error", "TEXT"),
    ("label_error_at", "TIMESTAMP WITH TIME ZONE"),
    # sha256 over the prompt fields that change what the judge sees.
    ("label_prompt_fingerprint", "VARCHAR(64)"),
    ("label_model", "VARCHAR(255)"),
)

# Both halves of the routine resume predicate. `label_status` filters it;
# `label_prompt_fingerprint` answers "adjudicated by WHICH judge", which is the
# staleness query.
_INDEXES = (
    ("ix_features_label_status", "label_status"),
    ("ix_features_label_prompt_fingerprint", "label_prompt_fingerprint"),
)


def upgrade() -> None:
    for column, ddl in _COLUMNS:
        op.execute(f"ALTER TABLE features ADD COLUMN IF NOT EXISTS {column} {ddl}")
    for index, column in _INDEXES:
        op.execute(f"CREATE INDEX IF NOT EXISTS {index} ON features ({column})")


def downgrade() -> None:
    for index, _column in reversed(_INDEXES):
        op.execute(f"DROP INDEX IF EXISTS {index}")
    for column, _ddl in reversed(_COLUMNS):
        op.execute(f"ALTER TABLE features DROP COLUMN IF EXISTS {column}")
