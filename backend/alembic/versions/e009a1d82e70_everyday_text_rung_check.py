"""everyday-text rung check: probe_monitors.rung_detail, and rungs re-stated under the new rule

Revision ID: e009a1d82e70
Revises: e008c3d5f7a9
Create Date: 2026-10-29

ENH-006 D82 (operator decision 2026-10-29). Rung 2 ("detects on unseen tasks") now also needs a completed
evaluation on an everyday-text set (view role `everyday_text`, at least `everyday_text_min_rows` rows) whose
false-alarm rate at the SHIPPED bar is at most `everyday_text_fpr_factor` x the probe's target rate.

* `rung_detail` (JSONB, NULL) — what `recompute_rung` decided beside the rung: `next_step` and the
  everyday-text verdict. NULL = not recomputed since this revision.
* Data: `probe_monitors.rung` is a CACHE of the ladder's decision. No `everyday_text` view can exist before
  this revision (the role is new), so EVERY probe at rung 2 or 3 fails the new rule by construction, and its
  cached rung is re-stated here exactly as `recompute_rung` would: rung 1 when an in-distribution set passed
  (the reason line `compute()` writes for that, "CI lower bound above 0.5 in-distribution on:"), else rung 0.
  The reason and next step are the ladder's own words ("not yet checked on everyday text — evaluate on an
  everyday-text set"). A cached local definition of such a probe is invalidated as `invalidate_definition`
  does on any rung change (cause "rung"); the file on disk, anything published to the Hub and every copy
  miLLM imported are NOT touched — an export records its rung as of export.

`ADD COLUMN ... NULL` rewrites nothing; the UPDATEs touch only probes at rung >= 2 (a handful of rows).

Downgrade drops the column. It does not restore the old rungs: the first evaluation, judge run or export
build under the previous code recomputes them from the evaluations, which are untouched.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "e009a1d82e70"
down_revision = "e008c3d5f7a9"
branch_labels = None
depends_on = None

#: The ladder's words (`probe_monitor_rung.EVERYDAY_NOT_CHECKED`), copied: a migration never imports the
#: service it predates. `test_everyday_text_rung.py` pins the two equal.
NOT_CHECKED = "not yet checked on everyday text — evaluate on an everyday-text set"
IN_DISTRIBUTION_PASSED = "%CI lower bound above 0.5 in-distribution on:%"


def upgrade() -> None:
    op.add_column("probe_monitors", sa.Column("rung_detail", JSONB(), nullable=True))
    bind = op.get_bind()
    new_rung = (
        "CASE WHEN rung_reasons::text LIKE :in_passed THEN 1 ELSE 0 END"
    )
    # 1 — invalidate the cached local definition first, while `rung` still holds the old value.
    bind.execute(
        sa.text(
            f"""
            UPDATE probe_monitors SET
              definition_build = COALESCE(definition_build, '{{}}'::jsonb) || jsonb_build_object(
                'invalidated', jsonb_build_object(
                  'at', to_char(now() AT TIME ZONE 'utc', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
                  'reason', 'rung changed ' || rung || ' -> ' || ({new_rung})
                            || ' (ENH-006 D82: the everyday-text check)',
                  'cause', 'rung',
                  'was', definition_path,
                  'was_sha256', definition_sha256,
                  'was_identity', CASE
                    WHEN definition_build->'resolved'->>'detector_digest' IS NOT NULL THEN jsonb_build_object(
                      'detector_digest', definition_build->'resolved'->'detector_digest',
                      'decision_digest', definition_build->'resolved'->'decision_digest',
                      'decision_revision', definition_build->'resolved'->'decision_revision',
                      'identity_source', definition_build->'resolved'->'identity_source')
                    ELSE 'null'::jsonb END)),
              definition_path = NULL,
              definition_built_at = NULL,
              definition_sha256 = NULL
            WHERE rung >= 2 AND definition_path IS NOT NULL
            """
        ),
        {"in_passed": IN_DISTRIBUTION_PASSED},
    )
    # 2 — re-state the rung, its reasons and its next step.
    bind.execute(
        sa.text(
            f"""
            UPDATE probe_monitors SET
              rung = {new_rung},
              rung_reasons = COALESCE(rung_reasons, '[]'::jsonb) || to_jsonb(CAST(:reason AS text)),
              rung_detail = jsonb_build_object(
                'next_step', CAST(:not_checked AS text),
                'everyday_text', jsonb_build_object(
                  'status', 'not_checked', 'passed', false,
                  'reason', CAST(:not_checked AS text), 'next_step', CAST(:not_checked AS text),
                  'target_fpr', target_fpr, 'factor', 2.0, 'min_rows', 1000,
                  'allowed_rate', CASE WHEN target_fpr IS NULL THEN NULL ELSE 2.0 * target_fpr END,
                  'sets', '[]'::jsonb),
                'set_by', 'migration e009a1d82e70')
            WHERE rung >= 2
            """
        ),
        {
            "in_passed": IN_DISTRIBUTION_PASSED,
            "reason": "rung 2 not reached: " + NOT_CHECKED,
            "not_checked": NOT_CHECKED,
        },
    )


def downgrade() -> None:
    op.drop_column("probe_monitors", "rung_detail")
