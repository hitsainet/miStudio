"""rung 2 needs a catch rate and a nonsense-text check: cached rungs re-stated under the new rule

Revision ID: e010d84a7c31
Revises: e009a1d82e70
Create Date: 2026-10-29

ENH-006 D84 (operator decision 2026-10-29: "I think the probe sucks."). Rung 2 ("detects on unseen tasks") now
also needs (a) at the SHIPPED bar, a catch rate of at least `rung2_min_recall` (0.5) on every counted
out-of-distribution set, and (b) a completed evaluation on a NONSENSE-TEXT set (view role `nonsense_text`, at
least `nonsense_text_min_rows` rows) whose false alarms at the shipped bar are at most
`nonsense_text_fpr_factor` x the target rate.

No schema change: `rung_detail` exists (e009a1d82e70) and `role` is a free string.

Data: `probe_monitors.rung` is a CACHE of the ladder's decision. No `nonsense_text` view can exist before this
revision (the role is new), so EVERY probe at rung 2 or 3 fails (b) by construction, and its cached rung is
re-stated here exactly as `recompute_rung` would cap it: rung 1 when an in-distribution set passed (the reason
line `compute()` writes for that, "CI lower bound above 0.5 in-distribution on:"), else rung 0. The reason and
next step are the ladder's own words ("not yet checked on nonsense text — evaluate on a nonsense-text set").
(a) needs the stored per-row scores and is decided by the next recompute; it can only lower a rung further.
A cached local definition of such a probe is invalidated as `invalidate_definition` does on any rung change
(cause "rung"); the file on disk, anything published to the Hub and every copy miLLM imported are NOT
touched — an export records its rung as of export.

The rung, reasons and detail BEFORE the restatement are kept in `rung_detail.previous`, so the downgrade puts
them back for every probe this revision restated and nothing has recomputed since. It does not rebuild an
invalidated definition (a cache, rebuilt on the next build).
"""

from alembic import op
import sqlalchemy as sa

revision = "e010d84a7c31"
down_revision = "e009a1d82e70"
branch_labels = None
depends_on = None

#: The ladder's words (`probe_monitor_rung.NONSENSE_NOT_CHECKED`), copied: a migration never imports the
#: service it predates. `test_rung2_tighten.py` pins the two equal.
NOT_CHECKED = "not yet checked on nonsense text — evaluate on a nonsense-text set"
IN_DISTRIBUTION_PASSED = "%CI lower bound above 0.5 in-distribution on:%"
SET_BY = "migration e010d84a7c31"


def upgrade() -> None:
    bind = op.get_bind()
    new_rung = "CASE WHEN rung_reasons::text LIKE :in_passed THEN 1 ELSE 0 END"
    # 1 — invalidate the cached local definition first, while `rung` still holds the old value.
    bind.execute(
        sa.text(
            f"""
            UPDATE probe_monitors SET
              definition_build = COALESCE(definition_build, '{{}}'::jsonb) || jsonb_build_object(
                'invalidated', jsonb_build_object(
                  'at', to_char(now() AT TIME ZONE 'utc', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
                  'reason', 'rung changed ' || rung || ' -> ' || ({new_rung})
                            || ' (ENH-006 D84: the catch-rate and nonsense-text checks)',
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
    # 2 — re-state the rung, its reasons and its next step; keep what was there for the downgrade.
    bind.execute(
        sa.text(
            f"""
            UPDATE probe_monitors SET
              rung = {new_rung},
              rung_reasons = COALESCE(rung_reasons, '[]'::jsonb) || to_jsonb(CAST(:reason AS text)),
              rung_detail = COALESCE(rung_detail, '{{}}'::jsonb) || jsonb_build_object(
                'next_step', CAST(:not_checked AS text),
                'nonsense_text', jsonb_build_object(
                  'kind', 'nonsense_text', 'status', 'not_checked', 'passed', false,
                  'reason', CAST(:not_checked AS text), 'next_step', CAST(:not_checked AS text),
                  'target_fpr', target_fpr, 'factor', 2.0, 'min_rows', 1000,
                  'allowed_rate', CASE WHEN target_fpr IS NULL THEN NULL ELSE 2.0 * target_fpr END,
                  'sets', '[]'::jsonb),
                'set_by', CAST(:set_by AS text),
                'previous', jsonb_build_object(
                  'rung', rung,
                  'rung_detail', COALESCE(rung_detail, 'null'::jsonb)))
            WHERE rung >= 2
            """
        ),
        {
            "in_passed": IN_DISTRIBUTION_PASSED,
            "reason": "rung 2 not reached: " + NOT_CHECKED,
            "not_checked": NOT_CHECKED,
            "set_by": SET_BY,
        },
    )


def downgrade() -> None:
    # Only probes this revision restated and nothing has recomputed since (a recompute rewrites
    # `rung_detail` without `set_by`, and its rung is then real evidence under whichever code ran it).
    op.get_bind().execute(
        sa.text(
            """
            UPDATE probe_monitors SET
              rung = (rung_detail->'previous'->>'rung')::int,
              rung_reasons = CASE
                WHEN rung_reasons->>(jsonb_array_length(rung_reasons) - 1) = CAST(:reason AS text)
                THEN rung_reasons - (jsonb_array_length(rung_reasons) - 1)
                ELSE rung_reasons END,
              rung_detail = CASE
                WHEN jsonb_typeof(rung_detail->'previous'->'rung_detail') = 'object'
                THEN rung_detail->'previous'->'rung_detail'
                ELSE NULL END
            WHERE rung_detail->>'set_by' = CAST(:set_by AS text)
            """
        ),
        {"set_by": SET_BY, "reason": "rung 2 not reached: " + NOT_CHECKED},
    )
