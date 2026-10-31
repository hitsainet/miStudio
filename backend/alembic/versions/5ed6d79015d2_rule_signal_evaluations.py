"""rule signal evaluations: view text/rule columns, rules ∪ probe evaluations, the rule-verdict cache

Revision ID: 5ed6d79015d2
Revises: b7e41c2d9a05
Create Date: 2026-10-24

Feature 040 slice S3 (FR-12 (rest), FR-14 (rest), FR-33 – FR-38; FTDD §4.4, §4.7; FTID §4.1).

* `probe_monitor_datasets`: `text_column` (String 255; the text rules read when it differs from the input
  column), `rule_columns` (JSONB; producer-computed verdict COLUMN NAMES plus the rule-set identity VALUES,
  checked against the manifest's listing), `rule_set_listing` (JSONB; the manifest's rule-set listing as read
  at registration, 040-T23);
* `probe_rules_evaluations` — partial unique index on `identity_sha256` over live rows (040-T29), the
  reaper's `(status, heartbeat_at)` index, and the body-immutable trigger (a completed result is never
  rewritten or deleted, FR-38);
* `rule_verdict_cache` — PK (view, rule-set digest, engine digest, reading digest) and a `last_used_at`
  index for pruning.

NO DEFAULT AND NO BACKFILL: NULL is "not declared".

`down_revision` re-read on 2026-10-24 by walking every revision: the head is 041's `b7e41c2d9a05`
(probe blend reports), which landed after 040 S2's `effadddf2061` — not S2 as FTASKS 20.1 expected.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "5ed6d79015d2"
down_revision = "b7e41c2d9a05"
branch_labels = None
depends_on = None

#: ⚠ FROZEN COPY of `models.rules_evaluation.BODY_IMMUTABLE_DDL`; pinned equal by
#: `test_rules_evaluation_migration.py`.
BODY_IMMUTABLE_DDL = (
    """
CREATE OR REPLACE FUNCTION probe_rules_evaluations_body_immutable() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF OLD.body IS NOT NULL THEN
            RAISE EXCEPTION 'probe_rules_evaluations: a completed result is never deleted';
        END IF;
        RETURN OLD;
    END IF;
    IF OLD.body IS NOT NULL AND NEW.body IS DISTINCT FROM OLD.body THEN
        RAISE EXCEPTION 'probe_rules_evaluations: a completed body is never rewritten';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
""",
    """
CREATE TRIGGER trg_probe_rules_evaluations_body_immutable
BEFORE UPDATE OR DELETE ON probe_rules_evaluations
FOR EACH ROW EXECUTE FUNCTION probe_rules_evaluations_body_immutable()
""",
)


def upgrade() -> None:
    op.add_column("probe_monitor_datasets", sa.Column("text_column", sa.String(length=255), nullable=True))
    op.add_column("probe_monitor_datasets", sa.Column("rule_columns", JSONB(), nullable=True))
    op.add_column("probe_monitor_datasets", sa.Column("rule_set_listing", JSONB(), nullable=True))

    op.create_table(
        "probe_rules_evaluations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("identity_sha256", sa.String(length=64), nullable=False),
        sa.Column("probe_id", sa.String(length=36), nullable=False),
        sa.Column("view_ids", JSONB(), nullable=False),
        sa.Column("behaviour_key", sa.String(length=64), nullable=False),
        sa.Column("behaviour_version", sa.Integer(), nullable=False),
        sa.Column("rule_set_digest", sa.String(length=64), nullable=True),
        sa.Column("engine_digest", sa.String(length=64), nullable=False),
        sa.Column("calibration", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("progress", JSONB(), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("task_queue_id", sa.String(length=36), nullable=True),
        sa.Column("error", JSONB(), nullable=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("body_sha256", sa.String(length=64), nullable=True),
        sa.Column("inputs", JSONB(), nullable=False),
        sa.Column("created_by", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
                           name="ck_probe_rules_evaluations_status"),
    )
    op.create_index("uq_probe_rules_evaluations_identity_live", "probe_rules_evaluations", ["identity_sha256"],
                    unique=True, postgresql_where=sa.text("status IN ('pending', 'running', 'completed')"))
    op.create_index("ix_probe_rules_evaluations_key_created", "probe_rules_evaluations",
                    ["behaviour_key", "created_at"])
    op.create_index("ix_probe_rules_evaluations_status_heartbeat", "probe_rules_evaluations",
                    ["status", "heartbeat_at"])
    for statement in BODY_IMMUTABLE_DDL:
        op.execute(statement)

    op.create_table(
        "rule_verdict_cache",
        sa.Column("view_id", sa.String(length=36), primary_key=True),
        sa.Column("rule_set_digest", sa.String(length=64), primary_key=True),
        sa.Column("engine_digest", sa.String(length=64), primary_key=True),
        sa.Column("reading_digest", sa.String(length=64), primary_key=True),
        sa.Column("n_rows", sa.Integer(), nullable=False),
        sa.Column("bytes", sa.Integer(), nullable=False),
        sa.Column("path", sa.String(length=1000), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("in_use_by", sa.String(length=36), nullable=True),
    )
    op.create_index("ix_rule_verdict_cache_last_used", "rule_verdict_cache", ["last_used_at"])


def downgrade() -> None:
    op.drop_index("ix_rule_verdict_cache_last_used", table_name="rule_verdict_cache")
    op.drop_table("rule_verdict_cache")
    op.execute("DROP TRIGGER IF EXISTS trg_probe_rules_evaluations_body_immutable ON probe_rules_evaluations")
    op.execute("DROP FUNCTION IF EXISTS probe_rules_evaluations_body_immutable()")
    op.drop_index("ix_probe_rules_evaluations_status_heartbeat", table_name="probe_rules_evaluations")
    op.drop_index("ix_probe_rules_evaluations_key_created", table_name="probe_rules_evaluations")
    op.drop_index("uq_probe_rules_evaluations_identity_live", table_name="probe_rules_evaluations")
    op.drop_table("probe_rules_evaluations")
    for column in ("rule_set_listing", "rule_columns", "text_column"):
        op.drop_column("probe_monitor_datasets", column)
