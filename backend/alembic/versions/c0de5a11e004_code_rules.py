"""code rules: operator code-rule documents and their per-hash operator approvals

Revision ID: c0de5a11e004
Revises: 5ed6d79015d2
Create Date: 2026-10-24

Feature 040 slice S4 (FR-31; FTDD Appendix C.5, C.6, C.8).

* `code_rules` — content-addressed CodeRuleDocuments (PK `code_sha256`, the exact hashed bytes in `canonical`);
* `code_rule_approvals` — the durable per-hash decision list; `origin` CHECK = 'operator' (an agent can never
  write a decision), non-empty `reason` and `changed_by`, an index on (code_sha256, id) for the latest-row read.

Both APPEND-ONLY (ORM refusal + trigger). No column on an existing table; no backfill.

`down_revision` re-read on 2026-10-24 by walking every revision's `down_revision`: one head, 040 S3's
`5ed6d79015d2`.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "c0de5a11e004"
down_revision = "5ed6d79015d2"
branch_labels = None
depends_on = None

#: ⚠ FROZEN COPY of `models.code_rule.APPEND_ONLY_DDL`; pinned equal by `test_code_rules_migration.py`.
APPEND_ONLY_DDL = (
    """
CREATE OR REPLACE FUNCTION code_rules_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION '% is append-only (% refused)', TG_TABLE_NAME, TG_OP;
END;
$$ LANGUAGE plpgsql
""",
    """
CREATE TRIGGER trg_code_rules_append_only
BEFORE UPDATE OR DELETE ON code_rules
FOR EACH ROW EXECUTE FUNCTION code_rules_append_only()
""",
    """
CREATE TRIGGER trg_code_rule_approvals_append_only
BEFORE UPDATE OR DELETE ON code_rule_approvals
FOR EACH ROW EXECUTE FUNCTION code_rules_append_only()
""",
)


def upgrade() -> None:
    op.create_table(
        "code_rules",
        sa.Column("code_sha256", sa.String(64), primary_key=True),
        sa.Column("canonical", sa.LargeBinary(), nullable=False),
        sa.Column("runner_contract", sa.String(64), nullable=False),
        sa.Column("functions", JSONB(), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("source_ref", JSONB(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("code_sha256 ~ '^[0-9a-f]{64}$'", name="ck_code_rules_sha_hex"),
        sa.CheckConstraint("source IN ('referenced', 'inline')", name="ck_code_rules_source"),
    )
    op.create_table(
        "code_rule_approvals",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("code_sha256", sa.String(64), sa.ForeignKey("code_rules.code_sha256", ondelete="RESTRICT"),
                  nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("changed_by", sa.String(100), nullable=False),
        sa.Column("origin", sa.String(16), nullable=False),
        sa.Column("approval_request_id", sa.String(36),
                  sa.ForeignKey("agent_approval_requests.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("action IN ('approve', 'revoke')", name="ck_code_rule_approvals_action"),
        sa.CheckConstraint("origin = 'operator'", name="ck_code_rule_approvals_origin_operator"),
        sa.CheckConstraint("length(btrim(reason)) > 0", name="ck_code_rule_approvals_reason"),
        sa.CheckConstraint("length(btrim(changed_by)) > 0", name="ck_code_rule_approvals_changed_by"),
    )
    op.create_index("ix_code_rule_approvals_sha_id", "code_rule_approvals", ["code_sha256", "id"])
    for statement in APPEND_ONLY_DDL:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_code_rule_approvals_append_only ON code_rule_approvals")
    op.execute("DROP TRIGGER IF EXISTS trg_code_rules_append_only ON code_rules")
    op.drop_index("ix_code_rule_approvals_sha_id", table_name="code_rule_approvals")
    op.drop_table("code_rule_approvals")
    op.drop_table("code_rules")
    op.execute("DROP FUNCTION IF EXISTS code_rules_append_only()")
