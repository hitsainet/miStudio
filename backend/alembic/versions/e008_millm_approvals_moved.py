"""drain miLLM requests from miStudio's approval queue: miLLM now decides its own (ENH-008)

Revision ID: e008b7c9d1e3
Revises: e008a1f2b3c4
Create Date: 2026-10-28

ENH-008 §3.2 step 6 (D1). miStudio's `millm_*` proxies stop filing in miStudio's queue: miLLM gates those actions
itself and the proxies relay its pending answer. A request still `pending`, or `approved` and not yet consumed, for
one of the four tools that used to file here can never be carried out any more (the tool refuses
`approval_request_id` with `approval_moved_to_millm`), so it is EXPIRED with a reason that says why. The rows stay,
and the Approvals page's History shows them. Every other tool's rows are untouched.

DATA ONLY; the downgrade does nothing (an expired request is not re-opened: the agent asks again in miLLM).
"""

from alembic import op
import sqlalchemy as sa

revision = "e008b7c9d1e3"
down_revision = "e008a1f2b3c4"
branch_labels = None
depends_on = None

#: ⚠ FROZEN: the four tools that filed in miStudio's queue before ENH-008. Written out, not imported from the
#: endpoint (which no longer lists them), so this migration means the same thing forever.
MOVED_TOOLS = (
    "millm_load_model",
    "millm_request_probe_limits",
    "millm_measure_probe_capacity",
    "millm_import_rule_set",
)
REASON = "moved: miLLM now decides its own requests; ask again (ENH-008)"


def upgrade() -> None:
    op.get_bind().execute(
        sa.text(
            """
            UPDATE agent_approval_requests
               SET status = 'expired',
                   reason = :reason,
                   decided_by = 'system',
                   resolved_at = COALESCE(resolved_at, now())
             WHERE tool_name IN :tools
               AND status IN ('pending', 'approved')
            """
        ).bindparams(sa.bindparam("tools", expanding=True)),
        {"reason": REASON, "tools": list(MOVED_TOOLS)},
    )


def downgrade() -> None:
    pass
