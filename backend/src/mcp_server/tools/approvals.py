"""Read one miStudio approval request (category: read) — MCP gap G10, 2026-10-21.

THE CHECK MUST COME BEFORE THE ACT. The caller-executed GPU tools (`start_probe_training_run`,
`evaluate_probe_monitor`, `millm_load_model`) answer a re-call with the approval id by CONSUMING
the approval and starting the job — so before this tool, the only way an agent could learn
whether an approval had been granted was the call that spends it. During the end-to-end probe
run of 2026-10-20 the pre-GPU safety checks had to be run before knowing whether the approval existed at all.

UNGATED AND IN `read`, the category every deployment enables. It used to be registered only
when `MCP_STEERING_APPROVAL=true` (inside `steering.py`), so with steering approval off — the
deployed setting — the tool did not exist even though probe and model-load approvals did.

WHO DECIDED: the suite has no login (operator decision 2026-10-06), so no approval records a
person's name. An approval or denial is the operator's act in miStudio's approvals banner; an
expiry is the TTL's (`mcp_approval_ttl_hours`). The answer says exactly that, never a name.
"""

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import BackendError, MiStudioClient
from ..config import MCPSettings

#: What an agent should do next, per status. A caller-executed approval is spent by RE-CALLING the
#: original tool with the same arguments plus the id; nothing here consumes it.
NEXT_STEP = {
    "pending": "The operator has not decided. Poll again later; do not re-submit.",
    "approved": "Approved and unspent. Re-call the original tool with the SAME arguments plus "
                "approval_request_id to use it (once). A steering approval has already been "
                "submitted by the backend: poll its steering_task_id.",
    "denied": "Denied. Do not retry the same request; ask the operator why (see reason).",
    "expired": "Expired unused. Call the original tool without approval_request_id to file a "
               "new request.",
    "consumed": "Already used once; the job it approved was submitted. A new request is needed "
                "to run it again.",
}

OPERATOR = ("operator (miStudio approvals banner; the suite has no login, so no name is "
            "recorded)")


def describe(row: dict[str, Any]) -> dict[str, Any]:
    """The agent-facing view of one approval row: status, who decided, when, and what next."""
    status = row.get("status")
    reason = row.get("reason") or ""
    if status in ("approved", "denied", "consumed"):
        decided_by = OPERATOR
    elif status == "expired":
        # Two expiries (mcp_approvals.expire_stale): pending too long (the operator never
        # decided), or approved and never used (the operator DID decide, then time ran out).
        decided_by = (OPERATOR if reason.startswith("expired: approved")
                      else "expiry (mcp_approval_ttl_hours elapsed while pending)")
    else:
        decided_by = None
    return {
        "approval_request_id": row.get("id"),
        "tool_name": row.get("tool_name"),
        "status": status,
        "decided": status != "pending",
        "decided_by": decided_by,
        "decided_at": row.get("resolved_at"),
        "consumed_at": row.get("consumed_at"),
        "created_at": row.get("created_at"),
        "reason": row.get("reason"),
        "steering_task_id": row.get("steering_task_id"),
        "payload": row.get("payload"),
        "next": NEXT_STEP.get(status, "Unknown status; read the approvals banner."),
    }


def register(mcp: MCPServer, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def get_approval_status(
        approval_request_id: Annotated[str, Field(description="The approval_request_id a gated miStudio tool returned (start_probe_training_run, evaluate_probe_monitor, millm_load_model, steer_* under approval mode)", min_length=1)],
    ) -> Any:
        """Read a miStudio operator-approval request WITHOUT spending it: status (pending |
        approved | denied | expired | consumed), who decided and when, when it was consumed,
        the reason, and what to do next.

        Use it BEFORE re-calling a gated GPU tool: that re-call consumes an approved request and
        starts the job at once. Read-only; never approves, denies or consumes. miDataworks'
        approvals are separate — read those with dataworks_get_approval_status."""
        try:
            row = await client.get(f"/mcp/approvals/{approval_request_id}")
        except BackendError as exc:
            if exc.status_code == 404:
                return {"approval_request_id": approval_request_id, "status": "not_found",
                        "next": "No such miStudio approval. A miDataworks approval id "
                                "(apr_…) is read with dataworks_get_approval_status."}
            raise
        return describe(row)
