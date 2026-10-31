"""miLLM's own approval queue (category: millm_models) — ENH-008, decisions D1 and D5–D8.

**miLLM decides its own requests.** An agent's model load, unload and delete, a probe-limit write, a
capacity measurement, an import of operator code and (contract v1.23) a telemetry-settings change are filed and decided IN miLLM, in its Approvals
page, from any client. miStudio's `millm_*` proxies used to file those in miStudio's queue and then
re-call with the approval id; they now RELAY: the request goes to miLLM's route with the
`X-miLLM-Agent` header (`millm_client.AGENT_HEADER`, on every request), miLLM stores it and answers
202 with `X-miLLM-Approval: pending`, and the tool returns miLLM's body unchanged plus
`"decide_in": "millm"`. Approving runs the stored request in miLLM, exactly once; the agent polls
`millm_get_approval_status` and never re-submits.

**FAILS CLOSED against a miLLM without a queue.** An older miLLM ignores the header and would run
the action with nobody approving it. So before a gated action is sent, `ApprovalQueueCheck` reads
`GET /api/approvals/actions` (cached on the health gate's window): a 404, or an action missing from
the list, answers `{"refused": "millm_has_no_approval_queue", …}` and sends nothing.

⚠ THE ONE `millm.get("/api/approvals/actions")` CALL LIVES IN THIS MODULE, beside the two tools that
read the same router. miLLM's `tests/unit/test_mcp_tool_paths_are_real.py` maps each `millm_*.py`
module's literal paths to the routers its surface names, so a call to `/api/approvals*` from
`millm_models.py` or `millm_behaviours.py` would fail that guard; those modules call the helper.
"""

import time
from typing import Annotated, Any, Literal, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import BackendError
from ..context import ExternalContext
from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient

#: The response header miLLM sets on a stored agent request (miLLM `api/approval_gate.py`).
APPROVAL_HEADER = "X-miLLM-Approval"

#: miStudio tool → the miLLM action that gates it (miLLM `APPROVAL_ACTIONS`; ENH-008 §3.1).
RELAYED_ACTIONS: dict[str, str] = {
    "millm_load_model": "model_load",
    "millm_unload_model": "model_unload",
    "millm_delete_model": "model_delete",
    "millm_request_probe_limits": "probe_limits_write",
    "millm_measure_probe_capacity": "capacity_measure",
    "millm_import_rule_set": "code_rule_import",
    # miLLM contract v1.23 (ENH-006 D76): an agent's write of a TELEMETRY_* setting.
    "millm_set_telemetry": "telemetry_write",
}

#: What an agent should do next, per miLLM status (ENH-008 §2.2's vocabulary).
NEXT_STEP: dict[str, str] = {
    "pending": "Waiting for the operator in miLLM's Approvals page. Poll again later; do not "
               "re-submit.",
    "executing": "Approved; miLLM is running it now. Poll again shortly.",
    "executed": "Approved and run by miLLM. `result` holds what it produced.",
    "failed": "Approved, then failed when miLLM ran it; `error` says why. Ask again if it still "
              "needs doing.",
    "rejected": "Rejected by the operator (see `reason`). Do not retry the same request.",
    "expired": "Expired without a decision. Call the original tool again to file a new request.",
}


def moved_refusal(tool_name: str) -> dict[str, Any]:
    """The answer to the deprecated `approval_request_id` (kept for one release, ENH-008 §3.2)."""
    return {
        "refused": "approval_moved_to_millm",
        "next": "millm_get_approval_status",
        "message": (
            f"{tool_name} no longer takes approval_request_id: miLLM now decides its own requests "
            "in its Approvals page, and an approved request runs there once, by itself. Call "
            f"{tool_name} again without approval_request_id to file the request in miLLM, then "
            "poll millm_get_approval_status with the approval_id it returns."
        ),
    }


def relayed(response: Any) -> Optional[dict[str, Any]]:
    """miLLM's pending-approval answer, plus `decide_in`, or `None` when the call simply ran.

    Pending means BOTH a 202 and `X-miLLM-Approval: pending`: a 202 alone is also how miLLM
    answers a load that has merely started.
    """
    status = getattr(response, "status", None)
    headers = getattr(response, "headers", None) or {}
    flag = next((v for k, v in headers.items() if k.lower() == APPROVAL_HEADER.lower()), None)
    if status != 202 or str(flag).strip().lower() != "pending":
        return None
    body = response.body if isinstance(response.body, dict) else {"body": response.body}
    return {**body, "decide_in": "millm"}


def _action_names(body: Any) -> set[str]:
    """The action names in `GET /api/approvals/actions`, whatever its outer shape: a list of names,
    a list of rows naming one each (`action`, `name` or `id`), or an object holding either under
    `actions` (or keyed by name). Anything else names nothing — which refuses (fail closed)."""
    if isinstance(body, dict):
        if "actions" in body:
            return _action_names(body["actions"])
        return {str(k) for k in body}
    names: set[str] = set()
    if isinstance(body, list):
        for row in body:
            if isinstance(row, str):
                names.add(row)
            elif isinstance(row, dict):
                for key in ("action", "name", "id"):
                    if isinstance(row.get(key), str):
                        names.add(row[key])
                        break
    return names


class ApprovalQueueCheck:
    """Does this miLLM have the approval queue that gates `action`? Fails closed.

    One per registering module; answers are reused for the health gate's window (`gate.ttl_s`).
    Only DEFINITE answers are cached (the list, or a 404): an error or an outage is asked again.
    """

    def __init__(self, millm: MiLLMClient, gate: HealthGate) -> None:
        self._millm = millm
        self._gate = gate
        self._cached: Optional[tuple[float, Optional[set[str]]]] = None

    def _ttl(self) -> float:
        ttl = getattr(self._gate, "ttl_s", None)
        return float(ttl) if isinstance(ttl, (int, float)) else 0.0

    async def _actions(self, millm: MiLLMClient) -> "set[str] | None | dict[str, Any]":
        """The action names; `None` for a miLLM with no queue (404); or the answer to return."""
        hit = self._cached
        if hit is not None and time.monotonic() - hit[0] < self._ttl():
            return hit[1]
        try:
            body = await millm.get("/api/approvals/actions")
        except BackendError as exc:
            if exc.status_code == 404:
                self._cached = (time.monotonic(), None)
                return None
            if exc.status_code == 0:
                self._gate.invalidate("millm")
                return {"unavailable": "millm", "reason": str(exc)}
            return {
                "refused": "millm_has_no_approval_queue",
                "reason": (
                    f"miLLM's approval queue could not be read (HTTP {exc.status_code}: "
                    f"{exc.detail}), so this request was NOT sent: without the queue nobody "
                    "would approve it. Try again later."
                ),
            }
        names = _action_names(body)
        self._cached = (time.monotonic(), names)
        return names

    async def refusal(self, action: str) -> Optional[dict[str, Any]]:
        """`None` when miLLM gates `action`; otherwise the answer to return (nothing was sent)."""
        found = await self._actions(self._millm)
        if isinstance(found, dict):
            return found
        if found is not None and action in found:
            return None
        why = ("has no approval queue (GET /api/approvals/actions is not served)" if found is None
               else f"does not gate the action {action!r}")
        return {
            "refused": "millm_has_no_approval_queue",
            "action": action,
            "reason": (
                f"This miLLM {why}, so it would run the request with nobody approving it. "
                "Nothing was sent. Upgrade miLLM to a release with its own Approvals page "
                "(ENH-008); until then the operator does this in miLLM's admin UI."
            ),
        }


def register(mcp: MCPServer, millm: MiLLMClient, gate: HealthGate,
             ctx: ExternalContext) -> None:  # noqa: ARG001
    @mcp.tool()
    @gated(gate, "millm")
    async def millm_list_approvals(
        status: Annotated[Optional[Literal["pending", "executing", "executed", "failed", "rejected", "expired"]], Field(description="Only requests in this state; omit for all")] = None,
        limit: Annotated[Optional[int], Field(description="Most recent first; 1-500 (miLLM's default 200)", ge=1, le=500)] = None,
    ) -> Any:
        """miLLM's approval queue, newest first, with the count still pending.

        The requests agents filed that wait for, or were decided by, the operator in miLLM's
        Approvals page. miLLM decides its own requests: a model load, unload or delete, a probe-limit change, a
        capacity measurement, an import of operator code and a telemetry-settings change. You cannot approve or reject one;
        miStudio's own requests (steering, probe GPU runs, code-rule runs) are read with
        get_approval_status instead."""
        return await millm.get("/api/approvals", status=status, limit=limit)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_get_approval_status(
        approval_id: Annotated[str, Field(description="The approval_id a gated millm_* tool returned with decide_in: millm (apr_…)", min_length=1)],
    ) -> Any:
        """One miLLM approval request: its status, who decided, when and why, and what it produced.

        Status is pending, executing, executed, failed, rejected or expired; the result or error
        appears once miLLM ran it, with what to do next. Read-only: it never approves, rejects or re-runs anything. An approved
        request runs in miLLM by itself, exactly once — never call the original tool again to
        carry it out."""
        try:
            row = await millm.get(f"/api/approvals/{approval_id}")
        except BackendError as exc:
            if exc.status_code == 404:
                return {"approval_id": approval_id, "status": "not_found",
                        "next": "No such miLLM approval. A miStudio approval id is read with "
                                "get_approval_status; a miDataworks one with "
                                "dataworks_get_approval_status."}
            raise
        if isinstance(row, dict):
            return {**row, "next": NEXT_STEP.get(str(row.get("status")),
                                                 "Unknown status; read miLLM's Approvals page.")}
        return row


__all__ = ["ApprovalQueueCheck", "RELAYED_ACTIONS", "moved_refusal", "relayed", "register"]
