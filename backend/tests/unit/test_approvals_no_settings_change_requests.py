"""ENH-006 D79 (2026-10-29): "Approve all settings changes" has nothing to batch in miStudio — pinned.

The operator decision: when two or more SETTINGS-CHANGE requests wait, one named button approves them all. miLLM,
miDataworks and miForge each queue agent settings writes for approval and build that button. miStudio's own queue
(`agent_approval_requests`, filed through `POST /mcp/approvals`) holds none:

* steering (`steer_*`, backend-executed), probe GPU runs (caller-executed) and code-rule runs (a durable record);
* runtime settings are never queued — `PUT`/`DELETE /environment/{key}` refuse agents outright (403), and no MCP
  tool writes them (`test_mcp_environment_tool.py::test_no_registered_tool_writes_the_environment`, ENH-003 D5);
* miLLM's settings requests are decided in miLLM (ENH-008 D1), never relayed into this queue.

So the honest build here is no SETTINGS batch button. (D81, the same day, added the code-rule batch:
`test_approve_batch_code_rules.py`; its `kind: "settings"` is refused 422.) This file keeps that true ON PURPOSE: every key of the LIVE
`APPROVABLE_TOOLS` table (the authority both the filing validator and approve/consume read) must be classified
below as a non-settings action. The day someone adds a tool to the queue, the exact-set test goes red and its
message asks the question: if the new tool changes a setting, build D79's "Approve all settings changes (N)"
(banner + Approvals page, `POST /mcp/approvals/approve-batch`, see the miLLM/miDataworks/miForge builds).

Negative controls (recorded in `0xcc/reviews/approve_all_settings_controls_2026-10-29.md`):
  N1 add `"set_environment_setting": (Execution.BACKEND, "/api/v1/environment/log_level")` to APPROVABLE_TOOLS →
     the exact-set test, the settings-endpoint test and the validator test go red.
  N2 drop `dependencies=[Depends(_refuse_agents)]` from `set_environment_override` → the agent-refusal test goes red.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from src.core.deps import get_db
from src.mcp_server.client import AGENT_HEADER
from src.api.v1.endpoints.mcp_approvals import APPROVABLE_TOOLS, ApprovalCreateRequest, Execution
from src.main import app

#: Every action miStudio's queue holds, and why it is not a settings change. Adding a key to APPROVABLE_TOOLS
#: without classifying it here fails `test_every_queued_action_is_classified_and_none_is_a_settings_change`.
NON_SETTINGS_ACTIONS: dict[str, str] = {
    "steer_compare": "GPU steering task",
    "steer_sweep": "GPU steering task",
    "steer_combined": "GPU steering task",
    "start_probe_training_run": "GPU probe training run",
    "evaluate_probe_monitor": "GPU probe evaluation",
    "start_probe_judge_run": "judge run against miLLM's GPU",
    "start_probe_spot_check": "judge spot-check against miLLM's GPU",
    "recalibrate_probe_monitor": "GPU re-scoring of a probe's negatives",
    "code_rule_approve": "operator approval to run code (a durable per-hash record)",
}

#: Route prefixes that write a miStudio setting. A queued backend-executed action must never call one.
SETTINGS_PREFIXES = ("/api/v1/environment", "/api/v1/settings")


def test_every_queued_action_is_classified_and_none_is_a_settings_change():
    live = set(APPROVABLE_TOOLS)
    known = set(NON_SETTINGS_ACTIONS)
    assert live == known, (
        f"miStudio's approval queue changed: added {sorted(live - known)}, removed {sorted(known - live)}. "
        "Classify each new action in NON_SETTINGS_ACTIONS. If it CHANGES A SETTING, ENH-006 D79 applies: build "
        "'Approve all settings changes (N)' (banner + Approvals page, POST /mcp/approvals/approve-batch, "
        "operator-only, pre-check before approving any, per-item results with a shared batch_id) as miLLM, "
        "miDataworks and miForge did, and replace this file with that feature's tests."
    )


def test_no_backend_executed_action_calls_a_settings_route():
    for tool, (kind, endpoint) in APPROVABLE_TOOLS.items():
        if kind is Execution.BACKEND:
            assert endpoint and not endpoint.startswith(SETTINGS_PREFIXES), (tool, endpoint)


@pytest.mark.parametrize("tool_name", ["set_environment_setting", "environment_set", "settings_update"])
def test_the_filing_validator_refuses_a_settings_change(tool_name):
    """`POST /mcp/approvals` validates `tool_name` against the same table: a settings write cannot be filed."""
    with pytest.raises(ValidationError):
        ApprovalCreateRequest(tool_name=tool_name, payload={"value": "DEBUG"})


def _agent_write(method: str) -> tuple[int, dict, list[str]]:
    """Send an agent's settings write through the LIVE app (no DB: `get_db` is overridden and records use)."""
    used: list[str] = []

    async def _no_db():
        used.append("db")
        yield None

    async def go():
        app.dependency_overrides[get_db] = _no_db
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.request(method, "/api/v1/environment/log_level", headers={AGENT_HEADER: "mcp"},
                                    json={"value": "DEBUG"} if method == "PUT" else None)
                return r.status_code, r.json(), used
        finally:
            app.dependency_overrides.pop(get_db, None)

    return asyncio.run(go())


@pytest.mark.parametrize("method", ["PUT", "DELETE"])
def test_settings_writes_refuse_agents_instead_of_queueing(method):
    """Runtime settings are the operator's, directly: an agent's settings write is a 403, never a pending request
    that a batch would approve."""
    status, body, _ = _agent_write(method)
    assert status == 403, (status, body)
    assert body["detail"]["code"] == "environment_operator_only"
    assert "approval_request_id" not in str(body)


def test_the_batch_route_has_no_settings_kind():
    """D81 built the batch for CODE RULES; there is still no settings batch, because there is nothing to batch."""
    from src.api.v1.endpoints.mcp_approvals import BATCH_KINDS

    assert set(BATCH_KINDS) == {"code_rules"}
    assert all(not tools & {"set_environment_setting"} for tools in BATCH_KINDS.values())
    paths = app.openapi()["paths"]
    assert "/api/v1/mcp/approvals/approve-batch" in paths
