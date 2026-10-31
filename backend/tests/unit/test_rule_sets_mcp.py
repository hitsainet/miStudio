"""The S3 agent surface on the MCP pod (Feature 040 S3; FR-39; FTASKS 23.1, 22.1 (a)).

* The six rule tools are in the LIVE registry of a server built in a subprocess with the MCP pod's kind of
  environment (no backend settings) and the default categories the deployment enables — not a module import.
  Control: drop `rule_sets` from `CATEGORY_MODULES["probe_monitors"]` → red.
* `recalibrate_probe_monitor`'s GPU arm (`recut_windows=true`, review R1-5): behind the operator approval gate
  (the request names the tool and the exact payload); with the gate off, ONE call with `recut_windows=true` and
  `require_positions` in the body. The arithmetic arm is unchanged (`test_reachability.py` pins it).
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
RULE_TOOLS = {"validate_rule_set", "evaluate_rule_signals", "evaluate_detector_with_rules",
              "get_detector_rules_evaluation", "list_detector_rules_evaluations", "cancel_detector_rules_evaluation",
              # Feature 040 S4 (FTDD Appendix C.6): an agent FILES a request; no tool approves or revokes.
              "request_code_rule_approval"}

PROBE = """
import asyncio, json
from src.mcp_server.config import MCPSettings
from src.mcp_server.server import build_server
mcp, _ = build_server(MCPSettings(), stdio=True)
print('TOOLS=' + json.dumps(sorted(t.name for t in asyncio.run(mcp.list_tools()))))
"""


def test_the_rule_tools_are_in_the_pods_live_registry():
    keep = {"PATH", "HOME", "LANG", "VIRTUAL_ENV", "TMPDIR"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.update(MCP_AUTH_TOKEN="x" * 32, PYTHONPATH=str(BACKEND), MCP_ALLOW_ANONYMOUS="true")
    proc = subprocess.run([sys.executable, "-c", PROBE], cwd=BACKEND, env=env, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    tools = set(json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("TOOLS="))[len("TOOLS="):]))
    assert RULE_TOOLS <= tools, sorted(RULE_TOOLS - tools)


MILLM_BEHAVIOUR_TOOLS = {
    "millm_list_behaviours", "millm_arm_behaviour", "millm_disarm_behaviour", "millm_get_probe_limit_recommendation",
    "millm_list_rule_validators", "millm_validate_rule_set", "millm_import_rule_set", "millm_list_rule_sets",
    "millm_get_rule_set", "millm_delete_rule_set", "millm_evaluate_rule_set", "millm_get_behaviour_rule_signals",
    "millm_set_behaviour_rule_signals", "millm_clear_behaviour_rule_signals",
    # ENH-006 D54: the capacity read and the two operator-approved writes.
    "millm_probe_capacity", "millm_request_probe_limits", "millm_measure_probe_capacity",
}


def test_the_millm_behaviour_proxies_are_in_the_deployed_pods_live_registry():
    """Review R1 (S3): the 14 miLLM 039 proxies in the live registry of a server built the way the DEPLOYED pod
    is (`k8s/base/mcp.yaml`'s `MCP_TOOL_CATEGORIES`, a miLLM URL set) — needing no miLLM checkout. The route
    parity guard needs the sibling and skips without it, so dropping `millm_behaviours` from
    `MILLM_CATEGORY_MODULES` went green wherever miLLM is absent (CI). Control: drop it → red."""
    import yaml

    manifest = BACKEND.parent / "k8s" / "base" / "mcp.yaml"
    categories = None
    for doc in yaml.safe_load_all(manifest.read_text()):
        for container in (((doc or {}).get("spec") or {}).get("template") or {}).get("spec", {}).get("containers", []):
            for e in container.get("env", []):
                if e.get("name") == "MCP_TOOL_CATEGORIES":
                    categories = e["value"]
    assert categories and "millm_probes" in categories.split(",")
    keep = {"PATH", "HOME", "LANG", "VIRTUAL_ENV", "TMPDIR"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.update(MCP_AUTH_TOKEN="x" * 32, PYTHONPATH=str(BACKEND), MCP_ALLOW_ANONYMOUS="true",
               MCP_TOOL_CATEGORIES=categories, MILLM_API_URL="http://127.0.0.1:9")
    proc = subprocess.run([sys.executable, "-c", PROBE], cwd=BACKEND, env=env, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    tools = set(json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("TOOLS="))[len("TOOLS="):]))
    assert MILLM_BEHAVIOUR_TOOLS <= tools, sorted(MILLM_BEHAVIOUR_TOOLS - tools)
    assert RULE_TOOLS <= tools


class Recorder:
    def __init__(self):
        self.calls = []

    async def post(self, path, json_body=None, **params):
        self.calls.append(("POST", path, json_body, params))
        if path == "/mcp/approvals":
            return {"id": "apr_1"}
        return {"task_id": "t", "status": "queued"}


def _tool(approval: bool):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import probe_monitors

    mcp = MCPServer("t")
    client = Recorder()
    probe_monitors.register(mcp, client, MCPSettings(auth_token="x" * 32, probe_gpu_approval=approval))
    return mcp._tool_manager, client


def test_the_gpu_recut_waits_for_the_operator():
    manager, client = _tool(approval=True)
    out = asyncio.run(manager.call_tool("recalibrate_probe_monitor", {
        "probe_id": "pm_1", "target_fpr": 0.01, "preview": False, "recut_windows": True, "require_positions": True},
        context=None))
    assert len(client.calls) == 1
    method, path, body, _ = client.calls[0]
    assert (method, path) == ("POST", "/mcp/approvals")
    assert body == {"tool_name": "recalibrate_probe_monitor", "payload": {
        "probe_id": "pm_1", "target_fpr": 0.01, "preview": False, "reason": "", "allow_fire_on_nothing": False,
        "require_positions": True}}
    assert "pending_approval" in json.dumps(out, default=str)


def test_the_gpu_recut_without_the_gate_is_one_call_with_the_flags():
    manager, client = _tool(approval=False)
    asyncio.run(manager.call_tool("recalibrate_probe_monitor", {
        "probe_id": "pm_1", "target_fpr": 0.01, "preview": False, "recut_windows": True, "require_positions": True},
        context=None))
    assert client.calls == [("POST", "/probe-monitors/probes/pm_1/recalibrate",
                             {"target_fpr": 0.01, "preview": False, "reason": "", "allow_fire_on_nothing": False,
                              "require_positions": True}, {"recut_windows": True})]


def test_require_positions_without_the_recut_is_refused_before_any_call():
    manager, client = _tool(approval=False)
    out = asyncio.run(manager.call_tool("recalibrate_probe_monitor", {
        "probe_id": "pm_1", "target_fpr": 0.01, "require_positions": True}, context=None))
    assert client.calls == [] and "recut_windows" in json.dumps(out, default=str)


def test_millm_delete_rule_set_needs_the_acknowledgement():
    """miLLM 039 proxies: deleting a rule set detaches it from every behaviour — refused here unless acknowledged,
    before any call (the `millm_delete_probe` rule)."""
    from unittest.mock import AsyncMock, MagicMock

    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import millm_behaviours

    mcp = MCPServer("t")
    millm = MagicMock()
    millm.delete = AsyncMock(return_value={"deleted": True})
    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, None))
    millm_behaviours.register(mcp, millm, gate, MagicMock())
    out = asyncio.run(mcp._tool_manager.call_tool("millm_delete_rule_set", {"rule_set_id": "rs_000000000001"},
                                                  context=None))
    assert "delete_not_acknowledged" in json.dumps(out, default=str)
    assert millm.delete.await_count == 0
