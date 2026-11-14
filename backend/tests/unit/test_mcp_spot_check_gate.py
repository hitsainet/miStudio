"""MCP: a spot-check judged by miServer waits for the operator (038 FR-14, D34), payload- and count-asserted.

Control C-SPOTGATE: remove the gate line in `start_probe_spot_check` → red.
"""
from __future__ import annotations

import asyncio

import pytest

from src.mcp_server.config import MCPSettings

ARGS = {"behaviour": "pii", "prompt_id": "jp_1", "endpoint": "http://millm-backend.millm.svc.cluster.local:8000/v1",
        "model": "qwen", "size": 50, "seed": 3}
BODY = {"behaviour": "pii", "prompt_id": "jp_1", "endpoint": ARGS["endpoint"], "model": "qwen", "size": 50,
        "window": "all", "seed": 3}


@pytest.fixture(autouse=True)
def _no_miserver_address_unless_the_test_sets_one(monkeypatch):
    """These cases state their miServer address through `miserver_api_url_override`; an empty override falls back to
    the environment, so a MILLM_API_URL/MISERVER_API_URL left by another test (or a developer shell) would give
    the judge gate a configured address and make it resolve hosts (042 FR-042.38). Pin "none configured"."""
    for name in ("MILLM_API_URL", "MISERVER_API_URL"):
        monkeypatch.delenv(name, raising=False)


class _Client:
    def __init__(self):
        self.calls = []

    async def post(self, path, json_body=None, **params):
        self.calls.append(("POST", path, json_body))
        if path == "/mcp/approvals":
            return {"id": "apr_1", "status": "pending"}
        return {"id": "psc_1", "status": "pending"}


def _call(args):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import probe_monitors

    client = _Client()
    mcp = MCPServer("t")
    probe_monitors.register(mcp, client, MCPSettings(auth_token="x" * 32, probe_gpu_approval=True))
    result = asyncio.run(mcp._tool_manager._tools["start_probe_spot_check"].fn(**args))  # noqa: SLF001
    return result, client.calls


def test_spot_check_needs_approval_on_millm():
    result, calls = _call(ARGS)
    assert result["status"] == "pending_approval"
    assert calls == [("POST", "/mcp/approvals", {"tool_name": "start_probe_spot_check", "payload": BODY})]


def test_the_approved_recall_consumes_then_submits_the_same_body():
    _, calls = _call({**ARGS, "approval_request_id": "apr_1"})
    assert calls == [
        ("POST", "/mcp/approvals/apr_1/consume", {"tool_name": "start_probe_spot_check", "payload": BODY}),
        ("POST", "/probe-monitors/spot-checks", BODY),
    ]


def test_a_judge_elsewhere_files_nothing():
    args = {**ARGS, "endpoint": "https://api.openai.com/v1"}
    _, calls = _call(args)
    assert calls == [("POST", "/probe-monitors/spot-checks", {**BODY, "endpoint": args["endpoint"]})]


def test_the_backend_accepts_it_as_caller_executed_and_says_it_plainly():
    from src.api.v1.endpoints import mcp_approvals

    assert "start_probe_spot_check" in mcp_approvals.CALLER_TOOLS
    text = mcp_approvals._summarize("start_probe_spot_check", BODY)
    assert text.startswith("Spot-check 50 turns of pii with qwen on miServer")
