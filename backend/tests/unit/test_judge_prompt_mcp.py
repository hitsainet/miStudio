"""MCP: a judge run against miLLM waits for the operator; the library tools are live (IDL-57).

`start_probe_judge_run` was not approval-gated at all. A judge run against miLLM is thousands of
generations on the GPU that serves miLLM's users and may load the judge model (evicting the served
one), so it now takes the caller-executed approval that training runs take — asserted by payload
AND call count, as `test_mcp_probe_gpu_approval.py` does for those. A judge elsewhere (an external
OpenAI-compatible endpoint) is one direct call, asserted in `test_reachability.EXPECTED_CALLS`.
"""
from __future__ import annotations

import asyncio

import pytest

from src.mcp_server.client import BackendError
from src.mcp_server.config import MCPSettings
from src.mcp_server.tools.probe_monitors import judge_endpoint_is_millm

ARGS = {"endpoint": "http://millm-backend.millm.svc.cluster.local:8000/v1", "model": "qwen",
        "dataset_ids": ["pmd_2"], "probe_id": "pm_1", "prompt_id": "jp_pii_presence_v1",
        "max_rows_per_set": 200}
BODY = {"endpoint": ARGS["endpoint"], "model": "qwen", "dataset_ids": ["pmd_2"],
        "max_rows_per_set": 200, "probe_id": "pm_1", "prompt_id": "jp_pii_presence_v1"}


class _Client:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or {}

    async def post(self, path, json_body=None, **params):
        self.calls.append(("POST", path, json_body))
        if path in self.fail:
            raise self.fail[path]
        if path == "/mcp/approvals":
            return {"id": "apr_1", "status": "pending"}
        return {"id": "pmj_1", "status": "queued"}


def _call(args, *, approval=True, client=None, millm_url=""):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import probe_monitors

    client = client or _Client()
    mcp = MCPServer("t")
    probe_monitors.register(
        mcp, client,
        MCPSettings(auth_token="x" * 32, probe_gpu_approval=approval, millm_api_url_override=millm_url),
    )
    result = asyncio.run(mcp._tool_manager._tools["start_probe_judge_run"].fn(**args))  # noqa: SLF001
    return result, client.calls


class TestAJudgeOnMiLLMWaitsForTheOperator:
    def test_the_first_call_files_an_approval_and_queues_NOTHING(self):
        result, calls = _call(ARGS)
        assert result["status"] == "pending_approval"
        assert calls == [("POST", "/mcp/approvals",
                          {"tool_name": "start_probe_judge_run", "payload": BODY})]

    def test_the_approved_recall_consumes_THEN_submits_the_same_body(self):
        _, calls = _call({**ARGS, "approval_request_id": "apr_1"})
        assert calls == [
            ("POST", "/mcp/approvals/apr_1/consume",
             {"tool_name": "start_probe_judge_run", "payload": BODY}),
            ("POST", "/probe-monitors/judge-runs", BODY),
        ]

    def test_a_refused_consume_submits_NOTHING(self):
        client = _Client(fail={"/mcp/approvals/apr_1/consume": BackendError(409, {"status": "pending"})})
        result, calls = _call({**ARGS, "approval_request_id": "apr_1"}, client=client)
        assert result["status"] == "pending"
        assert [c[1] for c in calls] == ["/mcp/approvals/apr_1/consume"]

    def test_with_approval_OFF_it_is_submitted_directly(self):
        _, calls = _call(ARGS, approval=False)
        assert calls == [("POST", "/probe-monitors/judge-runs", BODY)]

    def test_a_judge_ELSEWHERE_is_one_direct_call(self):
        args = {**ARGS, "endpoint": "https://api.openai.com/v1"}
        _, calls = _call(args)
        assert calls == [("POST", "/probe-monitors/judge-runs", {**BODY, "endpoint": args["endpoint"]})]

    def test_the_backend_accepts_it_as_CALLER_executed_and_summarises_it_plainly(self):
        from src.api.v1.endpoints import mcp_approvals

        assert "start_probe_judge_run" in mcp_approvals.CALLER_TOOLS
        text = mcp_approvals._summarize("start_probe_judge_run", BODY)
        assert text.startswith("Judge 1 set(s) with qwen on miLLM, up to 200 rows each")


@pytest.mark.parametrize(
    "endpoint,configured,expected",
    [
        ("http://millm-backend.millm.svc.cluster.local:8000/v1", "", True),
        ("http://k8s-millm.hitsai.local/v1", "", True),
        ("http://10.0.0.5:8000/v1", "http://10.0.0.5:8000", True),
        ("http://10.0.0.6:8000/v1", "http://10.0.0.5:8000", False),
        ("https://api.openai.com/v1", "http://millm:8000", False),
        ("not a url", "", False),
    ],
)
def test_which_endpoints_are_miLLM(endpoint, configured, expected):
    assert judge_endpoint_is_millm(endpoint, configured) is expected


def test_the_library_tools_are_in_the_LIVE_registry_under_probe_monitors(monkeypatch):
    """Built the way production builds it, with only `probe_monitors` enabled: removing the
    module from CATEGORY_MODULES makes every one of these disappear."""
    from src.mcp_server.server import build_server

    settings = MCPSettings(tool_categories="probe_monitors", allow_anonymous=True)
    mcp, _ = build_server(settings, stdio=True)
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    expected = {
        "list_judge_prompts", "get_judge_prompt", "create_judge_prompt", "update_judge_prompt",
        "new_judge_prompt_version", "clone_judge_prompt", "archive_judge_prompt",
        "unarchive_judge_prompt", "delete_judge_prompt", "preview_judge_prompt",
        "try_judge_prompt", "export_judge_prompt", "import_judge_prompt",
        "list_midataworks_rubrics_for_judging", "import_judge_prompt_from_rubric",
        "set_probe_behaviour",
    }
    assert expected <= names, sorted(expected - names)
