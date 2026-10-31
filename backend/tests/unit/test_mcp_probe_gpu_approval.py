"""The probe GPU starts and their approval (MCP gap closure, 2026-10-19).

`start_probe_training_run` and `evaluate_probe_monitor` each hold a GPU for minutes to hours, so
an agent's call waits for the operator — the `millm_load_model` caller-executed shape:

1. first call, no id → files `POST /mcp/approvals` with EXACTLY the payload, queues nothing;
2. re-call with the id → `POST /mcp/approvals/{id}/consume` with the SAME payload, then the job;
3. a consume refusal (pending, denied, consumed, mismatch) → a structured status and NO job;
4. `MCP_PROBE_GPU_APPROVAL=false` → the job is submitted directly, one call.

Asserted by payload AND call count — `test_reachability.py` exempts these two tools from its
one-call rule and names this file. Also here: the deployment states the flag (fails closed), the
batch scorer issues one call per input and validates them all first, the unacknowledged run
delete issues nothing, and the feature-by-index tool's training branch.

Controls (recorded in the 2026-10-19 hand-back; restored from saved bytes):
  C4 `if not settings.probe_gpu_approval: return None` → `return None` unconditionally
     → test_the_first_call_files_an_approval_and_queues_NOTHING red
  C5 drop `MCP_PROBE_GPU_APPROVAL` from k8s/base/mcp.yaml → test_the_DEPLOYMENT_states_it red
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from src.mcp_server.client import BackendError
from src.mcp_server.config import MCPSettings

REPO = Path(__file__).resolve().parents[3]

RUN_ARGS = {"model_id": "m_1", "train_dataset_id": "pmd_1", "eval_dataset_ids": ["pmd_2"],
            "config": {"layers": [11], "rules": ["mean"]}}
RUN_PAYLOAD = {"model_id": "m_1", "train_dataset_id": "pmd_1", "eval_dataset_ids": ["pmd_2"],
               "gpu": "auto", "config": {"layers": [11], "rules": ["mean"]}}
EVAL_ARGS = {"probe_id": "pm_1", "dataset_ids": ["pmd_2", "pmd_3"]}
EVAL_PAYLOAD = {"probe_id": "pm_1", "dataset_ids": ["pmd_2", "pmd_3"]}


class _Client:
    """A miStudio client stand-in: records every call; `fail` maps a path to an exception."""

    def __init__(self, fail=None):
        self.calls: list[tuple[str, str, object]] = []
        self.fail = fail or {}

    async def _do(self, method, path, body):
        self.calls.append((method, path, body))
        if path in self.fail:
            raise self.fail[path]
        if path == "/mcp/approvals":
            return {"id": "apr_1", "status": "pending"}
        return {"id": "pmr_9", "task_id": "t_9", "status": "queued"}

    async def post(self, path, json_body=None, **params):
        return await self._do("POST", path, json_body)

    async def get(self, path, **params):
        return await self._do("GET", path, params)

    async def delete(self, path):
        return await self._do("DELETE", path, None)


def _call(tool, args, *, approval=True, client=None):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import probe_monitors

    client = client or _Client()
    mcp = MCPServer("t")
    probe_monitors.register(
        mcp, client, MCPSettings(auth_token="x" * 32, probe_gpu_approval=approval)
    )
    result = asyncio.run(mcp._tool_manager._tools[tool].fn(**args))  # noqa: SLF001
    return result, client.calls


@pytest.mark.parametrize("tool,args,payload", [
    ("start_probe_training_run", RUN_ARGS, RUN_PAYLOAD),
    ("evaluate_probe_monitor", EVAL_ARGS, EVAL_PAYLOAD),
])
class TestTheApprovalSequence:
    def test_the_first_call_files_an_approval_and_queues_NOTHING(self, tool, args, payload):
        result, calls = _call(tool, args)
        assert calls == [("POST", "/mcp/approvals", {"tool_name": tool, "payload": payload})]
        assert result["status"] == "pending_approval"
        assert result["approval_request_id"] == "apr_1"

    def test_the_approved_recall_consumes_THEN_submits_the_same_payload(self, tool, args, payload):
        result, calls = _call(tool, {**args, "approval_request_id": "apr_1"})
        assert len(calls) == 2, calls
        assert calls[0] == ("POST", "/mcp/approvals/apr_1/consume",
                            {"tool_name": tool, "payload": payload})
        if tool == "start_probe_training_run":
            assert calls[1] == ("POST", "/probe-monitors/runs", payload)
        else:
            # The route's body IS the list.
            assert calls[1] == ("POST", "/probe-monitors/probes/pm_1/evaluate",
                                ["pmd_2", "pmd_3"])
        assert result["task_id"] == "t_9"

    @pytest.mark.parametrize("status_code,detail,expected", [
        (409, {"status": "pending", "reason": None}, "pending"),
        (409, {"status": "denied", "reason": "no"}, "denied"),
        (409, {"status": "consumed", "reason": None}, "consumed"),
        (404, "missing", "not_found"),
        (422, "the payload differs", "mismatch"),
    ])
    def test_a_refused_consume_submits_NOTHING(self, tool, args, payload, status_code, detail,
                                              expected):
        client = _Client(fail={"/mcp/approvals/apr_1/consume": BackendError(status_code, detail)})
        result, calls = _call(tool, {**args, "approval_request_id": "apr_1"}, client=client)
        assert [c[1] for c in calls] == ["/mcp/approvals/apr_1/consume"]
        assert result["status"] == expected and result["approval_request_id"] == "apr_1"

    def test_with_approval_OFF_the_job_is_submitted_directly(self, tool, args, payload):
        result, calls = _call(tool, args, approval=False)
        assert len(calls) == 1
        assert "/mcp/approvals" not in calls[0][1]
        assert result["task_id"] == "t_9"


class TestTheFlagFailsClosed:
    def test_the_code_default_is_ON(self):
        assert MCPSettings(allow_anonymous=True).probe_gpu_approval is True

    def test_the_DEPLOYMENT_states_it(self):
        text = (REPO / "k8s" / "base" / "mcp.yaml").read_text()
        m = re.search(r"name:\s*MCP_PROBE_GPU_APPROVAL\s*\n(?:\s*#[^\n]*\n)*\s*value:\s*\"(\w+)\"",
                      text)
        assert m and m.group(1) == "true", "k8s/base/mcp.yaml must set MCP_PROBE_GPU_APPROVAL=true"

    def test_COMPOSE_defaults_it_on(self):
        compose = (REPO / "docker-compose.yml").read_text()
        m = re.search(r"MCP_PROBE_GPU_APPROVAL:\s*\$\{MCP_PROBE_GPU_APPROVAL:-(\w+)\}", compose)
        assert m and m.group(1) == "true"

    def test_the_backend_accepts_both_tools_as_CALLER_executed(self):
        from src.api.v1.endpoints import mcp_approvals

        assert {"start_probe_training_run", "evaluate_probe_monitor"} <= mcp_approvals.CALLER_TOOLS


class TestTheBatchScorer:
    """Feature 041 T19: ONE POST to `…/score-batch` carrying every input; the cap is miStudio's runtime
    setting, enforced by the route (`test_family_scoring.py::test_single_and_family_caps_read_one_setting`)."""

    def test_one_call_carrying_every_input(self):
        from src.mcp_server.tools.probe_monitors import ProbeScoreInput

        inputs = [ProbeScoreInput(text="a"), ProbeScoreInput(text="b"),
                  ProbeScoreInput(messages=[{"role": "user", "content": "c"}])]
        result, calls = _call("score_probe_monitor_batch", {"probe_id": "pm_1", "inputs": inputs})
        assert calls == [("POST", "/probe-monitors/probes/pm_1/score-batch", {"inputs": [
            {"text": "a"}, {"text": "b"}, {"messages": [{"role": "user", "content": "c"}]}]})]

    def test_a_bad_input_is_refused_BEFORE_anything_is_queued(self):
        from src.mcp_server.tools.probe_monitors import ProbeScoreInput

        inputs = [ProbeScoreInput(text="a"), ProbeScoreInput()]
        result, calls = _call("score_probe_monitor_batch", {"probe_id": "pm_1", "inputs": inputs})
        assert calls == [] and result["error"] == "exactly_one_input"


def test_the_unacknowledged_run_delete_issues_NOTHING():
    result, calls = _call("delete_probe_training_run", {"run_id": "pmr_1"})
    assert calls == [] and result["refused"] == "delete_not_acknowledged"


def test_score_needs_exactly_one_input():
    result, calls = _call("score_probe_monitor", {"probe_id": "pm_1"})
    assert calls == [] and result["error"] == "exactly_one_input"


def test_the_dataset_view_omits_unset_optionals_and_keeps_the_manifest_name():
    _result, calls = _call("register_probe_dataset", {
        "name": "v", "dataset_id": "d", "input_column": "t", "label_column": "y",
        "label_mapping": {"1": "positive", "0": "negative"}})
    assert calls == [("POST", "/probe-monitors/datasets", {
        "name": "v", "dataset_id": "d", "input_column": "t", "label_column": "y",
        "label_mapping": {"1": "positive", "0": "negative"}, "role": "eval"})]


class TestFeatureByIndex:
    def _call(self, args):
        from mcp.server.mcpserver import MCPServer

        from src.mcp_server.tools import features

        client = _Client()
        mcp = MCPServer("t")
        features.register(mcp, client, MCPSettings(auth_token="x" * 32))
        result = asyncio.run(mcp._tool_manager._tools["get_feature_by_index"].fn(**args))  # noqa: SLF001
        return result, client.calls

    def test_the_training_branch(self):
        _result, calls = self._call({"training_id": "train_1", "feature_idx": 7})
        assert calls == [("GET", "/trainings/train_1/features/by-index/7", {})]

    @pytest.mark.parametrize("args", [{"feature_idx": 7},
                                      {"feature_idx": 7, "sae_id": "s", "training_id": "t"}])
    def test_exactly_one_owner_or_nothing_is_sent(self, args):
        result, calls = self._call(args)
        assert calls == [] and result["error"] == "exactly_one_owner"
