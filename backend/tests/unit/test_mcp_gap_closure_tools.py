"""The 2026-10-19 MCP gap closure: the new tools are on the BUILT server by name, and miLLM's
refusals reach an agent as structured results through the REAL client.

Registry presence is asserted against a hand-written list on purpose: a list derived from the
`@mcp.tool` decorators would lose a tool together with its decorator (the mutation that survived
the derived check on 2026-10-16), so the one expectation a deleted registration cannot edit is a
list of names.

The refusal tests drive the real `MiLLMClient` over an `httpx.MockTransport`, so the path under
test is the one production runs: HTTP status + OpenAI/management error envelope + response headers
→ `BackendError` (with headers) → `millm_send.refusal` → the tool's result. A stand-in client
would have to re-implement exactly the header plumbing being tested.

Controls (recorded in the 2026-10-19 hand-back; restored from saved bytes):
  C6 drop the `body["scoring_precision"] = scoring_precision` pass-through
     → EXPECTED_CALLS[millm_score_completion] and test_float32_TRAVELS_and_its_echo_is_reported red
  C7 delete the scoring_precision_unavailable branch of `refusal()`
     → test_a_precision_refusal_carries_its_REASON_from_the_header red
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from tests.support.external_ctx import make_ctx

NEW_TOOLS = {
    "probe_monitors": {
        "register_probe_dataset", "list_probe_datasets", "get_probe_dataset",
        "start_probe_training_run", "list_probe_training_runs",
        "get_probe_training_run", "cancel_probe_training_run", "delete_probe_training_run",
        "evaluate_probe_monitor", "get_probe_evaluation_scores", "score_probe_monitor",
        "score_probe_monitor_batch", "get_probe_score_result", "start_probe_judge_run",
        "list_probe_judge_runs", "cancel_probe_judge_run",
    },
    "read": {"get_feature_by_index", "get_training", "list_datasets", "get_dataset",
             "list_saes", "get_sae"},
    "jobs": {"list_gpus"},
    "groups": {"cancel_feature_grouping"},
    "circuits": {"cancel_circuit_capture", "cancel_circuit_discovery", "cancel_attribution_pass",
                 "cancel_edge_validation", "cancel_circuit_faithfulness",
                 "cancel_circuit_calibration", "cancel_steering_samples"},
    "jlens": {"jlens_probe", "get_jlens_probe", "jlens_token_check", "revalidate_jlens_artifact",
              "get_jlens_artifact_document", "validate_jlens_document"},
    "millm_inference": {"millm_embeddings", "millm_batch_lease"},
    "millm_probes": {"millm_get_probe", "millm_delete_probe", "millm_probe_parity",
                     "millm_probe_hub_search", "millm_probe_hub_definitions",
                     "millm_clear_probe_events", "millm_get_probe_event"},
    "millm_clusters": {"millm_cluster_hub_definitions"},
    "millm_sensing": {"millm_sensing_clear", "millm_get_sensing_event"},
    "dataworks": {"dataworks_list_detector_sends", "dataworks_get_detector_send",
                  "dataworks_resume_detector_send", "dataworks_refresh_probe_results",
                  "dataworks_get_probe_results"},
}


def _built(category, monkeypatch) -> set[str]:
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.server import build_server

    for spec in PRODUCTS:
        monkeypatch.setenv(spec.url_env, f"http://{spec.name}.test")
    mcp, _ = build_server(MCPSettings(tool_categories=category, allow_anonymous=True), stdio=True)
    return {t.name for t in asyncio.run(mcp.list_tools())}


@pytest.mark.parametrize("category", sorted(NEW_TOOLS))
def test_the_new_tools_reach_the_BUILT_server_by_name(category, monkeypatch):
    missing = sorted(NEW_TOOLS[category] - _built(category, monkeypatch))
    assert not missing, f"{category}: {missing} are not on the built server"


def test_the_deployment_enables_every_category_the_new_tools_live_in():
    """Layer four (memory: mcp-category-needs-deployment-enabling)."""
    from tests.unit.test_reachability import TestTheDEPLOYMENTEnablesWhatTheCodeRegisters as T

    deployed = T()._deployed_categories()
    assert set(NEW_TOOLS) <= deployed, sorted(set(NEW_TOOLS) - deployed)


# ── miLLM refusals through the real client ─────────────────────────────────────────────────────


def _tools_over(handler):
    """millm tool functions wired to a REAL MiLLMClient whose transport is `handler`."""
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.millm_client import MiLLMClient
    from src.mcp_server.tools import millm_inference, millm_models

    client = MiLLMClient("http://millm.test")
    client._http = httpx.AsyncClient(  # noqa: SLF001
        base_url="http://millm.test", transport=httpx.MockTransport(handler))
    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, "ok"))
    gate.invalidate = MagicMock()
    mcp = MCPServer("t")
    ctx = make_ctx()
    millm_inference.register(mcp, client, gate, ctx)
    millm_models.register(mcp, client, gate, ctx)
    return mcp._tool_manager._tools  # noqa: SLF001


def _openai_error(status, code, message="m", headers=None):
    def handler(request):
        handler.seen.append(json.loads(request.content or b"null"))
        return httpx.Response(
            status, headers=headers or {},
            json={"error": {"message": message, "type": "invalid_request_error",
                            "param": None, "code": code}})
    handler.seen = []
    return handler


def _run(tools, name, **args):
    return asyncio.run(tools[name].fn(**args))


class TestScoringPrecision:
    def test_float32_TRAVELS_and_its_echo_is_reported(self):
        def handler(request):
            handler.body = json.loads(request.content)
            return httpx.Response(200, headers={"X-miLLM-Scoring-Precision": "float32"},
                                  json={"choices": [], "scoring_precision": "float32"})

        result = _run(_tools_over(handler), "millm_score_completion", model="m", prompt=["a"],
                      scoring_precision="float32")
        assert handler.body["scoring_precision"] == "float32"
        assert result["millm_headers"]["scoring_precision"] == "float32"

    def test_omitted_is_NOT_sent(self):
        def handler(request):
            handler.body = json.loads(request.content)
            return httpx.Response(200, json={"choices": []})

        _run(_tools_over(handler), "millm_score_completion", model="m", prompt=["a"])
        assert "scoring_precision" not in handler.body

    @pytest.mark.parametrize("reason", ["memory", "device", "model_not_float32_castable", "busy"])
    def test_a_precision_refusal_carries_its_REASON_from_the_header(self, reason):
        handler = _openai_error(409, "scoring_precision_unavailable", "float32 unavailable",
                                headers={"X-miLLM-Scoring-Precision-Reason": reason})
        result = _run(_tools_over(handler), "millm_score_completion", model="m", prompt=["a"],
                      scoring_precision="float32")
        assert result["refused"] == "scoring_precision_unavailable"
        assert result["reason"] == reason
        assert result["status_code"] == 409

    def test_the_parameter_is_described_with_both_values(self):
        from src.mcp_server.tools.millm_inference import SCORE_COMPLETION_DESCRIPTION

        assert "float32" in SCORE_COMPLETION_DESCRIPTION
        assert "REFUSED, never downgraded" in SCORE_COMPLETION_DESCRIPTION


class TestNameRefusals:
    @pytest.mark.parametrize("code,next_tool", [
        ("model_not_generative", "millm_list_models"),
        ("ambiguous_model_name", "millm_list_models"),
    ])
    @pytest.mark.parametrize("tool,args", [
        ("millm_generate", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 4}),
        ("millm_score_completion", {"prompt": ["x"]}),
    ])
    def test_a_400_name_refusal_is_STRUCTURED(self, code, next_tool, tool, args):
        from src.mcp_server.tools.millm_inference import ChatMessage

        if "messages" in args:
            args = {**args, "messages": [ChatMessage(**m) for m in args["messages"]]}
        result = _run(_tools_over(_openai_error(400, code, "pick another")), tool, model="m",
                      **args)
        assert result["refused"] == code and result["next"] == next_tool
        assert result["status_code"] == 400 and result["hint"]

    def test_MODEL_NAME_TAKEN_on_download_is_STRUCTURED(self):
        def handler(request):
            return httpx.Response(409, json={
                "success": False, "data": None,
                "error": {"code": "MODEL_NAME_TAKEN", "message": "name 'x:Q8' is used by 7",
                          "details": {"name": "x:Q8", "model_id": 7}}})

        result = _run(_tools_over(handler), "millm_download_model", repo_id="o/x",
                      quantization="Q8")
        assert result["refused"] == "model_name_taken"
        assert result["next"] == "millm_download_model" and "custom_name" in result["hint"]

    def test_any_other_download_error_still_RAISES(self):
        from src.mcp_server.client import BackendError

        def handler(request):
            return httpx.Response(400, json={
                "success": False, "data": None,
                "error": {"code": "VALIDATION_ERROR", "message": "bad", "details": None}})

        with pytest.raises(BackendError):
            _run(_tools_over(handler), "millm_download_model", repo_id="o/x")


class TestEmbeddings:
    def test_the_request_and_its_shape(self):
        def handler(request):
            handler.request = request
            return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

        result = _run(_tools_over(handler), "millm_embeddings", model="e", input=["a"],
                      pooling="last")
        assert handler.request.url.path == "/v1/embeddings"
        assert json.loads(handler.request.content)["pooling"] == "last"
        assert handler.request.headers["X-miLLM-Load-Policy"] == "refuse"
        assert result["result"]["data"][0]["embedding"] == [0.1]

    def test_a_non_resident_model_is_REFUSED_not_loaded(self):
        result = _run(_tools_over(_openai_error(409, "model_not_resident")), "millm_embeddings",
                      model="e", input=["a"])
        assert result["refused"] == "model_not_resident"

    @pytest.mark.parametrize("inputs,error", [(["a", ""], "empty_input"),
                                              (["a"] * 129, "over_synchronous_cap")])
    def test_bad_inputs_are_refused_before_anything_is_sent(self, inputs, error):
        def handler(request):  # pragma: no cover - must not be reached
            raise AssertionError("sent")

        result = _run(_tools_over(handler), "millm_embeddings", model="e", input=inputs)
        assert result["error"] == error


class TestScopedAndAcknowledgedDeletes:
    def _tools(self):
        from mcp.server.mcpserver import MCPServer

        from src.mcp_server.tools import millm_probes, millm_sensing
        from tests.unit.test_reachability import RecordingClient

        client = RecordingClient()
        gate = MagicMock()
        gate.check = AsyncMock(return_value=(True, "ok"))
        mcp = MCPServer("t")
        millm_probes.register(mcp, client, gate, make_ctx())
        millm_sensing.register(mcp, client, gate, make_ctx())
        return mcp._tool_manager._tools, client  # noqa: SLF001

    @pytest.mark.parametrize("tool,args,error", [
        ("millm_clear_probe_events", {}, "scope_required"),
        ("millm_clear_probe_events", {"probe_id": "p", "all_probes": True}, "ambiguous_scope"),
        ("millm_sensing_clear", {}, "scope_required"),
        ("millm_sensing_clear", {"profile_id": "p", "all_profiles": True}, "ambiguous_scope"),
    ])
    def test_an_unscoped_clear_sends_NOTHING(self, tool, args, error):
        tools, client = self._tools()
        result = asyncio.run(tools[tool].fn(**args))
        assert client.calls == [] and result["error"] == error

    def test_clearing_ALL_is_explicit_and_unscoped_on_the_wire(self):
        tools, client = self._tools()
        asyncio.run(tools["millm_clear_probe_events"].fn(all_probes=True))
        assert client.calls == [("DELETE", "/api/probes/events", {"probe_id": None})]

    def test_an_unacknowledged_probe_delete_sends_NOTHING(self):
        tools, client = self._tools()
        result = asyncio.run(tools["millm_delete_probe"].fn(probe_id="pr_1"))
        assert client.calls == [] and result["refused"] == "delete_not_acknowledged"


def test_the_event_DETAIL_tools_say_they_return_user_content(monkeypatch):
    """Operator decision 2026-10-19: exposed, and the description must say so plainly — the
    agent reading the tool list is the only reader who can decide not to fetch."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.server import build_server

    for spec in PRODUCTS:
        monkeypatch.setenv(spec.url_env, f"http://{spec.name}.test")
    mcp, _ = build_server(MCPSettings(tool_categories="millm_probes,millm_sensing",
                                      allow_anonymous=True), stdio=True)
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    for name in ("millm_get_probe_event", "millm_get_sensing_event"):
        assert "DECODED USER PROMPT CONTENT" in (tools[name].description or ""), name
    assert "millm_get_probe_event" in tools["millm_probe_events"].description
