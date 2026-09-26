"""The `probe_monitors` MCP category (033 FR-9, task 5.3).

⚠ REGISTRATION IS THE POINT OF THIS FILE. This server once shipped 16 `millm_circuit_*` tools that
were fully implemented, unit-tested and documented in `docs/mcp-contract.md` while registered
nowhere — every test passed by importing the module directly, so the suite was green and the docs
said ✅ while no agent could call the feature. So the assertions here read the LIVE registry, the
built server, and the deployment's category list, never the module.

MUTATION CONTROLS (each verified to fail this file or test_reachability.py):
  M1  `probe_monitors` removed from `CATEGORY_MODULES`      → the registry test
  M2  removed from `VALID_CATEGORIES`                       → the valid-category test
  M3  removed from `DEFAULT_CATEGORIES`                     → the default test
  M4  removed from k8s `MCP_TOOL_CATEGORIES`                → test_reachability's deployment test
  M5  a tool's route re-pointed                             → test_reachability's payload assertion
  M6  the token added to the publish body                   → the secrecy test
  M7  a parameter's `Field(description=…)` dropped          → the described-parameter test
"""
import asyncio
import inspect

import pytest

from src.mcp_server.config import DEFAULT_CATEGORIES, VALID_CATEGORIES, MCPSettings
from src.mcp_server.tools import CATEGORY_MODULES, probe_monitors

EXPECTED_TOOLS = {
    "list_probe_monitors",
    "get_probe_monitor_report",
    "build_probe_definition",
    "export_probe_definition",
    "publish_probe_definition",
}


def _registered():
    from mcp.server.fastmcp import FastMCP

    from tests.unit.test_reachability import RecordingClient

    mcp = FastMCP("t")
    client = RecordingClient()
    probe_monitors.register(mcp, client, MCPSettings(auth_token="x" * 32))
    return mcp._tool_manager, client


class TestTheCategoryIsRegisteredEverywhereItHasToBe:
    def test_it_is_in_the_module_registry(self):
        assert "probe_monitors" in CATEGORY_MODULES
        assert probe_monitors in CATEGORY_MODULES["probe_monitors"]

    def test_it_is_a_valid_category(self):
        assert "probe_monitors" in VALID_CATEGORIES

    def test_it_is_ON_BY_DEFAULT(self):
        """Unlike `admin` (irreversible deletes) and the `millm_*` categories (need a URL), these
        are read-mostly plus two writes that go through the evidence gate."""
        assert "probe_monitors" in DEFAULT_CATEGORIES.split(",")

    def test_the_deployment_enables_it(self):
        """The layer that made 35 circuit tools unreachable in production while every code-level
        test was green: an explicit `MCP_TOOL_CATEGORIES` that omitted the category."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]
        manifests = [
            path
            for path in (root / "k8s").rglob("*.yaml")
            if "MCP_TOOL_CATEGORIES" in path.read_text()
        ]
        assert manifests, "no manifest sets MCP_TOOL_CATEGORIES"
        for path in manifests:
            assert "probe_monitors" in path.read_text(), (
                f"{path} sets MCP_TOOL_CATEGORIES without probe_monitors, so the tools are "
                f"registered and switched off in production"
            )


class TestEveryToolIsExposed:
    def test_all_five_are_registered(self):
        manager, _ = _registered()
        names = {tool.name for tool in manager.list_tools()}
        assert EXPECTED_TOOLS <= names, EXPECTED_TOOLS - names

    def test_no_extras_crept_in(self):
        """A tool added here without an `EXPECTED_CALLS` entry ships unverified."""
        manager, _ = _registered()
        names = {tool.name for tool in manager.list_tools()}
        assert names == EXPECTED_TOOLS, names ^ EXPECTED_TOOLS

    def test_every_tool_has_a_docstring_an_agent_can_act_on(self):
        manager, _ = _registered()
        for tool in manager.list_tools():
            assert tool.description and len(tool.description) > 80, tool.name

    def test_every_parameter_is_described(self):
        """An agent cannot infer from a bare `str` that `probe_id` wants `pm_…`, or that a rung
        below 2 needs a reason. The estate has a gate for this on other categories."""
        manager, _ = _registered()
        undescribed = []
        for tool in manager.list_tools():
            schema = tool.parameters or {}
            for name, spec in (schema.get("properties") or {}).items():
                if not spec.get("description"):
                    undescribed.append(f"{tool.name}.{name}")
        assert not undescribed, undescribed


class TestTheDescriptionsCarryTheEvidenceRule:
    """⚠ THE RUNG IS THE PRODUCT, NOT A DETAIL. An agent that exports a rung-1 probe and serves it
    as a monitor has made a claim the evidence does not support, and the tool description is the
    only place it will read."""

    def test_the_build_tool_explains_the_rung_ladder(self):
        manager, _ = _registered()
        tool = next(t for t in manager.list_tools() if t.name == "build_probe_definition")
        text = (tool.description or "") + str(tool.parameters)
        for phrase in ("rung", "acknowledge"):
            assert phrase in text.lower(), phrase

    def test_the_report_tool_says_to_read_the_rung_first(self):
        manager, _ = _registered()
        tool = next(t for t in manager.list_tools() if t.name == "get_probe_monitor_report")
        assert "rung" in (tool.description or "").lower()

    def test_the_publish_tool_states_the_manifest_is_merged(self):
        """A publisher that replaced the manifest would delete other people's probes; an agent
        should know it does not have to worry about that."""
        manager, _ = _registered()
        tool = next(t for t in manager.list_tools() if t.name == "publish_probe_definition")
        assert "merged" in (tool.description or "").lower()

    def test_the_publish_tool_says_the_token_is_not_a_parameter(self):
        manager, _ = _registered()
        tool = next(t for t in manager.list_tools() if t.name == "publish_probe_definition")
        assert "token" in (tool.description or "").lower()
        assert "token" not in (tool.parameters or {}).get("properties", {}), (
            "the token is a tool parameter, so a credential would pass through the agent transcript"
        )


class TestTheCallsThemselves:
    def test_the_publish_body_carries_no_credential(self):
        manager, client = _registered()
        asyncio.run(
            manager.call_tool(
                "publish_probe_definition", {"probe_id": "pm_1", "repo_id": "owner/probes"}
            )
        )
        _method, _path, payload = client.calls[0]
        assert "token" not in str(payload).lower()

    def test_an_acknowledgement_is_sent_only_when_asked_for(self):
        manager, client = _registered()
        asyncio.run(manager.call_tool("build_probe_definition", {"probe_id": "pm_1"}))
        assert "acknowledge_below_rung2" not in client.calls[0][2]["json_body"], (
            "an empty acknowledgement would record a judgement nobody made"
        )

    def test_the_acknowledgement_is_sent_when_it_is(self):
        manager, client = _registered()
        asyncio.run(
            manager.call_tool(
                "build_probe_definition",
                {"probe_id": "pm_1", "acknowledge_below_rung2_reason": "exploratory, not gating"},
            )
        )
        body = client.calls[0][2]["json_body"]
        assert body["acknowledge_below_rung2"] == {"reason": "exploratory, not gating"}

    def test_the_list_tool_sends_query_arguments_not_a_params_dict(self):
        """⚠ FOUND WHILE WRITING THESE. `MiStudioClient.get` takes `**params`, so `params={...}`
        sends a query field literally named "params". A path-only assertion could not tell."""
        manager, client = _registered()
        asyncio.run(manager.call_tool("list_probe_monitors", {"run_id": "pmr_1"}))
        _method, _path, payload = client.calls[0]
        assert "params" not in payload, payload
        assert payload.get("run_id") == "pmr_1"


class TestEveryRouteTheseToolsCallExists:
    """A tool posting to a path nobody serves is the unregistered-tool defect wearing another hat:
    the agent gets a 404 it cannot interpret and the contract says the capability is there."""

    def test_each_path_is_in_the_served_openapi(self):
        import re

        from src.main import app

        served = set(app.openapi()["paths"])
        source = inspect.getsource(probe_monitors)
        # The literal paths the tools pass to the client, with f-string holes generalised.
        raw = set(re.findall(r'"(/probe-monitors/[^"]*)"', source))
        assert raw, "no probe-monitor paths found in the module"
        for path in raw:
            pattern = re.sub(r"\{[^}]+\}", "{}", path)
            candidates = [
                served_path
                for served_path in served
                if re.sub(r"\{[^}]+\}", "{}", served_path).endswith(pattern)
            ]
            assert candidates, f"no served route matches {path}"


class TestThePublishPreflightDistinguishesRejectedFromUnreachable:
    """⚠ "A TOKEN IS PRESENT" IS NOT "A TOKEN WORKS", and the publish endpoint promised the second.

    It refused only on an ABSENT token, so an expired one produced a 202, a task that created
    nothing and a failure at the last step — the shape its own docstring says it exists to prevent.

    Found during 033 acceptance: this installation's stored token is 37 characters, begins `hf_`,
    and HuggingFace answers `Invalid user token`. Nothing had said so, because public downloads need
    no credential and no publish had ever run.

    The classification is the load-bearing part. A 401 is a refusal. A timeout or a 5xx is NOT — it
    does not prove the token is bad, and refusing on it would make publishing fail closed against an
    outage it has no business judging.
    """

    def _reject(self, exc: Exception, monkeypatch):
        import src.api.v1.endpoints.probe_monitors as endpoints

        class _Api:
            def __init__(self, token=None):
                self.token = token

            def whoami(self):
                raise exc

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
        return endpoints._huggingface_rejects("hf_whatever")

    def test_a_valid_token_is_not_refused(self, monkeypatch):
        import src.api.v1.endpoints.probe_monitors as endpoints
        import huggingface_hub

        class _Api:
            def __init__(self, token=None):
                pass

            def whoami(self):
                return {"name": "someone"}

        monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
        assert endpoints._huggingface_rejects("hf_good") is None

    def test_an_invalid_token_is_refused_with_an_actionable_reason(self, monkeypatch):
        reason = self._reject(Exception("Invalid user token."), monkeypatch)
        assert reason is not None
        assert "Settings" in reason
        assert "last step" in reason

    def test_a_401_response_is_refused_even_without_that_message(self, monkeypatch):
        class _Response:
            status_code = 401

        error = Exception("boom")
        error.response = _Response()
        assert self._reject(error, monkeypatch) is not None

    def test_a_403_is_refused_too(self, monkeypatch):
        class _Response:
            status_code = 403

        error = Exception("forbidden")
        error.response = _Response()
        assert self._reject(error, monkeypatch) is not None

    def test_a_TIMEOUT_IS_NOT_A_REFUSAL(self, monkeypatch):
        """The whole reason the classification exists: an outage must not block a publish."""
        assert self._reject(TimeoutError("hub unreachable"), monkeypatch) is None

    def test_a_5xx_IS_NOT_A_REFUSAL(self, monkeypatch):
        class _Response:
            status_code = 503

        error = Exception("service unavailable")
        error.response = _Response()
        assert self._reject(error, monkeypatch) is None
