"""MCP smoke tests for the Feature 016 circuit-discovery tools (R2 T4 /
SC-7). Each tool posts/gets the right path with the documented body shape,
via a mocked client. Uses the _register_and_get harness pattern."""

from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from src.mcp_server.client import MiStudioClient
from src.mcp_server.config import MCPSettings


def _register_and_get(tool_name):
    from mcp.server.fastmcp import FastMCP

    from src.mcp_server.tools import circuits

    mcp = FastMCP("test")
    client = MagicMock(spec=MiStudioClient)
    client.get = AsyncMock(return_value={"ok": 1})
    client.post = AsyncMock(return_value={"ok": 1})
    client.patch = AsyncMock(return_value={"ok": 1})
    client.delete = AsyncMock(return_value={"ok": 1})
    settings = MCPSettings(auth_token="t", _env_file=None)
    circuits.register(mcp, client, settings)
    return mcp._tool_manager._tools[tool_name], client  # noqa: SLF001


class TestCircuitDiscoveryToolSmoke:
    def test_start_circuit_capture_defaults_confirm_false(self):
        tool, client = _register_and_get("start_circuit_capture")
        anyio.run(lambda: tool.run({
            "dataset_id": "ds1", "layers": [{"layer": 13, "sae_id": "s1"}]}))
        args, kwargs = client.post.call_args
        assert args[0] == "/circuit-capture"
        assert kwargs["json_body"]["confirm"] is False
        assert kwargs["json_body"]["dataset_id"] == "ds1"

    def test_list_circuit_captures(self):
        tool, client = _register_and_get("list_circuit_captures")
        anyio.run(lambda: tool.run({}))
        client.get.assert_awaited_once_with("/circuit-capture")

    def test_run_circuit_discovery_passes_seed_refs(self):
        tool, client = _register_and_get("run_circuit_discovery")
        anyio.run(lambda: tool.run({
            "capture_run_id": "cap1", "mode": "seeded", "granularity": "cluster",
            "seed_refs": [{"layer": 13, "cluster_profile_id": "clp"}]}))
        args, kwargs = client.post.call_args
        assert args[0] == "/circuit-discovery"
        assert kwargs["json_body"]["seed_refs"] == [
            {"layer": 13, "cluster_profile_id": "clp"}]
        assert kwargs["json_body"]["mode"] == "seeded"

    def test_get_discovery_results_forwards_include_flag(self):
        tool, client = _register_and_get("get_discovery_results")
        anyio.run(lambda: tool.run({"run_id": "dsc1"}))
        args, kwargs = client.get.call_args
        assert args[0] == "/circuit-discovery/dsc1"
        assert kwargs.get("include_candidates") is True

    def test_run_attribution_pass(self):
        tool, client = _register_and_get("run_attribution_pass")
        anyio.run(lambda: tool.run({"run_id": "dsc1"}))
        args, _ = client.post.call_args
        assert args[0] == "/circuit-discovery/dsc1/attribution"


class TestValidationToolSmoke:
    """017 MCP tools (validation + manifests)."""

    def test_validate_circuit_edges(self):
        tool, client = _register_and_get("validate_circuit_edges")
        anyio.run(lambda: tool.run({"run_id": "dsc1", "ordering": "attr", "k": 10}))
        args, kwargs = client.post.call_args
        assert args[0] == "/circuit-discovery/dsc1/validate"
        assert kwargs["json_body"]["ordering"] == "attr"
        assert kwargs["json_body"]["k"] == 10

    def test_run_circuit_faithfulness(self):
        tool, client = _register_and_get("run_circuit_faithfulness")
        anyio.run(lambda: tool.run({"circuit_id": "crc_1", "mode": "both"}))
        args, kwargs = client.post.call_args
        assert args[0] == "/circuits/crc_1/faithfulness"
        assert kwargs["json_body"]["mode"] == "both"

    def test_get_validation_manifest(self):
        tool, client = _register_and_get("get_validation_manifest")
        anyio.run(lambda: tool.run({"manifest_id": "vman_1"}))
        client.get.assert_awaited_once_with("/validation-manifests/vman_1")

    def test_reproduce_validation(self):
        tool, client = _register_and_get("reproduce_validation")
        anyio.run(lambda: tool.run({"manifest_id": "vman_1"}))
        args, _ = client.post.call_args
        assert args[0] == "/validation-manifests/vman_1/reproduce"

    def test_list_validation_manifests_filters(self):
        tool, client = _register_and_get("list_validation_manifests")
        anyio.run(lambda: tool.run({"circuit_id": "crc_1"}))
        args, kwargs = client.get.call_args
        assert args[0] == "/validation-manifests"
        assert kwargs.get("circuit_id") == "crc_1"
