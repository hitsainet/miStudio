"""Every Feature 041 tool reaches the LIVE MCP registry of a server built the way the pod builds it
(FR-25; FTASKS 2.5, 9.4, 12.7, 13.5, 14.2).

The server is built with only the `probe_monitors` category on — the category `k8s/base/mcp.yaml` enables —
and its `list_tools()` is read: the set an agent sees, not the set the source declares. The expected set
is written here per slice (a tool dropped from the module, or the module dropped from
`CATEGORY_MODULES`, turns this red), and the module's declared tools must equal it.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

#: The 041 tools, by slice. S1: the family read.
EXPECTED_041_TOOLS = {
    "get_probe_family",
    # S2: blend reports and the suggested blend
    "evaluate_probe_blends", "get_blend_report", "list_blend_reports", "get_blend_report_scores",
    "rerun_blend_report", "cancel_blend_evaluation", "set_suggested_blend", "get_suggested_blend",
    # S3: family scoring, export and publish
    "score_probe_family", "score_probe_family_batch", "get_probe_family_score_result", "export_probe_family",
    "publish_probe_family",
}


def test_the_041_surface_is_fourteen_tools():
    assert len(EXPECTED_041_TOOLS) == 14

REPO = Path(__file__).resolve().parents[3]


def _live(category: str = "probe_monitors"):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mcp, _client = build_server(MCPSettings(tool_categories=category, allow_anonymous=True), stdio=True)
    return {tool.name: tool for tool in asyncio.run(mcp.list_tools())}


def test_every_041_tool_is_in_the_live_registry():
    live = _live()
    missing = sorted(EXPECTED_041_TOOLS - set(live))
    assert not missing, f"041 tools absent from the built server: {missing}"


def test_the_module_declares_exactly_the_expected_tools():
    from src.mcp_server.tools import probe_families
    from tests.unit.test_reachability import _tool_names_in

    assert _tool_names_in(probe_families) == EXPECTED_041_TOOLS


def test_the_family_tool_takes_the_routes_parameters():
    schema = getattr(_live()["get_probe_family"], "input_schema", None) or _live()["get_probe_family"].inputSchema
    assert set(schema["properties"]) == {"key", "model", "member_ids", "include_retired", "offset", "limit"}
    assert schema["required"] == ["key"]


def test_the_category_is_deployed():
    """The tools ride an existing category; the deployed manifest must enable it (miStudio memory
    `mcp-category-needs-deployment-enabling`)."""
    text = (REPO / "k8s" / "base" / "mcp.yaml").read_text()
    after = text.split("MCP_TOOL_CATEGORIES", 1)[1]
    value = next(line for line in after.splitlines() if line.strip().startswith("value:"))
    assert "probe_monitors" in value.split(":", 1)[1].split(",") or "probe_monitors" in value
