"""`millm_get_environment` / `millm_get_environment_history` (ENH-003 D5 for miLLM, 2026-10-21).

miLLM hosts no MCP server, so the read of its runtime settings is a `millm_*` tool here. The one
write is `millm_set_telemetry` (telemetry keys only, gated in miLLM; test_millm_set_telemetry.py);
PUT/DELETE /api/environment/{key} carry written exemptions in `test_mcp_millm_route_parity.py`,
which (with the miLLM sibling present) proves the routes are served and covered.

Driven through the LIVE registry of the built server and the REAL MiLLMClient (envelope unwrap) over
`httpx.MockTransport`.

Control (review record `0xcc/reviews/2026-10-21_mcp-gaps.md`): delete the `millm_get_environment`
registration (its `@mcp.tool()` line) → test_the_tools_are_on_the_built_server red.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx

from tests.support.external_ctx import make_ctx


def test_the_tools_are_on_the_built_server(monkeypatch):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
    mcp, _ = build_server(MCPSettings(tool_categories="millm_runtime", allow_anonymous=True),
                          stdio=True)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"millm_get_environment", "millm_get_environment_history"} <= names


def test_absent_without_millm(monkeypatch):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    monkeypatch.delenv("MILLM_API_URL", raising=False)
    mcp, _ = build_server(MCPSettings(tool_categories="read,millm_runtime", allow_anonymous=True),
                          stdio=True)
    assert "millm_get_environment" not in {t.name for t in asyncio.run(mcp.list_tools())}


def _tools(handler, gate_open=True):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.millm_client import MiLLMClient
    from src.mcp_server.tools import millm_runtime

    client = MiLLMClient("http://millm.test")
    client._http = httpx.AsyncClient(base_url="http://millm.test",  # noqa: SLF001
                                     transport=httpx.MockTransport(handler))
    gate = MagicMock()
    gate.check = AsyncMock(return_value=(gate_open, None if gate_open else "down"))
    gate.invalidate = MagicMock()
    mcp = MCPServer("t")
    millm_runtime.register(mcp, client, gate, make_ctx())
    return mcp._tool_manager  # noqa: SLF001


def _recording():
    def handler(request):
        handler.seen.append((request.method, request.url.path, dict(request.url.params),
                             request.content))
        return httpx.Response(200, json={"success": True, "data": {"settings": {"k": 1}}})
    handler.seen = []
    return handler


def test_read_issues_one_GET_and_unwraps_the_envelope():
    handler = _recording()
    out = asyncio.run(_tools(handler).call_tool("millm_get_environment", {}, context=None))
    assert handler.seen == [("GET", "/api/environment", {}, b"")]
    assert out == {"settings": {"k": 1}}


def test_history_sends_its_limit():
    handler = _recording()
    asyncio.run(_tools(handler).call_tool("millm_get_environment_history", {"limit": 7}, context=None))
    assert handler.seen == [("GET", "/api/environment/history", {"limit": "7"}, b"")]


def test_a_closed_gate_is_structured_unavailable_and_sends_nothing():
    handler = _recording()
    out = asyncio.run(_tools(handler, gate_open=False).call_tool("millm_get_environment", {}, context=None))
    assert handler.seen == []
    assert json.loads(json.dumps(out))["unavailable"] == "millm"


def test_the_only_millm_setting_write_is_the_telemetry_save():
    """D5 as amended by ENH-006 D75-D76 (miLLM contract v1.23): the one write route a registered
    tool calls is `PUT /api/environment`, and only `millm_set_telemetry` calls it. A tool calling
    PUT/DELETE /api/environment/{key} would contradict the parity exemptions' reason, and a
    second caller of the bulk save would not carry the telemetry-only key check."""
    import pytest

    from tests.support.mcp_route_map import called_routes, registered_tools

    mp = pytest.MonkeyPatch()
    try:
        covered = called_routes(registered_tools(mp), "millm")
    finally:
        mp.undo()
    writes = {k: sorted(v) for k, v in covered.items()
              if k[1].startswith("/api/environment") and k[0] != "GET"}
    assert writes == {("PUT", "/api/environment"): ["millm_set_telemetry"]}, writes
    assert ("GET", "/api/environment") in covered
