"""`get_environment_settings` is reachable by agents and reads the real route (Feature 036; D5; FTASKS 5.1).

* In the LIVE registry with the DEFAULT categories (what `k8s/base/mcp.yaml` deploys), not merely
  importable — memory: mcp-category-needs-deployment-enabling.
* Driven: the tool, called through the built server, issues exactly one `GET /environment` and returns
  the body unchanged (payload and call count).
* No write tool exists (D5): no registered tool calls `PUT`/`DELETE /environment/{key}`.

Control C-MCP: remove `environment` from `CATEGORY_MODULES["read"]` → the first two tests are red.
"""
from __future__ import annotations

import asyncio

from src.mcp_server.config import DEFAULT_CATEGORIES, MCPSettings
from src.mcp_server.server import build_server
from tests.support.mcp_route_map import called_routes, registered_tools


def _server():
    settings = MCPSettings(tool_categories=DEFAULT_CATEGORIES, allow_anonymous=True)
    return build_server(settings, stdio=True)


def test_the_tool_is_registered_with_the_default_categories():
    mcp, _client = _server()
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "get_environment_settings" in names


def test_the_tool_issues_one_get_and_returns_the_listing(monkeypatch):
    mcp, client = _server()
    calls = []
    listing = {"settings": {"log_level": {"value": "INFO"}}, "count": 1, "runtime_keys": [], "overrides": {}}

    async def fake_get(path, **params):
        calls.append((path, params))
        return listing

    monkeypatch.setattr(client, "get", fake_get)
    result = asyncio.run(mcp.call_tool("get_environment_settings", {}))
    assert calls == [("/environment", {})]
    text = str(result)
    assert "log_level" in text and "INFO" in text


def test_no_registered_tool_writes_the_environment(monkeypatch):
    routes = called_routes(registered_tools(monkeypatch), "mistudio", prefix="/api/v1")
    writes = {k for k in routes if k[1].startswith("/api/v1/environment") and k[0] != "GET"}
    assert writes == set(), f"D5: no MCP write path in v1, found {writes}"
