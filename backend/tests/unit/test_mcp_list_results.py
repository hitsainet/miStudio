"""L5 + F-6 (2026-10-21): a list tool's answer, end to end over the MCP HTTP transport.

L5: `list_probe_training_runs(status="running")` with nothing running used to arrive as ZERO
content blocks (`content: []`) — indistinguishable from a dropped response. F-6: a list arrived as
one block per item for the client to stitch.

The fix lives in ONE place, `server.normalise_list_results`, applied by `build_server`, so every
list tool gets it: `content` keeps one block per item (compatible), an empty list is one block
`[]`, and the whole list is ALSO `structuredContent.result`.

Driven over the real chain: JSON-RPC `tools/call` → the assembled HTTP app (bearer auth) → the
built server → the tool → the REAL miStudio route on the test database (ASGI transport).

Controls (review record `0xcc/reviews/2026-10-21_mcp-gaps.md`, restored from saved bytes):
  C-WRAP   delete `normalise_list_results(mcp)` in build_server → every test here red
  C-EMPTY  make an empty list produce `[]` content again → test_an_empty_filter_… red
"""
from __future__ import annotations

import json

import httpx
import pytest

TOKEN = "k" * 40


async def _call(name: str, arguments: dict, backend_app) -> dict:
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_http_app, build_server

    settings = MCPSettings(auth_token=TOKEN, tool_categories="probe_monitors,jobs",
                           _env_file=None)
    mcp, client = build_server(settings)
    client._http = httpx.AsyncClient(transport=httpx.ASGITransport(app=backend_app),  # noqa: SLF001
                                     base_url="http://backend/api/v1")
    app = build_http_app(mcp, settings)
    body = {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": name, "arguments": arguments}}
    async with mcp.session_manager.run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://localhost") as c:
            resp = await c.post("/mcp", content=json.dumps(body).encode(), headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200, resp.text[:400]
    return resp.json()["result"]


async def test_an_empty_filter_is_ONE_block_and_an_empty_structured_list(client):
    from src.main import app

    result = await _call("list_probe_training_runs", {"status": "running"}, app)
    assert result.get("isError") is not True, result
    assert result["content"] == [{"type": "text", "text": "[]"}]
    assert result["structuredContent"] == {"result": []}


async def test_a_non_empty_list_keeps_one_block_per_item_AND_one_array(client, async_session):
    from src.main import app
    from tests.unit.test_operations_api import _seed

    ids = await _seed(async_session, run_status="running")
    result = await _call("list_probe_training_runs", {"status": "running"}, app)
    blocks = [json.loads(b["text"]) for b in result["content"]]
    assert [b["id"] for b in blocks] == [ids["run"]], "one block per item, as before"
    array = result["structuredContent"]["result"]
    assert isinstance(array, list) and [r["id"] for r in array] == [ids["run"]]
    assert array == blocks


async def test_a_dict_answer_is_untouched(client):
    """Only lists change: a tool answering an object keeps MCPServer's single JSON block."""
    from src.main import app

    result = await _call("list_active_jobs", {}, app)
    (block,) = result["content"]
    assert "jobs" in json.loads(block["text"])
    assert "structuredContent" not in result or result["structuredContent"] is None


def test_every_list_capable_tool_of_the_built_server_is_wrapped(monkeypatch):
    """The wrapper reaches EVERY tool without an output schema, on the server production builds."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    for spec in PRODUCTS:
        monkeypatch.setenv(spec.url_env, f"http://{spec.name}.test")
    mcp, _ = build_server(MCPSettings(tool_categories=",".join(all_categories()),
                                      allow_anonymous=True), stdio=True)
    tools = mcp._tool_manager.list_tools()  # noqa: SLF001
    unwrapped = [t.name for t in tools if t.fn_metadata.output_schema is None
                 and not getattr(t.fn, "normalises_list_results", False)]
    assert len(tools) > 100 and not unwrapped, unwrapped


@pytest.mark.parametrize("items", [[], [1, {"a": 2}], ("x",)])
def test_list_result_shape(items):
    from src.mcp_server.server import list_result

    out = list_result(items)
    assert out.structured_content == {"result": list(items)}
    assert len(out.content) == max(1, len(items))
