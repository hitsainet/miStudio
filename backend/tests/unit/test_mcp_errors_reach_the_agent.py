"""An anticipated tool failure reaches the agent WITH its message (mcp 2.x, 2026-10-22).

mcp 1.x passed every exception's text to the client as "Error executing tool <name>: <text>".
mcp 2.x passes the text only for `ToolError` (and `ResourceError`); any other exception is a
crash, and the client reads just "Error executing tool <name>". Two messages here are the
instruction an agent acts on, so each must stay a `ToolError`:

* `BackendError` — the backend's 4xx/5xx detail, passed on verbatim by promise (BR-6.2): a
  protected-label 409, a labeling 503, the steering circuit breaker.
* the steering concurrency guardrail — "poll get_steering_result or cancel one".

Both are driven through the in-process MCP client over JSON-RPC (`mcp.Client`), against the
server `build_server` assembles, so the assertion is what an agent reads, not what a function
raised.
"""

from __future__ import annotations

import asyncio

import httpx

from src.mcp_server.config import MCPSettings
from src.mcp_server.server import build_server

DETAIL = "Label is protected (source=user); pass force=true to overwrite"


def _server(**overrides):
    settings = MCPSettings(
        tool_categories="models,steering", allow_anonymous=True, _env_file=None, **overrides
    )
    return build_server(settings, stdio=True)


def _call(mcp, name: str, args: dict):
    from mcp import Client

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(name, args)

    return asyncio.run(run())


def _text(result) -> str:
    return " ".join(getattr(block, "text", "") for block in result.content)


def test_a_backend_error_detail_reaches_the_agent():
    mcp, client = _server()
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(409, json={"detail": DETAIL})

    client._http = httpx.AsyncClient(  # noqa: SLF001 — the tools hold this client
        transport=httpx.MockTransport(respond), base_url="http://test/api/v1"
    )
    result = _call(mcp, "get_model", {"model_id": "m_00000001"})

    assert calls == [("GET", "/api/v1/models/m_00000001")], _text(result)
    assert result.is_error
    assert "backend 409" in _text(result) and DETAIL in _text(result), _text(result)


def test_the_steering_guardrail_message_reaches_the_agent(monkeypatch):
    from src.mcp_server.tools import steering

    mcp, client = _server(steering_max_concurrent=1)
    monkeypatch.setattr(steering, "_inflight", {"task_already_running"})
    posted = []

    def respond(request: httpx.Request) -> httpx.Response:
        posted.append(request.url.path)
        return httpx.Response(200, json={"task_id": "never"})

    client._http = httpx.AsyncClient(  # noqa: SLF001
        transport=httpx.MockTransport(respond), base_url="http://test/api/v1"
    )
    result = _call(
        mcp,
        "steer_compare",
        {"sae_id": "sae_x", "prompt": "p", "features": [{"feature_idx": 1, "layer": 2, "strength": 10.0}]},
    )

    assert posted == [], "the guardrail must refuse before anything is sent"
    assert result.is_error
    assert "Guardrail" in _text(result) and "get_steering_result" in _text(result), _text(result)
