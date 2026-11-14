"""miServer's `error.next` reaches the agent (042 FR-042.38 P0 half; FTDD §5.5, §17 C2; TPD-6; FTASKS 4.3).

miServer answers a refusal with an envelope whose `error.next` names the tool to call next (miLLM
FR-39.30). `MiServerClient` built `{code, message, details}` from it and DROPPED `next`, in both the envelope
unwrap and the export path. Driven here through the REAL tools of the server `build_server` assembles,
over the in-process MCP client, so the assertion is what an agent reads; miServer is an `httpx.MockTransport`.

Controls K-4.11 (the envelope path's `next` key deleted), K-4.12 (the export path's).
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from src.mcp_server import health_gate, miserver_client
from src.mcp_server.config import MCPSettings
from src.mcp_server.server import build_server

ENVELOPE = {"success": False, "data": None,
            "error": {"code": "MODEL_NOT_LOADED", "message": "No model is loaded.",
                      "details": {"model": None}, "next": "miserver_load_model"}}


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/health"):
        return httpx.Response(200, json={"status": "healthy"})
    return httpx.Response(409, json=ENVELOPE)


@pytest.fixture
def mcp(monkeypatch):
    real = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(_handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(miserver_client.httpx, "AsyncClient", patched)
    monkeypatch.setattr(health_gate.httpx, "AsyncClient", patched)
    monkeypatch.setenv("MISERVER_API_URL", "http://miserver.test")
    server, _ = build_server(MCPSettings(tool_categories="miserver_runtime,miserver_clusters", allow_anonymous=True,
                                         _env_file=None), stdio=True)
    return server


def _call(server, name, args):
    from mcp import Client

    async def run():
        async with Client(server) as client:
            return await client.call_tool(name, args)

    return asyncio.run(run())


def _detail(result) -> dict:
    text = " ".join(getattr(b, "text", "") for b in result.content)
    assert "backend 409:" in text, text
    return json.loads(text.split("backend 409:", 1)[1].strip())


def test_envelope_next_reaches_tool_result(mcp):
    detail = _detail(_call(mcp, "miserver_deactivate_cluster", {"cluster_id": "c1"}))
    assert detail == {"code": "MODEL_NOT_LOADED", "message": "No model is loaded.", "details": {"model": None},
                      "next": "miserver_load_model"}


def test_export_path_next_reaches_tool_result(mcp):
    detail = _detail(_call(mcp, "miserver_export_cluster", {"cluster_id": "c1"}))
    assert detail["next"] == "miserver_load_model" and detail["code"] == "MODEL_NOT_LOADED"


def test_error_fields_keeps_next():
    fields = miserver_client.error_fields(409, {"code": "X", "message": "m", "details": None, "next": "millm_x"})
    assert fields == {"code": "X", "message": "m", "details": None, "next": "millm_x"}
    assert "next" not in miserver_client.error_fields(409, {"code": "X", "message": "m"})
