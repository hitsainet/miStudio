"""`miserver_lease_status` without a model id answers "is ANY lease held" (MCP gap, 2026-11-10).

A drain check before a window wants one question answered — is any client holding miServer's model — and the
tool required a `model_id`. miServer keeps one lease per model and only the RESIDENT model can be leased (its
X-08; the registry self-heals a lease whose model stopped being resident), so the resident model's lease on
`GET /api/health/detailed` is every lease it holds. Through the built server and the real `MiServerClient` on a
fake miServer: exactly ONE request, to that route, and the answer says whether a lease is held. With a model id the
tool still reads that model's lease route (pinned in test_reachability.EXPECTED_CALLS).
"""
from __future__ import annotations

import asyncio

import httpx
import pytest


def _call(monkeypatch, health: dict, arguments: dict):
    from src.mcp_server import server as server_mod
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.health_gate import HealthGate
    from src.mcp_server.miserver_client import MiServerClient

    seen: list[tuple[str, str]] = []

    def fake(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/api/health/detailed":
            return httpx.Response(200, json={"success": True, "data": health, "error": None})
        return httpx.Response(200, json={"success": True, "data": {"model_id": 7, "lease": None}, "error": None})

    real = server_mod._client_for

    def capture(spec, url):
        client = real(spec, url)
        if isinstance(client, MiServerClient):
            client._http = httpx.AsyncClient(base_url=url, transport=httpx.MockTransport(fake))  # noqa: SLF001
        return client

    async def open_gate(self, product):
        return True, "ok"

    monkeypatch.setattr(server_mod, "_client_for", capture)
    monkeypatch.setattr(HealthGate, "check", open_gate)
    monkeypatch.setenv("MISERVER_API_URL", "http://miserver.test")
    mcp, _ = server_mod.build_server(MCPSettings(tool_categories="miserver_models", allow_anonymous=True,
                                                 _env_file=None), stdio=True)
    out = asyncio.run(mcp._tool_manager.call_tool("miserver_lease_status", arguments, context=None))  # noqa: SLF001
    return out, seen


LEASE = {"model_id": 7, "model_name": "llama", "holder": "agent:miworkbench-mcp", "reason": "labelling run",
         "acquired_at": "2026-11-10T22:00:00Z", "expires_at": "2026-11-10T23:00:00Z", "ttl_seconds": 3600,
         "seconds_remaining": 1200}


@pytest.mark.parametrize("lease", [LEASE, None])
def test_no_model_id_answers_whether_any_lease_is_held(monkeypatch, lease):
    out, seen = _call(monkeypatch, {"model_loaded": True, "model_name": "llama", "lease": lease}, {})
    assert seen == [("GET", "/api/health/detailed")]
    assert out["any_lease_held"] is (lease is not None)
    assert out["lease"] == lease and out["model_loaded"] is True and out["model_name"] == "llama"


def test_a_model_id_still_reads_that_models_lease(monkeypatch):
    _out, seen = _call(monkeypatch, {}, {"model_id": 7})
    assert seen == [("GET", "/api/models/7/lease")]
