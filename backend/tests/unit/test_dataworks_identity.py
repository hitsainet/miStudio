"""The identity this MCP server presents to miDataworks is `agent:miworkbench-mcp` (042 FR-042.16; FTASKS 12.7).

Through the LIVE registry: the real `build_server()` builds the miDataworks client from the product row
(`_client_for`), a tool call goes out over a fake miDataworks socket, and the one request the fake records carries
`X-Dataworks-Agent: agent:miworkbench-mcp` (the stored, older `agent:mistudio-mcp` is history, DR-17).
"""
from __future__ import annotations

import asyncio

import httpx


def test_dataworks_fake_records_miworkbench_identity(monkeypatch):
    from src.mcp_server import server as server_mod
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.dataworks_client import DataworksClient
    from src.mcp_server.products import DATAWORKS

    seen: list[httpx.Request] = []

    def fake(request: httpx.Request) -> httpx.Response:
        if request.url.path == DATAWORKS.health_path:
            return httpx.Response(200, json={"status": "ok"})
        seen.append(request)
        return httpx.Response(200, json={"items": [], "total": 0})

    real = server_mod._client_for
    built = []

    def capture(spec, url):
        client = real(spec, url)
        if isinstance(client, DataworksClient):
            client._http = httpx.AsyncClient(base_url=url, transport=httpx.MockTransport(fake))  # noqa: SLF001
            built.append(client)
        return client

    monkeypatch.setattr(server_mod, "_client_for", capture)
    from src.mcp_server.health_gate import HealthGate

    async def open_gate(self, product):  # the gate's own probe client is not the one under test
        return True, "ok"

    monkeypatch.setattr(HealthGate, "check", open_gate)
    monkeypatch.setenv(DATAWORKS.url_env, "http://dataworks.test")
    mcp, _ = server_mod.build_server(MCPSettings(tool_categories="read,dataworks", allow_anonymous=True,
                                                 _env_file=None), stdio=True)
    assert len(built) == 1, "the live build made no (or more than one) miDataworks client"

    asyncio.run(mcp._tool_manager.call_tool("dataworks_list_datasets", {}, context=None))  # noqa: SLF001
    assert len(seen) == 1, [r.url.path for r in seen]
    assert seen[0].headers["X-Dataworks-Agent"] == "agent:miworkbench-mcp" == DATAWORKS.identity
