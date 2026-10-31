"""Every request `MiLLMClient` sends says it is an agent: `X-miLLM-Agent` (ENH-008 S1; spec §3.2 step 1, §6 risk 1).

miLLM gates an agent's model load, unload and delete, probe-limit writes, capacity measurements and operator-code
imports in its own approval queue, and it tells an agent from the operator by this header ALONE. A request without
it is the operator's and runs at once — so a single code path of this client that forgot the header would let an
agent load a model with nobody approving it. The header is therefore checked on the TRANSPORT, for every way out of
the client: the verb methods (`get`, `post`, `put`, `delete`, `full=True`), `request`, multipart `upload`, JSON
Lines (`get_jsonl`) and `raw_get`; and a caller's own `X-miLLM-Agent` cannot replace it.

Mutation controls (0xcc/reviews/enh008_controls_2026-10-28.md): delete the stamp in `request_full`, in
`_get_jsonl_full` or in `raw_get` → red here.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from src.mcp_server.millm_client import AGENT_HEADER, JSONL_CONTENT_TYPE, MiLLMClient


class _Transport:
    """Records every request's headers; answers by path so each code path parses its own shape."""

    def __init__(self):
        self.seen: list[tuple[str, str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append((request.method, request.url.path, dict(request.headers)))
        if request.url.path.endswith(".jsonl"):
            return httpx.Response(200, text='{"a": 1}\n', headers={"content-type": JSONL_CONTENT_TYPE})
        return httpx.Response(200, json={"ok": True})


def _client(identity: str | None = None) -> tuple[MiLLMClient, _Transport]:
    transport = _Transport()
    client = MiLLMClient("http://millm") if identity is None else MiLLMClient("http://millm", identity=identity)
    client._http = httpx.AsyncClient(base_url="http://millm", transport=httpx.MockTransport(transport))  # noqa: SLF001
    return client, transport


#: Every way a request leaves the client.
CALLS = {
    "get": lambda c: c.get("/api/models"),
    "get_full": lambda c: c.get("/api/models", full=True),
    "get_with_headers": lambda c: c.get("/api/models", headers={"X-miLLM-Lease": "L1"}),
    "post": lambda c: c.post("/api/models/1/load", json_body={"x": 1}),
    "post_full": lambda c: c.post("/api/models/1/load", full=True),
    "put": lambda c: c.put("/api/models/1/probe-limits", json_body={"reason": "r"}),
    "delete": lambda c: c.delete("/api/models/1"),
    "request": lambda c: c.request("GET", "/api/health"),
    "upload": lambda c: c.post("/v1/files", upload={"filename": "a.jsonl", "content": b"{}", "purpose": "batch"}),
    "get_jsonl": lambda c: c.get_jsonl("/v1/files/f/content.jsonl"),
    "get_jsonl_via_get": lambda c: c.get("/v1/files/f/content.jsonl", jsonl=True),
    "raw_get": lambda c: c.raw_get("/api/clusters/c1/export"),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_every_request_carries_the_agent_header(name):
    client, transport = _client()
    asyncio.run(CALLS[name](client))
    assert len(transport.seen) == 1, transport.seen
    _method, _path, headers = transport.seen[0]
    assert headers.get(AGENT_HEADER.lower()) == "agent:mistudio-mcp", (
        f"{name} sent no {AGENT_HEADER}: miLLM would treat this agent request as the OPERATOR's and run a gated "
        "action with nobody approving it"
    )


def test_a_callers_header_cannot_replace_it():
    client, transport = _client()
    asyncio.run(client.post("/api/models/1/load", headers={AGENT_HEADER: ""}))
    assert transport.seen[0][2][AGENT_HEADER.lower()] == "agent:mistudio-mcp"


def test_the_identity_is_the_products_and_an_empty_one_falls_back():
    from src.mcp_server.products import MILLM

    client, transport = _client(identity=MILLM.identity)
    asyncio.run(client.get("/api/models"))
    assert transport.seen[0][2][AGENT_HEADER.lower()] == MILLM.identity
    client, transport = _client(identity="")
    asyncio.run(client.get("/api/models"))
    assert transport.seen[0][2][AGENT_HEADER.lower()] == "agent:mistudio-mcp"


def test_the_server_builds_its_millm_client_with_the_header():
    """Wiring, not the class: the client `build_server` hands the millm_* modules sends the header."""
    from src.mcp_server.server import _client_for
    from src.mcp_server.products import MILLM

    client = _client_for(MILLM, "http://millm")
    transport = _Transport()
    client._http = httpx.AsyncClient(base_url="http://millm", transport=httpx.MockTransport(transport))  # noqa: SLF001
    asyncio.run(client.get("/api/models"))
    assert transport.seen[0][2][AGENT_HEADER.lower()] == MILLM.identity


def test_the_body_is_untouched_by_the_header():
    client, transport = _client()
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={})

    client._http = httpx.AsyncClient(base_url="http://millm", transport=httpx.MockTransport(handler))  # noqa: SLF001
    asyncio.run(client.put("/api/models/1/probe-limits", json_body={"reason": "r"}))
    assert seen["body"] == {"reason": "r"}
