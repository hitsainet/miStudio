"""The MCP HTTP transport's own limits, measured through the real ASGI app (mcp 1.30, 2026-10-17).

Two properties, both read off the app `__main__.py` serves, never off settings alone:

* **The request-body bound admits what the tools admit, and is still a bound.** mcp 1.29 added a
  Streamable HTTP request-body limit, default 4 MiB, answered 413 before any tool runs.
  `millm_upload_batch_file` documents and enforces its own cap of `MAX_BATCH_BYTES` (16 MiB) of
  rows, which arrive as the tool call's JSON arguments - so on the SDK default every upload over
  4 MiB would be refused by the transport while the tool's description promised 16. The server
  now sets the limit from that cap; this file proves both a body above the old default gets
  through and a body above the new limit does not (the bound is the point of the SDK change).
* **The server keeps no sessions.** GHSA-84m7-p3x7-pcfv (unbounded session accumulation, fixed
  in mcp 1.30.0) does not affect a server built with `stateless_http=True`. That flag is the
  reason miStudio was never exposed; a response carrying an `mcp-session-id` would mean it was
  dropped.
"""

import json

import anyio
import httpx

from src.mcp_server.config import MCPSettings
from src.mcp_server.server import MCP_MAX_REQUEST_BODY_BYTES, build_server
from src.mcp_server.tools.millm_batches import MAX_BATCH_BYTES

SDK_DEFAULT_BODY_BYTES = 4 * 1024 * 1024
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _body(pad_bytes: int) -> bytes:
    """A `tools/list` request padded to roughly `pad_bytes` through `_meta` (extras allowed)."""
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {"_meta": {"pad": "x" * pad_bytes}},
        }
    ).encode()


def _post(*bodies: bytes) -> list[httpx.Response]:
    mcp, _ = build_server(MCPSettings(auth_token="secret-token", _env_file=None))
    app = mcp.streamable_http_app()

    async def run() -> list[httpx.Response]:
        # ASGITransport does not run the lifespan; the session manager must be running.
        async with mcp.session_manager.run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as c:
                return [await c.post("/mcp", content=b, headers=HEADERS) for b in bodies]

    return anyio.run(run)


def test_the_limit_covers_the_batch_tools_cap_with_headroom_and_stays_finite():
    assert MCP_MAX_REQUEST_BODY_BYTES >= MAX_BATCH_BYTES + 1024 * 1024
    assert MCP_MAX_REQUEST_BODY_BYTES <= 64 * 1024 * 1024


def test_a_body_over_the_sdk_default_reaches_the_server():
    (small, large) = _post(_body(1024), _body(SDK_DEFAULT_BODY_BYTES + 1024 * 1024))
    assert small.status_code == 200, small.text[:300]
    assert large.status_code == 200, large.text[:300]
    assert "tools" in large.json()["result"]


def test_a_body_at_the_batch_tools_cap_reaches_the_server():
    (resp,) = _post(_body(MAX_BATCH_BYTES))
    assert resp.status_code == 200, resp.text[:300]


def test_a_body_over_the_limit_is_still_refused():
    (resp,) = _post(_body(MCP_MAX_REQUEST_BODY_BYTES + 1024))
    assert resp.status_code == 413


def test_the_server_issues_no_session_ids():
    # `initialize` is the request a stateful server answers with a new `mcp-session-id`.
    initialize = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "transport-limits-test", "version": "0"},
            },
        }
    ).encode()
    for resp in _post(initialize, initialize):
        assert resp.status_code == 200, resp.text[:300]
        assert resp.json()["result"]["serverInfo"]["name"] == "mistudio"
        assert "mcp-session-id" not in {k.lower() for k in resp.headers}


def test_the_mcp_ingress_admits_what_the_server_admits():
    """`k8s/base/mcp.yaml` read "10m" while the batch tool admits 16 MiB, so the edge refused
    uploads the server and the tool description accepted. The edge must be at least as wide."""
    from pathlib import Path

    import yaml

    manifest = Path(__file__).resolve().parents[3] / "k8s" / "base" / "mcp.yaml"
    ingresses = [
        doc
        for doc in yaml.safe_load_all(manifest.read_text())
        if doc and doc.get("kind") == "Ingress"
        and doc["metadata"]["name"] == "mistudio-mcp-ingress"
    ]
    assert len(ingresses) == 1
    size = ingresses[0]["metadata"]["annotations"]["nginx.ingress.kubernetes.io/proxy-body-size"]
    assert size.endswith("m"), size
    assert int(size[:-1]) * 1024 * 1024 >= MCP_MAX_REQUEST_BODY_BYTES
