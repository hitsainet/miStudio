"""G1 (2026-10-21): the MCP server goes on public names, so prove first that the app it SERVES
refuses unauthenticated requests — the assembled app, `server.build_http_app`, which `__main__`
calls, not the middleware on a toy app (`test_mcp_server_foundation.TestBearerMiddleware` does
that, and would stay green if `__main__` stopped adding it).

Then the Ingress: the three suite-style hosts route to `mistudio-mcp:8765`, and a request carrying
the Host header the public proxy actually sends (`k8s-mcp-studio.hitsai.local`) is served, not
refused by the SDK's host check.

Controls (review record `0xcc/reviews/2026-10-21_mcp-gaps.md`):
  delete `app.add_middleware(BearerAuthMiddleware, …)` in build_http_app → the 401 tests red.
"""

import json
from pathlib import Path

import anyio
import httpx
import pytest
import yaml

from src.mcp_server.config import MCPSettings
from src.mcp_server.server import build_http_app, build_server

TOKEN = "t" * 40
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
LIST = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
MANIFEST = Path(__file__).resolve().parents[3] / "k8s" / "base" / "mcp.yaml"


def _send(*requests: dict) -> list[httpx.Response]:
    settings = MCPSettings(auth_token=TOKEN, _env_file=None)
    mcp, _ = build_server(settings)
    app = build_http_app(mcp, settings)

    async def run():
        async with mcp.session_manager.run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as c:
                out = []
                for r in requests:
                    out.append(await c.request(r.get("method", "POST"), r.get("path", "/mcp"),
                                               content=r.get("body", LIST),
                                               headers={**HEADERS, **r.get("headers", {})}))
                return out

    return anyio.run(run)


def test_no_token_is_401():
    (resp,) = _send({})
    assert resp.status_code == 401
    assert "tools" not in resp.text


def test_a_wrong_token_is_401():
    (resp,) = _send({"headers": {"Authorization": "Bearer " + "x" * 40}})
    assert resp.status_code == 401


def test_a_non_bearer_scheme_is_401():
    (resp,) = _send({"headers": {"Authorization": f"Basic {TOKEN}"}})
    assert resp.status_code == 401


def test_the_right_token_is_served():
    (resp,) = _send({"headers": {"Authorization": f"Bearer {TOKEN}"}})
    assert resp.status_code == 200, resp.text[:300]
    assert resp.json()["result"]["tools"]


def test_health_is_open_and_says_nothing_secret():
    (resp,) = _send({"method": "GET", "path": "/health", "body": b""})
    assert resp.status_code == 200
    assert TOKEN not in resp.text


@pytest.mark.parametrize("host", ["k8s-mcp-studio.hitsai.local", "mcp-studio.hitsai.local",
                                  "mcp-studio.hitsai.net"])
def test_the_host_the_proxy_sends_is_served(host):
    """The public proxy rewrites Host to k8s-*.hitsai.local. An SDK host allow-list that knew
    only localhost would answer 421 to every off-LAN call while the Ingress looked right."""
    (resp,) = _send({"headers": {"Authorization": f"Bearer {TOKEN}", "Host": host}})
    assert resp.status_code == 200, (host, resp.status_code, resp.text[:200])


def test_build_http_app_refuses_to_serve_without_a_token():
    settings = MCPSettings(auth_token="", allow_anonymous=True, _env_file=None)
    mcp, _ = build_server(settings, stdio=True)
    with pytest.raises(SystemExit):
        build_http_app(mcp, settings)


def test_main_serves_the_assembled_app():
    """`__main__` must serve `build_http_app`'s app, or the tests above prove a different app."""
    import ast
    import inspect

    from src.mcp_server import __main__ as main_module

    tree = ast.parse(inspect.getsource(main_module.main))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "build_http_app" in called
    assert not any(isinstance(n, ast.Attribute) and n.attr == "streamable_http_app"
                   for n in ast.walk(tree)), "main builds its own app beside build_http_app"


# ── the Ingress ─────────────────────────────────────────────────────────────────────────────


def _ingress() -> dict:
    docs = [d for d in yaml.safe_load_all(MANIFEST.read_text()) if d]
    return next(d for d in docs if d["kind"] == "Ingress"
                and d["metadata"]["name"] == "mistudio-mcp-ingress")


@pytest.mark.parametrize("host", ["mcp-studio.hitsai.net", "mcp-studio.hitsai.local",
                                  "k8s-mcp-studio.hitsai.local"])
def test_the_suite_hosts_route_the_whole_path_space_to_the_mcp_service(host):
    rules = {r["host"]: r for r in _ingress()["spec"]["rules"]}
    assert host in rules, sorted(rules)
    (path,) = rules[host]["http"]["paths"]
    assert path["path"] == "/" and path["pathType"] == "Prefix"
    assert path["backend"]["service"] == {"name": "mistudio-mcp", "port": {"number": 8765}}


def test_the_existing_hosts_are_kept():
    hosts = {r["host"] for r in _ingress()["spec"]["rules"]}
    assert {"k8s-mistudio.hitsai.local", "mcp-mistudio.hitsai.local",
            "mcp-mistudio.hitsai.net"} <= hosts


def test_the_pod_requires_the_token_from_the_secret():
    docs = [d for d in yaml.safe_load_all(MANIFEST.read_text()) if d]
    deploy = next(d for d in docs if d["kind"] == "Deployment")
    env = {e["name"]: e for e in deploy["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["MCP_AUTH_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "mistudio-secrets", "key": "mcp-auth-token"}
    assert "MCP_ALLOW_ANONYMOUS" not in env
