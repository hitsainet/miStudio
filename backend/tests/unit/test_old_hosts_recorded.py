"""A request on an OLD host name is recorded, by the real apps (042 FR-042.9; FTID §3.5; TI-9; FTASKS 3.15).

`legacy_names.record` is wrapped to count calls, so each test asserts the payload AND the count: one
record per request on an old host, none on a new one. The MCP case sends NO token: the bearer middleware
refuses it (401) and it must still count, because a request that arrived is evidence of use.

Controls: K-3.36 (the middleware line removed from `main.py`), K-3.37 (removed from the MCP HTTP app).
"""
from __future__ import annotations

import anyio
import httpx
import pytest

from src.core import legacy_names

TOKEN = "t" * 40


@pytest.fixture
def calls(monkeypatch):
    # Only host uses are asserted here: an old env name in the environment (a developer shell, or a test that
    # leaked one) would add an env use, so every ENV name is removed for the test.
    for row in legacy_names.ENV:
        monkeypatch.delenv(row.old, raising=False)
        monkeypatch.delenv(row.new, raising=False)
    legacy_names.reset()
    seen: list[tuple[str, str, str]] = []
    real = legacy_names.record

    def spy(kind, old, new):
        seen.append((kind, old, new))
        real(kind, old, new)

    monkeypatch.setattr(legacy_names, "record", spy)
    yield seen
    legacy_names.reset()


def _get(app, host: str, path: str, method: str = "GET", headers=None) -> httpx.Response:
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://{host}") as c:
            return await c.request(method, path, headers=headers or {})

    return anyio.run(run)


def test_api_records_old_host(calls):
    from src.main import app

    r = _get(app, "K8S-MISTUDIO.hitsai.local:443", "/api/v1/system/health")
    assert r.status_code == 200
    assert calls == [("host", "k8s-mistudio.hitsai.local", "k8s-miworkbench.hitsai.local")]
    assert [(u["kind"], u["old"]) for u in legacy_names.snapshot()["in_use"]] == [
        ("host", "k8s-mistudio.hitsai.local")]


def test_mcp_app_records_old_host_even_when_refused(calls):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_http_app, build_server

    settings = MCPSettings(auth_token=TOKEN, _env_file=None)
    mcp, _ = build_server(settings)
    app = build_http_app(mcp, settings)
    r = _get(app, "mcp-studio.hitsai.local", "/mcp", method="POST",
             headers={"Content-Type": "application/json"})
    assert r.status_code == 401, "no token: refused by the bearer middleware"
    assert calls == [("host", "mcp-studio.hitsai.local", "mcp-workbench.hitsai.local")]


def test_new_host_records_nothing(calls):
    from src.main import app

    for host in ("k8s-miworkbench.hitsai.local", "testserver", "10.0.0.1:8000"):
        assert _get(app, host, "/api/v1/system/health").status_code == 200
    assert calls == []
    assert legacy_names.snapshot()["in_use"] == []
