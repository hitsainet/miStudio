"""The MCP pod's old-name evidence reaches the backend (042 FR-042.9; FTDD §5.2, §5.3; CF-5 M6; FTASKS 3.12,
3.13).

The route is the REAL backend app (`POST /api/v1/system/legacy-names`). The push is driven through the REAL
MCP server's `/health` (the only thing that schedules it in production), with the server's own backend
client; time moves through `legacy_names_push._clock`. Payload AND call count are asserted.

Controls: K-3.32 (the scheduling line in `/health` deleted), K-3.33 (throttle 600 → 0), K-3.34 (no push
when there is nothing to report), K-3.35 (`pending` cleared on failure).
"""
from __future__ import annotations

import json

import httpx
import pytest
from sqlalchemy import text

from src.core import legacy_names
from src.mcp_server import legacy_names_push

TOKEN = "t" * 40
PATH = "/api/v1/system/legacy-names"
NAMES = sorted({n for r in legacy_names.ENV for n in (r.new, r.old)})


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    legacy_names.reset()
    yield
    legacy_names.reset()


@pytest.fixture
def tables(async_engine):
    from src.services import legacy_names_store

    legacy_names_store.dispose()
    with legacy_names_store.session_factory()() as db:
        db.execute(text("DELETE FROM legacy_name_uses"))
        db.execute(text("DELETE FROM legacy_name_reporters"))
        db.commit()
    yield legacy_names_store
    legacy_names_store.dispose()


async def _post(body) -> httpx.Response:
    from src.main import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as c:
        return await c.post(PATH, json=body)


def _use(kind="host", old="k8s-mistudio.hitsai.local", new="k8s-miworkbench.hitsai.local",
         last_seen="2026-11-08T10:00:00Z"):
    return {"kind": kind, "old": old, "new": new, "last_seen": last_seen}


# ── the route (3.12) ────────────────────────────────────────────────────────────────────────────────────
async def test_post_refuses_value_shaped_names_unknown_kind_and_over_64():
    cases = {
        "value-shaped": {"reporter": "mcp", "uses": [_use(kind="env", old="http://secret/x", new="MISERVER_API_URL")]},
        "space": {"reporter": "mcp", "uses": [_use(old="has space")]},
        "kind": {"reporter": "mcp", "uses": [_use(kind="header")]},
        "over-64": {"reporter": "mcp", "uses": [_use()] * 65},
        "reporter": {"reporter": "MCP!", "uses": []},
        "extra-field": {"reporter": "mcp", "uses": [], "process": "x"},
        "entry-extra": {"reporter": "mcp", "uses": [{**_use(), "process": "x"}]},
    }
    for name, body in cases.items():
        r = await _post(body)
        assert r.status_code == 422, (name, r.status_code, r.text)
    # A name the tables do not list is refused, so the table stays bounded (23 rows at most).
    r = await _post({"reporter": "mcp", "uses": [_use(kind="env", old="_".join(("MILLM", "SOMETHING", "ELSE")), new="MISERVER_X")]})
    assert r.status_code == 422 and "unknown_legacy_name" in r.text, r.text


async def test_post_store_failure_is_503_with_a_generic_body(monkeypatch):
    """A store that fails answers 503 with exactly `{"code": "legacy_names_store_unavailable"}`: the sender
    keeps its uses and retries. The exception's text and type never reach the caller (an unauthenticated
    route), and the failed report's also_set/conflicts are not noted."""
    from src.services import legacy_names_report, legacy_names_store

    planted = "PLANTED-SECRET /data/private/dsn"

    def boom(report, now):
        raise RuntimeError(planted)

    noted = []
    monkeypatch.setattr(legacy_names_store, "write", boom)
    monkeypatch.setattr(legacy_names_report, "note_remote_pairs", lambda *a: noted.append(a))
    r = await _post({"reporter": "mcp", "uses": [_use()]})
    assert r.status_code == 503, r.text
    assert r.json() == {"detail": {"code": "legacy_names_store_unavailable"}}
    assert planted not in r.text and "RuntimeError" not in r.text
    assert noted == []


async def test_post_empty_uses_moves_only_the_reporter(tables):
    r = await _post({"reporter": "mcp", "uses": []})
    assert r.status_code == 204, r.text
    with tables.session_factory()() as db:
        assert db.execute(text("SELECT count(*) FROM legacy_name_uses")).scalar() == 0
        assert db.execute(text("SELECT reporter FROM legacy_name_reporters")).scalars().all() == ["mcp"]


async def test_post_writes_uses_and_clamps_a_future_time(tables):
    r = await _post({"reporter": "mcp", "uses": [_use(last_seen="2099-01-01T00:00:00Z")]})
    assert r.status_code == 204, r.text
    with tables.session_factory()() as db:
        row = db.execute(text("SELECT kind, old, last_seen < now() + interval '1 minute' FROM legacy_name_uses")).one()
    assert row == ("host", "k8s-mistudio.hitsai.local", True), "a forged future time must be clamped"


# ── the push, through the MCP /health (3.13) ─────────────────────────────────────────────────────────────
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _mcp(monkeypatch, respond):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_http_app, build_server

    clock = Clock()
    monkeypatch.setattr(legacy_names_push, "_clock", clock)
    settings = MCPSettings(auth_token=TOKEN, _env_file=None)
    mcp, client = build_server(settings)
    posts: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == PATH:
            posts.append(json.loads(request.content))
        return await respond(request)

    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler),  # noqa: SLF001
                                     base_url="http://backend/api/v1")
    return mcp, build_http_app(mcp, settings), clock, posts


async def _health(mcp, app) -> dict:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as c:
        body = (await c.get("/health")).json()
    task = mcp.legacy_names_push.task
    if task is not None:
        await task
    return body


async def _ok(_request):
    return httpx.Response(204)


async def test_one_push_per_ten_minutes_with_uses(monkeypatch):
    mcp, app, clock, posts = _mcp(monkeypatch, _ok)
    legacy_names.record("host", "mcp-studio.hitsai.local", "mcp-workbench.hitsai.local")
    await _health(mcp, app)
    clock.t += 100
    await _health(mcp, app)
    clock.t += 100
    await _health(mcp, app)
    assert len(posts) == 1, posts
    assert posts[0]["reporter"] == "mcp"
    assert [(u["kind"], u["old"], u["new"]) for u in posts[0]["uses"]] == [
        ("host", "mcp-studio.hitsai.local", "mcp-workbench.hitsai.local")]
    clock.t += 401  # 601 s after the first attempt
    body = await _health(mcp, app)
    assert len(posts) == 2
    assert body["legacy_names_meta"]["push"]["pending"] == 0
    assert body["legacy_names_meta"]["push"]["last_error"] is None


async def test_heartbeat_push_with_no_uses_moves_reporters_mcp(monkeypatch, tables):
    """An empty push still goes, and the REAL backend route moves `reporters.mcp` on its health."""
    from src.main import app as backend

    async def to_backend(request: httpx.Request) -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=backend), base_url="http://backend") as c:
            r = await c.request(request.method, request.url.path, content=request.content,
                                headers={"content-type": "application/json"})
            return httpx.Response(r.status_code, content=r.content, headers=r.headers)

    mcp, app, _clock, posts = _mcp(monkeypatch, to_backend)
    await _health(mcp, app)
    assert posts == [{"reporter": "mcp", "uses": [], "also_set": [], "conflicts": []}]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=backend), base_url="http://testserver") as c:
        meta = (await c.get("/api/health")).json()["legacy_names_meta"]
    assert "mcp" in meta["reporters"], meta


async def test_failed_push_keeps_uses_pending_and_redelivers(monkeypatch):
    answers = [httpx.Response(503, json={"detail": "down"}), httpx.Response(204)]

    async def flaky(_request):
        return answers.pop(0)

    mcp, app, clock, posts = _mcp(monkeypatch, flaky)
    legacy_names.record("host", "k8s-mcp-studio.hitsai.local", "k8s-miworkbench.hitsai.local")
    await _health(mcp, app)
    state = mcp.legacy_names_push.as_dict()
    assert state["pending"] == 1 and state["last_error"] == "BackendError" and state["last_ok"] is None
    clock.t += 601
    body = await _health(mcp, app)
    # The answer shows the state BEFORE this push ran (the route never waits for it): still pending.
    assert body["legacy_names_meta"]["push"]["pending"] == 1
    assert len(posts) == 2
    assert posts[1]["uses"] == posts[0]["uses"] and posts[1]["uses"][0]["old"] == "k8s-mcp-studio.hitsai.local"
    push = mcp.legacy_names_push.as_dict()
    assert push["pending"] == 0 and push["last_error"] is None and push["last_ok"]


@pytest.mark.parametrize("reporter", ["api", "worker", "beat", "mcp:x", "evil"])
async def test_post_refuses_every_reporter_but_the_mcp_pod(tables, reporter):
    """Review rev-mistudio-p0 F5: the route stores nothing, and notes no pairs, for any reporter name but the
    MCP pod's. Heartbeats under the API's or a worker's name would otherwise make their window read measured."""
    from src.services import legacy_names_report

    before = dict(legacy_names_report._remote_pairs)
    pair = {"old": "MILLM_API_URL", "new": "MISERVER_API_URL"}
    r = await _post({"reporter": reporter, "uses": [_use()], "also_set": [pair], "conflicts": []})
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == {"code": "unknown_reporter"}
    with tables.session_factory()() as db:
        assert db.execute(text("SELECT count(*) FROM legacy_name_reporters")).scalar_one() == 0
        assert db.execute(text("SELECT count(*) FROM legacy_name_uses")).scalar_one() == 0
    assert legacy_names_report._remote_pairs == before

