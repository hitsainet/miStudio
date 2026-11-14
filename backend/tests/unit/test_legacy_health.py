"""The old-name evidence on both health endpoints (042 FR-042.9; FTDD §5.1, §5.3; BR-023; CF-4, CF-5, CF-18,
CF-24; FTASKS 3.10).

Every assertion is made on what the REAL ASGI apps answer: the backend's `main.app` (`/api/health`) and the
MCP server's assembled HTTP app (`build_http_app(build_server(...))`, `/health`). The block must be BR-023's
EXACTLY (four keys; entries `kind, old, new, last_seen`; pairs `old, new`) and `legacy_names_meta.error` must
always be present: S001's P7 sampler reads a missing key as an unmeasured day, and a failed store read must
never read as "nothing in use".

Controls: K-3.25 (route without the block), K-3.26 (MCP closure without it), K-3.27 (24 h filter dropped),
K-3.28 (sink no-op), K-3.29 (extra key), K-3.30 (`exclude_none`), K-3.31 (`error` left None on a failure).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import anyio
import httpx
import pytest
from sqlalchemy import text

from src.core import legacy_names
from tests.support import legacy_export

BLOCK_KEYS = {"in_use", "also_set", "conflicts", "retired_but_set"}
ENTRY_KEYS = {"kind", "old", "new", "last_seen"}
PAIR_KEYS = {"old", "new"}
BACKEND_META_KEYS = {"observing_since", "in_use_window_hours", "watching", "unobserved_hosts", "reporters",
                     "error"}
MCP_META_KEYS = {"observing_since", "in_use_window_hours", "watching", "unobserved_hosts", "error", "push"}
TOKEN = "t" * 40
NAMES = sorted({n for r in legacy_names.ENV for n in (r.new, r.old)})


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    legacy_names.reset()
    from src.services import legacy_names_report, legacy_names_store

    legacy_names_report.reset_remote_pairs()
    legacy_names_store.API_QUEUE.drain()
    yield
    legacy_names.reset()
    legacy_names_store.API_QUEUE.drain()


@pytest.fixture
def tables(async_engine):
    """The two tables exist (conftest builds the ORM schema) and start empty."""
    from src.services import legacy_names_store

    legacy_names_store.dispose()
    with legacy_names_store.session_factory()() as db:
        db.execute(text("DELETE FROM legacy_name_uses"))
        db.execute(text("DELETE FROM legacy_name_reporters"))
        db.commit()
    yield legacy_names_store
    legacy_names_store.dispose()


def backend_health(host: str = "testserver") -> dict:
    from src.main import app

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://{host}") as c:
            r = await c.get("/api/health")
            assert r.status_code == 200, r.text
            return r.json()

    return anyio.run(run)


def mcp_health() -> dict:
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_http_app, build_server

    settings = MCPSettings(auth_token=TOKEN, _env_file=None)
    mcp, _ = build_server(settings)
    app = build_http_app(mcp, settings)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as c:
            r = await c.get("/health")
            assert r.status_code == 200, r.text
            return r.json()

    body = anyio.run(run)
    task = mcp.legacy_names_push.task
    if task is not None and not task.done():
        task.cancel()
    return body


def _exact(block: dict) -> None:
    assert set(block) == BLOCK_KEYS, sorted(block)
    for e in block["in_use"]:
        assert set(e) == ENTRY_KEYS, e
        assert e["last_seen"].endswith("Z") and e["kind"] in ("env", "host", "category")
    for k in ("also_set", "conflicts", "retired_but_set"):
        for p in block[k]:
            assert set(p) == PAIR_KEYS, p


def test_backend_block_has_exact_br023_keys(tables, monkeypatch):
    monkeypatch.setenv("MILLM_API_URL", "http://old")
    monkeypatch.setenv("MIWORKBENCH_API_URL", "http://a")
    monkeypatch.setenv("MISTUDIO_API_URL", "http://b")
    body = backend_health()
    assert {"legacy_names", "legacy_names_meta"} <= set(body)
    _exact(body["legacy_names"])
    # Retired since ENH-009 P7: the old name set alone is reported in retired_but_set, never in in_use.
    assert {"old": "MILLM_API_URL", "new": "MISERVER_API_URL"} in body["legacy_names"]["retired_but_set"]
    assert set(body["legacy_names_meta"]) == BACKEND_META_KEYS, sorted(body["legacy_names_meta"])
    assert body["legacy_names_meta"]["in_use_window_hours"] == 24
    assert body["legacy_names_meta"]["watching"] == {"env": 6, "host": 10, "category": 0}  # P7 G2: no category alias


def test_mcp_block_has_exact_br023_keys(monkeypatch):
    monkeypatch.setenv("MILLM_API_URL", "http://old")
    body = mcp_health()
    _exact(body["legacy_names"])
    assert {"old": "MILLM_API_URL", "new": "MISERVER_API_URL"} in body["legacy_names"]["retired_but_set"]
    meta = body["legacy_names_meta"]
    assert set(meta) == MCP_META_KEYS, sorted(meta)
    assert set(meta["push"]) == {"last_ok", "last_error", "pending"}
    assert "error" in meta and meta["error"] is None


def test_old_url_alone_reported_retired_in_both(tables, monkeypatch):
    """ENH-009 P7: the row is retired, so the old name alone is not in use; both answers report it set."""
    monkeypatch.setenv("MILLM_API_URL", "http://old")
    for body in (backend_health(), mcp_health()):
        block = body["legacy_names"]
        assert block["retired_but_set"] == [{"old": "MILLM_API_URL", "new": "MISERVER_API_URL"}], block
        assert "MILLM_API_URL" not in {e["old"] for e in block["in_use"]}, block
        assert "http://old" not in str(body), "a value reached a health answer"


def test_new_url_alone_empty(tables, monkeypatch):
    monkeypatch.setenv("MISERVER_API_URL", "http://new")
    for body in (backend_health(), mcp_health()):
        block = body["legacy_names"]
        assert block == {"in_use": [], "also_set": [], "conflicts": [], "retired_but_set": []}, block


def test_both_set_also_set_and_conflicts(tables, monkeypatch):
    monkeypatch.setenv("MISERVER_API_URL", "http://new")
    monkeypatch.setenv("MILLM_API_URL", "http://old")
    monkeypatch.setenv("MIWORKBENCH_API_URL", "http://same")
    monkeypatch.setenv("MISTUDIO_API_URL", "http://same")
    for body in (backend_health(), mcp_health()):
        block = body["legacy_names"]
        assert block["also_set"] == [{"old": "MILLM_API_URL", "new": "MISERVER_API_URL"},
                                     {"old": "MISTUDIO_API_URL", "new": "MIWORKBENCH_API_URL"}]
        assert block["conflicts"] == [{"old": "MILLM_API_URL", "new": "MISERVER_API_URL"}]
        assert block["in_use"] == []


def test_window_23h_in_25h_out(tables):
    now = datetime.now(timezone.utc)
    with tables.session_factory()() as db:
        for old, new, age in (("k8s-mistudio.hitsai.local", "k8s-miworkbench.hitsai.local", 23),
                              ("mcp-studio.hitsai.local", "mcp-workbench.hitsai.local", 25)):
            db.execute(text("INSERT INTO legacy_name_uses VALUES ('host', :o, :n, :t, :t)"),
                       {"o": old, "n": new, "t": now - timedelta(hours=age)})
        db.commit()
    olds = {e["old"] for e in backend_health()["legacy_names"]["in_use"]}
    assert "k8s-mistudio.hitsai.local" in olds
    assert "mcp-studio.hitsai.local" not in olds


async def test_stored_use_survives_restart(tables, monkeypatch):
    """The API's own wiring end to end: the lifespan sets the queue sink and the flush task, a request on an
    old host is recorded by the middleware, the shutdown flush writes it — and after a simulated restart
    (empty registry) the health answer still lists it, read from the table."""
    from src import main
    from src.services import background_monitor

    class Inert:
        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(main, "get_background_monitor", lambda: Inert())
    monkeypatch.setattr(background_monitor, "_background_monitor", None)
    async with main.lifespan(main.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                     base_url="http://k8s-mistudio.hitsai.local") as c:
            assert (await c.get("/api/v1/system/health")).status_code == 200
    legacy_names.reset()  # a restarted process remembers nothing
    with tables.session_factory()() as db:
        rows = db.execute(text("SELECT kind, old, new FROM legacy_name_uses")).all()
        reporters = db.execute(text("SELECT reporter FROM legacy_name_reporters")).scalars().all()
    assert ("host", "k8s-mistudio.hitsai.local", "k8s-miworkbench.hitsai.local") in rows
    assert "api" in reporters
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://testserver") as c:
        body = (await c.get("/api/health")).json()
    assert "k8s-mistudio.hitsai.local" in {e["old"] for e in body["legacy_names"]["in_use"]}
    assert "api" in body["legacy_names_meta"]["reporters"]


def test_store_unreadable_answers_200_with_meta_error(monkeypatch):
    from src.services import legacy_names_store

    legacy_names_store.dispose()
    monkeypatch.setattr(legacy_names_store, "_url",
                        lambda: "postgresql://postgres:x@127.0.0.1:1/mistudio_test_p0_absent")
    monkeypatch.setenv("MILLM_API_URL", "http://old")
    try:
        body = backend_health()
    finally:
        legacy_names_store.dispose()
    meta = body["legacy_names_meta"]
    assert isinstance(meta["error"], str) and meta["error"], meta
    # The block still answers from this process's memory: a failed read never reads as "nothing set".
    assert {"old": "MILLM_API_URL", "new": "MISERVER_API_URL"} in body["legacy_names"]["retired_but_set"]


def test_meta_and_error_key_always_present(tables):
    body = backend_health()
    meta = body["legacy_names_meta"]
    assert "error" in meta and meta["error"] is None
    assert "unobserved_hosts" in meta and "reporters" in meta and "observing_since" in meta
    mcp_meta = mcp_health()["legacy_names_meta"]
    assert "error" in mcp_meta and mcp_meta["error"] is None


def test_unobserved_hosts_equal_export_false_hosts(tables):
    export = legacy_export.load()
    expected = sorted(h["old"] for h in export.hosts() if h["observed"] is False)
    assert backend_health()["legacy_names_meta"]["unobserved_hosts"] == expected
    assert mcp_health()["legacy_names_meta"]["unobserved_hosts"] == expected
