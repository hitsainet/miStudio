"""The API never blocks on the old-name store, and never loses a drained use (042 FR-042.9; FTID §3.3).

miLLM's P0 review found a cancelled flush that lost its drained uses, and miForge's an unbounded health
read. Here: the API's sink only queues; the flush is bounded (`FLUSH_TIMEOUT_S`) and puts its uses back on
a failure, a timeout or a cancellation; the lifespan starts the flush task and keeps a strong reference;
`/api/health` answers within its budget with `meta.error` when the store hangs.

Controls (recorded in the 042 P0 controls record): restore removed on CancelledError, on failure; the
`wait_for` bound removed; the lifespan task line removed; the health budget removed.
"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from src.core import legacy_names
from src.services import legacy_names_store as store

USE = {"kind": "host", "old": "k8s-mistudio.hitsai.local", "new": "k8s-miworkbench.hitsai.local",
       "last_seen": "2026-11-08T10:00:00Z"}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for row in legacy_names.ENV:  # the flush is asserted use by use; no env use may leak in
        monkeypatch.delenv(row.old, raising=False)
        monkeypatch.delenv(row.new, raising=False)
    legacy_names.reset()
    store.API_QUEUE.drain()
    yield
    legacy_names.reset()
    store.API_QUEUE.drain()


def _queue_with_use() -> store.ApiQueue:
    q = store.ApiQueue()
    q.sink({"reporter": "api", "uses": [dict(USE)], "also_set": [], "conflicts": []})
    return q


def test_the_api_sink_only_queues(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("the API sink touched the database on the request path")

    monkeypatch.setattr(store, "write", boom)
    legacy_names.set_sink(store.API_QUEUE.sink, "api")
    legacy_names.record("host", USE["old"], USE["new"])
    pending = store.API_QUEUE.pending()
    assert pending["reporter"] == "api" and [u["old"] for u in pending["uses"]] == [USE["old"]]


async def test_a_failed_flush_keeps_its_uses(monkeypatch):
    def fail(_report, now=None):
        raise RuntimeError("database down")

    monkeypatch.setattr(store, "write", fail)
    q = _queue_with_use()
    assert await store.flush_once(q) is False
    assert [u["old"] for u in q.pending()["uses"]] == [USE["old"]]


async def test_the_flush_is_bounded_and_keeps_its_uses(monkeypatch):
    monkeypatch.setattr(store, "write", lambda _report, now=None: time.sleep(1.0))
    q = _queue_with_use()
    t0 = time.monotonic()
    assert await store.flush_once(q, timeout=0.1) is False
    assert time.monotonic() - t0 < 0.8, "the flush waited for a hung store"
    assert [u["old"] for u in q.pending()["uses"]] == [USE["old"]]


async def test_a_cancelled_flush_keeps_its_uses(monkeypatch):
    monkeypatch.setattr(store, "write", lambda _report, now=None: time.sleep(0.5))
    q = _queue_with_use()
    task = asyncio.create_task(store.flush_once(q, timeout=5.0))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [u["old"] for u in q.pending()["uses"]] == [USE["old"]], "a cancelled flush lost its uses"


async def test_a_restore_keeps_newer_values():
    q = _queue_with_use()
    drained = q.drain()
    q.sink({"reporter": "api", "uses": [{**USE, "last_seen": "2026-11-08T11:00:00Z"}],
            "also_set": [], "conflicts": []})
    q.restore(drained)
    assert q.pending()["uses"][0]["last_seen"] == "2026-11-08T11:00:00Z"


async def test_the_lifespan_starts_the_flush_task_and_flushes_on_shutdown(monkeypatch):
    from src import main
    from src.services import background_monitor

    class Inert:
        async def start(self):
            return None

        async def stop(self):
            return None

    writes: list[dict] = []
    monkeypatch.setattr(store, "write", lambda report, now=None: writes.append(report))
    monkeypatch.setattr(main, "get_background_monitor", lambda: Inert())
    monkeypatch.setattr(background_monitor, "_background_monitor", None)
    monkeypatch.setattr(main.runtime_settings, "refresh_from_database_async", _noop)
    monkeypatch.setattr(main, "validate_schema_on_startup", _true)
    monkeypatch.setattr(main.judge_prompt_seeds, "ensure_seeds", _noop)
    async with main.lifespan(main.app):
        task = main._legacy_names_task  # noqa: SLF001
        assert task is not None and not task.done() and task.get_name() == "legacy-names-flush"
        assert legacy_names.reporter() == "api"
        await asyncio.sleep(0.05)  # the first loop step: refresh (the API's heartbeat) + flush
        assert [w["reporter"] for w in writes] == ["api"], writes
        n = len(writes)
        legacy_names.record("host", USE["old"], USE["new"])  # what the host middleware does per request
        assert len(writes) == n, "a request-path record wrote to the database (the API sink must only queue)"
    assert task.cancelled() or task.done()
    assert main._legacy_names_task is None  # noqa: SLF001
    assert any(u["old"] == USE["old"] for u in writes[-1]["uses"]), "the shutdown flush wrote nothing"


async def _noop(*_a, **_k):
    return None


async def _true(*_a, **_k):
    return True


async def test_a_hung_store_never_holds_health(monkeypatch):
    from src import main

    monkeypatch.setattr(store, "read", lambda: (time.sleep(2.0), ([], {}, None))[1])
    monkeypatch.setattr(main, "LEGACY_HEALTH_BUDGET_S", 0.2)
    t0 = time.monotonic()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://testserver") as c:
        r = await c.get("/api/health")
    assert time.monotonic() - t0 < 1.5
    assert r.status_code == 200
    assert r.json()["legacy_names_meta"]["error"] == "TimeoutError"
