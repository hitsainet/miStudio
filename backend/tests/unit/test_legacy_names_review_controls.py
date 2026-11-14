"""Regression tests for the mutation survivors of the independent P0 review (rev-mistudio-p0, 2026-11-08).

Each test below names the review control it closes; every one was red against its mutation (negative control)
and green on the fixed tree. They pin the store's own guarantees (the route and the flush were tested with the
store stubbed, so the store's bounds and its forward-only upserts were never exercised), the evidence bound and
the per-name throttle of `legacy_names.record`, the MCP pod's also_set/conflicts reaching the backend's health,
the POST route's bound, the dotenv old name staying in force, and the miServer provenance headers miWorkbench
reads under their frozen names.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import anyio
import httpx
import pytest
from sqlalchemy import text

from src.core import legacy_names

NAMES = sorted({n for r in legacy_names.ENV for n in (r.new, r.old)})
HOST = ("host", "k8s-mistudio.hitsai.local", "k8s-miworkbench.hitsai.local")
T1 = datetime(2026, 11, 8, 10, 0, tzinfo=timezone.utc)
T2 = T1 + timedelta(hours=1)


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
    legacy_names_report.reset_remote_pairs()
    legacy_names_store.API_QUEUE.drain()


@pytest.fixture
def store(async_engine):
    from src.services import legacy_names_store

    legacy_names_store.dispose()
    with legacy_names_store.session_factory()() as db:
        db.execute(text("DELETE FROM legacy_name_uses"))
        db.execute(text("DELETE FROM legacy_name_reporters"))
        db.commit()
    yield legacy_names_store
    legacy_names_store.dispose()


def _use(when: datetime, kind=HOST[0], old=HOST[1], new=HOST[2]) -> dict:
    return {"kind": kind, "old": old, "new": new, "last_seen": when.strftime("%Y-%m-%dT%H:%M:%SZ")}


# ── the store (R01–R04) ─────────────────────────────────────────────────────────────────────────────────────
def test_r02_a_late_older_write_never_moves_last_seen_back(store):
    store.write({"reporter": "mcp", "uses": [_use(T2)]}, now=T2)
    store.write({"reporter": "mcp", "uses": [_use(T1)]}, now=T2)  # a delayed flush of an older report
    with store.session_factory()() as db:
        assert db.execute(text("SELECT last_seen FROM legacy_name_uses")).scalar_one() == T2


def test_r04_a_late_older_report_never_moves_last_report_back(store):
    store.write({"reporter": "beat", "uses": []}, now=T2)
    store.write({"reporter": "beat", "uses": []}, now=T1)
    with store.session_factory()() as db:
        assert db.execute(text("SELECT last_report FROM legacy_name_reporters")).scalar_one() == T2


def test_r03_the_store_drops_a_use_its_tables_do_not_list(store):
    store.write({"reporter": "beat", "uses": [_use(T1, kind="env", old="NOT_A_ROW", new="ALSO_NOT")]}, now=T2)
    with store.session_factory()() as db:
        assert db.execute(text("SELECT count(*) FROM legacy_name_uses")).scalar_one() == 0


def test_r01_a_locked_table_fails_the_write_within_the_bound(store):
    """`lock_timeout`/`statement_timeout` are set on every store connection: a write that meets a held lock
    raises within about `BOUND_S` instead of waiting for the lock (a worker's prerun refresh and the API's
    flush would otherwise wait as long as the lock is held)."""
    import psycopg2

    from tests.support.test_databases import sync_test_url
    from tests.support.xdist_isolation import libpq_dsn

    holder = psycopg2.connect(libpq_dsn(sync_test_url()))
    try:
        cur = holder.cursor()
        cur.execute("LOCK TABLE legacy_name_uses IN ACCESS EXCLUSIVE MODE")  # held until rollback
        released = threading.Timer(4 * store.BOUND_S, holder.rollback)
        released.start()
        t0 = time.monotonic()
        with pytest.raises(Exception):  # noqa: B017 — LockNotAvailable or QueryCanceled, wrapped
            store.write({"reporter": "beat", "uses": [_use(T1)]}, now=T2)
        assert time.monotonic() - t0 < 2 * store.BOUND_S + 1.0, "the store waited for the lock"
        released.cancel()
    finally:
        holder.rollback()
        holder.close()


# ── the backend's answer (R07) ──────────────────────────────────────────────────────────────────────────────
def test_r07_a_newer_stored_use_wins_over_an_older_one_in_memory():
    from src.services import legacy_names_report

    now = T2 + timedelta(minutes=5)
    legacy_names.reset(clock=lambda: T1)
    legacy_names.record(*HOST)  # this process saw it at T1
    stored = [{"kind": HOST[0], "old": HOST[1], "new": HOST[2], "first_seen": T1, "last_seen": T2}]
    out = legacy_names_report.build(now=now, read=lambda: (stored, {}, None))
    (entry,) = [e for e in out["legacy_names"]["in_use"] if e["old"] == HOST[1]]
    assert entry["last_seen"] == "2026-11-08T11:00:00Z", entry


# ── the recorder (R15, R17) ─────────────────────────────────────────────────────────────────────────────────
def test_r15_record_drops_a_pair_the_tables_do_not_list():
    legacy_names.record("env", "NOT_A_ROW", "ALSO_NOT")
    legacy_names.record("host", HOST[1], "somewhere-else.example")
    assert legacy_names.report()["uses"] == []


def test_r17_one_name_reaches_the_sink_at_most_once_a_minute():
    mono = [100.0]
    legacy_names.reset(mono=lambda: mono[0])
    seen: list[dict] = []
    legacy_names.set_sink(seen.append, "api")
    for _ in range(5):
        legacy_names.record(*HOST)
    assert len(seen) == 1, f"{len(seen)} deliveries for one name within a minute"
    mono[0] += legacy_names.WRITE_INTERVAL_S
    legacy_names.record(*HOST)
    assert len(seen) == 2


# ── the MCP pod's pairs reach the backend (R19, R20) ────────────────────────────────────────────────────────
async def _post(body) -> httpx.Response:
    from src.main import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as c:
        return await c.post("/api/v1/system/legacy-names", json=body)


async def test_r19_the_push_carries_this_pods_also_set_and_conflicts(monkeypatch):
    from src.mcp_server.legacy_names_push import PushState

    monkeypatch.setenv("MISERVER_API_URL", "http://new:8000")
    monkeypatch.setenv("MILLM_API_URL", "http://old:8000")
    sent = []

    class Client:
        async def post(self, path, body):
            sent.append(body)
            return {}

    assert await PushState().push(Client())
    pair = {"old": "MILLM_API_URL", "new": "MISERVER_API_URL"}
    assert pair in sent[0]["also_set"] and pair in sent[0]["conflicts"], sent[0]


async def test_r20_posted_pairs_appear_in_the_backend_health(store, monkeypatch):
    from src.services import legacy_names_report

    pair = {"old": "MISTUDIO_API_URL", "new": "MIWORKBENCH_API_URL"}
    r = await _post({"reporter": "mcp", "uses": [], "also_set": [pair], "conflicts": [pair]})
    assert r.status_code == 204, r.text
    block = (await anyio.to_thread.run_sync(legacy_names_report.build))["legacy_names"]
    assert pair in block["also_set"] and pair in block["conflicts"], block


# ── the POST route's bound (R21) ────────────────────────────────────────────────────────────────────────────
async def test_r21_a_hung_store_answers_503_within_the_route_bound(monkeypatch):
    from src.api.v1.endpoints import system
    from src.services import legacy_names_store

    monkeypatch.setattr(legacy_names_store, "write", lambda report, now=None: time.sleep(1.5))
    monkeypatch.setattr(system, "LEGACY_NAMES_WRITE_TIMEOUT_S", 0.2)
    t0 = time.monotonic()
    r = await _post({"reporter": "mcp", "uses": []})
    assert r.status_code == 503 and time.monotonic() - t0 < 1.0, (r.status_code, time.monotonic() - t0)


# ── miServer's provenance headers, read under their frozen names (R29; register DR-08, Q-12) ───────────────
def test_r29_provenance_headers_read_under_the_names_miserver_sends():
    from src.mcp_server.miserver_send import provenance

    sent = {"X-miLLM-Steering": "none", "X-miLLM-Seed": "7", "X-miLLM-Constrained": "true",
            "X-miLLM-Ignored-Fields": "logit_bias", "X-miLLM-Scoring-Precision": "float32"}
    assert provenance(sent) == {"steering": "none", "seed": "7", "constrained": "true",
                                "ignored_fields": "logit_bias", "scoring_precision": "float32"}


# ── a dotenv old name stays in use while it is in force (F7; exposed by survivor R12) ─────────────────────
def test_f7_a_dotenv_old_name_stays_in_use_past_24_hours(tmp_path, monkeypatch):
    from src.core.config import Settings

    # The row is retired since ENH-009 P7; the in-force mechanism is pinned on it put back into transition.
    row = next(r for r in legacy_names.ENV if r.old == "MILLM_API_URL")
    live = legacy_names.EnvName(new=row.new, old=row.old, status="transition")
    monkeypatch.setitem(legacy_names._ENV_BY_NEW, row.new, live)  # noqa: SLF001
    monkeypatch.setitem(legacy_names._ENV_BY_OLD, row.old, live)  # noqa: SLF001
    monkeypatch.setattr(legacy_names, "ENV", tuple(live if r is row else r for r in legacy_names.ENV))
    now, mono = [T1], [100.0]
    legacy_names.reset(clock=lambda: now[0], mono=lambda: mono[0])
    env_file = tmp_path / ".env"
    env_file.write_text("MILLM_API_URL=http://from-dotenv\n")
    assert Settings(_env_file=env_file).miserver_api_url == "http://from-dotenv"
    now[0], mono[0] = T1 + timedelta(hours=25), mono[0] + 25 * 3600
    assert legacy_names.refresh(), "refresh did not run"
    in_use = {(u["kind"], u["old"]) for u in legacy_names.snapshot()["in_use"]}
    assert ("env", "MILLM_API_URL") in in_use, "a dotenv old name still in force aged out of the window"
