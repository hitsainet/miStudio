"""Discovery progress reaches the ROW, so REST and MCP see it while the run is live.

On hardware (2026-10-17) ``dsc_8b10fd9f32b8`` ran 52.5 minutes and ``GET
/circuit-discovery/{id}`` — which the MCP ``get_discovery_run`` tool wraps — read
``progress: 0.0`` the whole time: progress went only to the WebSocket callback. These
tests run the REAL discovery service on a synthetic store in a worker thread, pause it
mid-stage, and read progress through the REST route.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from src.core.database import get_db
from src.main import app
from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
from src.services import circuit_discovery_service as disc
from src.services import circuit_stats_service as stats
from tests.unit.test_circuit_discovery_service import _build_store, _capture_row


@pytest.fixture
def sync_db(async_engine):
    sync_url = async_engine.url.set(
        drivername=async_engine.url.drivername.replace("+asyncpg", "").replace("+psycopg", ""))
    engine = create_engine(sync_url)
    session = sessionmaker(bind=engine)()
    yield session
    session.rollback()
    session.query(CircuitDiscoveryRun).delete()
    session.query(CircuitCaptureRun).delete()
    session.commit()
    session.close()
    engine.dispose()


@pytest_asyncio.fixture
async def api(async_engine):
    maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async def per_request_session():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_db] = per_request_session
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://test/api/v1") as client:
        yield client
    app.dependency_overrides.clear()


class _Gate:
    """Pauses the null loop at chosen tests so the REST route can be read mid-run."""

    def __init__(self, pause_at):
        self.pause_at = set(pause_at)
        self.calls = 0
        self.reached = threading.Event()
        self.resume = threading.Event()

    def wrap(self, real):
        def null_test(*a, **k):
            j = self.calls
            self.calls += 1
            if j in self.pause_at:
                self.reached.set()
                assert self.resume.wait(30), "the test never resumed the run"
                self.resume.clear()
            return real(*a, **k)
        return null_test


async def _progress(client, run_id):
    r = await client.get(f"/circuit-discovery/{run_id}", params={"include_candidates": False})
    assert r.status_code == 200, r.text
    body = r.json()
    return body["status"], body["progress"]


@pytest.mark.asyncio
async def test_rest_reports_live_progress_while_discovery_advances(
        api, sync_db, tmp_path, monkeypatch):
    _build_store(tmp_path)
    cap = _capture_row(sync_db, tmp_path)
    run = disc.CircuitDiscoveryService.create_run(sync_db, {
        "capture_run_id": cap.id, "granularity": "feature",
        "mode": "open", "s_min": 1, "null_shuffles": 20})
    run_id = run.id
    # Every report persists; the throttle is pinned separately below.
    monkeypatch.setattr(disc, "PROGRESS_PERSIST_SECONDS", 0.0)
    gate = _Gate(pause_at=(0, 1))
    monkeypatch.setattr(stats, "null_test", gate.wrap(stats.null_test))

    socket = []
    worker = threading.Thread(target=lambda: disc.CircuitDiscoveryService.run(
        sync_db, run_id, progress_cb=socket.append))
    worker.start()
    seen = []
    try:
        for _ in gate.pause_at:
            assert await asyncio.to_thread(gate.reached.wait, 30), "the run never reached a pause"
            gate.reached.clear()
            seen.append(await _progress(api, run_id))
            gate.resume.set()
    finally:
        gate.resume.set()
        await asyncio.to_thread(worker.join, 60)
    final = await _progress(api, run_id)

    # Paused at the first null test: pair generation is done, so 50%.
    assert seen[0] == ("running", 50.0), seen
    # One null test later the row has moved on, still short of done.
    assert seen[1][0] == "running" and 50.0 < seen[1][1] < 100.0, seen
    assert final == ("completed", 100.0)
    # The socket still hears every persisted step.
    assert socket and socket == sorted(socket)
    # Pair generation moves per up-unit; it used to report only once a layer was done.
    assert any(0.0 < p < 50.0 for p in socket), socket


def test_progress_writes_are_throttled_by_time():
    """At most one row write per PROGRESS_PERSIST_SECONDS, except forced stage ends."""
    class _Db:
        commits = 0

        def commit(self):
            self.commits += 1

    class _Run:
        progress = 0.0

    now = [0.0]
    db, row, heard = _Db(), _Run(), []
    report = disc._Progress(db, row, heard.append, clock=lambda: now[0])
    report(10.0)                    # first write always lands
    now[0] = 0.5
    report(20.0)                    # inside the window: dropped
    assert (row.progress, db.commits) == (10.0, 1)
    report(50.0, force=True)        # a stage end always lands
    assert (row.progress, db.commits) == (50.0, 2)
    now[0] = 0.5 + disc.PROGRESS_PERSIST_SECONDS
    report(60.0)
    assert (row.progress, db.commits) == (60.0, 3)
    assert heard == [10.0, 50.0, 60.0]
