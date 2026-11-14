"""A progress emission re-entered on one thread is dropped, not deadlocked.

tqdm calls `close()` from `__del__`, and `TqdmWebSocketCallback.close()` emits. So the
garbage collector can start an emission from inside another one, while the thread holds
httpcore's connection-pool lock, which is not re-entrant. On 2026-10-29 that froze CI
shard 1 for 30 minutes, twice, inside `test_a_live_run_is_not_interrupted`; a worker
thread would freeze the same way, with no error.

The fake client below re-enters `emit_progress` from inside `post`, the way the collector
did, and holds a non-blocking stand-in for the pool lock so a second send is observable
rather than a hang.
"""

import threading

import pytest

from src.workers import websocket_emitter


class _Response:
    status_code = 200


class _ReentrantClient:
    """`post` re-enters emit_progress once, as tqdm's `__del__` did mid-send."""

    is_closed = False

    def __init__(self):
        self.calls = []
        self.inner_result = None
        self._pool_lock = threading.Lock()

    def post(self, url, json, headers, timeout):
        # The real pool lock blocks forever on re-entry; acquire(blocking=False)
        # turns that hang into a recorded failure.
        acquired = self._pool_lock.acquire(blocking=False)
        self.calls.append((json["event"], acquired))
        try:
            if len(self.calls) == 1:
                self.inner_result = websocket_emitter.emit_progress(
                    "datasets/ds-1/progress", "collected", {"progress": 100.0}
                )
            return _Response()
        finally:
            if acquired:
                self._pool_lock.release()


@pytest.fixture
def client(monkeypatch):
    fake = _ReentrantClient()
    monkeypatch.setattr(websocket_emitter, "_get_http_client", lambda: fake)
    return fake


def test_a_reentered_emission_never_reaches_the_client(client):
    outer = websocket_emitter.emit_progress(
        "datasets/ds-1/progress", "progress", {"progress": 50.0}
    )

    assert outer is True
    assert client.inner_result is False
    # One send, and it held the lock. Without the guard the inner send arrives
    # while the outer holds the lock — the deadlock in production.
    assert client.calls == [("progress", True)]


def test_the_guard_is_released_after_a_send(client):
    websocket_emitter.emit_progress("datasets/ds-1/progress", "progress", {"progress": 1.0})
    client.calls.clear()
    client.inner_result = None

    assert websocket_emitter.emit_progress(
        "datasets/ds-1/progress", "progress", {"progress": 2.0}
    ) is True
    assert client.calls[0] == ("progress", True)


def test_the_guard_is_released_when_the_send_raises(monkeypatch):
    class _Boom:
        is_closed = False

        def post(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(websocket_emitter, "_get_http_client", lambda: _Boom())
    assert websocket_emitter.emit_progress("c", "e", {}) is False

    fake = _ReentrantClient()
    monkeypatch.setattr(websocket_emitter, "_get_http_client", lambda: fake)
    assert websocket_emitter.emit_progress("c", "progress", {}) is True
    assert fake.calls[0] == ("progress", True)


def test_another_thread_is_not_blocked_by_this_threads_send(monkeypatch):
    """The flag is per thread: a send on one thread must not drop another's."""
    seen = []

    class _Blocking:
        is_closed = False

        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()

        def post(self, url, json, headers, timeout):
            seen.append(json["event"])
            if json["event"] == "slow":
                self.entered.set()
                self.release.wait(5)
            return _Response()

    fake = _Blocking()
    monkeypatch.setattr(websocket_emitter, "_get_http_client", lambda: fake)
    t = threading.Thread(target=websocket_emitter.emit_progress, args=("c", "slow", {}))
    t.start()
    assert fake.entered.wait(5)
    try:
        assert websocket_emitter.emit_progress("c", "other", {}) is True
    finally:
        fake.release.set()
        t.join(5)
    assert sorted(seen) == ["other", "slow"]
