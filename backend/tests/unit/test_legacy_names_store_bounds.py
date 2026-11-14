"""The old-name store recovers by itself when connections it holds go silent (review rev-mistudio-p0 F2).

The miForge P0 review found a pooled probe connection that black-holes and keeps health "unavailable" forever.
Here the store's engine was pooled (`pool_size=1, max_overflow=1`): two in-flight calls whose connections go
silent (a failover or a dropped NAT entry mid-flush) held both slots, so every later `read()` waited
`pool_timeout` and failed — `/api/health` answered `error: TimeoutError` and every flush failed — while the
database itself was healthy. `statement_timeout` is enforced by the server and cannot fire for a connection
the server never hears from again. The fix: no pool (each call opens a fresh, `connect_timeout`-bounded
connection) and TCP keepalive/user-timeout options on every connection.

The silence is made by a local TCP proxy that stops forwarding the connections open before the cut and keeps
forwarding new ones (the database stays healthy). Negative control: the pooled engine → red.
"""
from __future__ import annotations

import os
import socket
import threading
import time

import pytest
from sqlalchemy import text

from tests.support.test_databases import sync_test_url


class _Proxy:
    def __init__(self, upstream: tuple[str, int]) -> None:
        self.upstream = upstream
        self.cut = threading.Event()
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _pump(self, src, dst, before_cut: bool) -> None:
        try:
            while True:
                if before_cut and self.cut.is_set():
                    time.sleep(0.1)
                    continue
                data = src.recv(65536)
                if not data:
                    return
                dst.sendall(data)
        except OSError:
            return

    def _accept(self) -> None:
        while True:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            u = socket.create_connection(self.upstream)
            before = not self.cut.is_set()
            threading.Thread(target=self._pump, args=(c, u, before), daemon=True).start()
            threading.Thread(target=self._pump, args=(u, c, before), daemon=True).start()


def test_two_silent_connections_do_not_hold_the_store(async_engine, monkeypatch):
    from sqlalchemy.engine import make_url

    from src.services import legacy_names_store as store

    url = make_url(sync_test_url())
    proxy = _Proxy((url.host or "127.0.0.1", url.port or 5432))
    proxied = url.set(drivername="postgresql", host="127.0.0.1", port=proxy.port).render_as_string(hide_password=False)
    store.dispose()
    monkeypatch.setattr(store, "_url", lambda: proxied)
    try:
        assert store.read()[2] is None, "the store cannot read through the proxy at all"

        def in_flight():
            try:
                with store.session_factory()() as db:
                    db.execute(text("SELECT pg_sleep(1)"))
            except Exception:  # noqa: BLE001 — a silenced call may end in any error, or never
                pass

        hung = [threading.Thread(target=in_flight, daemon=True) for _ in range(2)]
        for t in hung:
            t.start()
        time.sleep(0.5)
        proxy.cut.set()  # both in-flight connections go silent; the database stays reachable
        time.sleep(1.0)
        t0 = time.monotonic()
        results = [store.read()[2] for _ in range(2)]
        assert results == [None, None], (
            f"the store still fails with the database healthy ({results}): silent connections hold it")
        assert time.monotonic() - t0 < 2 * store.BOUND_S + 1
    finally:
        store.dispose()
        proxy.sock.close()
