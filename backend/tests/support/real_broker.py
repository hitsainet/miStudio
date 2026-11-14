"""A real Redis for tests of kombu's own structures, on a database of its own per xdist worker.

`real_broker` yields `(client, url)`. A message put in `unacked` by kombu itself, not a hand-written imitation of it: a
fixture whose fields agree by construction is the usual reason a suite stays green over a real bug.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def real_broker():
    import redis

    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw")
    base = os.environ.get("REDIS_URL", "redis://localhost:6379/0").rsplit("/", 1)[0]
    url = f"{base}/{8 + int(worker or 0)}"
    client = redis.Redis.from_url(url, socket_connect_timeout=1.0, socket_timeout=5.0)
    try:
        client.ping()
    except redis.RedisError as exc:
        pytest.skip(f"NO REDIS AT {url}: {exc}. These tests read kombu's real structures.")
    client.flushdb()
    yield client, url
    client.flushdb()
