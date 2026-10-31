"""Parked GPU jobs come back when due, and a removed card's queue is rescued (Phase 3).

A job whose card is busy is parked in a Redis sorted set rather than handed to
Celery's ``countdown`` (which holds it unacked in a worker's memory, stranding it
for 12 hours if that worker is killed). The GPU supervisor republishes due
entries and moves messages off ``gpu.<uuid>`` queues whose card has no worker.

Redis is faked in memory (no broker in CI); the fake's list and sorted-set
semantics were checked against the real local Redis (redis-py 5.3.1) with the
same calls — see the Phase 3 notes in the plan.

MUTATION CONTROLS (2026-09-14; each alone, restored byte-identically, sha256 checked,
`git diff` clean) — both red:
  Q1 a parked entry is published even when another releaser removed it -> not published twice
  Q2 a queue with a live worker is swept too                             -> moved in order; nothing moves when live
The supervisor's own wiring of the tick (Q3, Q4) is recorded in test_gpu_supervisor.py.
"""

import fnmatch
import json

import pytest

from src.services import gpu_worker_queues as Q
from src.services.gpu_claim import AUTO_QUEUE, queue_for

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
GONE_UUID = "GPU-11111111-2222-3333-4444-555555555555"


class FakeRedis:
    """The calls gpu_worker_queues makes, with Redis's semantics."""

    def __init__(self):
        self.zsets = {}
        self.lists = {}

    @staticmethod
    def _b(value):
        return value.encode() if isinstance(value, str) else value

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update({self._b(m): float(s) for m, s in mapping.items()})

    def zrangebyscore(self, key, low, high, start=None, num=None):
        members = sorted((s, m) for m, s in self.zsets.get(key, {}).items() if s <= float(high))
        out = [m for _, m in members]
        return out[start:start + num] if num is not None else out

    def zrem(self, key, member):
        return 1 if self.zsets.get(key, {}).pop(self._b(member), None) is not None else 0

    def lpush(self, key, *values):
        for value in values:
            self.lists.setdefault(key, []).insert(0, self._b(value))

    def rpoplpush(self, source, destination):
        items = self.lists.get(source)
        if not items:
            return None
        value = items.pop()
        self.lists.setdefault(destination, []).insert(0, value)
        return value

    def rpop(self, key):
        items = self.lists.get(key)
        return items.pop() if items else None

    def scan_iter(self, match):
        keys = [k for k in self.lists if self.lists[k]] + list(self.zsets)
        return [k.encode() for k in keys if fnmatch.fnmatchcase(k, match)]

    def type(self, key):
        if self.lists.get(key):
            return b"list"
        return b"zset" if key in self.zsets else b"none"


def park(client, task_id, due_at):
    Q.park_job("src.workers.x.task", [task_id], {"gpu_request": "auto"},
               {"queue": AUTO_QUEUE, "task_id": task_id}, due_at=due_at, client=client)


class TestParkedJobs:
    def test_a_due_job_is_republished_once_with_its_arguments(self):
        client, published = FakeRedis(), []
        park(client, "due", due_at=100.0)
        park(client, "later", due_at=200.0)

        assert Q.release_due_jobs(client, published.append, now=150.0) == 1
        assert published == [{
            "task": "src.workers.x.task", "args": ["due"], "kwargs": {"gpu_request": "auto"},
            "options": {"queue": AUTO_QUEUE, "task_id": "due"},
        }]
        assert Q.release_due_jobs(client, published.append, now=150.0) == 0
        assert len(client.zsets[Q.WAITING_KEY]) == 1

    def test_a_job_another_releaser_took_is_not_published_twice(self):
        client, published = FakeRedis(), []
        park(client, "due", due_at=100.0)

        class Racing(FakeRedis):
            pass

        racing = Racing()
        racing.zsets = client.zsets
        original = racing.zrem

        def zrem_after_rival(key, member):
            original(key, member)   # the rival removed it first
            return 0

        racing.zrem = zrem_after_rival
        assert Q.release_due_jobs(racing, published.append, now=150.0) == 0
        assert published == []

    def test_a_failed_republish_keeps_the_job(self):
        client = FakeRedis()
        park(client, "due", due_at=100.0)

        def broker_down(entry):
            raise ConnectionError("broker unreachable")

        assert Q.release_due_jobs(client, broker_down, now=150.0) == 0
        (score,) = client.zsets[Q.WAITING_KEY].values()
        assert score == 150.0 + Q.REPUBLISH_RETRY_S

    def test_the_entry_is_json_a_restart_can_read(self):
        client = FakeRedis()
        park(client, "t", due_at=1.0)
        (raw,) = client.zsets[Q.WAITING_KEY]
        assert json.loads(raw)["options"]["task_id"] == "t"


class TestStrandedCardQueues:
    def test_a_removed_cards_jobs_move_to_the_shared_queue_in_order(self):
        client = FakeRedis()
        client.lpush(queue_for(GONE_UUID), "first", "second")   # published in that order
        client.lpush(queue_for(RTX_UUID), "stays")

        moved = Q.requeue_stranded_card_messages(client, [TI_UUID, RTX_UUID])

        assert moved == {queue_for(GONE_UUID): 2}
        assert client.rpop(AUTO_QUEUE) == b"first"
        assert client.rpop(AUTO_QUEUE) == b"second"
        assert client.lists[queue_for(RTX_UUID)] == [b"stays"]

    def test_a_priority_suffixed_list_of_a_removed_card_moves_too(self):
        client = FakeRedis()
        client.lpush(queue_for(GONE_UUID) + "\x06\x163", "prioritised")
        assert Q.requeue_stranded_card_messages(client, [RTX_UUID]) == {queue_for(GONE_UUID): 1}
        assert client.rpop(AUTO_QUEUE) == b"prioritised"

    def test_nothing_moves_when_every_queue_has_a_worker(self):
        client = FakeRedis()
        client.lpush(queue_for(TI_UUID), "a")
        client.lpush(AUTO_QUEUE, "b")
        assert Q.requeue_stranded_card_messages(client, [TI_UUID]) == {}
        assert client.lists[AUTO_QUEUE] == [b"b"]


class TestAnEmptyInventoryStrandsNothing:
    """Review round 1. With NO card in the inventory every card queue looks stranded.
    A pod whose NVML read failed at start (the node's driver has hung before) would
    move every job queued for a named card to the shared queue, where each one fails
    with "No GPU is visible" — irreversibly, for a card that is still there. An empty
    inventory cannot tell "no cards" from "could not read the cards", so it moves
    nothing and says so; a card that is really gone is swept once one card is seen.

    MUTATION CONTROL (2026-09-14; restored byte-identically, sha256 checked):
      Q5 the empty-inventory guard removed -> both tests below
    """

    def test_the_gpu_supervisor_moves_nothing_when_it_sees_no_card(self):
        client = FakeRedis()
        client.lpush(queue_for(RTX_UUID), "queued for the 3090")

        assert Q.requeue_stranded_card_messages(client, []) == {}
        assert client.lists[queue_for(RTX_UUID)] == [b"queued for the 3090"]
        assert not client.lists.get(AUTO_QUEUE)

    def test_the_steering_reconcile_moves_nothing_when_it_sees_no_card(self):
        client = FakeRedis()
        client.lpush("steering.gpu-247aa582-0d1b-e161-8156-983ed1fefc57", "a generation")

        assert Q.requeue_stranded_messages(client, [], pattern=Q.STEERING_QUEUE_PATTERN, target="steering") == {}
        assert not client.lists.get("steering")


class TestTheSupervisorTick:
    def test_it_throttles_itself_and_does_both_at_once_on_start(self, monkeypatch):
        clock = {"t": 0.0}
        calls = []
        monkeypatch.setattr(Q, "release_due_jobs", lambda client, publish, now: calls.append("release"))
        monkeypatch.setattr(Q, "requeue_stranded_card_messages", lambda client, live: calls.append(("sweep", tuple(live))))
        tick = Q.maintenance_tick([TI_UUID], client_factory=FakeRedis, publish=lambda e: None,
                                  clock=lambda: clock["t"], wall_clock=lambda: 1.0)

        tick()
        assert calls == ["release", ("sweep", (TI_UUID,))]
        clock["t"] = 1.0
        tick()
        assert len(calls) == 2
        clock["t"] = 5.0
        tick()
        assert calls[2:] == ["release"]
        clock["t"] = 60.0
        tick()
        assert calls[3:] == ["release", ("sweep", (TI_UUID,))]
