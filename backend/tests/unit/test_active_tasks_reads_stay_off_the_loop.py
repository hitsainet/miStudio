"""`/task-queue/active` reads Celery off the event loop, pinned on the REAL handler, with no bare wall-clock budget.

A load-robust proposal from review R3-D (2026-09-15) for `test_active_does_not_block_the_loop.py`. Whether
it replaces that file's `TestTheLoopStaysResponsive` is the integrator's call.

WHY. `TestTheLoopStaysResponsive` failed once in the round-2 full run while vitest and a production build
shared the CPU (a 17 ms stall against its 10 ms budget), then passed 4/4 twice alone. Two things are wrong
with it, and only the first is the flake:

1. It asserts a bare wall-clock bound, `stall < SLOW_S * 1000 / 2` = 10 ms, on the event loop's
   scheduling latency. A loaded runner exceeds 10 ms with nothing wrong (memory
   `check-ci-not-just-local-green`).
2. It never calls the handler. Its "fixed" coroutine is a copy of the handler's shape, so it measures
   asyncio, not `list_active_tasks`. Putting `_celery_view` back on the loop inside the handler leaves that
   class green; only the AST test beside it notices. That AST test's overlap half is a source substring
   (`"asyncio.gather" in source`), which a comment satisfies.

WHAT THIS FILE ASSERTS, on the real `list_active_tasks` with its database and Celery reads stood in:

- every row's Celery read runs exactly once, and none on the event loop's thread. This is thread identity,
  with no timing in it;
- the reads overlap: a second read is in flight while the first one is. The first read waits on an event
  with a generous timeout, which only a sequential handler waits out;
- the loop's worst stall while the handler runs is under half the stall of the defect's own shape, both
  measured in the same test. Load lengthens the blocking shape's stall and never shortens it, so load can
  fail this only if the loop's own jitter exceeds about 200 ms.

MUTATION CONTROLS: in the R3-D record, `review_sae_remediation_R3_D_2026-09-15.md`.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from src.api.v1.endpoints import task_queue

ROWS = 8
#: How long the first read waits for a second to start. A handler that overlaps its reads ends the wait
#: within milliseconds; only a sequential one waits it out, so it can be generous.
OVERLAP_WAIT_S = 5.0
#: A stand-in round-trip. Eight of them on the loop stall it at least 400 ms, whatever the load.
SLOW_S = 0.05

_FEDERATED = (
    "_federated_trainings",
    "_federated_extractions",
    "_federated_labeling",
    "_federated_pushes",
    "_federated_tokenizations",
    "_federated_activation_extractions",
)


@pytest.fixture
def active_rows(monkeypatch):
    """Eight running task-queue rows, and every read the handler makes besides Celery stood in."""
    rows = [
        SimpleNamespace(
            id=f"tq_{i}",
            task_id=f"celery-{i}",
            status="running",
            entity_id=f"entity_{i}",
            entity_type="model",
        )
        for i in range(ROWS)
    ]

    async def get_active_tasks(db):
        return list(rows)

    async def get_entity_info(db, entity_id, entity_type):
        return {"name": entity_id}

    async def no_rows(db, statuses, limit=25):
        return []

    monkeypatch.setattr(task_queue.TaskQueueService, "get_active_tasks", staticmethod(get_active_tasks))
    monkeypatch.setattr(task_queue.TaskQueueService, "get_entity_info", staticmethod(get_entity_info))
    monkeypatch.setattr(
        task_queue,
        "_serialize_task",
        lambda task, entity_info, can_retry=True: {
            "id": task.id,
            "entity_info": entity_info,
            "created_at": "",
        },
    )
    for name in _FEDERATED:
        monkeypatch.setattr(task_queue, name, no_rows)
    return rows


async def _call_handler():
    return await task_queue.list_active_tasks(db=object())


async def _worst_stall_ms(work) -> float:
    """The longest the loop was unable to run a 2 ms sleep while ``work`` was in flight."""
    stop = asyncio.Event()
    samples: list = []

    async def watchdog():
        while not stop.is_set():
            t0 = time.perf_counter()
            await asyncio.sleep(0.002)
            samples.append((time.perf_counter() - t0 - 0.002) * 1000)

    watcher = asyncio.create_task(watchdog())
    await asyncio.sleep(0.01)
    await work()
    stop.set()
    await watcher
    return max([s for s in samples if s > 0] or [0.0])


@pytest.mark.asyncio
async def test_each_rows_celery_read_runs_once_and_never_on_the_event_loop_thread(active_rows, monkeypatch):
    loop_thread = threading.get_ident()
    reads = []
    lock = threading.Lock()

    def view(task):
        with lock:
            reads.append((task.id, threading.get_ident()))
        return False, {}

    monkeypatch.setattr(task_queue, "_celery_view", view)
    result = await _call_handler()

    assert sorted(task_id for task_id, _ in reads) == sorted(row.id for row in active_rows), (
        f"each active row must be read from Celery exactly once; the reads were {reads}"
    )
    on_the_loop = [task_id for task_id, thread in reads if thread == loop_thread]
    assert not on_the_loop, (
        f"the Celery reads for {on_the_loop} ran on the event loop's own thread: every other request in "
        f"the process waits for them"
    )
    assert sorted(row["id"] for row in result["data"]) == sorted(row.id for row in active_rows)


@pytest.mark.asyncio
async def test_the_reads_overlap_rather_than_run_one_after_another(active_rows, monkeypatch):
    lock = threading.Lock()
    overlapped = threading.Event()
    state = {"in_flight": 0, "first_taken": False}

    def view(task):
        with lock:
            state["in_flight"] += 1
            if state["in_flight"] >= 2:
                overlapped.set()
            wait_here = not state["first_taken"]
            state["first_taken"] = True
        try:
            if wait_here:
                overlapped.wait(OVERLAP_WAIT_S)
        finally:
            with lock:
                state["in_flight"] -= 1
        return False, {}

    monkeypatch.setattr(task_queue, "_celery_view", view)
    await _call_handler()

    assert overlapped.is_set(), (
        f"no second Celery read started while the first was in flight ({OVERLAP_WAIT_S:.0f} s): the reads "
        f"run one after another, so the listing costs N round-trips of latency instead of one"
    )


@pytest.mark.asyncio
async def test_the_handler_stalls_the_loop_far_less_than_reading_on_it_would(active_rows, monkeypatch):
    def slow_view(task):
        time.sleep(SLOW_S)
        return False, {}

    monkeypatch.setattr(task_queue, "_celery_view", slow_view)

    handler_stall = await _worst_stall_ms(_call_handler)

    async def reads_on_the_loop():
        """The defect's own shape, measured in the same run: the calibration and the vacuity check."""
        for row in active_rows:
            slow_view(row)

    blocking_stall = await _worst_stall_ms(reads_on_the_loop)

    floor_ms = ROWS * SLOW_S * 1000
    assert blocking_stall >= 0.9 * floor_ms, (
        f"reading on the loop stalled it only {blocking_stall:.1f} ms against a {floor_ms:.0f} ms floor: the "
        f"probe cannot see a stall, so the assertion below would be vacuous"
    )
    assert handler_stall < blocking_stall / 2, (
        f"the handler stalled the loop {handler_stall:.1f} ms, not far below the {blocking_stall:.1f} ms of "
        f"reading on it: the Celery reads are back on the loop"
    )
