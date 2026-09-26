"""`/task-queue/active` must not stall the event loop.

MIS-E2E-102, verified by benchmark (task 15.4). `_celery_view` builds an
`AsyncResult` and reads Celery's Redis result backend — synchronous I/O, up to
three round-trips per row — and it was called directly inside the `async def`
handler. The Monitor page polls this endpoint continuously, so every poll froze
the whole process for the duration.

Measured on the path that changed, not a neighbouring one, with a 50ms stand-in
for a Redis round-trip over 12 active rows:

    variant                      wall ms   worst loop stall ms
    before (sync in coroutine)     600.9                 601.1
    after  (to_thread + gather)     53.5                   0.4

The first benchmark I wrote reported a 0.1ms stall for BOTH variants, because
its probe coroutines were gathered before the blocking call and completed on
the first yield — they never overlapped the stall. A loop-blocking measurement
has to sample while the work is in flight.

REVIEW R3-D (2026-09-15). The watchdog class that measured this (a bare 10 ms
wall-clock budget) never called the real handler — its coroutine copied the
handler's shape — and a check that `asyncio.gather` appeared in the source was
satisfied by a comment. Both are gone. The behaviour is pinned on the REAL
`list_active_tasks` in `test_active_tasks_reads_stay_off_the_loop.py`: reads run
off the loop's thread, overlap, and stall the loop less than reading on it,
measured in the same run, so load cannot fail it. What remains here is the
structural check that `_celery_view` is dispatched to a thread.
"""

import ast
import inspect



class TestTheHandlerDoesNotCallItSynchronously:
    def test_celery_view_is_dispatched_to_a_thread(self):
        from src.api.v1.endpoints import task_queue

        source = inspect.getsource(task_queue.list_active_tasks)
        tree = ast.parse(inspect.cleandoc(source))

        direct = []
        threaded = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name == "to_thread":
                args = [getattr(a, "id", "") for a in node.args]
                if "_celery_view" in args:
                    threaded = True
            elif name == "_celery_view":
                direct.append(node.lineno)

        assert threaded, (
            "_celery_view is not dispatched with asyncio.to_thread; its "
            "synchronous Redis reads run on the event loop"
        )
        assert not direct, (
            f"_celery_view is ALSO called directly at {direct} — one blocking "
            f"call is enough to stall the loop"
        )
