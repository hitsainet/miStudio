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
has to sample while the work is in flight; that is what the watchdog below does.
"""

import ast
import asyncio
import inspect
import time

import pytest


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

    def test_the_reads_are_overlapped_not_sequential(self):
        from src.api.v1.endpoints import task_queue

        source = inspect.getsource(task_queue.list_active_tasks)
        assert "asyncio.gather" in source, (
            "the per-row reads are dispatched but not overlapped, so the cost "
            "is still N sequential round-trips of latency"
        )


class TestTheLoopStaysResponsive:
    """The behavioural claim, measured rather than asserted structurally."""

    ROWS = 8
    SLOW_S = 0.02

    @staticmethod
    def _slow(_):
        time.sleep(TestTheLoopStaysResponsive.SLOW_S)
        return False, {}

    async def _worst_stall(self, work) -> float:
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
    async def test_threaded_reads_do_not_stall_the_loop(self):
        async def fixed():
            await asyncio.gather(
                *(asyncio.to_thread(self._slow, i) for i in range(self.ROWS))
            )

        stall = await self._worst_stall(fixed)
        budget = self.SLOW_S * 1000 / 2
        assert stall < budget, (
            f"the loop stalled {stall:.1f}ms; the reads are back on it"
        )

    @pytest.mark.asyncio
    async def test_the_probe_can_detect_a_stall(self):
        """Negative control in-test: the sequential shape MUST measure badly.

        Without this the test above passes on a machine where the watchdog
        never fires, and proves nothing.
        """
        async def blocking():
            for i in range(self.ROWS):
                self._slow(i)

        stall = await self._worst_stall(blocking)
        assert stall > self.SLOW_S * 1000, (
            f"the sequential version stalled only {stall:.1f}ms — the probe "
            f"cannot see a stall, so the assertion above is vacuous"
        )
