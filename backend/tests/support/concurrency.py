"""FR-11 helper: two behaviours evaluated concurrently give the bytes they give in sequence (Feature 040).

No code path may hold a process-wide "current behaviour" (spec D24). The cheapest way such a thing shows is
a module-level cache or global keyed WITHOUT the behaviour: run alone it is invisible, run interleaved with
a second behaviour it serves one behaviour's answer to the other. `assert_concurrent_equals_sequential`
runs `fn(item)` for each item alone, then all items interleaved on threads many times, and requires the
same bytes.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable, Sequence


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, default=str).encode()


def assert_concurrent_equals_sequential(fn: Callable[[Any], Any], items: Sequence[Any], rounds: int = 50) -> None:
    expected = [_bytes(fn(item)) for item in items]
    failures: list = []
    barrier = threading.Barrier(len(items))

    def worker(index: int) -> None:
        barrier.wait()
        for _ in range(rounds):
            got = _bytes(fn(items[index]))
            if got != expected[index]:
                failures.append((index, got[:200]))
                return

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(items))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures, f"interleaved results differ from sequential ones: {failures[:2]}"
