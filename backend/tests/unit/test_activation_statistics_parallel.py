"""The statistics phase must use the cores it has, and must not go silent.

Measured on a live job, 2026-09-12: a 10,000-sample extraction of
LFM2.5-1.2B-Instruct (2,048 tokens x 2,048 dims, five layers) finished its GPU
pass at 12:37 and then spent 17 minutes PER LAYER computing statistics on one
core of a 16-core node — 85 minutes, longer than the GPU pass itself.

Two defects, one phase:

  * SERIAL. Every chunk's statistics are independent sums, extremes and
    counts, which merge exactly, and NumPy releases the GIL inside each
    reduction. On the node: 3.8x at 4 threads, 6.5x at 12, results
    bit-identical. Disk (≈544 MB/s with parallel readers) is the next limit.
  * SILENT. Nothing wrote the extraction row during the phase. The janitor
    judges a PENDING task by the row's age, and was one sweep from failing that
    healthy job over valid output. It was kept alive by hand.
"""

import threading
import time
import types

import numpy as np
import pytest

import _cancel_ast as A
from src.services.activation_service import ActivationService

GB = 1024 ** 3


def _svc():
    return ActivationService.__new__(ActivationService)


def _array(n=23, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 3, size=(n, 16, 12)).astype(np.float16)


class TestParallelIsExact:
    @pytest.mark.parametrize("workers", [1, 2, 5])
    def test_any_worker_count_gives_the_direct_answer(self, workers):
        svc = _svc()
        array = _array()  # 23 rows at chunk 4 -> 6 chunks, the last one partial

        direct = svc._direct_statistics(array)
        chunked = svc._chunked_statistics(array, chunk_size=4, workers=workers)

        for key in direct:
            assert chunked[key] == pytest.approx(direct[key], rel=1e-9, abs=1e-12), key

    def test_the_final_partial_chunk_is_counted(self):
        svc = _svc()
        array = _array()
        array[-1] = 60000.0  # only in the 3-row remainder chunk

        stats = svc._chunked_statistics(array, chunk_size=4, workers=3)

        assert stats["max_activation"] == pytest.approx(60000.0), (
            "the trailing partial chunk was never reduced"
        )

    def test_extremes_compose_across_chunks_in_either_order(self):
        svc = _svc()
        array = _array()
        array[0, 0, 0] = -30000.0   # first chunk
        array[21, 0, 0] = 30000.0   # last chunk

        stats = svc._chunked_statistics(array, chunk_size=4, workers=4)

        assert stats["min_activation"] == pytest.approx(-30000.0)
        assert stats["max_activation"] == pytest.approx(30000.0)


class TestItReallyRunsConcurrently:
    def test_two_chunks_are_in_flight_at_the_same_time(self, monkeypatch):
        """A barrier that only two concurrent callers can pass. Run serially,
        the first chunk waits alone, the barrier times out, and this fails."""
        barrier = threading.Barrier(2, timeout=5)
        real = ActivationService._chunk_accumulators

        def gated(chunk, near_zero):
            barrier.wait()
            return real(chunk, near_zero)

        monkeypatch.setattr(ActivationService, "_chunk_accumulators", staticmethod(gated))

        stats = _svc()._chunked_statistics(_array(8), chunk_size=4, workers=2)

        assert stats["mean_magnitude"] is not None


class TestProgressAndCancellation:
    def test_every_chunk_is_reported_in_order_on_the_calling_thread(self):
        calls = []
        main = threading.get_ident()

        _svc()._chunked_statistics(
            _array(23), chunk_size=4, workers=3,
            on_chunk=lambda done, total: calls.append((done, total, threading.get_ident() == main)),
        )

        assert calls == [(i, 6, True) for i in range(1, 7)], (
            "progress must arrive once per chunk, in order, on the thread that "
            "owns the database session"
        )

    def test_a_raise_from_the_callback_abandons_the_queued_chunks(self, monkeypatch):
        reduced = []
        real = ActivationService._chunk_accumulators

        def counting(chunk, near_zero):
            reduced.append(1)
            time.sleep(0.01)
            return real(chunk, near_zero)

        monkeypatch.setattr(ActivationService, "_chunk_accumulators", staticmethod(counting))

        class Stop(BaseException):
            pass

        def stop_at_two(done, total):
            if done == 2:
                raise Stop()

        with pytest.raises(Stop):
            _svc()._chunked_statistics(
                _array(200), chunk_size=2, workers=2, on_chunk=stop_at_two
            )  # 100 chunks

        assert len(reduced) <= 10, (
            f"{len(reduced)} of 100 chunks ran after a cancel at chunk 2 — the "
            f"queued work was not abandoned, so Stop would wait out the layer"
        )


class TestWorkerCount:
    @pytest.mark.parametrize(
        "cpu, mem, chunk, expected",
        [
            (16, 1000 * GB, GB, 12),   # capped at the measured ceiling
            (4, 1000 * GB, GB, 2),     # two cores left for the pod
            (1, 1000 * GB, GB, 1),
            (16, 30 * GB, GB, 10),     # 3x each chunk's size in memory
            (16, None, GB, 4),         # unknown memory: small, not large
            (0, 0, GB, 1),             # never zero
        ],
    )
    def test_bounded_by_cores_memory_and_the_ceiling(self, cpu, mem, chunk, expected):
        assert ActivationService.statistics_worker_count(cpu, mem, chunk) == expected


class TestTheWiringReachesTheLiveJob:
    def test_large_layers_take_the_parallel_path_and_report_progress(self, monkeypatch):
        monkeypatch.setattr(ActivationService, "CHUNKED_STATISTICS_THRESHOLD_BYTES", 0)
        calls = []

        stats = _svc()._calculate_statistics(
            {"layer_a": _array(10), "layer_b": _array(6)},
            on_progress=lambda *args: calls.append(args),
        )

        assert set(stats) == {"layer_a", "layer_b"}
        assert {c[0] for c in calls} == {0, 1}, "a layer reported no progress"
        assert all(c[1] == 2 for c in calls)
        for layer in (0, 1):
            last = [c for c in calls if c[0] == layer][-1]
            assert last[2] == last[3], "a layer's progress never reached its last chunk"

    def test_extract_activations_hands_the_callback_to_the_statistics(self):
        calls = A.calls_named(ActivationService.extract_activations, "_calculate_statistics")
        assert calls, "extract_activations no longer computes statistics"
        passed = [A.keyword_of(c, "on_progress") for c in calls]
        assert any(getattr(v, "id", None) == "statistics_progress_callback" for v in passed), (
            "the statistics phase is not given the heartbeat, so it runs silent"
        )

    def test_the_task_builds_the_heartbeat_and_passes_it(self):
        from src.workers.model_tasks import extract_activations as task

        assert A.calls_named(task, "build_statistics_heartbeat"), (
            "the task never builds the statistics heartbeat"
        )
        assert A.passes_real_value(task, "extract_activations", "statistics_progress_callback"), (
            "the heartbeat is built and never handed to the extraction"
        )


class TestStatisticsHeartbeat:
    def _build(self, monkeypatch, times, cancel=None):
        from src.workers import model_tasks as MT

        writes, polls = [], []
        monkeypatch.setattr(
            MT, "record_progress", lambda kind, target, **kw: writes.append((kind, target, kw)) or True
        )
        monkeypatch.setattr(MT, "emit_extraction_progress", lambda **kw: None)

        def raise_if_cancelled(detail=""):
            polls.append(detail)
            if cancel is not None:
                raise cancel

        clock = iter(times).__next__
        cb = MT.build_statistics_heartbeat(
            "m_1", "ext_1", types.SimpleNamespace(raise_if_cancelled=raise_if_cancelled),
            clock=clock, interval_s=60,
        )
        return cb, writes, polls

    def test_the_first_call_touches_the_row(self, monkeypatch):
        cb, writes, _ = self._build(monkeypatch, [1000.0])

        cb(0, 5, 1, 100)

        assert len(writes) == 1
        kind, target, fields = writes[0]
        assert (kind, target) == ("activation_extraction", "ext_1")
        assert "updated_at" in fields, (
            "the write must move updated_at — that is the clock the janitor reads"
        )

    def test_writes_are_throttled_on_time(self, monkeypatch):
        cb, writes, _ = self._build(monkeypatch, [0.0, 10.0, 59.0, 61.0, 90.0, 125.0])

        for i in range(6):
            cb(0, 5, i + 1, 100)

        assert len(writes) == 3, "expected writes at t=0, t=61 and t=125 only"

    def test_cancellation_is_polled_on_every_call(self, monkeypatch):
        cb, _, polls = self._build(monkeypatch, [0.0, 1.0, 2.0])

        for i in range(3):
            cb(2, 5, i + 1, 100)

        assert len(polls) == 3, "the throttle must not delay a Stop"

    def test_a_cancel_stops_before_anything_is_written(self, monkeypatch):
        class Cancelled(BaseException):
            pass

        cb, writes, _ = self._build(monkeypatch, [0.0], cancel=Cancelled())

        with pytest.raises(Cancelled):
            cb(0, 5, 1, 100)
        assert writes == []
