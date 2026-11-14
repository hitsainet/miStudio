"""Each GPU worker's held memory is published so the API's forecast can credit it (ENH-006 G14).

On 2026-11-04 the request-time forecast refused a run against 54.79 GiB usable while the worker
that would run it held 19.2 GiB of heap it then reused. The API runs in another container and
cannot read the workers' processes, so the supervisor publishes each worker's ANONYMOUS memory to
the shared volume and the forecast reads it. Forecast-side tests are in
test_probe_run_memory_forecast.py (TestTheWorkerItRunsInIsCredited, and the route).
"""

import json

import pytest

from src.services import gpu_worker_memory as W
from src.workers import gpu_supervisor as S
from tests.unit.test_gpu_supervisor import RTX, Clock, Launcher, forever, run, stop_at

GIB = 1024**3


def test_anon_bytes_reads_rssanon_for_a_pid(tmp_path):
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "status").write_text(
        "VmRSS:\t 9000 kB\nRssAnon:\t 5000 kB\nRssFile:\t 4000 kB\n"
    )
    assert W.anon_bytes(42, proc=str(tmp_path)) == 5000 * 1024
    assert W.anon_bytes(43, proc=str(tmp_path)) is None


class Reader:
    def __init__(self, figures):
        self.figures, self.asked = figures, []

    def __call__(self, pid):
        self.asked.append(pid)
        return self.figures.get(pid)


def _status(card, pid, state="running"):
    return {"name": f"gpu-{card}", "card": card, "state": state, "pid": pid}


class TestPublishing:
    def test_each_running_worker_is_written_with_its_card_pid_and_figure(self, tmp_path):
        reader = Reader({11: 19 * GIB, 12: 2 * GIB})
        W.WorkerMemoryPublisher(str(tmp_path), read=reader, clock=lambda: 100.0)(
            [_status("GPU-a", 11), _status("GPU-b", 12)]
        )
        assert reader.asked == [11, 12]
        assert json.loads(W.entry_path(tmp_path, "GPU-a").read_text()) == {
            "card": "GPU-a",
            "pid": 11,
            "anon_bytes": 19 * GIB,
            "at": 100.0,
        }
        assert json.loads(W.entry_path(tmp_path, "GPU-b").read_text())["anon_bytes"] == 2 * GIB

    def test_a_worker_that_is_not_running_has_no_entry(self, tmp_path):
        W.entry_path(tmp_path, "GPU-a").write_text(
            json.dumps({"card": "GPU-a", "anon_bytes": 1, "at": 1.0})
        )
        W.WorkerMemoryPublisher(str(tmp_path), read=Reader({}), clock=lambda: 100.0)(
            [{"name": "gpu-a", "card": "GPU-a", "state": "restarting", "pid": None}]
        )
        assert not W.entry_path(tmp_path, "GPU-a").exists()

    def test_it_writes_at_most_every_few_seconds(self, tmp_path):
        clock = [100.0]
        reader = Reader({11: GIB})
        publish = W.WorkerMemoryPublisher(
            str(tmp_path), read=reader, clock=lambda: clock[0], every_s=10.0
        )
        publish([_status("GPU-a", 11)])
        clock[0] = 105.0
        publish([_status("GPU-a", 11)])
        clock[0] = 110.0
        publish([_status("GPU-a", 11)])
        assert reader.asked == [11, 11]

    def test_the_no_card_worker_has_its_own_entry(self, tmp_path):
        W.WorkerMemoryPublisher(str(tmp_path), read=Reader({7: GIB}), clock=lambda: 1.0)(
            [_status(None, 7)]
        )
        assert W.entry_path(tmp_path, None).name == "none.json"
        assert json.loads(W.entry_path(tmp_path, None).read_text())["anon_bytes"] == GIB


class TestReading:
    def write(self, directory, card, gib, at):
        W.entry_path(directory, card).write_text(
            json.dumps({"card": card, "pid": 1, "anon_bytes": gib * GIB, "at": at})
        )

    def test_a_named_card_reads_its_own_worker(self, tmp_path):
        self.write(tmp_path, "GPU-a", 19, 100.0)
        self.write(tmp_path, "GPU-b", 2, 100.0)
        assert W.held_bytes(str(tmp_path), "GPU-a", now=110.0) == 19 * GIB
        assert W.held_bytes(str(tmp_path), "GPU-c", now=110.0) == 0

    @pytest.mark.parametrize("gpu", ["auto", None, "", "0"])
    def test_a_job_any_card_may_take_is_credited_the_least(self, tmp_path, gpu):
        self.write(tmp_path, "GPU-a", 19, 100.0)
        self.write(tmp_path, "GPU-b", 2, 100.0)
        assert W.held_bytes(str(tmp_path), gpu, now=110.0) == 2 * GIB

    def test_a_stale_figure_is_ignored(self, tmp_path):
        self.write(tmp_path, "GPU-a", 19, 100.0)
        assert W.held_bytes(str(tmp_path), "GPU-a", now=100.0 + W.FRESH_S + 1) == 0
        assert W.held_bytes(str(tmp_path), "auto", now=100.0 + W.FRESH_S + 1) == 0

    def test_garbage_and_missing_directories_credit_nothing(self, tmp_path):
        (tmp_path / "GPU-a.json").write_text("not json")
        assert W.held_bytes(str(tmp_path), "GPU-a", now=1.0) == 0
        assert W.held_bytes(str(tmp_path / "missing"), "auto") == 0
        assert W.held_bytes(None, "auto") == 0

    def test_what_the_publisher_writes_is_what_the_reader_reads(self, tmp_path):
        W.WorkerMemoryPublisher(str(tmp_path), read=Reader({11: 19 * GIB}), clock=lambda: 500.0)(
            [_status("GPU-a", 11)]
        )
        assert W.held_bytes(str(tmp_path), "GPU-a", now=510.0) == 19 * GIB


class TestTheSupervisorPublishes:
    def test_it_hands_every_poll_the_workers_status(self, restore_signals):
        clock = Clock()
        launcher = Launcher(clock, {RTX.uuid: forever(clock)})
        seen = []
        run(
            launcher,
            clock,
            S.worker_specs([RTX], python="py"),
            on_worker_memory=lambda statuses: seen.append(statuses),
            on_tick=stop_at(clock, 5.0),
        )
        assert seen, "the supervisor never published"
        assert all(
            s[0]["card"] == RTX.uuid and s[0]["state"] == "running" and s[0]["pid"] for s in seen
        )

    def test_a_failing_publisher_never_stops_the_workers(self, restore_signals):
        clock = Clock()
        launcher = Launcher(clock, {RTX.uuid: forever(clock)})

        def boom(statuses):
            raise OSError("volume gone")

        assert (
            run(
                launcher,
                clock,
                S.worker_specs([RTX], python="py"),
                on_worker_memory=boom,
                on_tick=stop_at(clock, 5.0),
            )
            == 0
        )
        assert launcher.starts(RTX.uuid) == [0.0]


@pytest.fixture
def restore_signals():
    import signal

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)
