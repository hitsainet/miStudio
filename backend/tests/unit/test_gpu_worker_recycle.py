"""A GPU worker recycles itself after a job that left it holding host memory (ENH-006 G14).

On 2026-11-04 an idle GPU worker held 19.2 GiB after one probe evaluation, and the next probe
run was refused by its host-memory forecast until someone restarted the worker inside the pod.
These tests pin the worker half: when to recycle, the marker the supervisor reads, the
shutdown request (its target and that it is sent once), and that the check is wired into the
live Celery signal registry. The supervisor half is in test_gpu_supervisor.py
(TestRecycleAfterAJob, and main()'s wiring).
"""

import json
import time
from types import SimpleNamespace

import pytest

from src.workers import gpu_worker_recycle as R

GIB = 1024**3


class _Task:
    """A stand-in task: Celery signals key their cache on the sender, so it must be hashable."""

    def __init__(self, hostname, name):
        self.name = name
        self.request = SimpleNamespace(hostname=hostname)


def _task(hostname="gpu-0-247aa582@node", name="probe_monitors.train"):
    return _Task(hostname, name)


class Shutdowns:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, hostname):
        self.calls.append(hostname)
        if self.fail:
            raise RuntimeError("broker down")


class TestThreshold:
    def test_default_is_four_gib(self):
        assert R.threshold_bytes({}) == 4 * GIB

    @pytest.mark.parametrize("raw", ["", "  ", "nope", "0", "-3"])
    def test_a_bad_value_falls_back_to_the_default(self, raw):
        assert R.threshold_bytes({R.RECYCLE_RSS_ENV: raw}) == 4 * GIB

    def test_a_set_value_is_used(self):
        assert R.threshold_bytes({R.RECYCLE_RSS_ENV: "8.5"}) == int(8.5 * GIB)

    def test_at_the_threshold_recycles_and_below_does_not(self):
        assert R.should_recycle(4 * GIB, 4 * GIB) is True
        assert R.should_recycle(4 * GIB - 1, 4 * GIB) is False
        assert R.should_recycle(None, 4 * GIB) is False


def test_resident_bytes_reads_anonymous_memory_not_the_whole_rss(tmp_path):
    """File-backed pages (memory-mapped token files) are page cache the kernel drops by itself."""
    status = tmp_path / "status"
    status.write_text(
        "Name:\tpython\nVmRSS:\t 30000000 kB\nRssAnon:\t 20185088 kB\nRssFile:\t 9814912 kB\n"
    )
    assert R.resident_bytes(str(status)) == 20185088 * 1024
    assert R.resident_bytes(str(tmp_path / "missing")) is None


class TestAfterATask:
    def env(self, tmp_path, **extra):
        return {R.RECYCLE_DIR_ENV: str(tmp_path / "recycle"), **extra}

    def test_a_worker_holding_memory_writes_its_marker_and_asks_once_for_its_own_shutdown(
        self, tmp_path
    ):
        shutdowns = Shutdowns()
        asked = R.on_task_postrun(
            _task(),
            environ=self.env(tmp_path),
            read_rss=lambda: 19 * GIB,
            request_shutdown=shutdowns,
            pid=777,
            clock=lambda: 1000.0,
        )
        assert asked is True
        assert shutdowns.calls == ["gpu-0-247aa582@node"]
        marker = json.loads(R.marker_path(tmp_path / "recycle", 777).read_text())
        assert marker == {
            "pid": 777,
            "hostname": "gpu-0-247aa582@node",
            "rss_bytes": 19 * GIB,
            "threshold_bytes": 4 * GIB,
            "task": "probe_monitors.train",
            "at": 1000.0,
        }

    def test_below_the_threshold_nothing_happens(self, tmp_path):
        shutdowns = Shutdowns()
        assert (
            R.on_task_postrun(
                _task(),
                environ=self.env(tmp_path),
                read_rss=lambda: 2 * GIB,
                request_shutdown=shutdowns,
                pid=777,
            )
            is False
        )
        assert shutdowns.calls == []
        assert not (tmp_path / "recycle").exists()

    def test_a_worker_the_supervisor_did_not_start_never_recycles(self, tmp_path):
        shutdowns = Shutdowns()
        assert (
            R.on_task_postrun(
                _task(), environ={}, read_rss=lambda: 50 * GIB, request_shutdown=shutdowns, pid=777
            )
            is False
        )
        assert shutdowns.calls == []

    def test_an_unknown_hostname_does_not_shut_down_some_other_worker(self, tmp_path):
        shutdowns = Shutdowns()
        assert (
            R.on_task_postrun(
                _task(hostname=None),
                environ=self.env(tmp_path),
                read_rss=lambda: 19 * GIB,
                request_shutdown=shutdowns,
                pid=777,
            )
            is False
        )
        assert shutdowns.calls == []

    def test_a_failed_request_leaves_no_marker_so_a_later_crash_is_a_crash(self, tmp_path):
        shutdowns = Shutdowns(fail=True)
        assert (
            R.on_task_postrun(
                _task(),
                environ=self.env(tmp_path),
                read_rss=lambda: 19 * GIB,
                request_shutdown=shutdowns,
                pid=777,
            )
            is False
        )
        assert shutdowns.calls == ["gpu-0-247aa582@node"]
        assert not R.marker_path(tmp_path / "recycle", 777).exists()


class TestTheSupervisorReadsTheMarker:
    def test_a_fresh_marker_from_the_worker_is_consumed_once(self, tmp_path):
        R.on_task_postrun(
            _task(),
            environ={R.RECYCLE_DIR_ENV: str(tmp_path)},
            read_rss=lambda: 19 * GIB,
            request_shutdown=Shutdowns(),
            pid=31,
            clock=lambda: 5000.0,
        )
        assert R.consume_marker(str(tmp_path), 31, now=5010.0) is True
        assert R.consume_marker(str(tmp_path), 31, now=5010.0) is False

    def test_a_stale_marker_is_not_a_recycle_and_is_removed(self, tmp_path):
        R.marker_path(tmp_path, 31).write_text(json.dumps({"at": 0.0}))
        assert R.consume_marker(str(tmp_path), 31, now=R.MARKER_MAX_AGE_S + 1.0) is False
        assert not R.marker_path(tmp_path, 31).exists()

    @pytest.mark.parametrize("body", ["not json", json.dumps({"at": "yesterday"}), json.dumps([1])])
    def test_a_garbled_marker_is_not_a_recycle(self, tmp_path, body):
        R.marker_path(tmp_path, 31).write_text(body)
        assert R.consume_marker(str(tmp_path), 31, now=time.time()) is False

    def test_another_pids_marker_does_not_count(self, tmp_path):
        R.marker_path(tmp_path, 31).write_text(json.dumps({"at": time.time()}))
        assert R.consume_marker(str(tmp_path), 32) is False
        assert R.consume_marker(None, 31) is False


class TestWiredIntoTheLiveRegistry:
    def test_task_postrun_carries_exactly_one_recycle_receiver(self):
        from celery.signals import task_postrun

        import src.core.celery_app  # noqa: F401 - importing it is what connects the signals

        keys = [
            key for key, _ref in task_postrun.receivers if key[0] == "mistudio-gpu-worker-recycle"
        ]
        assert len(keys) == 1, keys

    def test_sending_task_postrun_reaches_the_recycle_check(self, tmp_path, monkeypatch):
        from celery.signals import task_postrun

        import src.core.celery_app  # noqa: F401

        calls = []
        monkeypatch.setattr(R, "on_task_postrun", lambda sender=None, **kw: calls.append(sender))
        sender = _task()
        task_postrun.send(
            sender=sender,
            task_id="t1",
            task=sender,
            args=(),
            kwargs={},
            retval=None,
            state="SUCCESS",
        )
        assert calls == [sender]

    def test_a_failing_check_never_raises_into_the_task(self, monkeypatch):
        def boom(sender=None, **kw):
            raise RuntimeError("bad")

        monkeypatch.setattr(R, "on_task_postrun", boom)
        R._postrun_receiver(_task())


def test_the_real_request_targets_this_worker_only(tmp_path, monkeypatch):
    """The default request goes through Celery's control API, addressed to this worker alone.

    A broadcast with no destination would shut down every worker on the broker.
    """
    from src.core.celery_app import celery_app

    sent = []
    monkeypatch.setattr(celery_app.control, "shutdown", lambda *a, **kw: sent.append((a, kw)))
    assert (
        R.on_task_postrun(
            _task(), environ={R.RECYCLE_DIR_ENV: str(tmp_path)}, read_rss=lambda: 19 * GIB, pid=9
        )
        is True
    )
    assert sent == [((), {"destination": ["gpu-0-247aa582@node"]})]
