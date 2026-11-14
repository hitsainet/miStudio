"""A redelivery of a task id that already completed does nothing; a second request runs (ENH-006 D116).

An evaluation, a re-cut or a score has no row to say it ended. Real Redis, the real task and the real wrapper: only the
service the task would call is a spy, so "does nothing" is a count. Driven from the registry for coverage, since the
stamp lives in `@gpu_job` and a new GPU task inherits it.

Controls: `0xcc/reviews/mcp_gaps_controls_2026-11-02.md`.
"""
from __future__ import annotations

import inspect
import json
import uuid
from unittest.mock import MagicMock

import pytest
from celery.exceptions import Ignore

from src.core.celery_app import celery_app
from src.services import probe_monitor_run as run_module
from src.services.gpu_dispatch import GPU_JOB_MARKER
from src.services.gpu_job_claim import JobHandoff
from src.workers import probe_monitor_tasks as tasks
from src.workers import task_completion
from tests.support.real_broker import real_broker  # noqa: F401 - a fixture

celery_app.loader.import_default_modules()


@pytest.fixture
def stamps(real_broker, monkeypatch):  # noqa: F811
    client, _url = real_broker
    monkeypatch.setattr(task_completion, "_client", lambda: client)
    monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: False)
    return client


def deliver(task, task_id, **kwargs):
    """One delivery, as a worker makes it: the request carries the id, and the task is called."""
    task.push_request(id=task_id, delivery_info={"redelivered": False})
    try:
        return task.run(**kwargs)
    finally:
        task.pop_request()


@pytest.fixture
def evaluate(monkeypatch, async_session):
    spy = MagicMock(return_value={"evaluated": 1})
    monkeypatch.setattr(run_module, "evaluate_probe", spy)
    monkeypatch.setattr(tasks, "_require_row", lambda *a, **k: MagicMock())
    return spy


class TestARedelivery:
    def test_the_second_delivery_of_an_id_does_nothing_and_stores_no_result(self, stamps, evaluate):
        task_id = str(uuid.uuid4())
        assert deliver(tasks.evaluate_probe_monitor, task_id, probe_id="pm", dataset_ids=["d"]) == {"evaluated": 1}
        with pytest.raises(Ignore):
            deliver(tasks.evaluate_probe_monitor, task_id, probe_id="pm", dataset_ids=["d"])
        assert evaluate.call_count == 1

    def test_a_deliberate_second_request_is_a_new_id_and_runs(self, stamps, evaluate):
        deliver(tasks.evaluate_probe_monitor, str(uuid.uuid4()), probe_id="pm", dataset_ids=["d"])
        deliver(tasks.evaluate_probe_monitor, str(uuid.uuid4()), probe_id="pm", dataset_ids=["d"])
        assert evaluate.call_count == 2

    def test_the_stamp_is_what_the_watch_and_the_next_delivery_read(self, stamps, evaluate):
        task_id = str(uuid.uuid4())
        deliver(tasks.evaluate_probe_monitor, task_id, probe_id="pm", dataset_ids=["d"])
        stamp = json.loads(stamps.get(task_completion.DONE_PREFIX + task_id))
        assert stamp["task"] == "src.workers.probe_monitor_tasks.evaluate_probe_monitor" and stamp["completed_at"]
        assert stamps.ttl(task_completion.DONE_PREFIX + task_id) > 43_200, "it must outlive the visibility timeout"


class TestWhatIsNotStamped:
    def test_a_failure_is_not_a_completion(self, stamps, monkeypatch, evaluate):
        evaluate.side_effect = RuntimeError("boom")
        task_id = str(uuid.uuid4())
        with pytest.raises(RuntimeError):
            deliver(tasks.evaluate_probe_monitor, task_id, probe_id="pm", dataset_ids=["d"])
        assert task_completion.completed_at(task_id) is None

    def test_a_hand_off_keeps_its_id_and_runs_when_it_comes_back(self, stamps, monkeypatch, evaluate):
        """A hand-off republishes the SAME id. Stamped, the job would vanish at its first busy card."""
        monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: True)
        monkeypatch.setattr("src.workers.gpu_job.requeue", MagicMock())
        import contextlib

        @contextlib.contextmanager
        def handing_off(ctx):
            raise JobHandoff("gpu.auto", "the card is busy")
            yield  # pragma: no cover

        monkeypatch.setattr("src.workers.gpu_job.claiming", handing_off)
        task_id = str(uuid.uuid4())
        with pytest.raises(Ignore):
            deliver(tasks.evaluate_probe_monitor, task_id, probe_id="pm", dataset_ids=["d"])
        assert task_completion.completed_at(task_id) is None

        monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: False)
        assert deliver(tasks.evaluate_probe_monitor, task_id, probe_id="pm", dataset_ids=["d"]) == {"evaluated": 1}
        assert evaluate.call_count == 1


class TestAnUnreadableBroker:
    def test_the_task_runs_as_it_always_did(self, monkeypatch, evaluate):
        monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: False)

        def no_broker():
            raise ConnectionError("down")

        monkeypatch.setattr(task_completion, "_client", no_broker)
        assert deliver(tasks.evaluate_probe_monitor, str(uuid.uuid4()), probe_id="pm", dataset_ids=["d"]) == {"evaluated": 1}


@pytest.mark.parametrize("name", sorted(n for n, t in celery_app.tasks.items() if hasattr(t.run, GPU_JOB_MARKER)))
def test_every_gpu_task_in_the_live_registry_inherits_it(name, monkeypatch):
    """A completed id reaches neither the row guard's answer nor the body."""
    monkeypatch.setattr(task_completion, "completed_at", lambda task_id, **k: "2026-11-02T00:00:00+00:00")
    monkeypatch.setattr("src.core.cancellation.terminal_status_of", lambda kind, target: None)
    body = MagicMock(side_effect=AssertionError("it ran"))
    monkeypatch.setattr("src.workers.gpu_job.per_card_workers", body)
    task = celery_app.tasks[name]
    kwargs = {p.name: "x" for p in inspect.signature(task.run).parameters.values()
              if p.default is inspect._empty and p.name != "self"}
    task.push_request(id="done-id")
    try:
        with pytest.raises(Ignore):
            task.run(**kwargs)
    finally:
        task.pop_request()
    body.assert_not_called()
