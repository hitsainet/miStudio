"""A GPU job is queued where the worker that should run it takes it (multi-GPU Phase 3).

``services/gpu_dispatch.gpu_queue_for`` is the one routing decision: a job naming
a card goes to ``gpu.<uuid>``; everything else to ``gpu.auto``.
``dispatch_gpu_task`` applies it in per-card mode to tasks marked for the GPU
queues and, otherwise, makes the very call the site made before Phase 3.

MUTATION CONTROLS (2026-09-14; each alone, restored byte-identically, sha256 checked,
`git diff` clean) — both red:
  D1 a named card is never given its own queue       -> four routing/helper tests
  D2 per-card mode never chooses the queue           -> four helper tests
The site-level bypasses (D3–D6) are recorded in test_gpu_job_wiring.py.
"""

from types import SimpleNamespace

import pytest

from src.core.config import settings
from src.services import gpu_dispatch as D
from src.services.gpu_claim import AUTO_QUEUE, queue_for
from src.workers.gpu_job import GpuJobSpec

RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"


def _marked(gpu_queue):
    def run():
        return None

    setattr(run, D.GPU_JOB_MARKER, GpuJobSpec(kind="k", handoff=gpu_queue, gpu_queue=gpu_queue))
    return run


class FakeTask:
    def __init__(self, gpu_queue=True):
        self.applied = []
        self.delayed = []
        self.run = _marked(gpu_queue)

    def apply_async(self, args=None, kwargs=None, **options):
        self.applied.append((args, kwargs, options))
        return SimpleNamespace(id=options.get("task_id", "generated"))

    def delay(self, *args, **kwargs):
        self.delayed.append((args, kwargs))
        return SimpleNamespace(id="generated")


@pytest.fixture
def per_card(monkeypatch):
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")


@pytest.fixture
def single(monkeypatch):
    monkeypatch.setattr(settings, "gpu_worker_mode", "single")


class TestTheRoutingDecision:
    def test_a_named_card_goes_to_its_own_queue(self):
        assert D.gpu_queue_for(RTX_UUID) == queue_for(RTX_UUID) == "gpu.gpu-247aa582-0d1b-e161-8156-983ed1fefc57"
        assert D.gpu_queue_for(RTX_UUID.lower()) == D.gpu_queue_for(f"  {RTX_UUID}  ")

    @pytest.mark.parametrize("requested", [None, "", "auto", "AUTO", "all", "ALL"])
    def test_auto_and_all_go_to_the_shared_queue(self, requested):
        assert D.gpu_queue_for(requested) == AUTO_QUEUE

    @pytest.mark.parametrize("requested", ["1", 0, "cuda:0", "GPU-", "gpu-zz", "3090"])
    def test_anything_that_is_not_a_card_uuid_never_names_a_queue(self, requested):
        """An unresolved index would name `gpu.gpu-1`, which nothing consumes."""
        assert D.gpu_queue_for(requested) == AUTO_QUEUE


class TestTheHelper:
    def test_per_card_mode_queues_a_named_card_job_on_its_card(self, per_card):
        task = FakeTask()
        D.dispatch_gpu_task(task, gpu_request=RTX_UUID, args=["run-1"], kwargs={"k": 1}, task_id="t-1")
        assert task.applied == [(["run-1"], {"k": 1}, {"task_id": "t-1", "queue": queue_for(RTX_UUID)})]
        assert task.delayed == []

    def test_per_card_mode_queues_an_auto_job_on_the_shared_queue(self, per_card):
        task = FakeTask()
        D.dispatch_gpu_task(task, gpu_request="auto", kwargs={"a": 2})
        assert task.applied == [(None, {"a": 2}, {"queue": AUTO_QUEUE})]

    def test_per_card_mode_leaves_a_task_that_claims_on_its_own_queue_alone(self, per_card):
        task = FakeTask(gpu_queue=False)
        D.dispatch_gpu_task(task, gpu_request=RTX_UUID, kwargs={"a": 2})
        assert task.delayed == [((), {"a": 2})]
        assert task.applied == []

    def test_an_unmarked_task_is_never_routed_to_a_gpu_queue(self, per_card):
        task = FakeTask()
        del task.run
        D.dispatch_gpu_task(task, gpu_request=RTX_UUID, kwargs={"a": 2}, kwargsrepr="x")
        assert task.applied == [(None, {"a": 2}, {"kwargsrepr": "x"})]

    def test_single_mode_makes_the_call_the_site_always_made(self, single):
        task = FakeTask()
        D.dispatch_gpu_task(task, gpu_request=RTX_UUID, args=["run-1"], kwargs={"k": 1})
        assert task.delayed == [(("run-1",), {"k": 1})]
        assert task.applied == []

    def test_single_mode_passes_options_and_arguments_exactly_as_given(self, single):
        task = FakeTask()
        D.dispatch_gpu_task(task, gpu_request=RTX_UUID, args=("x",), soft_time_limit=5)
        assert task.applied == [(("x",), None, {"soft_time_limit": 5})]

    def test_the_default_mode_is_single(self):
        from src.core.config import Settings

        assert Settings.model_fields["gpu_worker_mode"].default == "single"


class TestTheDelayShapedForm:
    def test_the_request_is_the_calls_own_keyword(self, per_card):
        task = FakeTask()
        D.gpu_delay(task)("run-1", gpu_request=RTX_UUID, other=3)
        assert task.applied == [(("run-1",), {"gpu_request": RTX_UUID, "other": 3}, {"queue": queue_for(RTX_UUID)})]

    def test_a_row_based_request_is_passed_explicitly(self, per_card):
        task = FakeTask()
        D.gpu_delay(task, RTX_UUID)("run-1", confirmed=True)
        assert task.applied == [(("run-1",), {"confirmed": True}, {"queue": queue_for(RTX_UUID)})]

    def test_single_mode_is_the_delay_it_replaced(self, single):
        task = FakeTask()
        D.gpu_delay(task, RTX_UUID)("run-1", confirmed=True)
        assert task.delayed == [(("run-1",), {"confirmed": True})]
