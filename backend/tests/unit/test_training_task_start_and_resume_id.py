"""A training task honours a Stop or Pause it finds at start, and the row names the task that runs it.

Review round 2 of the SAE training remediation (reviewer R2-A fixing R2-D's R2D-5 and
R2D-3, 2026-09-15). Driven through the real `train_sae_task` and `resume_training_task`
with the harness in test_resume_storage_plan.py.

R2D-5. The Stop endpoint marks any live row CANCELLED and revokes the task, but a revoke
is inert on a solo-pool worker and forgotten across a restart. The task's start then
wrote INITIALIZING over CANCELLED and trained to completion. The same bare write of
RUNNING after set-up overwrote a Stop or Pause landing while the model loaded. Both now
go through `start_refusal` (the training scope's `guard_allows`): a deleted or terminal
row returns before placement, with no card claimed and nothing created.

R2D-3. `resume_training_task` discarded the dispatched task's id, so the row kept the
PAUSED task's; the stuck-job janitor and `release_reaped_leases` judged a finished task.
The resume now records the dispatched id, and every task records its own id at start.

A paused training that never trained a step has no checkpoint: it resumes from step 0
(it used to raise "No complete checkpoint" with the row left RUNNING).

MUTATION CONTROLS: in .claude/context/sessions/review_sae_remediation_R2_A_2026-09-15.md.
"""

from types import SimpleNamespace

import pytest

from src.core import cancellation
from src.models.checkpoint import Checkpoint
from src.models.training import Training
from src.models.training_metric import TrainingMetric
from tests.unit.test_resume_storage_plan import _cached_world, _gpu_free_for, _no_post_run_evaluation


def _world(monkeypatch, tmp_path, **hp):
    world = _cached_world(monkeypatch, tmp_path, rows=4_000, seq=16, hp={"total_steps": 6, **hp})
    _no_post_run_evaluation(monkeypatch, world)
    world.gpu_free = _gpu_free_for(80_000)
    world.placed, world.created = [], []
    real_place, real_create = world.tasks.place_job, world.tasks.create_sae
    monkeypatch.setattr(world.tasks, "place_job", lambda *a, **k: world.placed.append(1) or real_place(*a, **k))
    monkeypatch.setattr(world.tasks, "create_sae", lambda *a, **k: world.created.append(1) or real_create(*a, **k))
    return world


@pytest.mark.parametrize(
    "status", ["CONTROL-initializing", "CONTROL-running", "cancelled", "paused", "failed", "deleted"],
)
def test_a_training_stopped_before_its_task_starts_returns_without_placing_or_training(monkeypatch, tmp_path, status):
    world = _world(monkeypatch, tmp_path)
    world.training.celery_task_id = "the-queued-task"
    if status.startswith("CONTROL"):
        world.training.status = status.split("-", 1)[1]
    elif status == "deleted":
        world.store[Training].remove(world.training)
    else:
        world.training.status = status

    result = world.run()

    if status.startswith("CONTROL"):
        assert result["status"] == "completed" and world.placed and world.created, result
        return
    expected = ("cancelled", "deleted") if status == "deleted" else (status, "stopped before its task started")
    assert (result["status"], result["reason"]) == expected, result
    assert world.placed == [] and world.created == [], "the task placed a card or built an SAE"
    assert not world.store.get(TrainingMetric) and not world.store.get(Checkpoint)
    if status != "deleted":
        assert world.training.status == status, f"the row was moved off {status!r} to {world.training.status!r}"


@pytest.mark.parametrize("status", ["CONTROL-no-stop", "cancelled", "paused"])
def test_a_stop_or_pause_landing_during_set_up_is_not_overwritten_with_running(monkeypatch, tmp_path, status):
    """The operator's Stop (or Pause) lands while the task is placing and loading: here,
    right after placement, before the RUNNING write."""
    world = _world(monkeypatch, tmp_path)
    placed = world.tasks.place_job

    def place_then_stop(*args, **kwargs):
        placement = placed(*args, **kwargs)
        if not status.startswith("CONTROL"):
            world.training.status = status
        return placement

    monkeypatch.setattr(world.tasks, "place_job", place_then_stop)
    result = world.run()

    if status.startswith("CONTROL"):
        assert result["status"] == "completed", result
        return
    assert (result["status"], result["reason"]) == (status, "stopped before its first step"), result
    assert world.training.status == status
    assert not world.store.get(TrainingMetric) and not world.store.get(Checkpoint), "the run trained"


def test_a_resumed_training_row_names_the_task_that_will_run_it(monkeypatch, tmp_path):
    world = _world(monkeypatch, tmp_path, total_steps=20)
    world.training.celery_task_id = "the-task-that-paused"
    world.pause(11)

    dispatched = []

    def gpu_delay(task, request):
        assert task is world.tasks.train_sae_task

        def delay(**kwargs):
            dispatched.append(kwargs)
            return SimpleNamespace(id="the-task-that-resumes")

        return delay

    monkeypatch.setattr(world.tasks, "gpu_delay", gpu_delay)
    result = world.tasks.resume_training_task.run(world.training.id)

    assert len(dispatched) == 1 and dispatched[0]["start_step"] == 11, dispatched
    assert dispatched[0]["checkpoint_id"] == result["checkpoint_id"] and result["task_id"] == "the-task-that-resumes"
    assert world.training.celery_task_id == "the-task-that-resumes", world.training.celery_task_id


def test_the_executing_task_records_its_own_id_when_it_starts(monkeypatch, tmp_path):
    world = _world(monkeypatch, tmp_path)
    world.training.celery_task_id = "a-task-that-no-longer-runs"
    world.task.push_request(id="the-executing-task")
    try:
        assert world.run()["status"] == "completed"
    finally:
        world.task.pop_request()
    assert world.training.celery_task_id == "the-executing-task"


@pytest.mark.parametrize("current_step", [0, 7], ids=["never-trained", "trained-without-a-checkpoint"])
def test_a_paused_training_without_a_checkpoint_resumes_from_step_zero_only_if_it_never_trained(
    monkeypatch, tmp_path, current_step
):
    """Paused while queued (the task now honours it), a training has no checkpoint and no
    step: it starts from 0. One that trained steps and has no checkpoint is still refused:
    starting it again from 0 would silently discard them."""
    world = _world(monkeypatch, tmp_path)
    world.training.status = "running"  # as TrainingService.resume_training leaves it
    world.training.current_step = current_step
    if current_step:
        with pytest.raises(ValueError, match="No complete checkpoint"):
            world.tasks.resume_training_task.run(world.training.id)
        assert world.dispatched == []
        return
    result = world.tasks.resume_training_task.run(world.training.id)
    assert (result["start_step"], result["checkpoint_id"]) == (0, None), result
    assert world.dispatched == [{"training_id": world.training.id, "start_step": 0, "checkpoint_id": None}]


def test_start_refusal_is_the_training_scopes_guard():
    """The refusal follows `guard_allows`, not a list of its own: every terminal value of
    the training scope refuses, every live one runs."""
    scope = cancellation.get_scope("training")
    for status in scope.terminal_values:
        row = SimpleNamespace(status=status, current_step=3)
        refused = __import__("src.workers.training_tasks", fromlist=["start_refusal"]).start_refusal(row, "x")
        assert refused == {"status": status, "step": 3, "reason": "stopped before x"}, status
    from src.workers.training_tasks import start_refusal

    for status in ("pending", "initializing", "running"):
        assert start_refusal(SimpleNamespace(status=status, current_step=0), "x") is None, status
    assert start_refusal(None, "x") == {"status": "cancelled", "step": 0, "reason": "deleted"}
