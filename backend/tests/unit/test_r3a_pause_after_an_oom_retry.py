"""Review round 3, R3-A: a Pause caught by the status check that follows an OOM retry (2026-09-15).

A PAUSE WRITES ITS OWN CHECKPOINT (R1-D L8, R1-A A16). It happens at the end-of-step check, which decides
the stop for the NEXT step, so the step that just ran can be checkpointed. An OOM `continue`s past that
check, and the loop's top-of-step check then catches the Pause instead. That check returned at once, with
no checkpoint:
- every step since the newest periodic checkpoint was repeated on resume;
- a run paused before its first periodic checkpoint could not be resumed at all (R2A-11 refuses a
  training that trained steps without one).

The Pause here is a real row status, set while step 14 draws its batch, exactly as the Pause endpoint would
set it. The OOM is raised by that draw (a refill), which keeps the batch size.

TWO STATUS-CHECK INTERVALS. At `log_interval` 1 the loop checks the status at the end of every step, so it
cannot tell whether the fix FORCES the end-of-step check of the step after a deferred Pause: R3-A's control
P2 (the forcing removed) survived on that fixture alone. At `log_interval` 5 the check after step 15 runs
only when forced.

NEGATIVE CONTROLS (review record):
- P1, the deferral removed: red on every oom case (the newest checkpoint is 10, or none).
- P2, the forcing removed: red on the oom cases at log_interval 5 (the run is not paused).
- P3, `trained_a_step` never set: red on every oom case.
"""

import pytest

from src.models.training_metric import TrainingMetric
from tests.unit.test_r1d_evaluation_seams import _cached_world

OOM_STEP = 14


def _oom():
    return RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")


@pytest.mark.parametrize("log_interval", [1, 5], ids=["checks-every-step", "checks-every-5-steps"])
@pytest.mark.parametrize("checkpoint_interval", [10, 50], ids=["after-a-periodic-checkpoint", "before-any-checkpoint"])
@pytest.mark.parametrize("oom", [False, True], ids=["CONTROL-no-oom", "oom-in-the-step-before-the-check"])
def test_a_pause_caught_after_an_oom_retry_checkpoints_every_step_it_trained_and_resumes(
    monkeypatch, tmp_path, checkpoint_interval, oom, log_interval
):
    from src.services.training_finalize_service import list_checkpoint_steps

    world = _cached_world(monkeypatch, tmp_path, rows=6, seq=8,
                          hp={"total_steps": 20, "checkpoint_interval": checkpoint_interval,
                              "log_interval": log_interval})
    tasks = world.tasks
    real_draw = tasks.draw_cached_batch
    attempts = []

    def draw(*args, **kwargs):
        attempts.append(len(attempts))
        if len(attempts) - 1 == OOM_STEP:
            world.training.status = "paused"  # the operator's Pause lands while step 14 draws
            if oom:
                raise _oom()
        return real_draw(*args, **kwargs)

    monkeypatch.setattr(tasks, "draw_cached_batch", draw)
    result = world.run()

    assert result["status"] == "paused", (
        f"the Pause during step {OOM_STEP} was not honoured before the run ended: {result}"
    )
    logged = sorted({m.step for m in world.store.get(TrainingMetric, []) if m.layer_idx is None})
    steps = list_checkpoint_steps(world.training.id)
    assert logged, "precondition: steps were trained and logged"
    # Steps 0..13 trained before the Pause landed (step 14 drew, and under `oom` failed).
    assert steps and max(steps) >= OOM_STEP - 1 and max(steps) >= max(logged), (
        f"paused with steps up to {OOM_STEP - 1} trained ({logged[-3:]} logged) but the newest checkpoint "
        f"is {steps[-1:] or 'none'}: the steps trained since it are lost"
    )

    monkeypatch.setattr(tasks, "draw_cached_batch", real_draw)
    world.training.status = "running"  # as TrainingService.resume_training sets it before dispatch
    assert world.resume()["status"] == "completed"
    assert world.dispatched[-1]["start_step"] == max(steps) + 1, world.dispatched
