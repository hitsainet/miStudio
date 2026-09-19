"""A pause saves a checkpoint at the last step it completed (R1-D L8).

Review round 1, 2026-09-15. Pausing saved nothing: the loop returned at its next
status check, so a pause lost every step since the newest periodic checkpoint — up to
`checkpoint_interval` steps (1,000 by default), which a resume then trained again.

The loop now checks for a stop at the END of each step (for the step that would run
next), and on a pause writes the same checkpoint a periodic save writes — weights,
`training_state.pt`, rows — before returning. That point is where a periodic save
happens, and nothing between it and the next step's check draws a random number or
moves the data stream, so resuming from a pause checkpoint continues the run exactly.

Driven through the real task with the resume-equivalence harness. The two resume tests
and the paused-at-first-step test were red on the code before the fix.

MUTATION CONTROLS (R1-A, 2026-09-15; applied alone, restored, sha256 verified):
  L8a the checkpoint condition loses `or pause_checkpoint`  -> both resume tests, first-step test
  L8b the end-of-step check never asks (stop_signal None)   -> both resume tests, the no-double-write test
  L8c the top check ignores `checked_step`                  -> SURVIVED (a second status query per
        interval, and a pause landing between the two returned without its checkpoint). New
        test_the_stop_is_checked_once_per_step; re-run: red
  L8d a cancel checkpoints too                              -> test_a_cancel_writes_no_checkpoint
  G1  check_stop drops lease_lost                           -> test_gpu_claim_review_r1 single-call guard
"""

import pytest

from tests.unit import test_training_resume_equivalence as E


def _steps(harness):
    return sorted(c.step for c in harness.store.get(E.Checkpoint, []))


@pytest.mark.parametrize(
    "name,stop_at",
    [("pool-standard-mid-decay", 13), ("rolling-jumprelu-accum", 14)],
    ids=["pool-between-checkpoints", "rolling-mid-accumulation"],
)
def test_a_pause_checkpoints_its_last_step_and_resumes_from_it_exactly(monkeypatch, tmp_path, name, stop_at):
    config = E.CONFIGS[name]
    straight = E._Harness(monkeypatch, tmp_path / "straight", config)
    assert straight.run()["status"] == "completed"

    paused = E._Harness(monkeypatch, tmp_path / "paused", config)
    paused.stop_at = stop_at
    assert paused.run()["status"] == "paused"
    last = stop_at - 1
    assert _steps(paused) == [5, 10, last], _steps(paused)
    assert paused.saved_state(last)["step"] == last

    result, kwargs, outcome = E._resume(paused)
    assert result["start_step"] == kwargs["start_step"] == stop_at
    assert outcome["status"] == "completed"
    E._assert_same_weights(straight.models[-1], paused.models[-1])
    E._assert_same_optimizer(straight.optimizers[-1], paused.optimizers[-1])
    assert paused.applied_lrs() == pytest.approx(straight.applied_lrs())
    assert paused.resamples == straight.resamples
    assert E._metric_rows(paused) == E._metric_rows(straight)


def test_a_pause_right_after_a_periodic_checkpoint_does_not_write_it_twice(monkeypatch, tmp_path):
    harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
    harness.stop_at = 11
    assert harness.run()["status"] == "paused"
    assert _steps(harness) == [5, 10]


def test_a_pause_before_any_step_ran_writes_no_checkpoint(monkeypatch, tmp_path):
    harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
    harness.stop_at = 0
    assert harness.run()["status"] == "paused"
    assert _steps(harness) == []


def test_a_resumed_run_paused_at_its_first_step_writes_no_new_checkpoint(monkeypatch, tmp_path):
    harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
    harness.stop_at = 13
    assert harness.run()["status"] == "paused"
    harness.stop_at = None
    harness.tasks.resume_training_task.run(harness.training.id)
    kwargs = harness.dispatched[-1]
    harness.stop_at = kwargs["start_step"]  # paused again before the resumed run trains a step
    assert harness.run(start_step=kwargs["start_step"], checkpoint_id=kwargs["checkpoint_id"])["status"] == "paused"
    assert _steps(harness) == [5, 10, 12]


def test_the_stop_is_checked_once_per_step(monkeypatch, tmp_path):
    """One status query per step, not two.

    The end-of-step check covers the next step, so the top-of-step check must skip a
    step already checked. Without that, every check interval queried the database
    twice, and a pause landing between the two queries returned WITHOUT its checkpoint
    (R1-A control L8c survived until this test).
    """
    harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
    wrapped = E._training_tasks.stop_signal_for
    asked = []

    def counting(row, step, lease_lost=None, **kw):
        asked.append(step)
        return wrapped(row, step, lease_lost, **kw)

    monkeypatch.setattr(E._training_tasks, "stop_signal_for", counting)
    assert harness.run()["status"] == "completed"
    assert asked == list(range(E.TOTAL)), asked  # log_interval 1: every step, once


def test_a_cancel_writes_no_checkpoint(monkeypatch, tmp_path):
    harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
    real_stop = E._REAL_STOP

    def stop(row, step, lease_lost=None, **kw):
        if step == 13:
            return {"status": "cancelled", "step": step}
        return real_stop(row, step, lease_lost, **kw)

    monkeypatch.setattr(E._training_tasks, "stop_signal_for", stop)
    assert harness.run()["status"] == "cancelled"
    assert _steps(harness) == [5, 10]
