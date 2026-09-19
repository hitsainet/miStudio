"""A resume continues the run it interrupted: the right checkpoint, durably written, in lockstep.

Debt lane WS-C, 2026-09-16. Seven tracked findings from the SAE training remediation
reviews, all in the checkpoint/resume path:

R2A-9  A step whose `training_state.pt` was gone resumed WEIGHTS-ONLY and silently:
       `select_resume_checkpoint` judged a step complete on its weights alone, so the
       newest step won even when an older one could have continued exactly. Adam, the
       LR warmup, the loss scale, the dead-latent statistics, the RNG and the data
       position all restarted to save re-training a handful of steps.
R2A-8  Neither the state nor the weights were fsynced, and the weights were not even
       written atomically: `save_file` wrote straight to the name a resume looks for.
       A power loss could leave a torn `checkpoint.safetensors`, which R2A-9's
       selection would then happily choose.
R2A-7  A refused resume raised, leaving the row RUNNING with its previous error
       cleared — `TrainingService.resume_training` sets that before dispatching — so
       the record said "running" with no worker, until `cleanup_stuck_trainings`
       reaped it as "no progress ... a crashed worker or system issue": a message for
       something that never happened. The status and the message must agree.
A9     After an OOM halved the batch and persisted it, a resumed process re-derived
       the accumulation window from the HALVED size. That changes the effective batch
       AND the training-step -> optimizer-step conversion the LR scheduler was built
       with, while the scheduler's restored position was recorded under the old one.
       The window now belongs to the run, is saved, and is reused.
R2C-5  The SAEs were stepped in one interleaved loop, so an OOM raised part-way
       through left the SAEs before it one optimizer step and one LR position ahead
       of the rest, permanently, on a batch the others never saw. The step is now two
       passes: every forward/backward, then every update.
R2C-8  Throughput was `absolute step / seconds since this process started`, so every
       resumed run reported nonsense and its slow-training alert — which fires BELOW
       a threshold — could never fire again.

MUTATION CONTROLS (each applied alone to the source, this file and the files named
run, the edit confirmed landed with `git diff`, bytes restored from a byte copy and
sha256 verified; the record is in the WS-C scratchpad notes). All RED:

  C1  `select_resume_checkpoint` ignores `has_state` (first pass takes complete[0])
      -> test_the_newest_step_that_can_continue_wins,
         test_a_weights_only_newer_step_is_passed_over
  C2  the weights-only fallback returns None instead of the newest complete step
      -> test_every_step_weights_only_still_resumes_from_the_newest
  C3  `_fsync_path(partial)` deleted from save_training_state
      -> test_the_training_state_is_fsynced_before_and_after_the_rename
  C4  `_fsync_directory` deleted from save_training_state -> the same test
  C5  save_checkpoint writes straight to the final path again (no .partial/rename)
      -> test_a_weights_write_that_dies_leaves_no_file_under_the_resume_name
  C6  the refusal does not write FAILED (status line deleted)
      -> test_a_refused_resume_leaves_a_row_whose_status_and_message_agree
  C7  the refusal writes FAILED but no error_message -> the same test
  C8  `resolve_grad_accum_steps` ignores the saved window (always derives)
      -> test_the_saved_accumulation_window_wins_over_the_reduced_batch_size,
         test_a_resumed_run_keeps_the_window_its_checkpoint_recorded
  C9  `grad_accum_steps` is not saved in the training state (None)
      -> test_a_checkpoint_records_the_accumulation_window,
         test_a_resumed_run_keeps_the_window_its_checkpoint_recorded
  C10 `_fresh_scaler_at` always returns GradScaler() -> test_an_oom_rebuild_keeps_the_calibrated_loss_scale
  C11 `_fresh_scaler_at` carries a non-finite scale forward -> the same test
  C12 the second pass is merged back into the first (update inside the per-SAE loop)
      -> test_an_oom_in_one_sae_leaves_no_sae_a_step_ahead
  C13 the OOM handler does not clear every SAE's gradients -> the same test
  C14 the log site measures the absolute step again (`steps_this_process = step`)
      -> test_the_log_site_measures_steps_since_this_run_started
  C15 `steps_per_minute` returns 0.0 instead of None at zero elapsed
      -> test_throughput_is_none_rather_than_a_division_by_zero

All fifteen were KILLED on their first run (2026-09-16), and the driver
(scratchpad `debt-c/mutate.py`) restores each file from a byte copy and asserts its
sha256 afterwards; `grep -rn "MUTANT C" src` is empty and `git diff --stat` covers
only the intended files.

TWO THINGS THIS FILE DELIBERATELY TESTS AT THE CALL SITE, not only as pure functions:

* C14 is the reason `test_the_log_site_measures_steps_since_this_run_started` exists.
  `steps_per_minute` alone is trivially correct; the DEFECT was in what it was called
  with, and a test of the function would have stayed green through it. The check walks
  the AST for the call and the assignment rather than scraping the source for a name —
  the comment above the call names both `steps_per_minute` and `start_step`, so a
  substring search matches the comment and passes for the wrong reason.
* C13 cannot bite at an accumulation window of 1, where every step begins with its own
  `zero_grad`. `TestAnOomClearsEverySaesGradients` uses batch 16 (window 4) so the OOM
  lands MID-window, which is the only arrangement where the abandoned gradients
  genuinely survive into the next step.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.models.checkpoint import Checkpoint
from src.models.training import Training
from src.services import checkpoint_service as CS
from src.services.checkpoint_service import CheckpointService
from src.workers import training_tasks
from src.workers.training_tasks import (
    _fresh_scaler_at,
    resolve_grad_accum_steps,
    steps_per_minute,
)
from tests.unit.test_resume_storage_plan import (
    _cached_world,
    _gpu_free_for,
    _no_post_run_evaluation,
)

KEYS = [(3, "residual"), (4, "residual")]


@contextmanager
def _captured_warnings(logger_name="src.services.checkpoint_service"):
    """The warnings one call emits, as a list of formatted messages.

    `caplog` is not available in this suite — the logging plugin is disabled — so the
    records are collected with a handler of our own. What the resume LOGS is part of
    these findings, not decoration: choosing an older step, or continuing weights-only,
    is a decision the operator has to be able to see.
    """
    messages = []

    class _Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    logger = logging.getLogger(logger_name)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _row(tmp_path, step, layer, *, exists=True, state=True):
    """A checkpoint row for one layer of one step, with its files on disk."""
    path = tmp_path / f"checkpoint_{step}" / f"layer_{layer}_residual" / "checkpoint.safetensors"
    if exists:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"w" * 100)
    if state:
        state_path = tmp_path / f"checkpoint_{step}" / CS.TRAINING_STATE_FILENAME
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(b"s" * 100)
    return SimpleNamespace(
        id=f"ckpt_{step}_{layer}",
        step=step,
        storage_path=str(path),
        is_best=False,
        extra_metadata={"layer_idx": layer, "hook_type": "residual"},
    )


def _step_rows(tmp_path, step, **kwargs):
    return [_row(tmp_path, step, layer, **kwargs) for _, layer in [(0, 3), (1, 4)]]


# ── R2A-9: which step a resume chooses ───────────────────────────────────────


class TestTheStepAResumeChooses:
    def test_the_newest_step_that_can_continue_wins(self, tmp_path):
        """Every step complete: the newest, exactly as before this fix."""
        rows = _step_rows(tmp_path, 100) + _step_rows(tmp_path, 200)
        step, chosen = CS.select_resume_checkpoint(rows, KEYS)
        assert step == 200
        assert {key: row.step for key, row in chosen.items()} == {KEYS[0]: 200, KEYS[1]: 200}

    def test_a_weights_only_newer_step_is_passed_over(self, tmp_path):
        """THE FINDING. Step 200 has its weights but no training_state.pt.

        Resuming from it restarts Adam, the warmup, the loss scale, the dead-latent
        statistics, the RNG and the data position. Step 100 can continue exactly, at
        the price of re-training 100 steps, and that is the better trade — but the
        run must be told it is making it.
        """
        rows = _step_rows(tmp_path, 100) + _step_rows(tmp_path, 200, state=False)

        with _captured_warnings() as warnings:
            step, chosen = CS.select_resume_checkpoint(rows, KEYS)

        assert step == 100, "the resume took a step it could not continue from"
        assert all(row.step == 100 for row in chosen.values())
        text = "\n".join(warnings)
        assert "200" in text and CS.TRAINING_STATE_FILENAME in text, text

    def test_every_step_weights_only_still_resumes_from_the_newest(self, tmp_path):
        """A training from before training_state.pt existed: a weights-only resume is
        better than refusing to resume at all, but it says so."""
        rows = _step_rows(tmp_path, 100, state=False) + _step_rows(tmp_path, 200, state=False)

        with _captured_warnings() as warnings:
            step, _ = CS.select_resume_checkpoint(rows, KEYS)

        assert step == 200
        text = "\n".join(warnings)
        assert "WEIGHTS" in text.upper(), text

    def test_an_incomplete_step_with_state_does_not_win(self, tmp_path):
        """State present but a layer's weights missing is still incomplete: the
        completeness rule comes first, the state preference second."""
        rows = _step_rows(tmp_path, 100) + [_row(tmp_path, 200, 3)]  # layer 4 never written
        step, _ = CS.select_resume_checkpoint(rows, KEYS)
        assert step == 100

    def test_no_complete_step_is_still_none(self, tmp_path):
        assert CS.select_resume_checkpoint([_row(tmp_path, 10, 3)], KEYS) is None

    def test_the_state_is_looked_for_beside_the_layer_directories(self, tmp_path):
        """`training_state_is_present` derives the step directory the same way
        `checkpoint_retention._training_state_size` does: two levels up from the row."""
        [row] = [_row(tmp_path, 300, 3)]
        assert CS.training_state_is_present(row.storage_path) is True
        (tmp_path / "checkpoint_300" / CS.TRAINING_STATE_FILENAME).unlink()
        assert CS.training_state_is_present(row.storage_path) is False


# ── R2A-8: a checkpoint that survives a power loss ───────────────────────────


class TestCheckpointsReachTheDisk:
    def _count_fsyncs(self, monkeypatch):
        calls = []
        real = os.fsync

        def counting_fsync(fd):
            calls.append(fd)
            return real(fd)

        monkeypatch.setattr(CS.os, "fsync", counting_fsync)
        return calls

    def test_the_training_state_is_fsynced_before_and_after_the_rename(self, tmp_path, monkeypatch):
        """The bytes, then the directory entry that names them.

        `os.replace` is atomic against a killed PROCESS, not against a power loss: the
        rename can reach the disk while the file's contents are still in the page
        cache, leaving `training_state.pt` present, the right size, and torn.
        """
        calls = self._count_fsyncs(monkeypatch)
        CS.save_training_state(tmp_path, {"step": 1, "saes": {}, "best_loss": 0.0})

        assert len(calls) >= 2, f"expected the file AND its directory to be fsynced, got {len(calls)}"
        assert CS.load_training_state(tmp_path)["step"] == 1

    def test_the_weights_are_fsynced_too(self, tmp_path, monkeypatch):
        calls = self._count_fsyncs(monkeypatch)
        model = torch.nn.Linear(4, 4)
        CheckpointService.save_checkpoint(
            model=model, optimizer=torch.optim.Adam(model.parameters()), step=1,
            storage_path=str(tmp_path / "layer_0_residual" / "checkpoint.safetensors"),
        )
        assert len(calls) >= 2, f"expected the file AND its directory to be fsynced, got {len(calls)}"

    def test_a_weights_write_that_dies_leaves_no_file_under_the_resume_name(
        self, tmp_path, monkeypatch
    ):
        """THE FINDING. `save_file` used to write straight to `checkpoint.safetensors`.

        A worker killed — or a node powered off — mid-write left a TORN file under
        exactly the name `select_resume_checkpoint` looks for, and the path existing
        is what that function tests. Written under a temporary name, a dead write
        leaves the resume name absent, which the selector already handles.
        """
        final = tmp_path / "layer_0_residual" / "checkpoint.safetensors"
        real_save_file = CS.save_file

        def dying_save_file(tensors, path, metadata=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"truncated by a power loss")
            raise OSError("No space left on device")

        monkeypatch.setattr(CS, "save_file", dying_save_file)
        model = torch.nn.Linear(4, 4)
        with pytest.raises(OSError):
            CheckpointService.save_checkpoint(
                model=model, optimizer=torch.optim.Adam(model.parameters()), step=1,
                storage_path=str(final),
            )

        assert not final.exists(), "a torn write is sitting under the name a resume reads"
        monkeypatch.setattr(CS, "save_file", real_save_file)
        CheckpointService.save_checkpoint(
            model=model, optimizer=torch.optim.Adam(model.parameters()), step=1,
            storage_path=str(final),
        )
        assert final.exists()

    def test_a_dead_write_is_not_selected_as_a_resume_point(self, tmp_path, monkeypatch):
        """The two findings together: the step whose weights never completed is not
        chosen, because its file is not there to be chosen."""
        rows = _step_rows(tmp_path, 100)
        step_dir = tmp_path / "checkpoint_200" / "layer_3_residual"
        step_dir.mkdir(parents=True)
        (tmp_path / "checkpoint_200" / CS.TRAINING_STATE_FILENAME).write_bytes(b"s")
        # Nothing under checkpoint.safetensors: the write died before the rename.
        rows = rows + [
            SimpleNamespace(
                id="ckpt_200_3", step=200, is_best=False,
                storage_path=str(step_dir / "checkpoint.safetensors"),
                extra_metadata={"layer_idx": 3, "hook_type": "residual"},
            )
        ]
        assert CS.select_resume_checkpoint(rows, KEYS)[0] == 100


# ── A9 / R2C-8: the small decisions, as pure functions ───────────────────────


class TestTheAccumulationWindowBelongsToTheRun:
    def test_a_run_with_no_saved_window_derives_one(self):
        assert resolve_grad_accum_steps(None, 16) == training_tasks.grad_accum_steps_for(16)
        assert resolve_grad_accum_steps(0, 16) == training_tasks.grad_accum_steps_for(16)

    def test_the_saved_accumulation_window_wins_over_the_reduced_batch_size(self):
        """THE FINDING. The run started at batch 64 (window 1). An OOM halved it to 32
        and persisted that, so deriving afresh would give window 2: a different
        effective batch AND a different step -> optimizer-step conversion from the one
        the restored LR scheduler state was written under."""
        assert training_tasks.grad_accum_steps_for(32) == 2, "fixture no longer exercises the gap"
        assert resolve_grad_accum_steps(1, 32) == 1

    def test_it_is_an_integer_even_from_a_float_state(self):
        assert resolve_grad_accum_steps(4.0, 8) == 4


class TestTheLossScaleSurvivesAnOomRebuild:
    def test_an_oom_rebuild_keeps_the_calibrated_loss_scale(self, monkeypatch):
        """A GradScaler's scale is calibrated over thousands of steps. Rebuilding at
        the 65536 default costs a run of overflowing, skipped updates."""
        built = []
        monkeypatch.setattr(
            training_tasks, "GradScaler", lambda **kwargs: built.append(kwargs) or SimpleNamespace(**kwargs)
        )
        _fresh_scaler_at(1024.0)
        assert built == [{"init_scale": 1024.0}]

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan"), None, "x"])
    def test_a_scale_worth_nothing_is_not_carried_forward(self, monkeypatch, bad):
        built = []
        monkeypatch.setattr(
            training_tasks, "GradScaler", lambda **kwargs: built.append(kwargs) or SimpleNamespace(**kwargs)
        )
        _fresh_scaler_at(bad)
        assert built == [{}], f"{bad!r} was carried forward as a loss scale"


class TestThroughputIsMeasuredOverThisProcess:
    def test_the_log_site_measures_steps_since_this_run_started(self):
        """THE CALL, not just the function.

        A pure function is only as good as its call site, and the defect lived in the
        call: `step / elapsed`. Read from the AST, never as a substring — the comment
        above the call names `steps_per_minute` and `start_step` too, so a text scrape
        matches the comment and passes for the wrong reason. That has happened five
        times in this repo.
        """
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(training_tasks))

        assigns = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "steps_this_process" for t in node.targets
            )
        ]
        assert assigns, "nothing computes steps_this_process"
        for assign in assigns:
            names = {n.id for n in ast.walk(assign.value) if isinstance(n, ast.Name)}
            assert {"step", "start_step"} <= names, (
                f"steps_this_process is computed from {names}, not from step and start_step"
            )

        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "steps_per_minute"
        ]
        assert calls, "nothing calls steps_per_minute"
        for call in calls:
            first = call.args[0]
            assert isinstance(first, ast.Name) and first.id == "steps_this_process", (
                f"steps_per_minute is called with {ast.dump(first)}"
            )

    def test_throughput_counts_only_this_processs_steps(self):
        """THE FINDING, in numbers: a run resumed at step 120,000 that has done 10
        steps in 30 seconds is doing 20 steps/min, not 240,000."""
        assert steps_per_minute(10, 30.0) == pytest.approx(20.0)

    @pytest.mark.parametrize("steps,elapsed", [(0, 5.0), (-3, 5.0), (10, 0.0), (10, -1.0)])
    def test_throughput_is_none_rather_than_a_division_by_zero(self, steps, elapsed):
        """The caller divides the target BY this to report how many times slower than
        target the run is, so 0.0 would be a ZeroDivisionError in an alert path."""
        assert steps_per_minute(steps, elapsed) is None


# ── the training task: R2A-7, A9 and R2C-5 through the real loop ─────────────


def _world(monkeypatch, tmp_path, *, layers=(0,), **hp):
    world = _cached_world(
        monkeypatch, tmp_path, rows=4_000, seq=16, layers=layers, hp={"total_steps": 12, **hp}
    )
    _no_post_run_evaluation(monkeypatch, world)
    # Sized for every SAE this run holds, not for one.
    world.gpu_free = _gpu_free_for(80_000, keys=len(layers))
    return world


class TestARefusedResumeIsHonest:
    def test_a_refused_resume_leaves_a_row_whose_status_and_message_agree(
        self, monkeypatch, tmp_path
    ):
        """THE FINDING. The run trained steps but has no checkpoint, so it cannot be
        resumed — starting again at 0 would discard them silently.

        `TrainingService.resume_training` has already set RUNNING and cleared the
        previous error by the time this task runs, so the refusal used to leave a row
        reading RUNNING with nothing running and nothing to explain why.
        """
        world = _world(monkeypatch, tmp_path)
        world.training.status = "running"  # as TrainingService.resume_training leaves it
        world.training.current_step = 7
        world.training.error_message = None

        with pytest.raises(ValueError, match="No complete checkpoint"):
            world.tasks.resume_training_task.run(world.training.id)

        assert world.dispatched == []
        assert world.training.status == "failed", (
            f"the row still reads {world.training.status!r} with no worker behind it"
        )
        assert world.training.error_message, "a FAILED row with no reason recorded"
        assert "No complete checkpoint" in world.training.error_message
        # The janitor's message is for a crashed worker; this is not one.
        assert "no progress" not in world.training.error_message.lower()


class TestTheAccumulationWindowSurvivesAResume:
    def test_a_checkpoint_records_the_accumulation_window(self, monkeypatch, tmp_path):
        """batch 16 with MIN_EFFECTIVE_BATCH 64 means a window of 4."""
        world = _world(monkeypatch, tmp_path, batch_size=16, checkpoint_interval=5)
        assert world.run()["status"] == "completed"

        state = world.state(10)
        assert state is not None
        assert state["grad_accum_steps"] == training_tasks.grad_accum_steps_for(16) == 4

    def test_a_resumed_run_keeps_the_window_its_checkpoint_recorded(self, monkeypatch, tmp_path):
        """THE FINDING. The row's batch size is halved between the pause and the
        resume, exactly as an OOM would halve and persist it. The resumed process must
        keep the window the interrupted one used, not derive a wider one."""
        world = _world(monkeypatch, tmp_path, batch_size=64, checkpoint_interval=5)
        world.pause(6)
        assert world.state(5)["grad_accum_steps"] == 1

        # The OOM's persisted reduction, as `train_sae_task` writes it.
        world.training.hyperparameters = {**world.training.hyperparameters, "batch_size": 32}
        assert training_tasks.grad_accum_steps_for(32) == 2, "fixture no longer exercises the gap"

        world.training.status = "running"
        assert world.resume()["status"] == "completed"

        assert world.state(10)["grad_accum_steps"] == 1, (
            "the resumed process re-derived the window from the reduced batch size"
        )


class TestAnOomLeavesTheSaesInLockstep:
    def test_an_oom_in_one_sae_leaves_no_sae_a_step_ahead(self, monkeypatch, tmp_path):
        """THE FINDING (R2C-5). Two SAEs; the SECOND one's forward raises a CUDA OOM
        at a step that CLOSES an accumulation window — the only kind of step at which
        the two arrangements can differ.

        Interleaved, the first SAE had already taken its optimizer and scheduler step
        for a step the handler then abandoned (`continue` moves to the next step, it
        does not retry this one), leaving it permanently one optimizer step and one LR
        position ahead of the second, on a batch the second never saw.
        """
        world = _world(monkeypatch, tmp_path, layers=(0, 1), batch_size=64, total_steps=8)

        steps = {"n": -1}
        real_draw = training_tasks.draw_cached_batch

        def counting_draw(*args, **kwargs):
            steps["n"] += 1
            return real_draw(*args, **kwargs)

        monkeypatch.setattr(training_tasks, "draw_cached_batch", counting_draw)

        # The schedulers the task builds, so the run's LR positions can be read after
        # it ends; each one holds the optimizer it steps.
        schedulers = []
        real_build = training_tasks.build_lr_scheduler

        def build_lr_scheduler(*args, **kwargs):
            schedulers.append(real_build(*args, **kwargs))
            return schedulers[-1]

        monkeypatch.setattr(training_tasks, "build_lr_scheduler", build_lr_scheduler)

        real_create = training_tasks.create_sae
        created = []
        oom = {"fired": False}

        def create_sae(*args, **kwargs):
            model = real_create(*args, **kwargs)
            created.append(model)
            index = len(created) - 1
            real_forward = model.forward

            def forward(*f_args, **f_kwargs):
                # The second SAE, once, on an update step (window is 1 at batch 64).
                if index == 1 and steps["n"] == 3 and not oom["fired"]:
                    oom["fired"] = True
                    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
                return real_forward(*f_args, **f_kwargs)

            model.forward = forward
            return model

        monkeypatch.setattr(training_tasks, "create_sae", create_sae)

        result = world.run()

        assert oom["fired"], "the OOM never happened; this test proved nothing"
        assert result["status"] == "completed", result
        assert len(schedulers) == 2, f"expected one scheduler per SAE, got {len(schedulers)}"

        positions = [scheduler.last_epoch for scheduler in schedulers]
        assert len(set(positions)) == 1, (
            f"the SAEs are at different points in the LR schedule: {positions}"
        )
        applied = [
            str(scheduler.optimizer.state_dict()["state"].get(0, {}).get("step"))
            for scheduler in schedulers
        ]
        assert len(set(applied)) == 1, (
            f"the SAEs have taken different numbers of optimizer steps: {applied}"
        )


class TestAnOomClearsEverySaesGradients:
    def test_the_abandoned_windows_gradients_are_dropped_for_every_sae(
        self, monkeypatch, tmp_path
    ):
        """An OOM mid-accumulation leaves the SAEs holding gradients for a batch the
        step never applied — and only the SAEs that got as far as their backward.

        The update closing that window would then apply the asymmetry, moving some SAEs
        on a batch the others never saw. The handler clears every SAE's gradients with
        `set_to_none=True`, so "all gradients are None at the next draw" distinguishes
        the fix from its absence: an uncleared SAE still holds tensors from the steps
        before the OOM.

        Batch 16 gives an accumulation window of 4, so the OOM lands MID-window, where
        gradients genuinely survive from one step to the next. At window 1 every step
        begins with a zero_grad and this control cannot bite.
        """
        world = _world(
            monkeypatch, tmp_path, layers=(0, 1), batch_size=16, total_steps=10
        )
        assert training_tasks.grad_accum_steps_for(16) == 4, "fixture no longer accumulates"

        steps = {"n": -1}
        observed = {}
        real_draw = training_tasks.draw_cached_batch
        created = []
        oom = {"fired": False}

        def counting_draw(*args, **kwargs):
            steps["n"] += 1
            if oom["fired"] and "after_oom" not in observed:
                observed["after_oom"] = [
                    [p.grad is None for p in model.parameters()] for model in created
                ]
            return real_draw(*args, **kwargs)

        monkeypatch.setattr(training_tasks, "draw_cached_batch", counting_draw)

        real_create = training_tasks.create_sae

        def create_sae(*args, **kwargs):
            model = real_create(*args, **kwargs)
            created.append(model)
            index = len(created) - 1
            real_forward = model.forward

            def forward(*f_args, **f_kwargs):
                # The second SAE, once, at a step that is NOT a window boundary.
                if index == 1 and steps["n"] == 2 and not oom["fired"]:
                    oom["fired"] = True
                    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
                return real_forward(*f_args, **f_kwargs)

            model.forward = forward
            return model

        monkeypatch.setattr(training_tasks, "create_sae", create_sae)

        result = world.run()

        assert oom["fired"], "the OOM never happened; this test proved nothing"
        assert result["status"] == "completed", result
        assert "after_oom" in observed, "the run never reached another draw after the OOM"
        for index, grads in enumerate(observed["after_oom"]):
            assert all(grads), (
                f"SAE {index} carried the abandoned window's gradients past the OOM: {grads}"
            )
