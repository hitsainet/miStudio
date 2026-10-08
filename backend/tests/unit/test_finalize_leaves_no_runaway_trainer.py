"""Stop & Finalize must actually stop the trainer, whichever terminal value wins.

Found on the 3090, 2026-09-16, on train_c29aa474. The operator pressed Stop &
Finalize at step ~87,750 of a 150,000-step run. The row was marked COMPLETED with
`finalized_from_step=80000` and the UI reported success — and the worker kept
training. It was still going two and a half hours later at step 143,750, holding
the card at 97%, writing metrics for a run the database called finished, with FVU
past 5.0 (a dictionary five times worse than predicting the mean). It stopped only
when the row was set to CANCELLED by hand, 6,250 steps before it would have
overwritten the operator's export with the collapsed weights.

THE RACE. `POST /trainings/{id}?action=stop_and_finalize` does three things:

    1. stop_training(...)                      -> writes CANCELLED
    2. revoke_task(..., terminate=True)        -> INERT; this pool is --pool=solo,
                                                  and revoke signals a pool child
                                                  that does not exist
    3. finalize_training_from_checkpoint_task  -> asynchronously writes COMPLETED

The loop polls every `status_check_interval` steps (25 here, ~4 s). On the real
run `updated_at` and `completed_at` were 3 SECONDS apart, so the finalize
overwrote CANCELLED with COMPLETED inside the poll window and the stop signal was
destroyed before it was ever read. From then on the loop saw COMPLETED, which
`stop_signal_for` did not recognise, and ran on forever. Recovery was welded shut
too: `TrainingService.stop_training` returns None for an already-terminal row, so
pressing Stop again wrote nothing.

THE FIX IS SCOPED TO THE LOOP, AND THAT SCOPE IS THE WHOLE DIFFICULTY. Both the
training loop and the post-run evaluation decide through `stop_signal_for`, and
they need OPPOSITE answers for COMPLETED:

    check_stop (the loop)        COMPLETED -> someone finalized underneath me: STOP
    post_run_stop_reason (eval)  COMPLETED -> the normal state: KEEP GOING

because R3-A (decided 2026-09-15) marks the row COMPLETED in the commit AFTER the
export and BEFORE the evaluation, so that a Stop during the evaluation cancels
only the evaluation and leaves the run completed with its final weights. A first
attempt at this fix made every terminal status a stop unconditionally and broke
that: nine tests went red, four of them
`test_r3a_stop_after_the_export.py::test_a_button_pressed_during_the_post_run_evaluation...`,
because the evaluation aborted at its first forward. Hence `terminal_is_a_stop`,
default False: the evaluation keeps its semantics, and only the loop opts in.

A SECOND ATTEMPT THEN BROKE 21 TESTS by adding the keyword alone: six test files
replace `stop_signal_for` with stubs of a fixed `(row, step, lease_lost=None)`
shape, and they do not merely accept the call, they FORWARD it. Widening those
stubs to `**kw` without forwarding `**kw` would have been worse than the TypeError
— the loop's rule would have silently stopped applying inside exactly the tests
that drive the loop, and everything would have gone green.

`start_refusal` has always consulted the registry through `guard_allows`, which is
why a task REFUSES TO START on a COMPLETED row. The loop could not STOP on one.
Deriving both from the scope's `terminal_values` closes that asymmetry; a second
hardcoded list was the defect, so these tests derive their cases FROM THE REGISTRY
and a terminal value added later is covered without editing this file.

MUTATION CONTROLS (2026-09-16; each applied alone, the source restored
byte-identically and the sha256 verified):
  F1 the terminal branch back to `status == CANCELLED` only   -> 6 failed
  F2 the terminal check moved BELOW the lease check           -> SURVIVED, then
     pinned by test_a_terminal_row_beats_a_lease_loss            -> red
  F3 the PAUSED branch removed                                -> SURVIVED, then
     pinned by test_the_paused_branch_is_load_bearing            -> 2 failed
  F4 a literal terminal set that omits "failed"               -> 4 failed
  F5 the loop's call site drops terminal_is_a_stop=True       -> 1 failed

TWO CONTROLS SURVIVED THEIR FIRST RUN, both for the same reason — the mutation
changed nothing the tests could observe:

  * F3: deleting the explicit PAUSED branch was a no-op, because "paused" is in
    terminal_values and the generic branch returned the identical dict.
  * F2: the ordering test used a CANCELLED row, and CANCELLED has its own branch
    ABOVE both the terminal check and the lease check, so moving the terminal
    check could not affect it. Only a row that is COMPLETED/FAILED *and* has lost
    its lease distinguishes the two orderings, and nothing exercised that.

F2's ordering is not cosmetic: terminal-first reports `{"status": "completed"}`,
lease-first reports a lease-loss FAILURE — and `stop_now` writes FAILED plus an
error message onto the row for that branch, mislabelling an operator's deliberate
stop as a crash.

WHAT IS STILL NOT PINNED, on the record: a literal set EQUIVALENT to the
registry's passes every test here, because the two agree today. The derivation
protects the future, not the present.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.cancellation import get_scope, guard_allows
from src.models.training import TrainingStatus
from src.workers import training_tasks
from src.workers.training_tasks import GPU_LEASE_LOST, start_refusal, stop_signal_for


TERMINAL = sorted(get_scope("training").terminal_values)


def _row(status):
    return SimpleNamespace(status=status, current_step=7)


def _loop(row, step, **kw):
    """How the training loop asks."""
    return stop_signal_for(row, step, terminal_is_a_stop=True, **kw)


# ── the defect itself: the LOOP stops on any terminal status ─────────────────

def test_a_completed_row_stops_the_loop():
    """THE REGRESSION. A finalize marks the row COMPLETED under a running loop."""
    assert _loop(_row(TrainingStatus.COMPLETED.value), 87_750) == {
        "status": "completed", "step": 87_750,
    }


def test_a_failed_row_stops_the_loop():
    assert _loop(_row(TrainingStatus.FAILED.value), 12) == {"status": "failed", "step": 12}


@pytest.mark.parametrize("status", TERMINAL)
def test_every_terminal_value_stops_the_loop(status):
    """Derived from the registry, so a new terminal value cannot reopen this hole."""
    signal = _loop(_row(status), 99)
    assert signal is not None, f"{status} is terminal for the guard but not a stop"
    assert signal["step"] == 99


def test_the_stop_check_and_the_start_refusal_agree():
    """A row that cannot START a task must not keep one RUNNING — the asymmetry
    the defect lived in."""
    for status in TERMINAL:
        refuses_to_start = start_refusal(_row(status), "training") is not None
        stops_the_loop = _loop(_row(status), 5) is not None
        assert refuses_to_start == stops_the_loop, (
            f"{status}: start_refusal={refuses_to_start} but the loop={stops_the_loop}"
        )


# ── the other half: the EVALUATION must NOT stop on COMPLETED (R3-A) ────────

def test_a_completed_row_does_not_stop_the_post_run_evaluation():
    """R3-A, decided 2026-09-15: the row is COMPLETED before the evaluation runs.

    Treating that as a stop aborts every post-run evaluation at its first forward,
    which is exactly what the first attempt at this fix did.
    """
    assert stop_signal_for(_row(TrainingStatus.COMPLETED.value), 4) is None


def test_the_evaluation_still_stops_on_a_real_cancel():
    assert stop_signal_for(_row(TrainingStatus.CANCELLED.value), 4) == {
        "status": "cancelled", "step": 4,
    }


def test_the_loop_opts_in_and_the_evaluation_does_not():
    """THE WIRING, by AST. A capability is not shipped until a test fails when its
    wiring is removed: the loop's status check must pass terminal_is_a_stop=True,
    and `post_run_stop_reason` must not."""
    tree = ast.parse(Path(training_tasks.__file__).read_text())

    def calls_in(name):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        return [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "stop_signal_for"]

    (loop_call,) = calls_in("train_sae_task")
    opts_in = [k for k in loop_call.keywords if k.arg == "terminal_is_a_stop"]
    assert opts_in, "the loop's status check does not opt in to terminal_is_a_stop"
    assert getattr(opts_in[0].value, "value", None) is True

    (eval_call,) = calls_in("post_run_stop_reason")
    assert not [k for k in eval_call.keywords if k.arg == "terminal_is_a_stop"], (
        "the post-run evaluation must NOT treat COMPLETED as a stop (R3-A)"
    )


# ── ordering: the operator's outcome beats a lease loss ──────────────────────

def test_the_operator_stop_still_beats_a_lease_loss():
    """Pinned by test_gpu_claim_review_r1 too: the operator's stop wins over a lapse."""
    assert stop_signal_for(
        _row(TrainingStatus.CANCELLED.value), 50, lease_lost="its lease lapsed"
    ) == {"status": "cancelled", "step": 50}
    assert _loop(
        _row(TrainingStatus.CANCELLED.value), 50, lease_lost="its lease lapsed"
    ) == {"status": "cancelled", "step": 50}


@pytest.mark.parametrize(
    "status", [TrainingStatus.COMPLETED.value, TrainingStatus.FAILED.value]
)
def test_a_terminal_row_beats_a_lease_loss(status):
    """MUTATION F2. The terminal check must sit ABOVE the lease check.

    The CANCELLED case cannot detect the ordering — CANCELLED has its own branch
    above both — so F2 survived until this test existed. A run finalized out from
    under the loop that ALSO lost its lease must report the finalize, not a
    failure: `stop_now` writes FAILED and an error message onto the row for a
    lease-loss stop, which would record the operator's deliberate stop as a crash.
    """
    assert _loop(_row(status), 60, lease_lost="its lease lapsed") == {
        "status": status, "step": 60,
    }


def test_a_live_row_that_lost_its_lease_still_fails():
    assert stop_signal_for(
        _row(TrainingStatus.RUNNING.value), 50, lease_lost="its lease lapsed"
    ) == {
        "status": "failed", "step": 50, "reason": GPU_LEASE_LOST,
        "detail": "its lease lapsed",
    }


# ── behaviour that must NOT change ───────────────────────────────────────────

def test_a_running_row_keeps_going():
    assert stop_signal_for(_row(TrainingStatus.RUNNING.value), 50) is None
    assert _loop(_row(TrainingStatus.RUNNING.value), 50) is None


def test_a_paused_row_is_still_reported_as_paused():
    assert stop_signal_for(_row(TrainingStatus.PAUSED.value), 7) == {
        "status": "paused", "step": 7,
    }


def test_the_paused_branch_is_load_bearing(monkeypatch):
    """MUTATION F3. PAUSED is reported as paused WITHOUT relying on it being terminal.

    Deleting the explicit branch survived at first, because the generic terminal
    branch returned the identical dict while "paused" remained in terminal_values.
    Pinned here against a guard that calls every row live: with the branch a pause
    is still a pause; without it the loop sails past a paused row and keeps
    training, losing the checkpoint a resume depends on.
    """
    monkeypatch.setattr(
        training_tasks, "guard_allows",
        lambda kind, current, incoming=None, **kw: True,
    )
    assert _loop(_row(TrainingStatus.PAUSED.value), 7) == {"status": "paused", "step": 7}


def test_a_deleted_row_is_still_a_stop():
    assert stop_signal_for(None, 425) == {
        "status": "cancelled", "step": 425, "reason": "deleted",
    }
    assert _loop(None, 425) == {"status": "cancelled", "step": 425, "reason": "deleted"}


def test_the_registry_is_the_only_authority():
    """No third copy of the terminal list: for the LOOP, `guard_allows` decides."""
    for status in (*TERMINAL, TrainingStatus.RUNNING.value, TrainingStatus.INITIALIZING.value):
        live = guard_allows("training", status, TrainingStatus.RUNNING.value)
        assert (_loop(_row(status), 3) is None) == live, (
            f"{status}: guard_allows says live={live}, the loop disagrees"
        )
