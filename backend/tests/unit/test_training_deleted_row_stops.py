"""A deleted training stops its worker instead of crashing it.

Found by the multi-GPU Phase 1 hardware acceptance on mcs-lnxhost02, 2026-09-14.
Stop then Delete on a running training (the UI allows Delete once the row reads
cancelled) removed the row before the loop's next status check. The check read
`training.status` off None, the task died with AttributeError at
training_tasks.py:1632, and the failed task's tensors stayed allocated — 18 GB on
the 3090 — until the worker ran its next task, so an Auto job placed in between
would have seen that card as nearly full. A training whose row was still there
at the check stopped cleanly and freed its memory.

The loop's decision now lives in `stop_signal_for`, which treats a missing row as
a stop, and the training cancellation scope says the same (`missing_row`). The
worker path for a row deleted while queued is covered in
test_training_gpu_placement.py.

MUTATION CONTROLS (2026-09-14; each applied alone, the source restored
byte-identically):
  D1 stop_signal_for returns None for a missing row      -> test_a_deleted_row_is_a_stop fails
  D2 the loop reads training.status again, no helper call -> test_the_step_loop_decides_with_stop_signal_for fails
  D3 the training scope's missing_row back to "continue"  -> test_the_registry_treats_a_deleted_training_as_cancelled fails
  D4 the placement read drops its None check              -> test_a_training_deleted_while_queued_stops_instead_of_crashing fails
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models.training import TrainingStatus
from src.workers import training_tasks
from src.workers.training_tasks import stop_signal_for


def test_a_deleted_row_is_a_stop():
    assert stop_signal_for(None, 425) == {"status": "cancelled", "step": 425, "reason": "deleted"}


@pytest.mark.parametrize(
    "status, expected",
    [
        (TrainingStatus.PAUSED.value, {"status": "paused", "step": 7}),
        (TrainingStatus.CANCELLED.value, {"status": "cancelled", "step": 7}),
        (TrainingStatus.RUNNING.value, None),
    ],
)
def test_the_row_status_decides_otherwise(status, expected):
    assert stop_signal_for(SimpleNamespace(status=status), 7) == expected


def _task_loops():
    tree = ast.parse(Path(training_tasks.__file__).read_text())
    task = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_sae_task"
    )
    return [node for node in ast.walk(task) if isinstance(node, ast.For)]


def test_the_step_loop_decides_with_stop_signal_for():
    """The CALL, found by walking the task's syntax tree, not a text search."""
    loops = _task_loops()
    called = {
        call.func.id
        for loop in loops
        for call in ast.walk(loop)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "stop_signal_for" in called

    # Nothing in the loops READS `training.status` directly any more; writes on
    # the failure path are fine.
    direct_reads = [
        node.lineno
        for loop in loops
        for node in ast.walk(loop)
        if isinstance(node, ast.Attribute) and node.attr == "status"
        and isinstance(node.value, ast.Name) and node.value.id == "training"
        and isinstance(node.ctx, ast.Load)
    ]
    assert not direct_reads, f"the loop reads training.status directly at lines {direct_reads}"


def test_the_registry_treats_a_deleted_training_as_cancelled():
    from src.core.cancellation import get_scope

    assert get_scope("training").missing_row == "cancelled"
