"""R2-D seam reproductions, review round 2 of the SAE training remediation (2026-09-15).

REPORT-ONLY. R2-D changes no product code; each test below reproduces a defect at
`integrate/sae-training-remediation` 7649ddbd and is marked strict xfail with
``raises=AssertionError``, so a harness error is a real error and never passes as the
expected failure, and a fix turns the xfail into XPASS (a red suite) until the marker is
removed. Where a CONTROL variant exists it is the same fixture without the triggering
variable, and it passes: the failure is caused by that variable, not by the fixture.

  R2D-1  (FIXED by R3-A; the xfail is removed.)
         Stop & Finalize pressed during the post-run evaluation replaces the export the
         loop wrote from the FINAL weights with the newest periodic checkpoint's, stamps
         `finalized_from_step`, and the evaluation's numbers then describe weights that
         are no longer on disk. The loop writes `community_format/` (training_tasks.py
         ~3286) and only then runs the evaluation (~3335) with the row still RUNNING, so
         `stop_training` accepts the Stop and the endpoint queues a finalize from
         `checkpoint_step=None` (trainings.py ~364-378).
  R2D-2  The MCP `list_trainings` tool pages with `offset`; the live route declares
         `page` and `limit` only, so every page an agent asks for is page 1.
  R2D-3  A resumed training's row never names the Celery task that runs it:
         `resume_training_task` dispatches `train_sae` through `gpu_delay` and discards
         the result, and the control endpoint calls `resume_training` without a task id.
         The row keeps the id of the task that PAUSED, so the janitor judges the resumed
         run by a finished task (spared while its result lives, reaped by row age once it
         expires), and `release_reaped_leases` targets the wrong id.

NEGATIVE CONTROL WHEN FIXED: remove the xfail marker, confirm the test passes, then revert
the fix and confirm it fails with its assertion message. Record both in the fixing lane's
review record.

The cached-path harness is R1-D's (`test_r1d_evaluation_seams._cached_world`): the real
`train_sae_task` on CPU over real extractions, with an in-memory session.
"""

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from tests.unit.test_r1d_evaluation_seams import _cached_world


# ── R2D-1: Stop & Finalize during the post-run evaluation ───────────────────


def _file_hashes(root: Path) -> dict:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _press_stop_and_finalize(monkeypatch, world):
    """What the card's Stop & Finalize button does, through the real endpoint and task.

    The endpoint's async session is the only stand-in: `TrainingService.get_training` reads
    the world's row, and the real `stop_training` rule and the real finalize task run on it.
    The queued finalize runs synchronously, as the worker would run it moments later.
    """
    from src.api.v1.endpoints import trainings as endpoint
    from src.schemas.training import TrainingControlRequest
    from src.services import training_service
    from src.workers import training_finalize_tasks

    class _AsyncSession:
        async def commit(self):
            return None

        async def refresh(self, obj):
            return None

    async def get_training(db, training_id):
        return world.training if world.training.id == training_id else None

    monkeypatch.setattr(training_service.TrainingService, "get_training", staticmethod(get_training))
    monkeypatch.setattr(training_service, "_emit_training_event_sync", lambda **kwargs: None)
    monkeypatch.setattr(endpoint, "revoke_task", lambda *args, **kwargs: None)
    task = training_finalize_tasks.finalize_training_from_checkpoint_task
    monkeypatch.setattr(task, "delay", lambda *args, **kwargs: task.run(*args, **kwargs))

    from fastapi import HTTPException

    try:
        return asyncio.run(endpoint.control_training(
            TrainingControlRequest(action="stop_and_finalize"),
            training_id=world.training.id,
            db=_AsyncSession(),
        ))
    except HTTPException as exc:
        # A refusal is a legitimate fix (the export of the final weights already exists).
        # A 5xx is the endpoint's catch-all for an error, which is the harness's problem.
        if exc.status_code >= 500:
            raise
        return {"refused": exc.status_code, "message": str(exc.detail)}


@pytest.mark.parametrize(
    "operator_presses_stop_and_finalize",
    [
        False,
        # R2D-1 FIXED by R3-A (the run is COMPLETED once its export is saved; a Stop stops only the
        # evaluation; Finalize refuses a full-length export): the strict xfail is removed.
        True,
    ],
    ids=["CONTROL-no-stop", "R2D-1-stop-and-finalize-during-the-evaluation"],
)
def test_stop_and_finalize_during_the_post_run_evaluation_keeps_the_export_of_the_final_weights(
    monkeypatch, tmp_path, operator_presses_stop_and_finalize
):
    """20 steps, checkpoints at 5, 10 and 15. The loop exports step-20 weights, then runs
    the evaluation; the operator presses Stop & Finalize while it runs.

    The evaluation is enabled, as it is by default, so the Stop reaches the post-run record
    the task writes with COMPLETED. R3-A's negative controls for this test and for each layer
    of the fix are in its review record (reverting all three layers turns this red with the
    original assertion; each layer alone has its own test in test_r3a_stop_after_the_export.py)."""
    world = _cached_world(monkeypatch, tmp_path, rows=6, seq=8,
                          hp={"total_steps": 20, "checkpoint_interval": 5, "evaluate_ce_delta": True})
    world.training.celery_task_id = "the-training-task"
    tid = world.training.id
    community = world.data_dir / "trainings" / tid / "community_format"
    newest_checkpoint = world.data_dir / "trainings" / tid / "checkpoints" / "checkpoint_15"
    seen = {}

    def evaluate(task, **kwargs):
        # The loop wraps this call in a catch-all (training_tasks.py ~3347), so an error
        # raised here would vanish and leave the export untouched: a pass for the wrong
        # reason. Everything is recorded and re-raised after the run instead.
        try:
            seen["final"] = {
                key: {name: t.detach().cpu().clone() for name, t in sae.state_dict().items()}
                for key, sae in kwargs["models"].items()
            }
            seen["export_existed"] = community.is_dir()
            seen["export_before"] = _file_hashes(community) if seen["export_existed"] else {}
            seen["status_during_evaluation"] = world.training.status
            if operator_presses_stop_and_finalize:
                seen["response"] = _press_stop_and_finalize(monkeypatch, world)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            seen["error"] = exc
        return {"status": "cancelled" if operator_presses_stop_and_finalize else "completed"}

    monkeypatch.setattr(world.tasks, "run_post_run_evaluation", evaluate)
    world.run()
    if "error" in seen:
        raise RuntimeError(f"the harness failed inside the evaluation window: {seen['error']!r}") from seen["error"]
    if not seen.get("export_existed"):
        raise RuntimeError("the loop ran the evaluation before writing its export; this fixture assumes the reverse")

    # THE FIXTURE CAN TELL THE TWO EXPORTS APART: the newest checkpoint's weights are not
    # the final weights, so a finalize from it necessarily changes what is on disk.
    (final_weights,) = seen["final"].values()
    checkpoint_files = sorted(newest_checkpoint.rglob("*.safetensors"))
    assert len(checkpoint_files) == 1, checkpoint_files
    stored = {name.removeprefix("model."): t for name, t in load_file(str(checkpoint_files[0])).items()}
    shared = sorted(set(stored) & set(final_weights))
    assert shared, (sorted(stored), sorted(final_weights))
    assert any(not torch.equal(stored[name], final_weights[name]) for name in shared), (
        "the step-15 checkpoint equals the final weights; this fixture could not see a replaced export"
    )

    after = _file_hashes(community)
    assert after == seen["export_before"], (
        f"the export written from the final weights (step 20) was replaced after a Stop & Finalize "
        f"during the post-run evaluation (row status during it: {seen['status_during_evaluation']!r}); "
        f"now status={world.training.status!r}, "
        f"finalized_from_step={getattr(world.training, 'finalized_from_step', None)!r}, "
        f"endpoint said {getattr(seen.get('response'), 'get', lambda k: None)('message')!r}"
    )
    # THE DECIDED OUTCOME, not only the unchanged export (review R3-A).
    assert world.training.status == "completed", world.training.status
    assert getattr(world.training, "finalized_from_step", None) is None
    if operator_presses_stop_and_finalize:
        response = seen["response"]
        assert response.get("success") is True and response["status"] == "completed", response
        assert world.training.evaluation.get("stop_requested_by") == "stop_and_finalize", world.training.evaluation


# ── R2D-2: the MCP training list pages with a parameter the route ignores ───


class _RecordingClient:
    def __init__(self):
        self.calls = []

    async def get(self, path, **params):
        # As the real client does: None values are not sent.
        self.calls.append(("GET", path, {k: v for k, v in params.items() if v is not None}))
        return {"data": []}


@pytest.mark.parametrize(
    "tool, arguments",
    [
        ("list_extractions", {"limit": 10, "offset": 20}),
        # R2D-2 FIXED (02f899f3): list_trainings sends page; the xfail is removed.
        ("list_trainings", {"limit": 10, "offset": 20}),
    ],
    ids=["CONTROL-list_extractions", "R2D-2-list_trainings"],
)
def test_an_mcp_list_tool_sends_only_query_parameters_its_live_route_declares(tool, arguments):
    from mcp.server.fastmcp import FastMCP

    from src.main import app
    from src.mcp_server.tools import discovery

    mcp = FastMCP("r2d")
    client = _RecordingClient()
    discovery.register(mcp, client, SimpleNamespace())
    asyncio.run(mcp._tool_manager._tools[tool].fn(**arguments))

    assert len(client.calls) == 1, client.calls
    method, path, sent = client.calls[0]
    # The MCP client's base URL is {backend}/api/v1 (mcp_server/client.py).
    operation = app.openapi()["paths"]["/api/v1" + path][method.lower()]
    declared = {p["name"] for p in operation.get("parameters", []) if p.get("in") == "query"}
    # The agent's paging reached the request in SOME form (offset 20 at limit 10 is page 3).
    assert sent.get("offset") == 20 or sent.get("page") == 3, sent
    assert set(sent) <= declared, (
        f"{tool} sends {sorted(set(sent) - declared)} to GET /api/v1{path}, which declares only "
        f"{sorted(declared)}: FastAPI drops the rest, so the agent's paging is silently ignored"
    )


# ── R2D-5: a Stop on a queued training is overwritten when the task starts ──


@pytest.mark.parametrize(
    "stopped_while_queued",
    [
        False,
        # R2D-5 FIXED by R2-A (start_refusal): the strict xfail is removed.
        True,
    ],
    ids=["CONTROL-not-stopped", "R2D-5-stopped-while-queued"],
)
def test_a_training_stopped_while_queued_does_not_run_when_its_task_starts(monkeypatch, tmp_path, stopped_while_queued):
    """The operator stops a training that is still waiting for its card (status set by the
    API's `stop_training`, which accepts any non-terminal row). The task starts later."""
    world = _cached_world(monkeypatch, tmp_path, rows=6, seq=8, hp={"total_steps": 6, "checkpoint_interval": 5})
    world.training.celery_task_id = "the-queued-task"
    if stopped_while_queued:
        world.training.status = "cancelled"

    steps = []
    real_log = world.task.log_metric

    def log_metric(*args, **kwargs):
        steps.append(kwargs.get("step"))
        return real_log(*args, **kwargs)

    monkeypatch.setattr(world.task, "log_metric", log_metric)
    result = world.run()

    if not stopped_while_queued:
        assert steps and result.get("status") == "completed", result
        return
    assert world.training.status == "cancelled" and not steps, (
        f"a training stopped while queued ran {len(set(steps))} logged steps and ended "
        f"{world.training.status!r} (task result {result.get('status')!r}): the start writes "
        f"INITIALIZING over CANCELLED (training_tasks.py:779)"
    )


# ── R2D-3: a resumed training's row names the task that paused, not the one that runs ──


# R2D-3 FIXED by R2-A (the resume writes the dispatched task id): the strict xfail is removed.
def test_a_resumed_training_row_names_the_task_that_will_run_it(monkeypatch, tmp_path):
    world = _cached_world(monkeypatch, tmp_path, rows=6, seq=8)
    world.training.celery_task_id = "the-task-that-paused"
    world.pause(stop_at=12)

    dispatched = []

    def gpu_delay(task, request):
        def delay(**kwargs):
            dispatched.append(kwargs)
            return SimpleNamespace(id="the-task-that-resumes")
        return delay

    monkeypatch.setattr(world.tasks, "gpu_delay", gpu_delay)
    result = world.tasks.resume_training_task.run(world.training.id)

    assert result["status"] == "queued" and len(dispatched) == 1, (result, dispatched)
    assert world.training.celery_task_id == "the-task-that-resumes", (
        f"the resumed run is queued as 'the-task-that-resumes' but its row names "
        f"{world.training.celery_task_id!r}: the janitor's task_looks_alive and "
        f"release_reaped_leases judge a task that has already finished"
    )
