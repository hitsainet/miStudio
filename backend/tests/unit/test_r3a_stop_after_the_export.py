"""Review round 3, R3-A: Stop, Pause, Stop & Finalize and Finalize once a training's full-length export is saved.

R2D-1 and R2D-10, fixed 2026-09-15. Record: .claude/context/sessions/review_sae_remediation_R3_A_2026-09-15.md.

THE DEFECT. The loop wrote `community_format/` from its final weights and then ran the post-run
evaluation, for minutes, with the row still RUNNING. A Stop there left CANCELLED beside a
full-length export (import locked), Stop & Finalize replaced that export with the newest periodic
checkpoint's and stamped `finalized_from_step`, and a Pause left PAUSED, so a resume retrained up
to a checkpoint interval and evaluated again.

DECIDED BEHAVIOUR (user, 2026-09-15). Once training has written its full-length export:
- a Stop cancels only the evaluation and leaves the run COMPLETED;
- Finalize refuses to overwrite a full-length export (409 with the reason);
- a Pause during the evaluation does not leave a paused run with a complete export.

THE FIX, in four places, each pinned below.
- `train_sae_task` marks the row COMPLETED in the commit after the export, before the evaluation,
  with a `pending` post-run record, or `cancelled` when a Stop or Pause landed while the last steps
  and the export ran (`completion_after_export`).
- The control route turns a Stop on a COMPLETED row into a stop request on its evaluation record
  (`request_evaluation_stop`), and refuses a Pause on one with a 409.
- The evaluation reads that request before the model load and between batches, and its own record
  writes carry it (`carry_stop_request`).
- The finalize route and `finalize_from_checkpoint` refuse a full-length export, force or not
  (`full_length_export_refusal`).
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException

from src.models.training import Training, TrainingStatus
from tests.unit.test_r1d_evaluation_seams import _cached_world, _on_the_fly_world
from tests.unit.test_r2d_seam_reproductions import _file_hashes


class _AsyncSession:
    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1

    async def refresh(self, obj):
        return None


def _community(world):
    return world.data_dir / "trainings" / world.training.id / "community_format"


def _write_export(data_dir, training_id, *, step, layers=("layer_0_residual",)):
    for name in layers:
        folder = data_dir / "trainings" / training_id / "community_format" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "cfg.json").write_text(json.dumps({"mistudio_checkpoint_step": step}))


# ── the operator's buttons, through the real routes, during the REAL evaluation ──


def _routes(monkeypatch, world):
    """The real control and finalize routes over the world's row. Revokes and queued finalizes are recorded."""
    from src.api.v1.endpoints import trainings as endpoint
    from src.services import training_service
    from src.workers import training_finalize_tasks

    async def get_training(db, training_id):
        return world.training if world.training.id == training_id else None

    monkeypatch.setattr(training_service.TrainingService, "get_training", staticmethod(get_training))
    monkeypatch.setattr(training_service, "_emit_training_event_sync", lambda **kwargs: None)
    revoked, queued = [], []
    monkeypatch.setattr(endpoint, "revoke_task", lambda *args, **kwargs: revoked.append(args))

    # `apply_async`, not `delay`: since OSD-36c the endpoint mints the Celery id so
    # the `task_queue` row can carry it, which `.delay` cannot express. `queued`
    # keeps the positional args so every assertion below reads unchanged.
    def dispatch(*args, **kwargs):
        queued.append(tuple(kwargs.get("args", args)))
        return SimpleNamespace(id=kwargs.get("task_id", "the-finalize-task"))

    monkeypatch.setattr(
        training_finalize_tasks.finalize_training_from_checkpoint_task, "apply_async", dispatch
    )

    async def _row(db, **kwargs):
        return SimpleNamespace(id="tq_1")

    monkeypatch.setattr(endpoint.TaskQueueService, "create_task_entry", staticmethod(_row))
    return endpoint, revoked, queued


def _press(endpoint, training_id, action):
    from src.schemas.training import TrainingControlRequest

    try:
        if action.startswith("finalize"):
            return asyncio.run(endpoint.finalize_training(
                training_id=training_id, checkpoint_step=None, allow_failed=False,
                force=action == "finalize-force", db=_AsyncSession(),
            ))
        return asyncio.run(endpoint.control_training(
            TrainingControlRequest(action=action), training_id=training_id, db=_AsyncSession(),
        ))
    except HTTPException as exc:
        if exc.status_code >= 500:
            raise
        return {"refused": exc.status_code, "detail": str(exc.detail)}


@pytest.mark.parametrize(
    "action", ["CONTROL-no-press", "stop", "stop_and_finalize", "pause", "finalize", "finalize-force"]
)
def test_a_button_pressed_during_the_post_run_evaluation_leaves_the_run_completed_with_its_final_weights(
    monkeypatch, tmp_path, action
):
    """The REAL post-run evaluation on the tiny Llama (on the fly, 2 layers: one cross-entropy batch is
    2 + 3 x 2 = 8 forwards). The button is pressed through the real route at the evaluation's first
    forward. Checkpoints exist (step 2 of 4), so a finalize had something to rebuild from.

    NEGATIVE CONTROLS (review record, R3-A table):
    - the COMPLETED write after the export removed: red on every case (status during the evaluation);
    - the stop-request read removed from the evaluation's `stop_reason`: red on stop, stop_and_finalize;
    - the Stop on a COMPLETED row not turned into a stop request: red on stop, stop_and_finalize;
    - the Pause refusal removed: red on pause (400, not 409);
    - the finalize route's full-length refusal removed: red on finalize-force.
    """
    world = _on_the_fly_world(
        monkeypatch, tmp_path, rows_per_dataset=2_000,
        hp={"total_steps": 4, "checkpoint_interval": 2, "holdout_fraction": 0.2, "holdout_eval_tokens": 256,
            "evaluate_ce_delta": True, "evaluation_token_budget": 12_800},
    )
    endpoint, revoked, queued = _routes(monkeypatch, world)
    community = _community(world)
    seen = {"forwards": 0}

    def hook(module, inputs, output):
        evaluation = world.training.evaluation or {}
        if evaluation.get("status") != "running":
            return
        seen["forwards"] += 1
        if seen["forwards"] != 1:
            return
        try:
            seen["status_during"] = world.training.status
            seen["export_before"] = _file_hashes(community)
            if action != "CONTROL-no-press":
                seen["response"] = _press(endpoint, world.training.id, action)
        except BaseException as exc:  # noqa: BLE001 - surfaced below, not swallowed by the evaluation
            seen["error"] = exc

    handle = world.model.lm_head.register_forward_hook(hook)
    try:
        result = world.run()
    finally:
        handle.remove()

    if "error" in seen:
        raise RuntimeError(f"the press failed inside the evaluation: {seen['error']!r}") from seen["error"]
    evaluation = world.training.evaluation
    assert (world.data_dir / "trainings" / world.training.id / "checkpoints" / "checkpoint_2").is_dir()
    assert seen["forwards"] >= 1, f"precondition: the evaluation never ran a forward ({evaluation})"
    assert seen["status_during"] == TrainingStatus.COMPLETED.value, (
        f"the row was {seen['status_during']!r} while its post-run evaluation ran; a run whose "
        "full-length export is saved is COMPLETED before it is evaluated"
    )
    assert seen["export_before"], "precondition: no export while the evaluation ran"
    assert _file_hashes(community) == seen["export_before"], "the export of the final weights changed"
    assert world.training.status == TrainingStatus.COMPLETED.value and result["status"] == "completed", result
    assert getattr(world.training, "finalized_from_step", None) is None
    assert queued == [] and revoked == [], (queued, revoked)

    if action == "CONTROL-no-press":
        assert evaluation["status"] == "completed", evaluation
        assert seen["forwards"] > 8, f"precondition: only {seen['forwards']} forwards"
    elif action in ("stop", "stop_and_finalize"):
        response = seen["response"]
        assert response.get("success") is True, response
        assert response["status"] == TrainingStatus.COMPLETED and "evaluation" in response["message"], response
        # The finished record names who stopped it in its reason; the request keys live only on a
        # pending or running record (`carry_stop_request`).
        assert evaluation["status"] == "cancelled", evaluation
        assert f"the operator stopped the evaluation ({action} at" in evaluation["reason"], evaluation
        assert "stop_requested_at" not in evaluation, evaluation
        assert seen["forwards"] - 1 <= 8, f"{seen['forwards'] - 1} forwards ran after the Stop"
    elif action == "pause":
        assert seen["response"].get("refused") == 409, seen["response"]
        assert "nothing to pause" in seen["response"]["detail"] and "send stop" in seen["response"]["detail"]
        assert evaluation["status"] == "completed", evaluation
    else:
        assert seen["response"].get("refused") == 409, seen["response"]
        assert "final weights" in seen["response"]["detail"], seen["response"]
        assert evaluation["status"] == "completed", evaluation


# ── the task: what the row becomes once the export is saved ─────────────────


@pytest.mark.parametrize(
    "status, expected",
    [
        (None, {"complete": False, "evaluate": False, "stopped_as": "deleted"}),
        ("pending", {"complete": True, "evaluate": True, "stopped_as": None}),
        ("initializing", {"complete": True, "evaluate": True, "stopped_as": None}),
        ("running", {"complete": True, "evaluate": True, "stopped_as": None}),
        ("cancelled", {"complete": True, "evaluate": False, "stopped_as": "cancelled"}),
        ("paused", {"complete": True, "evaluate": False, "stopped_as": "paused"}),
        ("completed", {"complete": True, "evaluate": False, "stopped_as": "completed"}),
        ("failed", {"complete": False, "evaluate": False, "stopped_as": "failed"}),
    ],
)
def test_completion_after_export_covers_every_status(status, expected):
    from src.workers.training_tasks import completion_after_export

    row = None if status is None else SimpleNamespace(status=status)
    assert completion_after_export(row) == expected


def test_the_run_is_completed_with_a_pending_record_before_its_post_run_evaluation_starts(monkeypatch, tmp_path):
    """NEGATIVE CONTROLS: the COMPLETED write after the export removed -> red (status); the pending
    record not written -> red; the training:completed emit moved back after the evaluation -> red."""
    from src.workers import websocket_emitter

    world = _cached_world(monkeypatch, tmp_path, rows=6, seq=8,
                          hp={"total_steps": 6, "checkpoint_interval": 5, "evaluate_ce_delta": True})
    events = []
    monkeypatch.setattr(websocket_emitter, "emit_training_progress", lambda **kw: events.append(kw["event"]) or True)
    seen = {}

    def evaluate(task, **kwargs):
        seen.update(
            status=world.training.status, progress=world.training.progress,
            completed_at=world.training.completed_at, evaluation=dict(world.training.evaluation or {}),
            events=list(events), export=_community(world).is_dir(),
        )
        return {"status": "completed"}

    monkeypatch.setattr(world.tasks, "run_post_run_evaluation", evaluate)
    result = world.run()

    assert result == {"status": "completed", "steps": 6, "final_loss": result["final_loss"]}, result
    assert seen["export"], "precondition: the export is written before the evaluation"
    assert seen["status"] == TrainingStatus.COMPLETED.value and seen["progress"] == 100.0, seen
    assert seen["completed_at"] is not None
    assert seen["evaluation"]["status"] == "pending" and seen["evaluation"]["trigger"] == "post_run", seen
    assert "training:completed" in seen["events"], seen["events"]


@pytest.mark.parametrize("landed", ["CONTROL-nothing", "cancelled", "paused", "completed-by-a-raced-finalize"])
def test_a_stop_or_pause_landing_while_the_export_is_written_completes_the_run_without_evaluating_it(
    monkeypatch, tmp_path, landed
):
    """A Stop or Pause after the loop's last status check (the check runs every <= 25 steps and never
    at the last step), or a Stop & Finalize whose finalize ran before the export. All steps trained
    and the export holds the final weights, so the run is COMPLETED, and the job stops: no evaluation.

    NEGATIVE CONTROLS: `completion_after_export` evaluating a CANCELLED/PAUSED row -> red;
    not completing it -> red; `finalized_from_step` not cleared -> red on the raced finalize."""
    from src.services.checkpoint_service import CheckpointService

    world = _cached_world(monkeypatch, tmp_path, rows=6, seq=8,
                          hp={"total_steps": 6, "checkpoint_interval": 5, "evaluate_ce_delta": True})
    real_save = CheckpointService.save_multilayer_community_checkpoint

    def save(*args, **kwargs):
        written = real_save(*args, **kwargs)
        if landed in ("cancelled", "paused"):
            world.training.status = landed
        elif landed.startswith("completed"):
            world.training.status = "completed"
            world.training.finalized_from_step = 5
        return written

    monkeypatch.setattr(CheckpointService, "save_multilayer_community_checkpoint", staticmethod(save))
    calls = []
    monkeypatch.setattr(world.tasks, "run_post_run_evaluation", lambda task, **kw: calls.append(kw) or {})
    result = world.run()

    assert _community(world).is_dir()
    assert world.training.status == TrainingStatus.COMPLETED.value and world.training.progress == 100.0
    assert world.training.finalized_from_step is None if hasattr(world.training, "finalized_from_step") else True
    assert result["status"] == "completed", result
    document = world.training.evaluation
    if landed == "CONTROL-nothing":
        assert len(calls) == 1 and document["status"] == "pending" and "reason" not in result, (calls, document)
        return
    assert calls == [], "the evaluation ran although the job was stopped as it finished"
    assert document["status"] == "cancelled" and document["trigger"] == "post_run", document
    assert "not run" in document["reason"] and "not run" in result["reason"], (document, result)


def test_a_row_deleted_while_the_export_is_written_is_not_evaluated(monkeypatch, tmp_path):
    from src.services.checkpoint_service import CheckpointService

    world = _cached_world(monkeypatch, tmp_path, rows=6, seq=8, hp={"total_steps": 6, "evaluate_ce_delta": True})
    real_save = CheckpointService.save_multilayer_community_checkpoint

    def save(*args, **kwargs):
        written = real_save(*args, **kwargs)
        world.store[Training].remove(world.training)
        return written

    monkeypatch.setattr(CheckpointService, "save_multilayer_community_checkpoint", staticmethod(save))
    calls = []
    monkeypatch.setattr(world.tasks, "run_post_run_evaluation", lambda task, **kw: calls.append(kw) or {})
    assert world.run() == {"status": "cancelled", "step": 6, "reason": "deleted"}
    assert calls == []


def test_a_failed_row_is_not_overwritten_when_the_export_is_saved(monkeypatch, tmp_path):
    """Tracked debt, pinned: a FAILED written while the export ran (the janitor, a lost lease) is left
    FAILED and the run is not evaluated. It is not reachable in practice (review record)."""
    from src.services.checkpoint_service import CheckpointService

    world = _cached_world(monkeypatch, tmp_path, rows=6, seq=8, hp={"total_steps": 6, "evaluate_ce_delta": True})
    real_save = CheckpointService.save_multilayer_community_checkpoint

    def save(*args, **kwargs):
        written = real_save(*args, **kwargs)
        world.training.status = "failed"
        return written

    monkeypatch.setattr(CheckpointService, "save_multilayer_community_checkpoint", staticmethod(save))
    calls = []
    monkeypatch.setattr(world.tasks, "run_post_run_evaluation", lambda task, **kw: calls.append(kw) or {})
    assert world.run()["status"] == "failed"
    assert world.training.status == "failed" and calls == []


# ── Finalize refuses the full-length export ────────────────────────────────


def test_full_length_export_step_reads_the_step_each_layer_recorded(monkeypatch, tmp_path):
    from src.core.config import settings
    from src.services.training_finalize_service import full_length_export_step

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    assert full_length_export_step("none", 6) is None
    _write_export(tmp_path, "loop", step=6, layers=("layer_0_residual", "layer_1_residual"))
    assert full_length_export_step("loop", 6) == 6
    assert full_length_export_step("loop", None) is None
    assert full_length_export_step("loop", 7) is None
    _write_export(tmp_path, "finalized", step=5)
    assert full_length_export_step("finalized", 6) is None
    legacy = tmp_path / "trainings" / "legacy" / "community_format" / "layer_0_residual"
    legacy.mkdir(parents=True)
    (legacy / "cfg.json").write_text(json.dumps({"d_in": 8}))
    assert full_length_export_step("legacy", 6) is None
    stray = tmp_path / "trainings" / "stray" / "community_format" / "notes"
    stray.mkdir(parents=True)
    (stray / "cfg.json").write_text(json.dumps({"mistudio_checkpoint_step": 6}))
    assert full_length_export_step("stray", 6) is None


def test_the_finalize_task_refuses_a_real_runs_full_length_export(monkeypatch, tmp_path):
    """NEGATIVE CONTROL: the refusal in `finalize_from_checkpoint` removed -> red (the export is
    rebuilt from checkpoint 5 and the row is stamped finalized_from_step)."""
    from src.services.training_finalize_service import FinalizeError
    from src.workers import training_finalize_tasks, websocket_emitter

    world = _cached_world(monkeypatch, tmp_path, rows=6, seq=8, hp={"total_steps": 6, "checkpoint_interval": 5})
    monkeypatch.setattr(world.tasks, "run_post_run_evaluation", lambda task, **kw: {"status": "skipped"})
    assert world.run()["status"] == "completed"
    assert (world.data_dir / "trainings" / world.training.id / "checkpoints" / "checkpoint_5").is_dir()
    before = _file_hashes(_community(world))
    events = []
    monkeypatch.setattr(websocket_emitter, "emit_training_progress", lambda **kw: events.append(kw["event"]) or True)

    with pytest.raises(FinalizeError, match="final weights"):
        training_finalize_tasks.finalize_training_from_checkpoint_task.run(world.training.id, None)

    assert _file_hashes(_community(world)) == before
    assert world.training.status == TrainingStatus.COMPLETED.value
    assert getattr(world.training, "finalized_from_step", None) is None
    assert events == ["training:finalize_failed"]


def _finalize_route(monkeypatch, row):
    from src.api.v1.endpoints import trainings as endpoint
    from src.workers import training_finalize_tasks

    async def get_training(db, training_id):
        return row

    monkeypatch.setattr(endpoint.TrainingService, "get_training", staticmethod(get_training))
    queued = []

    # `apply_async`, not `delay`: since OSD-36c the endpoint mints the Celery id so
    # the `task_queue` row can carry it, which `.delay` cannot express. `queued`
    # keeps the positional args so every assertion below reads unchanged.
    def dispatch(*args, **kwargs):
        queued.append(tuple(kwargs.get("args", args)))
        return SimpleNamespace(id=kwargs.get("task_id", "the-finalize-task"))

    monkeypatch.setattr(
        training_finalize_tasks.finalize_training_from_checkpoint_task, "apply_async", dispatch
    )

    async def _row(db, **kwargs):
        return SimpleNamespace(id="tq_1")

    monkeypatch.setattr(endpoint.TaskQueueService, "create_task_entry", staticmethod(_row))
    return endpoint, queued


@pytest.mark.parametrize(
    "status, flags",
    [("completed", {"force": True}), ("completed", {}), ("cancelled", {}), ("failed", {"allow_failed": True})],
    ids=["completed-force", "completed", "cancelled", "failed-allow_failed"],
)
def test_the_finalize_route_refuses_a_full_length_export_whatever_the_flags(monkeypatch, tmp_path, status, flags):
    """NEGATIVE CONTROL: the route's refusal removed -> red on completed-force, cancelled and failed-allow_failed."""
    from src.core.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    _write_export(tmp_path, "t1", step=6)
    endpoint, queued = _finalize_route(
        monkeypatch, SimpleNamespace(id="t1", status=status, hyperparameters={"total_steps": 6})
    )
    with pytest.raises(HTTPException) as refused:
        asyncio.run(endpoint.finalize_training(
            training_id="t1", checkpoint_step=None, allow_failed=flags.get("allow_failed", False),
            force=flags.get("force", False), db=_AsyncSession(),
        ))
    assert refused.value.status_code == 409 and "final weights" in refused.value.detail
    assert queued == []


def test_force_still_refinalizes_a_run_that_was_finalized_early(monkeypatch, tmp_path):
    """CONTROL for the refusal: an export written at a checkpoint's step is replaceable, as before."""
    from src.core.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    _write_export(tmp_path, "t1", step=5)
    endpoint, queued = _finalize_route(
        monkeypatch, SimpleNamespace(id="t1", status="completed", hyperparameters={"total_steps": 6})
    )
    result = asyncio.run(endpoint.finalize_training(
        training_id="t1", checkpoint_step=None, allow_failed=False, force=True, db=_AsyncSession(),
    ))
    assert result["data"]["status"] == "queued" and queued == [("t1", None)]


# ── the control route ───────────────────────────────────────────────────────


def _control_route(monkeypatch, row, *, service_returns=None):
    """The real control route with the service's writes faked: `service_returns` is what pause/stop return."""
    from src.api.v1.endpoints import trainings as endpoint
    from src.services import training_finalize_service
    from src.workers import training_finalize_tasks

    async def service(db, training_id):
        return service_returns

    async def get_training(db, training_id):
        return row

    for name in ("pause_training", "stop_training"):
        monkeypatch.setattr(endpoint.TrainingService, name, staticmethod(service))
    monkeypatch.setattr(endpoint.TrainingService, "get_training", staticmethod(get_training))
    revoked, queued = [], []
    monkeypatch.setattr(endpoint, "revoke_task", lambda *args, **kwargs: revoked.append(args))
    monkeypatch.setattr(training_finalize_service, "list_checkpoint_steps", lambda training_id: [5])
    # `apply_async` since OSD-36c: Stop & Finalize goes through `_queue_visible`,
    # which mints the Celery id so the task_queue row can carry it. `queued` keeps
    # the positional args, so the assertions read unchanged.
    def _dispatch(*args, **kwargs):
        queued.append(tuple(kwargs.get("args", args)))
        return SimpleNamespace(id=kwargs.get("task_id", "the-finalize-task"))

    monkeypatch.setattr(
        training_finalize_tasks.finalize_training_from_checkpoint_task, "apply_async", _dispatch
    )
    return endpoint, revoked, queued


def _control(endpoint, action, session):
    from src.schemas.training import TrainingControlRequest

    return asyncio.run(endpoint.control_training(TrainingControlRequest(action=action), training_id="t1", db=session))


@pytest.mark.parametrize("action", ["stop", "stop_and_finalize"])
@pytest.mark.parametrize("evaluation_status", ["pending", "running"])
def test_a_stop_on_a_completed_training_requests_only_its_evaluations_stop(monkeypatch, action, evaluation_status):
    """NEGATIVE CONTROL: `_stop_the_evaluation_of_a_completed_training` not called -> red (400)."""
    row = SimpleNamespace(id="t1", status="completed", celery_task_id="c1", hyperparameters={"total_steps": 6},
                          evaluation={"status": evaluation_status, "trigger": "post_run", "task_id": "c1"})
    endpoint, revoked, queued = _control_route(monkeypatch, row)
    session = _AsyncSession()
    result = _control(endpoint, action, session)

    assert result["success"] is True and result["status"] == TrainingStatus.COMPLETED, result
    assert "evaluation is being stopped" in result["message"], result
    assert row.status == "completed" and row.evaluation["status"] == evaluation_status
    assert row.evaluation["stop_requested_by"] == action and row.evaluation["stop_requested_at"]
    assert session.commits == 1 and revoked == [] and queued == []


@pytest.mark.parametrize("evaluation", [None, {"status": "completed"}, {"status": "cancelled"}])
def test_a_stop_on_a_completed_training_with_nothing_evaluating_is_still_refused(monkeypatch, evaluation):
    row = SimpleNamespace(id="t1", status="completed", celery_task_id="c1", evaluation=evaluation)
    endpoint, revoked, queued = _control_route(monkeypatch, row)
    session = _AsyncSession()
    with pytest.raises(HTTPException) as refused:
        _control(endpoint, "stop", session)
    assert refused.value.status_code == 400 and session.commits == 0 and row.evaluation == evaluation


@pytest.mark.parametrize("evaluation, says_stop", [({"status": "running"}, True), ({"status": "completed"}, False)])
def test_a_pause_on_a_completed_training_is_refused_with_the_reason(monkeypatch, evaluation, says_stop):
    """NEGATIVE CONTROL: `_refuse_pausing_a_completed_training` not called -> red (400, not 409)."""
    row = SimpleNamespace(id="t1", status="completed", evaluation=evaluation)
    endpoint, _revoked, _queued = _control_route(monkeypatch, row)
    with pytest.raises(HTTPException) as refused:
        _control(endpoint, "pause", _AsyncSession())
    assert refused.value.status_code == 409 and "nothing to pause" in refused.value.detail
    assert ("send stop" in refused.value.detail) is says_stop
    assert row.status == "completed"


def test_a_pause_the_service_refuses_for_another_reason_keeps_its_400(monkeypatch):
    endpoint, _revoked, _queued = _control_route(monkeypatch, SimpleNamespace(id="t1", status="failed", evaluation=None))
    with pytest.raises(HTTPException) as refused:
        _control(endpoint, "pause", _AsyncSession())
    assert refused.value.status_code == 400


@pytest.mark.parametrize("export_step, finalizes", [(6, False), (5, True)], ids=["full-length", "CONTROL-checkpoint-step"])
def test_stop_and_finalize_that_lands_as_the_run_finishes_does_not_finalize(monkeypatch, tmp_path, export_step, finalizes):
    """The Stop reached a live row (the service stopped it), but the loop's export is already on disk.

    NEGATIVE CONTROL: the route's full-length check removed -> red on full-length (a finalize is queued)."""
    from src.core.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    _write_export(tmp_path, "t1", step=export_step)
    stopped = SimpleNamespace(id="t1", status="cancelled", celery_task_id=None, hyperparameters={"total_steps": 6})
    endpoint, _revoked, queued = _control_route(monkeypatch, stopped, service_returns=stopped)
    result = _control(endpoint, "stop_and_finalize", _AsyncSession())
    assert result["success"] is True
    if finalizes:
        assert queued == [("t1", None)], queued
    else:
        assert queued == [] and "full-length export is saved" in result["message"], (queued, result)


# ── the evaluation reads the operator's Stop from its record ────────────────


def test_request_evaluation_stop_marks_only_a_pending_or_running_record():
    from src.services import training_evaluation as te

    for document in (None, {}, {"status": "completed"}, {"status": "cancelled"}, {"status": "failed"}):
        assert te.request_evaluation_stop(document, requested_by="stop") is None
        assert te.requested_stop_reason(document) is None
    for status in ("pending", "running"):
        marked = te.request_evaluation_stop({"status": status, "task_id": "x"}, requested_by="stop", now="T")
        assert marked == {"status": status, "task_id": "x", "stop_requested_at": "T", "stop_requested_by": "stop"}
        assert "operator" in te.requested_stop_reason(marked)
        assert te.requested_stop_reason({"status": status}) is None


def test_carry_stop_request_keeps_a_stop_only_for_the_same_evaluation_while_it_runs():
    from src.services import training_evaluation as te

    stored = te.request_evaluation_stop({"status": "running", "task_id": "a"}, requested_by="stop", now="T")
    carried = te.carry_stop_request(stored, {"status": "running", "task_id": "a", "progress": 1})
    assert carried["stop_requested_at"] == "T" and carried["progress"] == 1
    assert "stop_requested_at" not in te.carry_stop_request(stored, {"status": "cancelled", "task_id": "a"})
    assert "stop_requested_at" not in te.carry_stop_request(stored, {"status": "running", "task_id": "b"})
    assert "stop_requested_at" not in te.carry_stop_request({"status": "running", "task_id": "a"},
                                                           {"status": "running", "task_id": "a"})
    newer = te.carry_stop_request(stored, {"status": "running", "task_id": "a", "stop_requested_at": "U"})
    assert newer["stop_requested_at"] == "U"


def _evaluate(tmp_path, row, *, load, should_stop=None):
    from test_training_evaluation import _Db, _get_db, _jumprelu, _tokenization

    from src.services import training_evaluation as te

    db = _Db(row)
    document = te.run_evaluation(
        get_db=_get_db(db), training_id="t1", hp={"seed": 3}, saes={(1, "residual"): _jumprelu()},
        sources=[te.EvalSource(label="ext", dataset_path=_tokenization(tmp_path), rows_read=4)],
        load_base_model=load, trigger="post_run", token_budget=24, batch_tokens=8, should_stop=should_stop,
    )
    return document, db


def _counting_llama():
    from test_training_evaluation import _llama

    model = _llama()
    forwards = []
    model.register_forward_hook(lambda module, inputs, output: forwards.append(1))
    return model, forwards


def test_an_operators_stop_written_to_the_record_stops_the_evaluation_at_its_next_batch(tmp_path):
    """The Stop lands while the model loads. NEGATIVE CONTROL: the record read removed from
    `stop_reason` -> red (the evaluation completes)."""
    from test_training_evaluation import _Row

    from src.services import training_evaluation as te

    control_row = _Row(id="t1", evaluation=None)
    control_model, control_forwards = _counting_llama()
    document, _db = _evaluate(tmp_path / "control", control_row, load=lambda: (control_model, None))
    assert document["status"] == "completed", document.get("reason")

    row = _Row(id="t1", evaluation=None)
    model, forwards = _counting_llama()

    def load():
        row.evaluation = te.request_evaluation_stop(row.evaluation, requested_by="stop")
        return model, None

    document, db = _evaluate(tmp_path / "stopped", row, load=load)
    assert document["status"] == "cancelled" and "operator" in document["reason"], document
    assert [w["status"] for w in db.writes].count("completed") == 0
    assert len(forwards) < len(control_forwards), (len(forwards), len(control_forwards))
    assert len(forwards) <= 2, f"{len(forwards)} forwards ran after the Stop"


def test_a_stop_pressed_while_the_evaluation_was_pending_stops_it_before_the_model_loads(tmp_path):
    """The training wrote `pending` with COMPLETED; the operator pressed Stop before the evaluation's first
    write. NEGATIVE CONTROLS: `write_evaluation` not carrying the request -> red (the model loads);
    the check before the model load removed -> red (the model loads)."""
    from test_training_evaluation import _Row

    from src.services import training_evaluation as te

    pending = te.post_run_placeholder(task_id=None)
    row = _Row(id="t1", evaluation=te.request_evaluation_stop(pending, requested_by="stop_and_finalize"))
    loads = []

    def load():
        loads.append(True)
        return _counting_llama()[0], None

    document, db = _evaluate(tmp_path, row, load=load)
    assert loads == [], "the base model loaded although the evaluation had been stopped while pending"
    assert document["status"] == "cancelled" and "stop_and_finalize" in document["reason"], document
    assert db.writes[0]["status"] == "running" and db.writes[0]["stop_requested_by"] == "stop_and_finalize"


def test_the_post_run_placeholder_is_pending_or_a_cancelled_record_with_its_reason():
    from src.services import training_evaluation as te

    pending = te.post_run_placeholder(task_id="t", now="T")
    assert pending["status"] == "pending" and pending["trigger"] == "post_run" and pending["task_id"] == "t"
    assert te.requested_stop_reason(pending) is None
    cancelled = te.post_run_placeholder(task_id="t", not_run_reason="not run: because", now="T")
    assert cancelled["status"] == "cancelled" and cancelled["reason"] == "not run: because"
    assert cancelled["status"] not in te.ACTIVE_STATUSES
    assert torch.is_tensor(torch.zeros(1))  # the module imports torch for the tiny models above
