"""A full checkpoint volume PAUSES a training; it never fails it (R2D-7 / R3D-15).

THE DEFECT THIS CLOSES. Running out of disk raised OSError inside the training
loop, and the task's outer handler marks the run FAILED. That is the worst of all
outcomes, because a FAILED run's checkpoints are precisely what its resume needs
(`TrainingService.resume_training` accepts PAUSED **or** FAILED) — and the same
full disk is what may have truncated the newest one. So the run died in the state
whose recovery depends on files the failure itself had put at risk.

Three guards, at three ranges, all ending in a PAUSE:

  create     the forecast does not fit          -> 409, no row, nothing queued
  task start the volume cannot hold the run     -> paused before a card is claimed
  each save  no room for the step about to run  -> paused with the last checkpoint intact
  mid-save   ENOSPC despite the check           -> torn directory removed, then paused

REACHABILITY. Every guard is asserted CALLED by walking the AST, not by searching
the text: this file's own explanatory comments name each function, and a
substring search would match them and pass for the wrong reason. That is not a
hypothetical here — it has happened five times in one arc in this repository.

MUTATION CONTROLS are recorded in the scratchpad record for this arc; each was
applied alone, this file run, the bytes restored and the sha256 verified.
"""

import ast
import errno
import inspect
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import func, select

from src.models.dataset import Dataset, DatasetStatus
from src.models.model import Model, ModelStatus, QuantizationFormat
from src.models.training import Training
from src.schemas.training import TrainingCreate, TrainingHyperparameters
from src.services import checkpoint_disk as CD
from src.workers import training_tasks as TT
from tests.unit import test_training_resume_equivalence as E

#: The guards are the subject here, so they face the real volume. conftest's
#: `_a_roomy_checkpoint_volume` stubs `free_bytes_for` everywhere else; a test in
#: this file that wants a full or roomy disk sets its own value, and one that
#: means to read the machine must not silently get 8 TiB instead.
pytestmark = pytest.mark.real_checkpoint_disk


# ─────────────────────────── the loop, driven for real ───────────────────────────


def _recording_progress(monkeypatch):
    """Capture what the worker writes to the row, so the PAYLOAD can be asserted.

    The resume harness patches `record_progress` to a no-op, so this replaces it
    again after the harness is built. "It was called" is not enough: a pause that
    wrote the wrong status, or no reason, would pass that.
    """
    calls = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(TT, "record_progress", recorder)
    return calls


def _steps(harness):
    return sorted(c.step for c in harness.store.get(E.Checkpoint, []))


class TestTheLoopPausesRatherThanFails:
    def test_a_volume_that_fills_mid_run_pauses_before_the_next_save(self, monkeypatch, tmp_path):
        """Room is re-checked before EVERY save, because free space is shared.

        The first reading (the task's start-up check) is roomy, so the run
        starts; every reading after it is not, so the first periodic save is
        refused BEFORE it writes anything.
        """
        harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
        calls = _recording_progress(monkeypatch)

        readings = {"n": 0}

        def shrinking(_path):
            readings["n"] += 1
            return 10 * CD.GiB if readings["n"] == 1 else 4096

        monkeypatch.setattr(CD, "free_bytes_for", shrinking)

        result = harness.run()

        assert result["status"] == "paused", result
        assert result["reason"] == TT.DISK_FULL, result
        assert _steps(harness) == [], "a save was attempted on a volume with no room"

        assert calls, "the pause was never written to the row"
        _, kwargs = calls[-1]
        assert kwargs["status"] == "paused"
        assert "GiB free" in kwargs["error_message"], kwargs["error_message"]
        assert "resume" in kwargs["error_message"].lower()

    def test_the_run_is_not_marked_failed(self, monkeypatch, tmp_path):
        """The whole point: FAILED is what made a full disk unrecoverable."""
        harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
        calls = _recording_progress(monkeypatch)
        monkeypatch.setattr(CD, "free_bytes_for", lambda _p: 4096)

        result = harness.run()

        assert result["status"] == "paused"
        assert harness.training.status != "failed"
        assert all(kw.get("status") != "failed" for _a, kw in calls), calls

    def test_the_disk_is_checked_before_the_saes_are_built(self, monkeypatch, tmp_path):
        """Refusing early is the point: no card claimed, no weights allocated."""
        harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
        _recording_progress(monkeypatch)
        monkeypatch.setattr(CD, "free_bytes_for", lambda _p: 4096)

        result = harness.run()

        assert result["status"] == "paused"
        assert harness.models == [], "the SAEs were built before the disk was checked"

    def test_a_roomy_volume_still_trains_to_completion(self, monkeypatch, tmp_path):
        """The guard must not refuse a run that fits — the control for every test above.

        Without this, making `fits` always False would pass the whole class.
        """
        harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
        _recording_progress(monkeypatch)
        monkeypatch.setattr(CD, "free_bytes_for", lambda _p: 500 * CD.GiB)

        result = harness.run()

        assert result["status"] == "completed", result
        assert _steps(harness) == [5, 10, 15], _steps(harness)


class TestEnospcMidSave:
    def _fail_at_step_10(self, monkeypatch):
        real = TT.save_training_state

        def failing(step_dir, state):
            if Path(step_dir).name == "checkpoint_10":
                raise OSError(errno.ENOSPC, "No space left on device")
            return real(step_dir, state)

        monkeypatch.setattr(TT, "save_training_state", failing)

    def test_the_torn_step_is_removed_and_the_run_pauses(self, monkeypatch, tmp_path):
        """The weights were already written when the state file failed.

        That leaves a `checkpoint_10/` holding layer directories and no training
        state — a directory `training_finalize_service.list_checkpoint_steps`
        scans for and would finalize from. It has to go.
        """
        harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
        calls = _recording_progress(monkeypatch)
        self._fail_at_step_10(monkeypatch)

        result = harness.run()

        assert result["status"] == "paused", result
        assert result["reason"] == TT.DISK_FULL, result
        assert harness.step_dir(5).is_dir(), "the last complete checkpoint was lost"
        assert not harness.step_dir(10).exists(), (
            "a torn checkpoint directory was left on disk, where a finalize can see it"
        )
        assert _steps(harness) == [5], _steps(harness)

        _, kwargs = calls[-1]
        assert kwargs["status"] == "paused"
        assert "filled while writing step 10" in kwargs["error_message"]

    def test_the_weights_really_were_written_before_the_failure(self, monkeypatch, tmp_path):
        """Otherwise the removal above would be deleting nothing.

        A fixture where the step directory never existed would pass the test
        above for entirely the wrong reason, so prove the directory IS created
        by letting the same run write it and checking before the cleanup.
        """
        harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
        _recording_progress(monkeypatch)

        seen = {}
        real_remove = TT.remove_partial_step

        def watching(step_dir):
            path = Path(step_dir)
            seen["existed"] = path.is_dir()
            seen["files"] = sorted(p.name for p in path.rglob("*") if p.is_file())
            return real_remove(step_dir)

        monkeypatch.setattr(TT, "remove_partial_step", watching)
        self._fail_at_step_10(monkeypatch)

        harness.run()

        assert seen.get("existed"), "no partial directory existed to remove"
        assert any(name.endswith(".safetensors") for name in seen["files"]), seen

    def test_a_non_disk_oserror_still_fails_the_run(self, monkeypatch, tmp_path):
        """A permission error is a real failure and must stay loud.

        Swallowing every OSError as "disk full" would turn genuine breakage into
        a silent pause — a worse defect than the one being fixed.
        """
        harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
        _recording_progress(monkeypatch)
        real = TT.save_training_state

        def failing(step_dir, state):
            if Path(step_dir).name == "checkpoint_10":
                raise OSError(errno.EACCES, "Permission denied")
            return real(step_dir, state)

        monkeypatch.setattr(TT, "save_training_state", failing)

        with pytest.raises(OSError) as caught:
            harness.run()
        assert caught.value.errno == errno.EACCES

    def test_a_resume_after_the_pause_picks_the_last_complete_step(self, monkeypatch, tmp_path):
        """The pause has to leave something resumable — that is its whole purpose."""
        harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["pool-standard-mid-decay"])
        _recording_progress(monkeypatch)
        self._fail_at_step_10(monkeypatch)

        assert harness.run()["status"] == "paused"

        harness.training.status = "running"
        result = harness.tasks.resume_training_task.run(harness.training.id)
        assert harness.dispatched, result
        assert harness.dispatched[-1]["start_step"] == 6, harness.dispatched


# ──────────────────────────── the create-time refusal ────────────────────────────


async def _seed(async_session):
    model = Model(
        id="m_disk", name="tiny", repo_id="org/tiny", status=ModelStatus.READY.value,
        quantization=QuantizationFormat.FP16.value, architecture="llama", params_count=1_000,
    )
    dataset = Dataset(
        id=uuid.uuid4(), name="corpus", source="HuggingFace", status=DatasetStatus.READY,
    )
    async_session.add_all([model, dataset])
    await async_session.commit()
    return str(dataset.id)


def _request(dataset_id):
    return TrainingCreate(
        model_id="m_disk",
        dataset_ids=[dataset_id],
        hyperparameters=TrainingHyperparameters(
            hidden_dim=8, latent_dim=16, l1_alpha=0.001, learning_rate=0.0003,
            batch_size=64, total_steps=100, training_layers=[3], hook_types=["residual"],
        ),
    )


@pytest.mark.asyncio
async def test_the_endpoint_answers_409_naming_the_numbers(client, async_session, monkeypatch):
    """409 Conflict, not 400: the request is fine, the machine is not."""
    dataset_id = await _seed(async_session)
    monkeypatch.setattr(CD, "free_bytes_for", lambda _p: 4096)

    with patch("src.services.training_service._emit_training_event_sync"):
        response = await client.post(
            "/api/v1/trainings", json=_request(dataset_id).model_dump(mode="json")
        )

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    for fragment in ("Not enough disk", "free on", "reserve", "Short by"):
        assert fragment in detail, f"{fragment!r} missing from {detail!r}"


@pytest.mark.asyncio
async def test_the_refused_training_leaves_no_row_behind(client, async_session, monkeypatch):
    """Refused BEFORE the row exists, so nothing is queued and nothing is orphaned."""
    dataset_id = await _seed(async_session)
    monkeypatch.setattr(CD, "free_bytes_for", lambda _p: 4096)

    with patch("src.services.training_service._emit_training_event_sync"):
        response = await client.post(
            "/api/v1/trainings", json=_request(dataset_id).model_dump(mode="json")
        )

    assert response.status_code == 409
    count = (await async_session.execute(select(func.count()).select_from(Training))).scalar()
    assert count == 0, "a training row was created for a run that was refused"


@pytest.mark.asyncio
async def test_a_request_that_fits_is_still_created(client, async_session, monkeypatch):
    """The control: the refusal must not be unconditional.

    Without this, hard-coding `fits = False` would pass both tests above.
    """
    dataset_id = await _seed(async_session)
    monkeypatch.setattr(CD, "free_bytes_for", lambda _p: 500 * CD.GiB)

    with patch("src.services.training_service._emit_training_event_sync"), \
            patch("src.api.v1.endpoints.trainings.dispatch_gpu_task"):
        response = await client.post(
            "/api/v1/trainings", json=_request(dataset_id).model_dump(mode="json")
        )

    assert response.status_code == 201, response.text


@pytest.mark.asyncio
async def test_the_create_path_asks_about_the_hyperparameters_it_is_creating(
    async_session, monkeypatch
):
    """Assert the PAYLOAD and the CALL COUNT, not merely that something was called.

    A verdict taken on the wrong hyperparameters would forecast the wrong run and
    still answer "it fits".
    """
    from src.services import training_service

    dataset_id = await _seed(async_session)
    seen = []
    real = training_service.checkpoint_disk_verdict

    async def recording(db, hyperparameters, **kwargs):
        seen.append((hyperparameters, kwargs))
        return await real(db, hyperparameters, **kwargs)

    monkeypatch.setattr(training_service, "checkpoint_disk_verdict", recording)
    monkeypatch.setattr(CD, "free_bytes_for", lambda _p: 500 * CD.GiB)

    with patch("src.services.training_service._emit_training_event_sync"):
        await training_service.TrainingService.create_training(
            async_session, _request(dataset_id)
        )

    assert len(seen) == 1, f"the disk verdict was taken {len(seen)} times"
    hyperparameters, _ = seen[0]
    assert hyperparameters["latent_dim"] == 16
    assert hyperparameters["total_steps"] == 100


# ─────────────────────────────── reachability ───────────────────────────────


def _called_names(module) -> set:
    """Every function CALLED in a module, read from the AST.

    Not a substring search over the source: this file and the modules it guards
    both discuss these functions in prose, and a text search matches the prose.
    """
    tree = ast.parse(inspect.getsource(module))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_node = node.func
        if isinstance(func_node, ast.Attribute):
            names.add(func_node.attr)
        elif isinstance(func_node, ast.Name):
            names.add(func_node.id)
    return names


class TestEveryGuardIsReachable:
    @pytest.mark.parametrize("name", [
        "verdict_from_sync_session",   # the run-level check at task start
        "room_for_one_step",           # the per-save check
        "remove_partial_step",         # the ENOSPC cleanup
        "is_out_of_space",             # telling a full disk from a real failure
        "pause_for_disk",              # pausing instead of failing
        "checkpoint_step_bytes",       # the per-step forecast the save check uses
    ])
    def test_the_worker_calls_it(self, name):
        assert name in _called_names(TT), (
            f"{name} is implemented and tested and NOTHING IN THE WORKER CALLS IT; "
            f"the guard does not exist for any real training"
        )

    def test_the_per_save_forecast_uses_the_larger_save_shape(self):
        """The worker must size its per-save check against a mid-accumulation step.

        MUTATION CONTROL M19 SURVIVED the behavioural tests: flipping this
        keyword to False changed no outcome there, because their fake free-space
        readings lie far outside the range where the two shapes differ. A step
        that ends inside a gradient-accumulation window also saves gradients — a
        third more, 805 MB on the real three-layer run — so checking against the
        smaller shape lets a save BEGIN that cannot finish, which is exactly what
        leaves a torn checkpoint behind.

        Asserted on the AST rather than the text: this file's own prose names the
        keyword, and a substring search would match the prose and pass.
        """
        tree = ast.parse(inspect.getsource(TT))
        choices = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", "") != "checkpoint_step_bytes":
                continue
            for keyword in node.keywords:
                if keyword.arg == "include_grads":
                    choices.append(ast.literal_eval(keyword.value))

        assert choices == [True], (
            f"the worker's per-save forecast does not use the mid-accumulation "
            f"(gradient-saving) shape (found include_grads={choices})"
        )

    def test_the_create_path_calls_the_verdict(self):
        from src.services import training_service

        assert "checkpoint_disk_verdict" in _called_names(training_service), (
            "create_training does not ask whether the run will fit"
        )

    def test_the_endpoint_maps_the_refusal_to_409(self):
        """Walk to the handler and read its status code, rather than grepping for 409."""
        from src.api.v1.endpoints import trainings

        tree = ast.parse(inspect.getsource(trainings))
        codes = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            caught = node.type
            names = []
            if isinstance(caught, ast.Name):
                names = [caught.id]
            elif isinstance(caught, ast.Tuple):
                names = [e.id for e in caught.elts if isinstance(e, ast.Name)]
            if "InsufficientCheckpointDisk" not in names:
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "HTTPException":
                    for keyword in inner.keywords:
                        if keyword.arg == "status_code":
                            codes.append(ast.literal_eval(keyword.value))

        assert codes == [409], (
            f"the create endpoint does not turn InsufficientCheckpointDisk into a 409 "
            f"(found {codes})"
        )

    def test_the_route_is_in_the_live_registry(self):
        """`app.routes` is not a route list here — read the OpenAPI paths."""
        from src.main import app

        assert "post" in app.openapi()["paths"]["/api/v1/trainings"]

    def test_the_refusal_is_not_a_value_error(self):
        """A ValueError subclass would be caught by the 400 handler above it.

        The endpoint maps ValueError to 400. If `InsufficientCheckpointDisk`
        inherited from it, this refusal would silently become a 400 — the wrong
        answer, chosen by nobody.
        """
        from src.services.training_service import InsufficientCheckpointDisk

        assert not issubclass(InsufficientCheckpointDisk, ValueError)
