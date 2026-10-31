"""Tests for finalizing a stopped training from a checkpoint.

Cancelling a training skips the loop's community_format export, leaving usable
checkpoints that no downstream consumer can read. These tests pin the recovery
path against a REAL JumpReLU checkpoint round-trip.

MUTATION CONTROLS (each must turn a test red):
  * read dims from hyperparameters instead of tensor shapes -> dim-drift test fails
  * drop the layer_{idx}_{hook} -> layer_{idx} fallback -> legacy test fails
  * remove the finalize route/task registration -> reachability tests fail
"""

import pytest
import torch

from src.ml.sparse_autoencoder import create_sae
from src.services.checkpoint_service import CheckpointService
from src.services.training_finalize_service import (
    FinalizeError,
    build_model_for_checkpoint,
    list_checkpoint_steps,
    read_architecture_from_checkpoint,
    read_dims_from_checkpoint,
    resolve_finalize_step,
    resolve_layer_dirs,
)

HIDDEN = 32
LATENT = 128


def _write_jumprelu_checkpoint(path, hidden=HIDDEN, latent=LATENT, step=1000):
    """Save a real JumpReLU SAE exactly as the training loop would."""
    model = create_sae(
        architecture_type="jumprelu",
        hidden_dim=hidden,
        latent_dim=latent,
        l1_alpha=1e-3,
        initial_threshold=0.5,
        bandwidth=0.01,
        sparsity_coeff=1e-3,
        normalize_decoder=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    path.parent.mkdir(parents=True, exist_ok=True)
    CheckpointService.save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        storage_path=str(path),
        extra_metadata={"layer_idx": 34, "hook_type": "residual"},
    )
    return model


class TestCheckpointDiscovery:
    def test_lists_steps_ascending(self, tmp_path, monkeypatch):
        from src.services import training_finalize_service as svc

        monkeypatch.setattr(svc.settings, "data_dir", tmp_path)
        root = tmp_path / "trainings" / "t1" / "checkpoints"
        for step in (4000, 1000, 2000):
            (root / f"checkpoint_{step}" / "layer_34_residual").mkdir(parents=True)
        assert list_checkpoint_steps("t1") == [1000, 2000, 4000]

    def test_no_checkpoints_returns_empty(self, tmp_path, monkeypatch):
        from src.services import training_finalize_service as svc

        monkeypatch.setattr(svc.settings, "data_dir", tmp_path)
        assert list_checkpoint_steps("nope") == []

    def test_ignores_non_checkpoint_directories(self, tmp_path, monkeypatch):
        from src.services import training_finalize_service as svc

        monkeypatch.setattr(svc.settings, "data_dir", tmp_path)
        root = tmp_path / "trainings" / "t1" / "checkpoints"
        (root / "checkpoint_1000").mkdir(parents=True)
        (root / "community_format").mkdir(parents=True)
        (root / "not_a_checkpoint").mkdir(parents=True)
        assert list_checkpoint_steps("t1") == [1000]

    def test_resolve_defaults_to_newest(self, tmp_path, monkeypatch):
        from src.services import training_finalize_service as svc

        monkeypatch.setattr(svc.settings, "data_dir", tmp_path)
        root = tmp_path / "trainings" / "t1" / "checkpoints"
        for step in (1000, 8000, 4000):
            (root / f"checkpoint_{step}").mkdir(parents=True)
        assert resolve_finalize_step("t1", None) == 8000

    def test_resolve_rejects_unknown_step(self, tmp_path, monkeypatch):
        from src.services import training_finalize_service as svc

        monkeypatch.setattr(svc.settings, "data_dir", tmp_path)
        (tmp_path / "trainings" / "t1" / "checkpoints" / "checkpoint_1000").mkdir(parents=True)
        with pytest.raises(FinalizeError, match="no checkpoint at step 9999"):
            resolve_finalize_step("t1", 9999)

    def test_resolve_errors_when_nothing_on_disk(self, tmp_path, monkeypatch):
        from src.services import training_finalize_service as svc

        monkeypatch.setattr(svc.settings, "data_dir", tmp_path)
        with pytest.raises(FinalizeError, match="no on-disk checkpoints"):
            resolve_finalize_step("t1", None)


class TestLayerDirResolution:
    def test_current_layout(self, tmp_path):
        for name in ("layer_34_residual", "layer_35_residual"):
            d = tmp_path / name
            d.mkdir(parents=True)
            (d / "checkpoint.safetensors").write_bytes(b"x")
        found = resolve_layer_dirs(tmp_path)
        assert set(found) == {(34, "residual"), (35, "residual")}

    def test_legacy_layout_defaults_to_residual(self, tmp_path):
        """Legacy single-hook runs wrote layer_{idx}/ with no hook suffix."""
        d = tmp_path / "layer_7"
        d.mkdir(parents=True)
        (d / "checkpoint.safetensors").write_bytes(b"x")
        found = resolve_layer_dirs(tmp_path)
        assert set(found) == {(7, "residual")}

    def test_directory_without_checkpoint_file_is_skipped(self, tmp_path):
        (tmp_path / "layer_34_residual").mkdir(parents=True)
        assert resolve_layer_dirs(tmp_path) == {}

    def test_missing_dir_returns_empty(self, tmp_path):
        assert resolve_layer_dirs(tmp_path / "nope") == {}


class TestDimsFromCheckpoint:
    def test_reads_dims_from_tensor_shapes(self, tmp_path):
        """THE drift guard.

        The training task overwrites hp['hidden_dim'] in memory after peeking at
        the activation file and never persists it, so hyperparameters can lie.
        The checkpoint tensors cannot.
        """
        ckpt = tmp_path / "checkpoint.safetensors"
        _write_jumprelu_checkpoint(ckpt, hidden=HIDDEN, latent=LATENT)
        assert read_dims_from_checkpoint(ckpt) == (HIDDEN, LATENT)

    def test_records_architecture(self, tmp_path):
        ckpt = tmp_path / "checkpoint.safetensors"
        _write_jumprelu_checkpoint(ckpt)
        assert read_architecture_from_checkpoint(ckpt) == "JumpReLUSAE"

    def test_rebuilt_model_loads_stale_hyperparameters(self, tmp_path):
        """Rebuild must succeed even when hp['hidden_dim'] disagrees with the file."""
        ckpt = tmp_path / "checkpoint.safetensors"
        original = _write_jumprelu_checkpoint(ckpt, hidden=HIDDEN, latent=LATENT)

        stale_hp = {
            "hidden_dim": HIDDEN * 2,  # WRONG on purpose — the drift bug
            "latent_dim": LATENT * 2,
            "architecture_type": "jumprelu",
            "sparsity_coeff": 1e-3,
        }
        model, resolved = build_model_for_checkpoint(stale_hp, ckpt)
        CheckpointService.load_checkpoint(str(ckpt), model=model, device="cpu")

        for key, tensor in original.state_dict().items():
            assert torch.allclose(model.state_dict()[key], tensor), f"{key} mismatch"

        # The resolved config must carry the TRUE dims, because cfg.json is
        # written from these — not from the stale hyperparameters.
        assert resolved["hidden_dim"] == HIDDEN
        assert resolved["latent_dim"] == LATENT
        assert resolved["architecture_type"] == "jumprelu"

    def test_unreadable_checkpoint_raises(self, tmp_path):
        ckpt = tmp_path / "checkpoint.safetensors"
        from safetensors.torch import save_file

        save_file({"model.not_an_encoder": torch.zeros(4)}, str(ckpt))
        with pytest.raises(FinalizeError, match="cannot determine SAE dimensions"):
            read_dims_from_checkpoint(ckpt)


class TestFinalizeReachability:
    """A capability is not shipped until removing its wiring turns a test red."""

    def test_finalize_route_is_registered(self):
        from src.api.v1.endpoints.trainings import router

        paths = {
            (r.path, m) for r in router.routes
            if hasattr(r, "methods") for m in r.methods
        }
        assert ("/trainings/{training_id}/finalize", "POST") in paths

    def test_prune_routes_are_registered(self):
        from src.api.v1.endpoints.trainings import router

        paths = {
            (r.path, m) for r in router.routes
            if hasattr(r, "methods") for m in r.methods
        }
        assert ("/trainings/{training_id}/checkpoints/prune-preview", "GET") in paths
        assert ("/trainings/{training_id}/checkpoints/prune", "POST") in paths

    def test_delete_checkpoint_route_is_registered(self):
        """The frontend has called this route since before it existed (404ing)."""
        from src.api.v1.endpoints.trainings import router

        paths = {
            (r.path, m) for r in router.routes
            if hasattr(r, "methods") for m in r.methods
        }
        assert ("/trainings/{training_id}/checkpoints/{checkpoint_id}", "DELETE") in paths

    def test_literal_prune_routes_precede_parameterised_checkpoint_route(self):
        """FastAPI matches in declaration order.

        If /checkpoints/{checkpoint_id} were declared first it would swallow
        "prune-preview" as a checkpoint id and the preview would 404.
        """
        from src.api.v1.endpoints.trainings import router

        order = [r.path for r in router.routes if hasattr(r, "methods")]
        assert order.index("/trainings/{training_id}/checkpoints/prune-preview") < order.index(
            "/trainings/{training_id}/checkpoints/{checkpoint_id}"
        )

    def test_tasks_are_in_the_live_celery_registry(self):
        """Importable is not the same as registered — assert the registry."""
        from src.core.celery_app import celery_app

        registered = set(celery_app.tasks.keys())
        assert (
            "src.workers.training_finalize_tasks.finalize_training_from_checkpoint"
            in registered
        )
        assert "src.workers.prune_checkpoints.prune_checkpoints" in registered
        assert (
            "src.workers.prune_checkpoints.prune_single_training_checkpoints"
            in registered
        )

    def test_prune_is_on_the_beat_schedule(self):
        from src.core.celery_app import celery_app

        assert "prune-checkpoints" in celery_app.conf.beat_schedule

    @pytest.mark.parametrize(
        "task_name",
        [
            "src.workers.training_finalize_tasks.finalize_training_from_checkpoint",
            "src.workers.prune_checkpoints.prune_checkpoints",
            "src.workers.prune_checkpoints.prune_single_training_checkpoints",
        ],
    )
    def test_tasks_actually_route_to_low_priority(self, task_name):
        """Ask the ROUTER where the task goes — not the config we just wrote.

        Reading task_routes back asserts nothing: Celery matches globs against
        the TASK NAME, so a short name silently falls through to the default
        queue while the config still 'looks' correct. That exact bug shipped
        here once (finalize landed on the shared `datasets` queue, behind
        multi-hour dataset downloads).
        """
        from src.core.celery_app import celery_app

        route = celery_app.amqp.router.route({}, task_name)
        assert route["queue"].name == "low_priority"

    def test_beat_schedule_references_a_registered_task(self):
        """A beat entry naming a task that isn't registered fires into the void."""
        from src.core.celery_app import celery_app

        beat_task = celery_app.conf.beat_schedule["prune-checkpoints"]["task"]
        assert beat_task in celery_app.tasks

    def test_control_schema_accepts_stop_and_finalize(self):
        """The request Literal gates the action — a missing entry 422s."""
        from src.schemas.training import TrainingControlRequest

        assert TrainingControlRequest(action="stop_and_finalize").action == (
            "stop_and_finalize"
        )

    def test_response_schema_exposes_finalized_from_step(self):
        from src.schemas.training import TrainingResponse

        assert "finalized_from_step" in TrainingResponse.model_fields
