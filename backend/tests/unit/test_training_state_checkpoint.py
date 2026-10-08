"""Checkpoints that can resume: training_state.pt, choosing a step, and deleting it (tracker item 4).

The end-to-end equivalence (2N straight == N, checkpoint, resume, N) is in
test_training_resume_equivalence.py; this file pins the pieces it rests on.

MUTATION CONTROLS (2026-09-15, WS-LOOP; applied alone, this file and the
resume-equivalence file run, bytes restored, sha256 verified). All red:
  B13 torch.load(weights_only=False)            -> test_a_pickled_object_is_refused
  B14 optimizer state not restored               -> round trip, both equivalence configs
  B15 scheduler state not restored               -> round trip, both equivalence configs
  B16 dead-latent tracker not restored           -> SURVIVED the first run: after seven
        steps every latent of the fixture had just fired, so its counts were zero like a
        fresh tracker's. The fixture now sets distinct non-zero counts; re-run: red
        (the round-trip test)
  B17 accumulated gradients not restored         -> grads round trip, equivalence [accum]
  B18 resume selection takes the OLDEST step     -> newest_complete_step_not_the_best,
        both equivalence configs, the legacy resume (4 failed)
  B19 selection accepts an incomplete step       -> the three selection tests
  B20 state not deleted with the step's last layer -> both deletion tests
  B21 torch RNG not restored                     -> rng round trip, equivalence [jumprelu]
  B22 retention estimate ignores the state       -> test_retention_counts_the_state_once_per_step

REVIEW ROUND 1 (R1-A, 2026-09-15; same procedure, record in
.claude/context/sessions/review_sae_remediation_R1_A_2026-09-15.md):
  L12 GradScaler state not restored              -> SURVIVED: the round-trip scaler never left
        its initial scale. New test_a_grad_scaler_that_skipped_steps_resumes_mid_accumulation
        (two overflows, saved mid-window); re-run: red
  L16 training state written in place            -> SURVIVED: nothing wrote over an existing
        state and failed. New test_a_failed_rewrite_leaves_the_previous_state_loadable; re-run: red
  L18 numpy RNG not restored                     -> test_rng_states_continue_the_same_sequences
Not controllable on this machine: the CUDA RNG capture (`get_rng_state_all`), which needs a GPU.
"""

import pickle
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.ml.sparse_autoencoder import create_sae
from src.services import checkpoint_service as CS
from src.services.checkpoint_retention import RetentionPolicy, plan_from_checkpoints
from src.services.checkpoint_service import CheckpointService
from src.services.dead_latent_resampling import DeadLatentTracker
from src.services.lr_schedule import build_lr_scheduler

KEYS = [(3, "residual"), (4, "residual")]


def _objects(seed=0):
    torch.manual_seed(seed)
    models = {k: create_sae("jumprelu", hidden_dim=8, latent_dim=16, sparsity_coeff=1e-3) for k in KEYS}
    optimizers = {k: torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.0, 0.999)) for k, m in models.items()}
    schedulers = {
        k: build_lr_scheduler(o, total_steps=50, warmup_steps=5, decay_steps=20) for k, o in optimizers.items()
    }
    scalers = {k: torch.amp.GradScaler("cpu") for k in KEYS}
    trackers = {k: DeadLatentTracker(16) for k in KEYS}
    return models, optimizers, schedulers, scalers, trackers


def _train(objects, steps, seed):
    models, optimizers, schedulers, scalers, trackers = objects
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randn(32, 8, generator=gen)
        for k in KEYS:
            optimizers[k].zero_grad()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, z, losses = models[k](x, return_loss=True)
            scalers[k].scale(losses["loss"]).backward()
            scalers[k].unscale_(optimizers[k])
            scalers[k].step(optimizers[k])
            scalers[k].update()
            schedulers[k].step()
            trackers[k].update(z)


def _state(objects, step=7, **extra):
    models, optimizers, schedulers, scalers, trackers = objects
    return CS.build_training_state(
        step=step, sae_keys=KEYS, optimizers=optimizers, schedulers=schedulers, scalers=scalers,
        dead_latent_trackers=trackers,
        activation_ema={k: torch.full((16,), float(i)) for i, k in enumerate(KEYS)},
        firing_rate={k: torch.full((16,), 0.5) for k in KEYS},
        best_loss=0.25, activation_source=None, models=models, **extra,
    )


def _fresh_with_weights(objects, seed=99):
    fresh = _objects(seed=seed)
    for k in KEYS:
        fresh[0][k].load_state_dict(objects[0][k].state_dict())
    return fresh


def _assert_optimizers_equal(a, b):
    for k in KEYS:
        sa, sb = a[k].state_dict(), b[k].state_dict()
        assert sa["param_groups"] == sb["param_groups"]
        assert sa["state"].keys() == sb["state"].keys()
        for pid in sa["state"]:
            for name, value in sa["state"][pid].items():
                assert torch.equal(value, sb["state"][pid][name]), (k, pid, name)


class TestTheStateRoundTrips:
    def test_rng_states_continue_the_same_sequences(self, tmp_path):
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        saved = CS.capture_rng_state()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        torch.save(saved, tmp_path / "rng.pt")

        random.seed(2)
        np.random.seed(2)
        torch.manual_seed(2)
        CS.restore_rng_state(torch.load(tmp_path / "rng.pt", weights_only=True))
        assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected

    def test_optimizer_scheduler_scaler_and_trackers_are_restored_into_fresh_objects(self, tmp_path):
        run = _objects()
        _train(run, steps=7, seed=5)
        # Distinct, non-zero dead counts. After seven steps every latent of this
        # tiny SAE has just fired, so its counts were all zero — the same as a
        # fresh tracker's — and a restore that skipped the tracker still passed
        # (control B16 survived until this line).
        for i, k in enumerate(KEYS):
            run[4][k].steps_since_fired += torch.arange(16) + i
        step_dir = tmp_path / "checkpoint_7"
        CS.save_training_state(step_dir, _state(run))
        assert not (step_dir / "training_state.pt.partial").exists()

        resumed = _fresh_with_weights(run)
        loaded = CS.restore_training_state(
            CS.load_training_state(step_dir), sae_keys=KEYS, optimizers=resumed[1], schedulers=resumed[2],
            scalers=resumed[3], dead_latent_trackers=resumed[4], models=resumed[0],
        )

        _assert_optimizers_equal(run[1], resumed[1])
        for k in KEYS:
            assert resumed[2][k].last_epoch == run[2][k].last_epoch == 7
            assert resumed[2][k].get_last_lr() == run[2][k].get_last_lr()
            assert resumed[3][k].get_scale() == run[3][k].get_scale()
            assert torch.equal(resumed[4][k].steps_since_fired, run[4][k].steps_since_fired)
        assert loaded["step"] == 7 and loaded["best_loss"] == 0.25
        assert torch.equal(loaded["activation_ema"][KEYS[1]], torch.ones(16))

        # And both continue identically.
        _train(run, steps=4, seed=6)
        _train(resumed, steps=4, seed=6)
        for k in KEYS:
            for name, value in run[0][k].state_dict().items():
                assert torch.equal(value, resumed[0][k].state_dict()[name]), (k, name)

    def test_gradients_saved_mid_accumulation_are_restored(self, tmp_path):
        run = _objects()
        _train(run, steps=2, seed=1)
        for k in KEYS:
            for p in run[0][k].parameters():
                p.grad = torch.full_like(p, 0.125)
        CS.save_training_state(tmp_path / "checkpoint_2", _state(run, step=2, include_grads=True))
        resumed = _fresh_with_weights(run)
        CS.restore_training_state(
            CS.load_training_state(tmp_path / "checkpoint_2"), sae_keys=KEYS, optimizers=resumed[1],
            schedulers=resumed[2], scalers=resumed[3], dead_latent_trackers=resumed[4], models=resumed[0],
        )
        for k in KEYS:
            assert all(torch.all(p.grad == 0.125) for p in resumed[0][k].parameters())

    def test_a_grad_scaler_that_skipped_steps_resumes_mid_accumulation(self, tmp_path):
        """The loop's FP16 conditions on CPU: a scaler that backed off, a checkpoint mid-window.

        The round-trip test above never moved its scaler off the initial scale, so a
        restore that skipped the scaler still passed (R1-A control L12 survived).
        Here two optimizer windows overflow (the scale backs off twice) and the
        state is saved on the FIRST step of an accumulation window, when the saved
        gradients are scaled by the backed-off scale. A fresh scaler at the initial
        scale would unscale them by the wrong factor.
        """
        accum = 2

        def scaler():
            return torch.amp.GradScaler("cpu", init_scale=2.0**10, growth_interval=3)

        def run_steps(objects, start, stop, overflow_at=()):
            models, optimizers, schedulers, scalers, trackers = objects
            for step in range(start, stop):
                x = torch.randn(32, 8, generator=torch.Generator().manual_seed(1000 + step))
                for k in KEYS:
                    if step % accum == 0:
                        optimizers[k].zero_grad()
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        _, z, losses = models[k](x, return_loss=True)
                    scalers[k].scale(losses["loss"] / accum).backward()
                    if (step + 1) % accum == 0:
                        if step in overflow_at:
                            next(models[k].parameters()).grad.fill_(float("inf"))
                        scalers[k].unscale_(optimizers[k])
                        scalers[k].step(optimizers[k])
                        scalers[k].update()
                        schedulers[k].step()
                    trackers[k].update(z)

        straight = _objects()
        straight[3].update({k: scaler() for k in KEYS})
        run_steps(straight, 0, 12, overflow_at=(3, 5))

        run = _objects()
        run[3].update({k: scaler() for k in KEYS})
        run_steps(run, 0, 7, overflow_at=(3, 5))  # step 6 opens a window: mid-accumulation
        assert all(run[3][k].get_scale() == 2.0**8 for k in KEYS), "precondition: the scale backed off twice"
        assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in run[0][k].parameters()) for k in KEYS)
        CS.save_training_state(tmp_path / "checkpoint_6", _state(run, step=6, include_grads=True))

        resumed = _fresh_with_weights(run)
        resumed[3].update({k: scaler() for k in KEYS})
        CS.restore_training_state(
            CS.load_training_state(tmp_path / "checkpoint_6"), sae_keys=KEYS, optimizers=resumed[1],
            schedulers=resumed[2], scalers=resumed[3], dead_latent_trackers=resumed[4], models=resumed[0],
        )
        run_steps(resumed, 7, 12, overflow_at=(3, 5))

        _assert_optimizers_equal(straight[1], resumed[1])
        for k in KEYS:
            assert resumed[3][k].state_dict() == straight[3][k].state_dict()
            for name, value in straight[0][k].state_dict().items():
                assert torch.equal(value, resumed[0][k].state_dict()[name]), (k, name)

    def test_a_state_for_other_saes_is_refused(self, tmp_path):
        run = _objects()
        CS.save_training_state(tmp_path / "c", _state(run))
        other = _objects()
        with pytest.raises(ValueError, match="SAEs"):
            CS.restore_training_state(
                CS.load_training_state(tmp_path / "c"), sae_keys=[KEYS[0]], optimizers=other[1],
                schedulers=other[2], scalers=other[3], dead_latent_trackers=other[4],
            )


class TestTheFileIsTrustedOnlyAsData:
    def test_a_pickled_object_is_refused(self, tmp_path):
        step_dir = tmp_path / "checkpoint_1"
        step_dir.mkdir()
        torch.save(
            {"format": CS.TRAINING_STATE_FORMAT, "version": CS.TRAINING_STATE_VERSION, "x": SimpleNamespace(a=1)},
            step_dir / CS.TRAINING_STATE_FILENAME,
        )
        with pytest.raises(pickle.UnpicklingError):
            CS.load_training_state(step_dir)

    def test_a_weights_only_checkpoint_has_no_state(self, tmp_path):
        assert CS.load_training_state(tmp_path) is None

    def test_a_failed_rewrite_leaves_the_previous_state_loadable(self, tmp_path, monkeypatch):
        """A write that dies part-way must not replace the state already on disk.

        Every earlier test checked only that no `.partial` file was left after a
        write that succeeded, so writing in place survived (R1-A control L16).
        """
        run = _objects()
        _train(run, steps=2, seed=1)
        step_dir = tmp_path / "checkpoint_2"
        CS.save_training_state(step_dir, _state(run, step=2))

        def dying_save(obj, f, *args, **kwargs):
            Path(f).write_bytes(b"truncated by a full disk")
            raise OSError("No space left on device")

        monkeypatch.setattr(CS.torch, "save", dying_save)
        with pytest.raises(OSError):
            CS.save_training_state(step_dir, _state(run, step=2))
        monkeypatch.undo()
        assert CS.load_training_state(step_dir)["step"] == 2

    def test_a_partial_write_is_not_a_state(self, tmp_path):
        (tmp_path / (CS.TRAINING_STATE_FILENAME + ".partial")).write_bytes(b"truncated")
        assert CS.load_training_state(tmp_path) is None

    def test_another_format_or_version_is_refused(self, tmp_path):
        torch.save({"format": "something else"}, tmp_path / CS.TRAINING_STATE_FILENAME)
        with pytest.raises(ValueError, match="not a miStudio"):
            CS.load_training_state(tmp_path)
        torch.save({"format": CS.TRAINING_STATE_FORMAT, "version": 99}, tmp_path / CS.TRAINING_STATE_FILENAME)
        with pytest.raises(ValueError, match="version"):
            CS.load_training_state(tmp_path)


def _row(tmp_path, step, layer, *, exists=True, is_best=False, metadata=True):
    path = tmp_path / f"checkpoint_{step}" / f"layer_{layer}_residual" / "checkpoint.safetensors"
    if exists:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"w" * 100)
    return SimpleNamespace(
        id=f"ckpt_{step}_{layer}", step=step, storage_path=str(path), is_best=is_best,
        extra_metadata={"layer_idx": layer, "hook_type": "residual"} if metadata else {},
        created_at=datetime.now(timezone.utc) - timedelta(days=3), file_size_bytes=None,
    )


class TestWhichCheckpointAResumeUses:
    def test_the_newest_complete_step_not_the_best(self, tmp_path):
        rows = [
            _row(tmp_path, 100, 3, is_best=True), _row(tmp_path, 100, 4, is_best=True),
            _row(tmp_path, 200, 3), _row(tmp_path, 200, 4),
            _row(tmp_path, 300, 3),  # layer 4 never written
        ]
        step, chosen = CS.select_resume_checkpoint(rows, KEYS)
        assert step == 200
        assert {k: r.step for k, r in chosen.items()} == {KEYS[0]: 200, KEYS[1]: 200}

    def test_a_step_whose_file_is_gone_is_skipped(self, tmp_path):
        rows = [_row(tmp_path, 100, 3), _row(tmp_path, 100, 4), _row(tmp_path, 200, 3), _row(tmp_path, 200, 4, exists=False)]
        assert CS.select_resume_checkpoint(rows, KEYS)[0] == 100

    def test_legacy_rows_are_keyed_by_their_directory(self, tmp_path):
        rows = [_row(tmp_path, 50, 3, metadata=False), _row(tmp_path, 50, 4, metadata=False)]
        assert CS.select_resume_checkpoint(rows, KEYS)[0] == 50

    def test_no_complete_step_is_none(self, tmp_path):
        assert CS.select_resume_checkpoint([_row(tmp_path, 10, 3)], KEYS) is None


class TestTheStateIsDeletedWithItsStep:
    def _step(self, tmp_path):
        rows = [_row(tmp_path, 400, 3), _row(tmp_path, 400, 4)]
        state = tmp_path / "checkpoint_400" / CS.TRAINING_STATE_FILENAME
        state.write_bytes(b"s" * 1000)
        return rows, state

    def test_it_goes_with_the_last_layer_and_is_counted(self, tmp_path):
        rows, state = self._step(tmp_path)
        assert CheckpointService.delete_checkpoint_files(rows[0].storage_path) == 100
        assert state.exists(), "the state went while a layer of its step remained"
        assert CheckpointService.delete_checkpoint_files(rows[1].storage_path) == 1100
        assert not state.exists()
        assert not (tmp_path / "checkpoint_400").exists()

    def test_a_state_that_cannot_be_deleted_raises_and_a_retry_removes_it(self, tmp_path, monkeypatch):
        rows, state = self._step(tmp_path)
        CheckpointService.delete_checkpoint_files(rows[0].storage_path)
        real_unlink = Path.unlink

        def refuse_state(self, *args, **kwargs):
            if self.name == CS.TRAINING_STATE_FILENAME:
                raise PermissionError("read-only")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", refuse_state)
        with pytest.raises(OSError):
            CheckpointService.delete_checkpoint_files(rows[1].storage_path)
        assert state.exists()
        monkeypatch.setattr(Path, "unlink", real_unlink)
        # The prune kept the step's rows; its next pass finds the layer file gone.
        assert CheckpointService.delete_checkpoint_files(rows[1].storage_path) == 1000
        assert not state.exists()

    def test_retention_counts_the_state_once_per_step(self, tmp_path):
        rows = [_row(tmp_path, 1, 3), _row(tmp_path, 1, 4), _row(tmp_path, 2, 3), _row(tmp_path, 2, 4)]
        for step in (1, 2):
            (tmp_path / f"checkpoint_{step}" / CS.TRAINING_STATE_FILENAME).write_bytes(b"s" * 1000)
        plan = plan_from_checkpoints(
            "train_x", "completed", rows,
            RetentionPolicy(enabled=True, dry_run=False, keep_last=1, keep_best=False, min_age_hours=0),
        )
        assert plan.prunable_steps == [1]
        assert plan.estimated_bytes == 100 + 100 + 1000


class TestWeightsFilesAreWeights:
    def test_save_checkpoint_writes_weights_and_load_reports_no_optimizer_state(self, tmp_path):
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(model.parameters())
        model(torch.ones(1, 4)).sum().backward()
        opt.step()
        path = tmp_path / "checkpoint.safetensors"
        CheckpointService.save_checkpoint(model=model, optimizer=opt, step=1, storage_path=str(path))
        loaded = CheckpointService.load_checkpoint(str(path))
        assert set(loaded["model_state"]) == {"weight", "bias"}
        assert loaded["optimizer_state"] == {}
