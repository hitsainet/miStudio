"""R1D-3, copied from reviewer R1-D's seam reproductions (review round 1, 2026-09-15).

Source: `.claude/worktrees/agent-afddbbff77e161652/backend/tests/unit/test_r1d_seam_reproductions.py`
(R1-D's record: `.claude/context/sessions/review_sae_remediation_R1_D_2026-09-15.md`). R1-A found
the same defect independently (A1) and fixed it in `67ed92b2` (`discard_metrics_after_step`); these
two tests are R1-D's, kept verbatim apart from the scaffolding the other R1D tests need.

  R1D-3  Resuming a run paused after a logged step past its newest checkpoint re-logs those steps,
         and `uq_training_metrics_tid_step_layer` rejects the per-layer rows: the resumed run FAILS
         at its first re-logged step. The COMPANION proves the real constraint and the real
         `log_metric` do that.

WHAT IS REAL. `train_sae_task` and `resume_training_task` end to end on the CPU over real `.npy`
extractions, the storage plan, the fixed-pool sampler, checkpoints and `training_state.pt` on disk.
Faked: the database session (in-memory rows; it enforces the one unique constraint R1D-3 needs), GPU
placement (the CPU), model loading and WebSocket emits.

AFTER R1-D L8 (`6524c8e1`) A PAUSE CHECKPOINTS THE STEP IT STOPPED AFTER, so this pause-at-13
resumes at 13, re-runs no logged step, and passes whether or not the discard exists. It stays as
R1-D's reproduction of the paused path. The discard is guarded by the CRASH variant in
`test_training_resume_equivalence.py` (a worker lost two steps past a checkpoint): control C1r (the
resume's `discard_metrics_after_step` call removed) turns that test red and leaves this one green.

THE SESSION (review round 2, R2-B). This file kept a private in-memory session that enforced the
OLD key, (training, step, layer), after migration d5a1f3c7e9b2 replaced it with (training, step,
layer, COALESCE(hook_type, '')). It now uses the one shared harness,
`test_training_resume_equivalence._Session`, and the COMPANION runs one insert sequence, with two
hook types on a layer, a NULL-hook row and aggregates, through BOTH that session and the real
table the unit suite builds, and requires the same accept/refuse outcome for every insert.

MUTATION CONTROLS ON THE SHARED SESSION (R2-B; one line broken at a time, this file run, bytes
restored, sha256 and `git diff` verified; the table is in the R2-B record):
  S1 the shared `_metric_key` drops the hook             -> the COMPANION (the session refuses an
                                                            mlp row the real table accepts)
  S2 `log_metric` does not put hook_type on the row (K3) -> the COMPANION (the real table refuses
                                                            the second hook)
  S3 pause checkpoint condition AND the resume's discard
     removed together (L8a + C1)                          -> the pause-at-13 test (IntegrityError),
                                                            as with the private session
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sqlalchemy.exc import IntegrityError

from src.models.activation_extraction import ActivationExtraction
from src.models.model import Model
from src.models.training import Training
from src.models.training_metric import TrainingMetric
from src.services.gpu_placement import Placement
from tests.unit.test_training_resume_equivalence import _Session

NEW_KEY = "uq_training_metrics_tid_step_layer_hook"


# ── the world a training runs in ────────────────────────────────────────────


class _World:
    def __init__(self, monkeypatch, tmp_path, training, rows):
        from src.core import database
        from src.services import activation_service
        from src.workers import base_task, training_tasks, websocket_emitter

        self.tasks = training_tasks
        self.task = training_tasks.train_sae_task
        self.training = training
        self.store = {Training: [training], **rows}
        self.data_dir = tmp_path / "data"
        session = _Session(self.store)

        @contextmanager
        def get_sync_db():
            yield session

        memory = {"total_gb": 0.01, "total_mb": 10.0, "fits_in_6gb": True, "available_gpu_gb": 20.0,
                  "per_layer_gb": 0.01, "max_layers_in_6gb": 10}
        monkeypatch.setattr(base_task, "get_sync_db", get_sync_db)
        monkeypatch.setattr(database, "get_sync_db", get_sync_db)
        monkeypatch.setattr(training_tasks, "estimate_training_memory", lambda **kw: memory)
        monkeypatch.setattr(training_tasks, "estimate_multilayer_training_memory", lambda **kw: memory)
        monkeypatch.setattr(training_tasks, "place_job", lambda *a, **k: Placement(card=None, device=torch.device("cpu")))
        monkeypatch.setattr(training_tasks, "record_progress", lambda *a, **k: True)
        monkeypatch.setattr(
            training_tasks.TrainingValidator, "validate_sparsity_config", staticmethod(lambda hp: ([], []))
        )
        monkeypatch.setattr(training_tasks.settings, "data_dir", self.data_dir)
        monkeypatch.setattr(websocket_emitter, "emit_training_progress", lambda *a, **k: True)
        monkeypatch.setattr(websocket_emitter, "emit_checkpoint_created", lambda *a, **k: True)

        self.ram_bytes = 10**10
        monkeypatch.setattr(activation_service, "_available_memory_bytes", lambda: self.ram_bytes)

        def mem_get_info(device=None):
            raise RuntimeError("no CUDA device in this test")

        monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)

        self.stop_at = None
        real_stop = training_tasks.stop_signal_for

        def stop(row, step, lease_lost=None, **kw):
            if self.stop_at is not None and step == self.stop_at:
                return {"status": "paused", "step": step}
            return real_stop(row, step, lease_lost, **kw)

        monkeypatch.setattr(training_tasks, "stop_signal_for", stop)
        self.dispatched = []
        monkeypatch.setattr(training_tasks, "gpu_delay", lambda task, request: lambda **kw: self.dispatched.append(kw))
        monkeypatch.setattr(self.task, "_activation_stream", None, raising=False)

    def run(self, **kwargs):
        try:
            return self.task.run(self.training.id, **kwargs)
        finally:
            self.task._close_activation_stream()

    def pause(self, stop_at):
        self.stop_at = stop_at
        result = self.run()
        self.stop_at = None
        assert result["status"] == "paused", result
        return result

    def resume(self):
        self.tasks.resume_training_task.run(self.training.id)
        assert len(self.dispatched) == 1, self.dispatched
        kwargs = self.dispatched[-1]
        return self.run(start_step=kwargs["start_step"], checkpoint_id=kwargs["checkpoint_id"])


def _training_row(training_id, hp, **fields):
    base = dict(
        id=training_id, model_id="m_x", status="pending", current_step=0, current_loss=None,
        dataset_id="ds_a", dataset_ids=["ds_a", "ds_b"], extraction_id=None, extraction_ids=None,
        gpu_request="auto", gpu_uuid=None, gpu_uuids=None, checkpoint_dir=None,
        error_message=None, error_traceback=None, completed_at=None, progress=0.0, evaluation=None,
        hyperparameters=hp,
    )
    base.update(fields)
    return SimpleNamespace(**base)


D = 8
LATENT = 16
BATCH = 64


def _extraction(root, name, *, rows, seq, layers, seed):
    folder = root / name
    folder.mkdir(parents=True)
    gen = np.random.default_rng(seed)
    for layer in layers:
        np.save(folder / f"layer_{layer}_residual.npy", gen.normal(size=(rows, seq, D)).astype(np.float32))
    (folder / "metadata.json").write_text(json.dumps({
        "num_samples_processed": rows, "layer_indices": list(layers), "hook_types": ["residual"],
    }))
    return SimpleNamespace(id=f"ext_{name}", status="completed", output_path=str(folder), dataset_id=f"ds_{name}")


def _cached_world(monkeypatch, tmp_path, *, rows, seq, layers=(0,), hp=None):
    extractions = [
        _extraction(tmp_path, name, rows=rows, seq=seq, layers=layers, seed=i + 1)
        for i, name in enumerate(("a", "b"))
    ]
    training = _training_row(
        "train_r1d_cached",
        {
            "hidden_dim": D, "latent_dim": LATENT, "batch_size": BATCH, "learning_rate": 1e-3,
            "total_steps": 20, "checkpoint_interval": 5, "log_interval": 1, "seed": 7,
            "training_layers": list(layers), "hook_types": ["residual"],
            "architecture_type": "standard_saelens", "l1_alpha": 1e-3, "warmup_steps": 0,
            "sparsity_warmup_steps": 0, "resample_dead_neurons": False,
            "evaluate_ce_delta": False, "grad_clip_norm": None, **(hp or {}),
        },
        extraction_ids=[e.id for e in extractions],
    )
    rows_by_model = {
        Model: [SimpleNamespace(id="m_x", repo_id="org/tiny", quantization="FP16", file_path=None,
                                params_count=None, architecture_config=None, architecture="llama")],
        ActivationExtraction: extractions,
    }
    return _World(monkeypatch, tmp_path, training, rows_by_model)


# ── R1D-3: resume re-inserts the metric rows logged after the checkpoint ────


def test_resuming_a_run_paused_past_a_logged_step_does_not_fail_on_its_own_metric_rows(monkeypatch, tmp_path):
    """R1D-3 (HIGH). log_interval 1, checkpoint every 5, paused at step 13: steps 11 and 12
    were logged, the newest checkpoint is step 10, so the resumed run trains and logs 11
    again. With the production defaults (checkpoint 1000, log 100) any pause more than 100
    steps after a checkpoint does the same."""
    world = _cached_world(monkeypatch, tmp_path, rows=6, seq=8)
    world.pause(13)
    logged = sorted(m.step for m in world.store[TrainingMetric] if m.layer_idx == 0)
    assert logged == list(range(13)), logged  # precondition: 11 and 12 are on record

    try:
        outcome = world.resume()
    except IntegrityError as exc:
        pytest.fail(f"the resumed run FAILED re-logging a step it had already logged before the pause: {exc}")
    assert outcome["status"] == "completed", outcome
    per_layer = sorted(m.step for m in world.store[TrainingMetric] if m.layer_idx == 0)
    assert per_layer == list(range(20)), per_layer


#: One insert sequence at one step: (layer_idx, hook_type, accepted?). Aggregates repeat freely;
#: a per-layer or held-out row is unique per hook; a NULL hook keeps the old per-layer
#: uniqueness and is a different key from 'residual'.
KEY_SEQUENCE = [
    (None, None, True), (None, None, True),
    (0, None, True), (-1, None, True),
    (0, "mlp", True), (-1, "mlp", True),
    (0, "residual", True),
    (0, None, False), (-1, None, False),
    (0, "mlp", False), (-1, "mlp", False), (0, "residual", False),
    (1, "residual", True), (1, "attention", True), (1, "attention", False),
]


def _session_outcomes():
    """KEY_SEQUENCE through the shared in-memory session every training harness here uses."""
    session, outcomes = _Session({}), []
    for layer_idx, hook, _ in KEY_SEQUENCE:
        row = TrainingMetric(training_id="train_r1d_companion", step=11, loss=1.0, layer_idx=layer_idx, hook_type=hook)
        try:
            session.add(row)
            outcomes.append(True)
        except IntegrityError as exc:
            assert NEW_KEY in str(exc), exc
            outcomes.append(False)
    return outcomes


async def test_COMPANION_the_real_constraint_rejects_a_rewritten_per_layer_row_through_log_metric(
    async_engine, monkeypatch
):
    """The real table (created from the ORM by `async_engine`, the unit suite's schema path) and
    the real `TrainingTask.log_metric` accept and refuse exactly what KEY_SEQUENCE says, and the
    shared `_Session` the training harnesses run on agrees with them insert for insert."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import NullPool

    from src.core.config import settings
    from src.workers import base_task, training_tasks

    url = str(settings.database_url_sync)
    assert "test" in url, f"refusing to write metrics to {url}"
    engine = create_engine(url, poolclass=NullPool)
    make = sessionmaker(bind=engine)

    @contextmanager
    def get_sync_db():
        session = make()
        # No trainings row is needed for a constraint check: FK triggers off.
        session.execute(text("SET session_replication_role = replica"))
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(base_task, "get_sync_db", get_sync_db)
    task = training_tasks.train_sae_task
    real = []
    try:
        for layer_idx, hook, _ in KEY_SEQUENCE:
            try:
                task.log_metric(training_id="train_r1d_companion", step=11, loss=1.0, layer_idx=layer_idx, hook_type=hook)
                real.append(True)
            except IntegrityError as exc:
                assert NEW_KEY in str(exc), exc
                real.append(False)
        with engine.connect() as conn:
            stored = sorted(
                (-10**9 if layer is None else layer, hook or "")
                for layer, hook in conn.execute(text(
                    "SELECT layer_idx, hook_type FROM training_metrics WHERE training_id = 'train_r1d_companion'"
                )).all()
            )
    finally:
        engine.dispose()

    expected = [accepted for _, _, accepted in KEY_SEQUENCE]
    assert real == expected, list(zip(KEY_SEQUENCE, real))
    assert _session_outcomes() == expected, list(zip(KEY_SEQUENCE, _session_outcomes()))
    assert stored == sorted(
        (-10**9 if layer is None else layer, hook or "") for layer, hook, accepted in KEY_SEQUENCE if accepted
    ), stored
