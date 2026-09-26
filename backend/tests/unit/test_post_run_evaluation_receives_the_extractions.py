"""The cached-activation training hands the post-run evaluation THIS training's extractions.

SAE TRAINING REMEDIATION, ITEM 6. The post-run evaluation reads only the rows each
extraction never read (``>= max_samples``). It learns which extractions those are
from the ``extractions`` argument ``train_sae_task`` passes to
``run_post_run_evaluation``. Pass the wrong list, an empty one, or ``None``, and
the evaluation either reads rows the SAE trained on or records "skipped", with
every other test green.

WHY A RUN AND NOT THE SYNTAX TREE. Mutation control P3 (``extractions=None``) was
caught only by an AST keyword check, which a call sending ``extractions=[]`` or
another training's list would also satisfy. This runs the real task on the cached
path, on CPU, over tiny ``.npy`` extractions. Stubbed: the database session, the
memory estimate, placement, the export writer, WebSocket emits, the progress guard,
and the post-run step itself, which records what it receives. The training loop,
the extraction loading and the call site are real.

The extractions are listed in the training in the REVERSE of their row order in
the session, and their ``max_samples`` differ, so "the same ids, in the training's
order, with their max_samples" cannot be satisfied by construction.

MUTATION CONTROLS (2026-09-15; applied alone by the WS-EVAL runner, this file run,
source restored and checked by sha256). All went red:
  P3r extractions=None                                   -> test_the_post_run_step_receives_this_trainings_extractions_in_its_order
  P3b extractions=[]                                     -> the same test
  P3c extractions=list(reversed(extractions))            -> the same test
The AST keyword test alone caught only P3; it would have passed P3b and P3c.
  P7  the pre-evaluation release_job_memory() replaced by pass -> test_it_runs_after_the_export_with_the_trained_saes_and_the_placement
      (review R1-C: the release is now release_before_evaluation(), on both paths; R1-C's R4e
      removes that call and the order test goes red, as do R1D-4/R1D-5)
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models.activation_extraction import ActivationExtraction
from src.models.model import Model
from src.models.training import Training
from src.services.gpu_placement import Placement

D, SEQ = 16, 4


def _matches(row, condition) -> bool:
    """Evaluate ``Model.column == value`` against an in-memory row."""
    left, right = getattr(condition, "left", None), getattr(condition, "right", None)
    key = getattr(left, "key", None)
    if key is None or not hasattr(right, "value"):
        return True
    return getattr(row, key, None) == right.value


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter_by(self, **criteria):
        return _Query(r for r in self._rows if all(getattr(r, k, None) == v for k, v in criteria.items()))

    def filter(self, *conditions):
        return _Query(r for r in self._rows if all(_matches(r, c) for c in conditions))

    def order_by(self, *columns):
        return self

    def populate_existing(self):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _Session:
    def __init__(self, rows):
        self.rows = rows
        self.added = []

    def query(self, model):
        return _Query(self.rows.get(model, []))

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def _extraction(tmp_path, ext_id, rows, max_samples, seed):
    out = tmp_path / "activations" / ext_id
    out.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    np.save(out / "layer_0_residual.npy", rng.standard_normal((rows, SEQ, D)).astype(np.float32))
    (out / "metadata.json").write_text(json.dumps({
        "extraction_id": ext_id,
        "dataset_path": str(tmp_path / "no_such_tokenization"),
        "layer_indices": [0],
        "hook_types": ["residual"],
        "max_samples": max_samples,
        "num_samples_processed": rows,
    }))
    return SimpleNamespace(id=ext_id, output_path=str(out), status="completed", max_samples=max_samples,
                           dataset_id=f"ds_{ext_id}")


@pytest.fixture
def cached_run(monkeypatch, tmp_path):
    from src.workers import base_task, training_tasks, websocket_emitter

    first = _extraction(tmp_path, "ext_m_x_a", rows=5, max_samples=5, seed=1)
    second = _extraction(tmp_path, "ext_m_x_b", rows=3, max_samples=3, seed=2)
    training = SimpleNamespace(
        id="train_cached1", model_id="m_x", status="pending", current_step=0, current_loss=None,
        dataset_id="ds_x", dataset_ids=["ds_x"],
        # The training's order is b, a; the session holds a, b.
        extraction_id=second.id, extraction_ids=[second.id, first.id],
        hyperparameters={
            "hidden_dim": D, "latent_dim": 32, "batch_size": 64, "learning_rate": 1e-3,
            "total_steps": 3, "seed": 7, "training_layers": [0], "hook_types": ["residual"],
            "architecture_type": "standard", "l1_alpha": 1e-3, "log_interval": 1,
            "checkpoint_interval": 1000, "resample_dead_neurons": False, "warmup_steps": 0,
            "sparsity_warmup_steps": 0,
        },
        gpu_request=None, gpu_uuid=None, gpu_uuids=None, checkpoint_dir=None,
        error_message=None, error_traceback=None, completed_at=None, progress=0.0,
    )
    session = _Session({
        Training: [training],
        Model: [SimpleNamespace(id="m_x", repo_id="org/tiny", quantization="FP16", file_path=None)],
        ActivationExtraction: [first, second],
    })

    @contextmanager
    def get_sync_db():
        yield session

    state = SimpleNamespace(training=training, events=[], evaluations=[], placement=Placement(
        card=None, device=torch.device("cpu"),
    ))

    def estimate(**kwargs):
        return {"total_gb": 0.01, "total_mb": 10.0, "fits_in_6gb": True, "available_gpu_gb": 20.0}

    def export(**kwargs):
        state.events.append("export")
        return {}

    def evaluate(task, **kwargs):
        # SNAPSHOT AT CALL TIME. The task pops every SAE out of `models` right after
        # this returns; a recorder that kept the dict would see it empty.
        state.events.append("evaluate")
        state.evaluations.append({
            **kwargs,
            "models": dict(kwargs["models"]),
            "extractions": None if kwargs["extractions"] is None else list(kwargs["extractions"]),
        })
        return {"status": "completed"}

    monkeypatch.setattr(base_task, "get_sync_db", get_sync_db)
    monkeypatch.setattr(training_tasks, "estimate_training_memory", estimate)
    monkeypatch.setattr(training_tasks, "place_job", lambda *a, **k: state.placement)
    monkeypatch.setattr(training_tasks, "record_progress", lambda *a, **k: True)
    monkeypatch.setattr(training_tasks, "run_post_run_evaluation", evaluate)
    # The release before the evaluation is `release_before_evaluation` since review R1-C
    # (for R1-D's R1D-4/R1D-5): it closes the source in place on BOTH paths, where
    # `release_job_memory` closed and detached it on the cached path only. That the
    # memory is actually freed is R1D-4/R1D-5's weakref tests; this pins the ORDER.
    real_release = training_tasks.TrainingTask.release_before_evaluation

    def release(task):
        # The cached path frees its buffer before a base model loads for the evaluation.
        state.events.append("release")
        return real_release(task)

    monkeypatch.setattr(training_tasks.TrainingTask, "release_before_evaluation", release)
    monkeypatch.setattr(training_tasks.CheckpointService, "save_multilayer_community_checkpoint",
                        staticmethod(export))
    monkeypatch.setattr(
        training_tasks.TrainingValidator, "validate_sparsity_config", staticmethod(lambda hp: ([], []))
    )
    monkeypatch.setattr(training_tasks.settings, "data_dir", tmp_path)
    monkeypatch.setattr(websocket_emitter, "emit_training_progress", lambda *a, **k: True)

    state.first, state.second = first, second
    state.run = lambda: training_tasks.train_sae_task.run(training.id)
    return state


def test_the_post_run_step_receives_this_trainings_extractions_in_its_order(cached_run):
    result = cached_run.run()

    assert result["status"] == "completed", result
    assert len(cached_run.evaluations) == 1, "the post-run evaluation must run exactly once"
    [call] = cached_run.evaluations
    extractions = call["extractions"]
    assert extractions is not None, "a cached-activation run passed no extractions"
    assert [e.id for e in extractions] == cached_run.training.extraction_ids == ["ext_m_x_b", "ext_m_x_a"]
    assert [e.max_samples for e in extractions] == [3, 5]
    assert [e.output_path for e in extractions] == [cached_run.second.output_path, cached_run.first.output_path]


def test_it_runs_after_the_export_with_the_trained_saes_and_the_placement(cached_run):
    cached_run.run()

    # Export, then free the training buffer, then evaluate: the base model loads into
    # the memory the buffer held.
    assert cached_run.events == ["export", "release", "evaluate"]
    [call] = cached_run.evaluations
    assert call["training_id"] == "train_cached1"
    assert call["placement"] is cached_run.placement
    assert call["base_model"] is None, "a cached run holds no base model; the step must load one"
    assert list(call["models"]) == [(0, "residual")]
    assert call["sae_mb"] == 10.0
