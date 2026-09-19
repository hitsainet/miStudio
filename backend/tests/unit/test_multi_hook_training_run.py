"""A training over two hook types and two layers runs to completion, on REAL Postgres (review R1-A, A5).

Before migration d5a1f3c7e9b2 every per-SAE metric row was keyed (training, step, layer), so
the second hook type's row at the first log step violated the table's unique key and the
training FAILED at step 0. Its held-out row hit the same key inside a broad except and was
silently dropped. Nothing in the suite had run a training with more than one hook type.

WHAT IS REAL. train_sae_task and resume_training_task end to end on the CPU over real .npy
extractions (a residual AND an mlp file per layer), real SAEs, the fixed pool, the held-out
split and evaluation, checkpoints and training state on disk, and every training_metrics
row written by the task's own log_metric into a real Postgres table built exactly as the
migrations build it (test_training_metrics_hook_key.replay_migrated_table). Faked, as in
test_training_resume_equivalence: every OTHER row (in memory), GPU placement, model loading
and WebSocket emits.

MUTATION CONTROLS (one line broken at a time, this file run, bytes restored, sha256 and the
working tree verified clean; the full table is in the A5 record):
  M1 the per-SAE log_metric call drops hook_type      -> both tests (IntegrityError, FAILED at step 0)
  M2 the held-out log_metric call drops hook_type     -> both tests (the mlp held-out rows are missing)
  K3 (was M3) log_metric does not put hook_type on the row     -> both tests
  M4 the resume's discard_metrics_after_step removed  -> the resume test (R1-A control C1, re-run)
"""

import json
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models.training_metric import TrainingMetric
from tests.unit import test_training_resume_equivalence as E
from tests.unit.test_training_metrics_hook_key import replay_migrated_table, scratch_engine

LAYERS = [0, 1]
HOOKS = ["residual", "mlp"]
SAES = [(layer, hook) for layer in LAYERS for hook in HOOKS]

CONFIG = dict(
    # No useful buffer on the card in either process, so both plan cpu_all and the
    # saved plan is reused on resume (the harness plans for real since R1D-1/R1D-2).
    gpu_tokens=(0, 0),
    source_kind="fixed_activation_pool",
    kill_latents=[],
    hp=dict(
        architecture_type="standard_saelens", batch_size=64, learning_rate=1e-3, l1_alpha=1e-3,
        warmup_steps=2, training_layers=LAYERS, hook_types=HOOKS,
        holdout_fraction=0.34, holdout_eval_tokens=16, holdout_eval_chunk_tokens=8,
        resample_dead_neurons=False,
    ),
    resamples_after_checkpoint=False,
)


def _multi_hook_extraction(root, name, seed):
    """E._extraction, with a DIFFERENT activation file for every (layer, hook)."""
    folder = root / name
    folder.mkdir(parents=True)
    gen = np.random.default_rng(seed)
    for layer in LAYERS:
        for offset, hook in enumerate(HOOKS):
            scale = 1.0 + 2.0 * offset + layer
            acts = (scale * gen.normal(size=(E.ROWS, E.SEQ, E.D))).astype(np.float32)
            np.save(folder / ("layer_" + str(layer) + "_" + hook + ".npy"), acts)
    metadata = dict(num_samples_processed=E.ROWS, layer_indices=LAYERS, hook_types=HOOKS)
    (folder / "metadata.json").write_text(json.dumps(metadata))
    return SimpleNamespace(id="ext_m_" + name, status="completed", output_path=str(folder), dataset_id="ds_" + name)


class _MetricsOnPostgres(E._Session):
    """The equivalence harness's in-memory session, with training_metrics on real Postgres.

    Adds, queries and deletes of TrainingMetric go to a live SQLAlchemy session; commit
    commits it (rolling back and re-raising on failure, as get_sync_db does), so the
    unique index is enforced exactly where production enforces it.
    """

    def __init__(self, store, real):
        super().__init__(store)
        self.real = real

    def query(self, model):
        if model is TrainingMetric:
            return self.real.query(model)
        return super().query(model)

    def add(self, obj):
        if isinstance(obj, TrainingMetric):
            self.real.add(obj)
            return
        super().add(obj)

    def commit(self):
        try:
            self.real.commit()
        except BaseException:
            self.real.rollback()
            raise

    def rollback(self):
        self.real.rollback()


@pytest.fixture
def world(monkeypatch, tmp_path):
    engine = scratch_engine("multi_hook_run")
    with engine.begin() as conn:
        replay_migrated_table(conn)
    real = Session(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(E, "_extraction", _multi_hook_extraction)
    monkeypatch.setattr(E, "_Session", lambda store: _MetricsOnPostgres(store, real))

    def harness(name):
        return E._Harness(monkeypatch, tmp_path / name, CONFIG)

    def rows():
        real.rollback()
        return real.execute(text(
            "SELECT step, layer_idx, hook_type, loss FROM training_metrics ORDER BY step, layer_idx, hook_type"
        )).all()

    def clear():
        real.rollback()
        real.execute(text("DELETE FROM training_metrics"))
        real.commit()

    try:
        yield SimpleNamespace(harness=harness, rows=rows, clear=clear)
    finally:
        real.close()
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS training_metrics"))
        engine.dispose()


def _expected_keys(steps):
    keys = []
    for step in steps:
        keys.append((step, None, None))
        for layer, hook in SAES:
            keys.append((step, layer, hook))
            keys.append((step, -1 - layer, hook))
    return sorted(keys, key=repr)


def _keys(rows):
    return sorted(((r.step, r.layer_idx, r.hook_type) for r in rows), key=repr)


def test_a_two_hook_two_layer_training_completes_and_logs_every_sae_once_per_step(world):
    harness = world.harness("straight")
    outcome = harness.run()
    assert outcome["status"] == "completed", outcome

    rows = world.rows()
    counts = Counter((r.step, r.layer_idx, r.hook_type) for r in rows)
    assert max(counts.values()) == 1, [k for k, n in counts.items() if n > 1][:5]
    assert _keys(rows) == _expected_keys(range(E.TOTAL)), "every SAE, in-sample and held-out, once per log step"

    # The two hooks of a layer are two SAEs, not one row written twice: different data,
    # different losses.
    by_step = defaultdict(dict)
    for r in rows:
        by_step[r.step][(r.layer_idx, r.hook_type)] = r.loss
    for step, losses in by_step.items():
        for layer in LAYERS:
            assert losses[(layer, "residual")] != losses[(layer, "mlp")], (step, layer)

    # The aggregate (what trainings.current_loss and the progress event carry) is the
    # mean over every SAE of every hook: one value per step, never one hook's.
    for step, losses in by_step.items():
        in_sample = [losses[key] for key in SAES]
        assert losses[(None, None)] == pytest.approx(sum(in_sample) / len(in_sample)), step


def test_a_multi_hook_run_that_crashed_past_its_checkpoint_resumes_without_colliding(world):
    """A worker lost two steps after the step-10 checkpoint: steps 11 and 12 were logged
    for every SAE of both hooks, and the resume re-runs them."""
    straight = world.harness("straight")
    assert straight.run()["status"] == "completed"
    straight_rows = [(r.step, r.layer_idx, r.hook_type, r.loss) for r in world.rows()]
    world.clear()

    resumed = world.harness("resumed")
    resumed.crash_at = 13
    with pytest.raises(RuntimeError, match="worker lost"):
        resumed.run()
    resumed.crash_at = None
    logged = _keys(r for r in world.rows() if r.step in (11, 12))
    assert logged == _expected_keys([11, 12]), "precondition: steps after the checkpoint were logged for every SAE"

    _, kwargs, outcome = E._resume(resumed)
    assert kwargs["start_step"] == 11
    assert outcome["status"] == "completed", outcome
    assert [(r.step, r.layer_idx, r.hook_type, r.loss) for r in world.rows()] == straight_rows
