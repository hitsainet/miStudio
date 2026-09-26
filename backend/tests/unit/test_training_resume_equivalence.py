"""A resumed training continues exactly where it stopped — driven through the real task.

Tracker items 4 (full resume), 1 (resampling) and 7 (LR decay). `train_sae_task`
runs end to end on the CPU over real `.npy` extractions, real SAEs, real
optimizers and the real rolling buffer; only the database, GPU placement and
WebSocket emission are replaced. Two runs of one configuration are compared:

  STRAIGHT   20 steps.
  RESUMED    stopped by a pause at step 11 (after the step-10 checkpoint), then
             `resume_training_task` picks the checkpoint and dispatches the task
             again, which builds every object anew. The global RNGs are scrambled
             in between, as a fresh worker process would have them.

and required to agree EXACTLY: every weight, every optimizer moment and step
count, the learning rate every optimizer step APPLIED, every resample (its step,
latents and source rows), the per-step dead-neuron metric, the density log lines
and the is_best flags of the checkpoint rows. The fixtures are set so the things a
resume must restore actually matter: gradient accumulation with the checkpoint
mid-cycle, resamples both before and after the checkpoint, and a checkpoint inside
the LR decay window.

HARNESS NOTES.
* The first version wrapped whatever `training_tasks.create_sae` (and Adam, and
  the resample) currently WAS, so the second run's harness wrapped the first
  run's recorders and `straight.models[-1] is resumed.models[-1]`: the weight and
  optimizer comparisons compared an object with itself and passed. The real
  functions are captured once at import, and the comparisons assert the objects
  are distinct.
* The per-step `learning_rate` metric was read AFTER the step's scheduler update
  (it logged the next window's rate), so a restarted scheduler and a restored one
  could log the same value at the first resumed step. The recording Adam logs the
  rate each `step()` applies. Since review R2-C (L3) the metric carries the rate
  its step applies; test_logged_step_describes_that_step.py pins it.
* The standard configuration first had no dead latent at all, so it tested no
  resampling. Four of its latents are now killed at creation.

MUTATION CONTROLS (2026-09-15, WS-LOOP; one line of src/workers/training_tasks.py
broken at a time, this file and test_training_gpu_placement.py run, bytes restored,
sha256 verified). All red:
  B28 the loop's resample gate removed          -> both configs ("nothing was ever resampled")
  B29 trackers never updated                    -> both configs
  B30 resampled latents keep their dead count   -> both configs (the saved count is not 0)
  B31 the gate told lr_decay_steps=0            -> SURVIVED the first run: no dead latent
        falls due inside the window in these fixtures. The gate's payload is now
        asserted; re-run: red (both configs)
  B32 the checkpoint writes no training state   -> both configs, the legacy test
  B33 gradients not saved mid-accumulation      -> [rolling-jumprelu-accum]
  B34 the loop resumes AT the checkpoint step   -> both configs
  B35 resume_training_task dispatches AT it     -> first NOT RUN (its target also matched
        the loop's line); re-targeted and re-run: red (both configs)
  B36 b_dec re-initialised on resume            -> both configs
  B37 thresholds re-calibrated on resume        -> [jumprelu], never_recalibrates, legacy
  B38 RNG not restored                          -> [rolling-jumprelu-accum]
  B39 source position not restored              -> both configs
  B40 no sampler for the fixed pool             -> [pool-standard-mid-decay]
  B41 best loss not restored                    -> [rolling-jumprelu-accum] (is_best flags)
  B42 activity EMA / firing rate not restored   -> both configs (dead_neurons, density lines)
  B43 a checkpoint missing a layer resumes      -> the partial-resume test
  B44 ste_bandwidth for every architecture      -> [pool-standard], partial-resume test
  B45 scheduler without the accumulation factor -> [rolling-jumprelu-accum]
  B46 scheduler without the decay               -> [pool-standard-mid-decay]

REVIEW ROUND 1 (R1-A, 2026-09-15; same procedure; record in
.claude/context/sessions/review_sae_remediation_R1_A_2026-09-15.md):
  The fake session appended anything, and every test paused on the step right after
  the checkpoint, so a resume re-logging steps could never collide. The session now
  enforces uq_training_metrics_tid_step_layer; the new pause-at-13 test was red
  (IntegrityError) before discard_metrics_after_step existed.
  C1 the resume's metric discard removed         -> the pause-at-13 test
  C2 discard boundary `>=` (drops the checkpoint's own step) -> both equivalence configs,
        the pause-at-13 test, the real-Postgres discard test
  L5 the loop tells the gate warmup 0 / sparsity warmup 0 -> SURVIVED: the sparsity warmup was
        0 in both configs and warmup ended before the first resample. The pool config now
        has a sparsity warmup and the payload check covers both; re-run: red (both configs)
  L25 resume_training_task ignores missing files -> SURVIVED: no resume ever lost a file.
        New test_the_resume_task_skips_a_newest_step_whose_file_is_gone_and_the_run_recovers;
        re-run: red
  L26 a state for another step accepted          -> SURVIVED. New
        test_a_training_state_for_another_step_is_refused; re-run: red
  Re-run of WS-LOOP's own fixes, still red: B32, B34, B38, B39, B45.
"""

import json
import logging
import operator
import random
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sqlalchemy.exc import IntegrityError

from src.ml import sparse_autoencoder as SAE
from src.models.activation_extraction import ActivationExtraction
from src.models.checkpoint import Checkpoint
from src.models.model import Model
from src.models.training import Training
from src.models.training_metric import TrainingMetric
from src.services import activation_buffer as _activation_buffer
from src.services.checkpoint_service import load_training_state
from src.services.gpu_placement import Placement
from src.services.lr_schedule import lr_multiplier
from src.workers import training_tasks as _training_tasks

# The REAL functions, captured before any harness wraps them.
_REAL_CREATE_SAE = _training_tasks.create_sae
_REAL_RESAMPLE = _training_tasks.resample_dead_latents
_REAL_RESAMPLE_DUE = _training_tasks.resample_due
_REAL_STOP = _training_tasks.stop_signal_for
_REAL_ADAM = torch.optim.Adam
_REAL_CALIBRATE = SAE.JumpReLUSAE.calibrate_thresholds
_REAL_PLAN = _activation_buffer.plan_activation_storage

D, LATENT, SEQ, ROWS = 8, 16, 8, 6
TOTAL = 20
STOP_AT = 11

#: The planner's floor for a useful buffer, lowered for these fixtures: at the
#: production floor (50,000 tokens) a 96-token pool always loads whole, so no
#: rolling run this size could exist. Nothing else about the planner is changed.
MIN_USEFUL_TOKENS = 32

CONFIGS = {
    # Rolling buffer, gradient accumulation (batch 16 -> 4 steps), JumpReLU with
    # resamples allowed at 5, 10 and 15 (the decay window starts at 16).
    # MEMORY: the card holds 48 tokens when the run starts and 60 when it resumes, so
    # the resumed process's own plan (60 tokens, other quotas) differs from the
    # checkpoint's; the checkpoint's plan fits, so it must be the one used.
    "rolling-jumprelu-accum": dict(
        gpu_tokens=(48, 60),
        plans=(("gpu_rolling", 48), ("gpu_rolling", 60)),
        source_kind="rolling_activation_buffer",
        kill_latents=[],
        hp=dict(
            architecture_type="jumprelu", batch_size=16, learning_rate=1e-3, sparsity_coeff=1e-3,
            warmup_steps=2, lr_decay_steps=4, target_l0=0.01,
            resample_dead_neurons=True, resample_interval=5, dead_neuron_threshold=2,
        ),
        resamples_after_checkpoint=True,
    ),
    # Fixed pool, no accumulation, the standard SAE with four latents killed at
    # creation, the checkpoint INSIDE the decay window (it starts at 8).
    # MEMORY: no useful buffer on the card at first (the pool loads into RAM); at
    # resume the card could hold it (a fresh plan says gpu_all). The saved cpu_all fits.
    "pool-standard-mid-decay": dict(
        gpu_tokens=(0, 200),
        plans=(("cpu_all", 96), ("gpu_all", 96)),
        source_kind="fixed_activation_pool",
        kill_latents=[0, 1, 2, 3],
        hp=dict(
            architecture_type="standard_saelens", batch_size=64, learning_rate=1e-3, l1_alpha=1e-3,
            # A sparsity warmup, so the gate's sparsity_warmup_steps payload is not
            # 0 by construction (R1-A control L5 survived while it was).
            warmup_steps=2, lr_decay_steps=12, sparsity_warmup_steps=3,
            resample_dead_neurons=True, resample_interval=5, dead_neuron_threshold=2,
        ),
        resamples_after_checkpoint=False,
    ),
}


_COMPARISONS = {
    operator.gt: operator.gt, operator.ge: operator.ge,
    operator.lt: operator.lt, operator.le: operator.le, operator.ne: operator.ne,
}


class _Query:
    def __init__(self, rows, store=None, model=None):
        self._rows = list(rows)
        self._store, self._model = store, model

    def _narrowed(self, rows):
        return _Query(rows, self._store, self._model)

    def filter_by(self, **criteria):
        return self._narrowed(r for r in self._rows if all(getattr(r, k, None) == v for k, v in criteria.items()))

    def filter(self, *conditions):
        rows = self._rows
        for condition in conditions:
            left, right = getattr(condition, "left", None), getattr(condition, "right", None)
            if left is None or right is None or not hasattr(right, "value"):
                continue
            compare = _COMPARISONS.get(getattr(condition, "operator", None), operator.eq)
            rows = [r for r in rows if getattr(r, left.key, None) is not None
                    and compare(getattr(r, left.key), right.value)]
        return self._narrowed(rows)

    def order_by(self, *columns):
        return self

    def populate_existing(self):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)

    def delete(self, synchronize_session=None):
        doomed = {id(r) for r in self._rows}
        kept = [r for r in self._store.get(self._model, []) if id(r) not in doomed]
        self._store[self._model] = kept
        return len(doomed)


def _metric_key(metric):
    """The key `uq_training_metrics_tid_step_layer_hook` enforces, as Postgres evaluates it.

    (training_id, step, layer_idx, COALESCE(hook_type, '')). A NULL layer_idx is never
    equal to anything, so aggregated rows are not keyed at all (None is returned).
    Migration d5a1f3c7e9b2; `test_training_metrics_hook_key.py` proves the real index
    agrees with this on real Postgres.
    """
    if metric.layer_idx is None:
        return None
    return (metric.training_id, metric.step, metric.layer_idx, metric.hook_type or "")


class _Session:
    """The database the task sees. It enforces the one constraint a resume can hit.

    `training_metrics` is unique on (training_id, step, layer_idx, COALESCE(hook_type,
    '')) since review R1-A A5 (it was (training_id, step, layer_idx), which a training
    over two hook types violated at its first log step), and Postgres treats NULL
    layer_idx rows as distinct. A session that appended anything let a resume
    re-insert the metrics of steps it re-runs, which the real table refuses.
    """

    def __init__(self, store):
        self.store = store

    def query(self, model):
        return _Query(self.store.get(model, []), self.store, model)

    def add(self, obj):
        key = _metric_key(obj) if isinstance(obj, TrainingMetric) else None
        if key is not None and any(_metric_key(m) == key for m in self.store.get(TrainingMetric, [])):
            raise IntegrityError(
                "INSERT INTO training_metrics", key,
                Exception('duplicate key value violates unique constraint "uq_training_metrics_tid_step_layer_hook"'),
            )
        self.store.setdefault(type(obj), []).append(obj)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def _extraction(root: Path, name: str, seed: int) -> SimpleNamespace:
    folder = root / name
    folder.mkdir(parents=True)
    gen = np.random.default_rng(seed)
    centres = gen.normal(size=(3, D))
    labels = gen.integers(0, 3, size=(ROWS, SEQ))
    acts = (centres[labels] + 0.3 * gen.normal(size=(ROWS, SEQ, D))).astype(np.float32)
    np.save(folder / "layer_0_residual.npy", acts)
    (folder / "metadata.json").write_text(json.dumps({
        "num_samples_processed": ROWS, "layer_indices": [0], "hook_types": ["residual"],
    }))
    return SimpleNamespace(id=f"ext_m_{name}", status="completed", output_path=str(folder), dataset_id=f"ds_{name}")


class _Harness:
    """One training row and its world. Its recorders wrap the REAL functions."""

    def __init__(self, monkeypatch, tmp_path, config):
        from src.core import database
        from src.services import activation_buffer, activation_service
        from src.workers import base_task, websocket_emitter

        training_tasks = _training_tasks
        self.tasks = training_tasks
        self.data_dir = tmp_path / "data"
        self.hp = {
            "hidden_dim": D, "latent_dim": LATENT, "total_steps": TOTAL, "checkpoint_interval": 5,
            "log_interval": 1, "seed": 7, "training_layers": [0], "hook_types": ["residual"],
            "sparsity_warmup_steps": 0, "evaluate_ce_delta": False, "grad_clip_norm": None,
            **config["hp"],
        }
        extractions = [_extraction(tmp_path, "a", 1), _extraction(tmp_path, "b", 2)]
        self.training = SimpleNamespace(
            id="train_resume", model_id="m_x", status="pending", current_step=0, current_loss=None,
            dataset_id="ds_a", dataset_ids=["ds_a", "ds_b"], extraction_id=None,
            extraction_ids=[e.id for e in extractions], hyperparameters=dict(self.hp),
            gpu_request="auto", gpu_uuid=None, gpu_uuids=None, checkpoint_dir=None,
            error_message=None, error_traceback=None, completed_at=None, progress=0.0,
        )
        self.store = {
            Training: [self.training],
            Model: [SimpleNamespace(id="m_x", repo_id="org/tiny", quantization="FP16", file_path=None,
                                    params_count=None, architecture_config=None)],
            ActivationExtraction: extractions,
        }
        session = _Session(self.store)

        @contextmanager
        def get_sync_db():
            yield session

        # THE REAL PLANNER over per-process memory readings (review round 1, R1D-1). This
        # used to be a constant `plan_activation_storage`, so a resumed process planned
        # the checkpoint's buffer whatever memory it saw, and the resume that fails on
        # every card passed here.
        self.config = config
        self.plans = []
        self.gpu_free = _gpu_free_for(config["gpu_tokens"][0], self.hp["batch_size"])
        self.ram_bytes = 10**10

        def plan(total, gpu_capacity, useful, ram_capacity):
            self.plans.append(_REAL_PLAN(total, gpu_capacity, min(useful, MIN_USEFUL_TOKENS), ram_capacity))
            return self.plans[-1]

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
        monkeypatch.setattr(activation_service, "_available_memory_bytes", lambda: self.ram_bytes)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (int(self.gpu_free), 24 * 1024**3))
        monkeypatch.setattr(activation_buffer, "plan_activation_storage", plan)
        monkeypatch.setattr(websocket_emitter, "emit_training_progress", lambda *a, **k: True)
        monkeypatch.setattr(websocket_emitter, "emit_checkpoint_created", lambda *a, **k: True)

        # What THIS harness's runs built and did.
        self.models, self.optimizers, self.resamples, self.calibrations = [], [], [], []
        models, optimizers, resamples, calibrations = self.models, self.optimizers, self.resamples, self.calibrations
        due = {"step": None}
        kill = list(config["kill_latents"])

        def recording_create(**kwargs):
            model = _REAL_CREATE_SAE(**kwargs)
            if kill:
                with torch.no_grad():
                    model.encoder.bias[kill] = -1e6
            models.append(model)
            return model

        class RecordingAdam(_REAL_ADAM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.applied_lrs = []
                optimizers.append(self)

            def step(self, closure=None):
                self.applied_lrs.append(self.param_groups[0]["lr"])
                return super().step(closure)

        self.due_calls = []

        def recording_due(step, **kwargs):
            self.due_calls.append(kwargs)
            answer = _REAL_RESAMPLE_DUE(step, **kwargs)
            if answer:
                due["step"] = step
            return answer

        def recording_resample(*args, **kwargs):
            result = _REAL_RESAMPLE(*args, **kwargs)
            resamples.append((due["step"], result.latents.tolist(), result.source_rows.tolist()))
            return result

        def counting_calibrate(model, *args, **kwargs):
            calibrations.append(1)
            return _REAL_CALIBRATE(model, *args, **kwargs)

        monkeypatch.setattr(training_tasks, "create_sae", recording_create)
        monkeypatch.setattr(training_tasks.optim, "Adam", RecordingAdam)
        monkeypatch.setattr(training_tasks, "resample_due", recording_due)
        monkeypatch.setattr(training_tasks, "resample_dead_latents", recording_resample)
        monkeypatch.setattr(SAE.JumpReLUSAE, "calibrate_thresholds", counting_calibrate)

        self.stop_at = None
        #: A worker that dies before `crash_at` runs: no pause, no checkpoint.
        self.crash_at = None

        def stop(row, step, lease_lost=None, **kw):
            if self.crash_at is not None and step == self.crash_at:
                raise RuntimeError(f"worker lost before step {step}")
            if self.stop_at is not None and step == self.stop_at:
                return {"status": "paused", "step": step}
            return _REAL_STOP(row, step, lease_lost, **kw)

        monkeypatch.setattr(training_tasks, "stop_signal_for", stop)
        self.dispatched = []
        monkeypatch.setattr(training_tasks, "gpu_delay", lambda task, request: lambda **kw: self.dispatched.append(kw))

    def run(self, **kwargs):
        try:
            return self.tasks.train_sae_task.run(self.training.id, **kwargs)
        finally:
            self.tasks.train_sae_task._close_activation_stream()

    def applied_lrs(self):
        return [lr for optimizer in self.optimizers for lr in optimizer.applied_lrs]

    def metric_series(self, name):
        return {m.step: getattr(m, name) for m in self.store.get(TrainingMetric, []) if m.layer_idx is None}

    def checkpoint_flags(self):
        return sorted((c.step, c.is_best) for c in self.store.get(Checkpoint, []))

    def step_dir(self, step):
        return self.data_dir / "trainings" / self.training.id / "checkpoints" / f"checkpoint_{step}"

    def saved_state(self, step):
        return load_training_state(self.step_dir(step))


def _scramble_global_rngs():
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    torch.rand(37)


def _gpu_free_for(tokens, batch_size):
    """The free-memory reading at which the cached budget leaves ``tokens`` of buffer per layer."""
    from src.workers.training_tasks import sae_buffer_budget

    held_back = -sae_buffer_budget(0, num_keys=1, hidden_dim=D, latent_dim=LATENT, batch_size=batch_size)["available"]
    return held_back + int(np.ceil(tokens * D * 4 / 0.9)) if tokens else 0


def _resume(harness):
    harness.stop_at = None
    # The row is live when a resume is dispatched: `TrainingService.resume_training` sets
    # it RUNNING first. A harness "crash" raises inside the task, whose handler records
    # FAILED, which a real lost worker never gets to write. A task starting on a terminal
    # row returns without training (review round 2, R2D-5).
    harness.training.status = "running"
    # A resumed run is a new process, reading whatever memory is free then.
    harness.gpu_free = _gpu_free_for(harness.config["gpu_tokens"][1], harness.hp["batch_size"])
    result = harness.tasks.resume_training_task.run(harness.training.id)
    assert len(harness.dispatched) == 1, harness.dispatched
    kwargs = harness.dispatched[-1]
    _scramble_global_rngs()
    outcome = harness.run(start_step=kwargs["start_step"], checkpoint_id=kwargs["checkpoint_id"])
    return result, kwargs, outcome


def _density_lines(caplog):
    return [r.getMessage() for r in caplog.records if "density" in r.getMessage()]


def _assert_same_weights(a, b):
    assert a is not b, "the comparison would be with itself"
    for name, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[name]), f"{name} differs"


def _assert_same_optimizer(a, b):
    assert a is not b, "the comparison would be with itself"
    sa, sb = a.state_dict(), b.state_dict()
    assert sa["param_groups"] == sb["param_groups"]
    assert sa["state"].keys() == sb["state"].keys() and sa["state"]
    for pid, entry in sa["state"].items():
        for name, value in entry.items():
            assert torch.equal(value, sb["state"][pid][name]), f"param {pid} {name} differs"


@pytest.mark.parametrize("name", list(CONFIGS))
def test_a_resumed_run_equals_an_uninterrupted_one(monkeypatch, tmp_path, caplog, name):
    config = CONFIGS[name]
    hp = config["hp"]
    caplog.set_level(logging.INFO, logger=_training_tasks.logger.name)

    straight = _Harness(monkeypatch, tmp_path / "straight", config)
    caplog.clear()
    assert straight.run()["status"] == "completed"
    straight_density = _density_lines(caplog)
    assert len(straight.models) == 1 and len(straight.optimizers) == 1

    resumed = _Harness(monkeypatch, tmp_path / "resumed", config)
    resumed.stop_at = STOP_AT
    caplog.clear()
    assert resumed.run()["status"] == "paused"
    resamples_before_stop = len(resumed.resamples)
    result, kwargs, outcome = _resume(resumed)
    resumed_density = _density_lines(caplog)

    # Neither harness recorded the other's runs.
    assert len(straight.models) == 1 and len(resumed.models) == 2

    # The resume chose the NEWEST checkpoint (step 10, not the best one) and
    # continues after it.
    assert result["start_step"] == 11 and kwargs["start_step"] == 11
    chosen = next(c for c in resumed.store[Checkpoint] if c.id == kwargs["checkpoint_id"])
    assert chosen.step == 10
    assert outcome["status"] == "completed"

    # The checkpoint recorded the data source's position, whichever source fed the run.
    assert resumed.saved_state(10)["activation_source"]["kind"] == config["source_kind"]

    # THE REAL SCENARIO (review round 1, R1D-1). The resumed process read different
    # memory and planned a different buffer; the checkpoint recorded its own plan, the
    # plan still fits, so the source was built from it and the resume is exact.
    first_plan, resumed_fresh_plan = config["plans"]
    assert straight.plans == [first_plan], straight.plans
    assert resumed.plans == [first_plan, resumed_fresh_plan], resumed.plans
    saved_plan = resumed.saved_state(10)["activation_plan"]
    assert (saved_plan["mode"], saved_plan["buffer_tokens"]) == first_plan, saved_plan
    history = resumed.saved_state(15)["resume_history"]
    assert [(r["activation_plan"], r["bit_identical"]) for r in history] == [("reused", True)], history
    # Reused means the MODE and the SIZE too, not only the quotas: with only the quotas
    # reused the batches still agree, so this is what sees it (control M2).
    assert history[0]["plan"] == history[0]["saved_plan"], history[0]
    assert straight.saved_state(15)["resume_history"] == [], "an uninterrupted run recorded a resume"

    # Preconditions: the fixture really exercised what a resume must restore.
    performed = [(step, latents) for step, latents, _ in straight.resamples if latents]
    assert performed, "nothing was ever resampled"
    if config["resamples_after_checkpoint"]:
        assert any(latents for _, latents, _ in straight.resamples[resamples_before_stop:]), (
            "no resample after the checkpoint, so the RNG restore is untested"
        )

    # Resampling never runs inside the LR decay window — and the loop really tells
    # the gate where that window is. (No dead latent happens to fall due inside it
    # in these fixtures, so without the payload check a gate told decay=0 passed:
    # control B31.)
    assert all(step < TOTAL - hp["lr_decay_steps"] for step, _ in performed), performed
    assert straight.due_calls, "the loop never consulted the resample gate"
    assert all(
        call["lr_decay_steps"] == hp["lr_decay_steps"] and call["total_steps"] == TOTAL
        and call["warmup_steps"] == hp["warmup_steps"]
        and call["sparsity_warmup_steps"] == hp.get("sparsity_warmup_steps", 0)
        for call in straight.due_calls
    ), straight.due_calls[:3]
    # A resampled latent restarts its dead count: the checkpoint written at the
    # same step records zero for it.
    checked = 0
    for step, latents in performed:
        if step % 5 == 0 and resumed.step_dir(step).exists():
            dead_state = straight.saved_state(step)["saes"]["layer_0_residual"]["dead_latents"]
            counts = dead_state["steps_since_fired"]
            assert all(int(counts[i]) == 0 for i in latents), (step, latents, counts.tolist())
            # ...and WHICH latents, not just that their count restarted (acceptance
            # item 16, controls R1/R2). The counter above reads zero for a latent
            # that merely fired this step too, so it cannot identify the revived
            # set; this is the only record that can.
            recorded = dead_state["resampled"]
            at_step = [t for s, t in recorded if s == step]
            assert at_step, (step, [s for s, _ in recorded])
            assert sorted(int(i) for i in at_step[0]) == sorted(int(i) for i in latents), (
                step, at_step[0].tolist(), latents
            )
            checked += 1
    assert checked, "no resample coincided with a checkpoint"

    _assert_same_weights(straight.models[-1], resumed.models[-1])
    _assert_same_optimizer(straight.optimizers[-1], resumed.optimizers[-1])
    assert resumed.resamples == straight.resamples
    assert resumed.applied_lrs() == pytest.approx(straight.applied_lrs())
    assert resumed.metric_series("dead_neurons") == straight.metric_series("dead_neurons")
    assert resumed_density == straight_density and straight_density
    assert resumed.checkpoint_flags() == straight.checkpoint_flags()

    # And the curve applied is the configured one: optimizer step e covers
    # training steps e*k .. e*k + k - 1 and reads the factor at e*k.
    accum = max(1, 64 // hp["batch_size"]) if hp["batch_size"] < 64 else 1
    expected = [
        hp["learning_rate"] * lr_multiplier(
            e * accum, total_steps=TOTAL, warmup_steps=hp["warmup_steps"], decay_steps=hp["lr_decay_steps"]
        )
        for e in range(TOTAL // accum)
    ]
    assert straight.applied_lrs() == pytest.approx(expected)


def _metric_rows(harness):
    return sorted(
        (m.step, -10**9 if m.layer_idx is None else m.layer_idx, m.loss)
        for m in harness.store.get(TrainingMetric, [])
    )


def test_a_resume_re_runs_logged_steps_without_colliding_or_duplicating(monkeypatch, tmp_path):
    """Interrupted two steps AFTER the step-10 checkpoint.

    Every other test here stopped at step 11, the step right after the checkpoint,
    so nothing had been logged that the resume re-runs. A run interrupted several
    logged steps past its newest checkpoint resumes from that checkpoint and logs
    those steps again: the table refuses the per-layer rows
    (uq_training_metrics_tid_step_layer) and the resume fails, and the aggregated
    rows (NULL layer, never refused) would be doubled.

    A CRASH, not a pause: a pause now checkpoints the step it stopped after (R1-D L8,
    test_pause_saves_a_checkpoint.py), so it no longer re-runs anything. A worker that
    dies still does.
    """
    config = CONFIGS["pool-standard-mid-decay"]
    straight = _Harness(monkeypatch, tmp_path / "straight", config)
    assert straight.run()["status"] == "completed"

    resumed = _Harness(monkeypatch, tmp_path / "resumed", config)
    resumed.crash_at = 13
    with pytest.raises(RuntimeError, match="worker lost"):
        resumed.run()
    resumed.crash_at = None
    logged = {m.step for m in resumed.store[TrainingMetric] if m.layer_idx == 0}
    assert {11, 12} <= logged, "precondition: steps after the checkpoint were logged before the crash"
    assert max(c.step for c in resumed.store[Checkpoint]) == 10, "precondition: the crash saved nothing"

    _, kwargs, outcome = _resume(resumed)
    assert kwargs["start_step"] == 11
    assert outcome["status"] == "completed"
    assert _metric_rows(resumed) == _metric_rows(straight)


def test_the_resume_task_skips_a_newest_step_whose_file_is_gone_and_the_run_recovers(monkeypatch, tmp_path):
    """Rows say step 10 is whole; its file says otherwise. The resume takes step 5.

    Every other resume here had every file on disk, so `resume_training_task`
    could ignore the file check entirely (R1-A control L25 survived).
    """
    harness = _Harness(monkeypatch, tmp_path, CONFIGS["pool-standard-mid-decay"])
    harness.stop_at = STOP_AT
    assert harness.run()["status"] == "paused"
    (harness.step_dir(10) / "layer_0_residual" / "checkpoint.safetensors").unlink()

    result, kwargs, outcome = _resume(harness)

    chosen = next(c for c in harness.store[Checkpoint] if c.id == kwargs["checkpoint_id"])
    assert chosen.step == 5
    assert result["start_step"] == kwargs["start_step"] == 6
    assert outcome["status"] == "completed"
    assert (harness.step_dir(10) / "layer_0_residual" / "checkpoint.safetensors").exists()


def test_a_training_state_for_another_step_is_refused(monkeypatch, tmp_path):
    """A step directory holding another step's state must not resume as if it matched.

    No test ever put a mismatched state in place (R1-A control L26 survived).
    """
    harness = _Harness(monkeypatch, tmp_path, CONFIGS["pool-standard-mid-decay"])
    harness.stop_at = STOP_AT
    assert harness.run()["status"] == "paused"
    shutil.copyfile(harness.step_dir(5) / "training_state.pt", harness.step_dir(10) / "training_state.pt")
    step10 = next(c for c in harness.store[Checkpoint] if c.step == 10)
    harness.stop_at = None
    with pytest.raises(ValueError, match="holds training state for step 5"):
        harness.run(start_step=11, checkpoint_id=step10.id)


def test_the_resumed_run_never_recalibrates_thresholds(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, CONFIGS["rolling-jumprelu-accum"])
    harness.stop_at = STOP_AT
    assert harness.run()["status"] == "paused"
    assert len(harness.calibrations) == 1, "precondition: a fresh JumpReLU run calibrates once"
    _resume(harness)
    assert len(harness.calibrations) == 1


def test_a_checkpoint_missing_a_layer_is_refused_rather_than_resumed_partly(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, CONFIGS["pool-standard-mid-decay"])
    harness.stop_at = STOP_AT
    assert harness.run()["status"] == "paused"
    step10 = next(c for c in harness.store[Checkpoint] if c.step == 10)
    (harness.step_dir(10) / "layer_0_residual" / "checkpoint.safetensors").unlink()
    harness.stop_at = None
    with pytest.raises(ValueError, match="partial resume"):
        harness.run(start_step=11, checkpoint_id=step10.id)


def test_a_legacy_weights_only_checkpoint_resumes_with_a_warning(monkeypatch, tmp_path, caplog):
    harness = _Harness(monkeypatch, tmp_path, CONFIGS["rolling-jumprelu-accum"])
    harness.stop_at = STOP_AT
    assert harness.run()["status"] == "paused"
    removed = list(harness.data_dir.rglob("training_state.pt"))
    assert removed, "precondition: the run wrote training state"
    for state in removed:
        state.unlink()

    with caplog.at_level(logging.WARNING, logger=harness.tasks.logger.name):
        result, kwargs, outcome = _resume(harness)

    assert outcome["status"] == "completed"
    assert "LEGACY CHECKPOINT" in caplog.text and "warmup restarts" in caplog.text
    assert len(harness.calibrations) == 1, "the loaded thresholds were re-calibrated"
    resumed_optimizer = harness.optimizers[-1]
    # The warmup really restarted: the first update after the resume ran at factor 0.
    assert resumed_optimizer.applied_lrs[0] == 0.0
    # And Adam really restarted: its step count covers the resumed updates only
    # (training steps 11, 15, 19 end accumulation windows of 4).
    assert int(resumed_optimizer.state_dict()["state"][0]["step"]) == 3
