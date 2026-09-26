"""R1-D seam reproductions R1D-4..R1D-7, carried into R1-C with the fixes (2026-09-15).

Copied from R1-D's `test_r1d_seam_reproductions.py` (worktree agent-afddbbff77e161652):
its in-memory session and `_World` harness verbatim, and only the four reproductions in
R1-C's files. R1D-1/2/3 stay with the integrator. Each FAILED at 60d2c05d:

  R1D-4  the cached path freed neither its activation pool/buffer nor its last optimizer
         before the evaluation loaded a base model (step-loop locals still referenced them);
  R1D-5  the on-the-fly path released nothing before its evaluation;
  R1D-6  a Stop during the evaluation was overwritten with COMPLETED;
  R1D-7  a Stop during the evaluation did not stop it (35 of 36 forwards ran after it).

The fixes and their negative controls are recorded in R1-C's review record,
.claude/context/sessions/review_sae_remediation_R1_C_2026-09-15.md.

THE SESSION (review round 2, R2-B). This file kept a private copy of the in-memory session
that enforced the OLD key, (training, step, layer), and treated every filter condition as
equality. It now uses the one shared harness, `test_training_resume_equivalence._Session`,
which enforces the key migration d5a1f3c7e9b2 built, (training, step, layer,
COALESCE(hook_type, '')), and honours comparison operators; `test_r1d3_resume_metric_collision`'s
COMPANION proves that session and the real table agree. Negative controls re-run on the
shared session are in the R2-B record.
"""

import gc
import json
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models.activation_extraction import ActivationExtraction
from src.models.dataset import Dataset
from src.models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from src.models.model import Model
from src.models.training import Training
from src.services.gpu_placement import Placement
from tests.unit.test_training_resume_equivalence import _Session

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

        # THE MEMORY READINGS, per process. A resumed run is a new process on a
        # card and a host whose free memory is whatever it is at that moment.
        self.ram_bytes = 10**10
        monkeypatch.setattr(activation_service, "_available_memory_bytes", lambda: self.ram_bytes)
        self.gpu_free = None  # None: no reading, so the task takes its own fallback

        def mem_get_info(device=None):
            if self.gpu_free is None:
                raise RuntimeError("no CUDA device in this test")
            return int(self.gpu_free), 24 * 1024**3

        monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)

        self.plans = []
        real_plan = training_tasks.activation_buffer.plan_activation_storage

        def plan(*args):
            self.plans.append(real_plan(*args))
            return self.plans[-1]

        monkeypatch.setattr(training_tasks.activation_buffer, "plan_activation_storage", plan)

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


# ── the cached path ─────────────────────────────────────────────────────────

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


def _gpu_free_for(tokens_per_layer, *, keys=1):
    """The free-memory reading at which the cached budget leaves ``tokens_per_layer`` of buffer."""
    from src.workers.training_tasks import sae_buffer_budget

    held_back = -sae_buffer_budget(0, num_keys=keys, hidden_dim=D, latent_dim=LATENT, batch_size=BATCH)["available"]
    return held_back + int(np.ceil(tokens_per_layer * D * 4 * keys / 0.9)) + 1024


# ── the on-the-fly path ─────────────────────────────────────────────────────

VOCAB, WIDTH, HIDDEN = 512, 16, 32
KEYS = [(1, "residual"), (2, "residual")]


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2,
    )
    return LlamaForCausalLM(config).eval()


def _tokenized(rows, seed):
    from datasets import Dataset as ArrowDataset

    rng = np.random.default_rng(seed)
    ids, masks = [], []
    for r in range(rows):
        real = WIDTH if r % 11 else 6
        row = rng.integers(2, VOCAB, WIDTH).tolist()
        ids.append(row[:real] + [0] * (WIDTH - real))
        masks.append([1] * real + [0] * (WIDTH - real))
    return ArrowDataset.from_dict({"input_ids": ids, "attention_mask": masks})


def _on_the_fly_world(monkeypatch, tmp_path, *, rows_per_dataset, hp=None):
    import datasets as hf_datasets

    from src.workers import training_tasks

    model = _tiny_llama()
    corpora = {"ds_a": _tokenized(rows_per_dataset, 1), "ds_b": _tokenized(rows_per_dataset, 2)}
    training = _training_row(
        "train_r1d_fly",
        {
            "hidden_dim": HIDDEN, "latent_dim": 64, "batch_size": BATCH, "learning_rate": 1e-3,
            "total_steps": 12, "log_interval": 1, "checkpoint_interval": 5, "seed": 5,
            "training_layers": [1, 2], "hook_types": ["residual"], "architecture_type": "jumprelu",
            "sparsity_coeff": 1e-3, "sparsity_warmup_steps": 0, "warmup_steps": 0,
            "resample_dead_neurons": False, "evaluate_ce_delta": False, **(hp or {}),
        },
        model_id="m_tiny",
    )
    rows_by_model = {
        Model: [SimpleNamespace(id="m_tiny", repo_id="org/tiny-llama", quantization="FP16", file_path=None,
                                params_count=None, architecture="llama",
                                architecture_config={"hidden_size": HIDDEN})],
        Dataset: [SimpleNamespace(id="ds_a"), SimpleNamespace(id="ds_b")],
        DatasetTokenization: [
            SimpleNamespace(dataset_id=name, model_id="m_tiny", status=TokenizationStatus.READY,
                            tokenized_path=f"/tok/{name}", tokenizer_repo_id="org/tiny-llama",
                            vocab_size=VOCAB, max_length=WIDTH)
            for name in corpora
        ],
    }
    world = _World(monkeypatch, tmp_path, training, rows_by_model)

    def by_name(path):
        return corpora[str(path).rstrip("/").rsplit("/", 1)[-1]]

    monkeypatch.setattr(training_tasks, "select_tokenization_for_model",
                        lambda candidates, model_id, ds_id: next(c for c in candidates if c.dataset_id == ds_id))
    monkeypatch.setattr(training_tasks, "load_from_disk", by_name)
    # The post-run evaluation opens the held-out rows' tokenization itself.
    monkeypatch.setattr(hf_datasets, "load_from_disk", by_name)
    monkeypatch.setattr(
        training_tasks, "load_model_from_hf",
        lambda **kw: (model, SimpleNamespace(pad_token_id=0, eos_token_id=1, vocab_size=VOCAB), model.config, {}),
    )
    world.model, world.corpora = model, corpora
    return world



# ── R1D-4 / R1D-5: memory still held when the evaluation loads a model ──────


def _record_optimizers(monkeypatch, tasks, memory):
    real_adam = torch.optim.Adam

    class RecordingAdam(real_adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            memory.append((f"optimizer {sum(1 for label, _ in memory if label.startswith('optimizer'))}",
                           weakref.ref(self)))

    monkeypatch.setattr(tasks.optim, "Adam", RecordingAdam)


def _spy_evaluation(monkeypatch, tasks, memory, alive, calls):
    def evaluate(task, **kwargs):
        gc.collect()
        alive.extend(label for label, ref in memory if ref() is not None)
        calls.append(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(tasks, "run_post_run_evaluation", evaluate)


@pytest.mark.parametrize("storage", ["gpu_all", "gpu_rolling"])
def test_the_cached_path_frees_its_activations_and_optimizers_before_the_evaluation(monkeypatch, tmp_path, storage):
    """R1D-4 (HIGH on cards where the buffer was sized to fill them). Two layers. When
    `run_post_run_evaluation` starts, nothing the training alone needed may still be alive:
    the comment above the release says the base model loads into that memory."""
    from src.services import activation_buffer

    if storage == "gpu_all":
        world = _cached_world(monkeypatch, tmp_path, rows=6, seq=8, layers=(0, 1), hp={"total_steps": 4})
    else:
        world = _cached_world(monkeypatch, tmp_path, rows=4_000, seq=16, layers=(0, 1), hp={"total_steps": 4})
        world.gpu_free = _gpu_free_for(80_000, keys=2)

    memory, alive, calls = [], [], []
    real_sampler_init = activation_buffer.FixedPoolSampler.__init__

    def sampler_init(self, tensors, keys, *, seed):
        for key in keys:
            memory.append((f"pool {key}", weakref.ref(tensors[key])))
        real_sampler_init(self, tensors, keys, seed=seed)

    real_storage_for = activation_buffer.RollingActivationBuffer._storage_for

    def storage_for(self, key, total):
        storage = real_storage_for(self, key, total)
        if not any(ref() is storage for _, ref in memory):
            memory.append((f"buffer {key}", weakref.ref(storage)))
        return storage

    monkeypatch.setattr(activation_buffer.FixedPoolSampler, "__init__", sampler_init)
    monkeypatch.setattr(activation_buffer.RollingActivationBuffer, "_storage_for", storage_for)
    _record_optimizers(monkeypatch, world.tasks, memory)
    _spy_evaluation(monkeypatch, world.tasks, memory, alive, calls)

    assert world.run()["status"] == "completed"
    assert world.plans[0][0] == storage, world.plans
    kinds = {label.split()[0] for label, _ in memory}
    assert {"optimizer", "pool" if storage == "gpu_all" else "buffer"} <= kinds, memory  # precondition
    assert len(calls) == 1 and calls[0]["base_model"] is None
    assert alive == [], f"still alive when the evaluation starts to load the base model: {alive}"


def test_the_on_the_fly_path_frees_its_buffer_and_optimizers_before_the_evaluation(monkeypatch, tmp_path):
    """R1D-5 (HIGH). The on-the-fly buffer is sized from free memory AFTER the model and the
    SAEs, so the evaluation — which keeps the model and adds full-vocabulary logits — has
    only the reserves left unless the buffer and the optimizer state go first."""
    from src.services import model_activation_source as MAS

    world = _on_the_fly_world(monkeypatch, tmp_path, rows_per_dataset=400, hp={"total_steps": 4})
    memory, alive, calls = [], [], []
    real_storage_for = MAS.ModelActivationSource._storage_for

    def storage_for(self, key, total):
        storage = real_storage_for(self, key, total)
        if not any(ref() is storage for _, ref in memory):
            memory.append((f"buffer {key}", weakref.ref(storage)))
        return storage

    monkeypatch.setattr(MAS.ModelActivationSource, "_storage_for", storage_for)
    _record_optimizers(monkeypatch, world.tasks, memory)
    _spy_evaluation(monkeypatch, world.tasks, memory, alive, calls)

    assert world.run()["status"] == "completed"
    kinds = {label.split()[0] for label, _ in memory}
    assert {"optimizer", "buffer"} <= kinds, memory  # precondition
    assert len(calls) == 1 and calls[0]["base_model"] is world.model
    assert alive == [], f"still alive while the evaluation runs the model: {alive}"


# ── R1D-6 / R1D-7: a Stop during the post-run evaluation ────────────────────


@pytest.mark.parametrize("written", ["cancelled", "paused"])
def test_a_stop_during_the_post_run_evaluation_leaves_the_run_completed(monkeypatch, tmp_path, written):
    """R1D-6, AS DECIDED IN REVIEW ROUND 3 (R3-A, R2D-1/R2D-10). R1-C's fix kept a Stop or Pause
    that landed during the evaluation: CANCELLED or PAUSED beside a full-length export, import
    locked, and a Finalize that replaced the final weights. The decision since: once the
    full-length export is saved the run is COMPLETED and stays so; a Stop cancels only the
    evaluation.

    The row is COMPLETED before the evaluation starts, and the Stop and Pause routes refuse a
    COMPLETED row, so the only way CANCELLED or PAUSED reaches it is a request that read it RUNNING
    and wrote after that commit. The task repairs it at the end.

    NEGATIVE CONTROL: the end-of-task repair removed -> red on both."""
    world = _cached_world(monkeypatch, tmp_path, rows=6, seq=8, hp={"total_steps": 4})
    seen = {}

    def evaluate(task, **kwargs):
        seen["status_at_start"] = world.training.status
        world.training.status = written
        return {"status": "completed"}

    monkeypatch.setattr(world.tasks, "run_post_run_evaluation", evaluate)
    result = world.run()
    assert seen["status_at_start"] == "completed", seen
    assert world.training.status == "completed" and result["status"] == "completed", (
        f"a {written!r} written after the run completed with its export was left as {world.training.status!r}"
    )


@pytest.mark.parametrize(
    "stop_during_evaluation", [False, True], ids=["CONTROL-no-stop", "R1D-7-stop-at-the-first-evaluation-forward"]
)
def test_a_stop_during_the_post_run_evaluation_stops_it_within_one_batch(monkeypatch, tmp_path, stop_during_evaluation):
    """R1D-7 (MEDIUM). The REAL post-run evaluation on the tiny Llama: 800 held-out rows,
    a budget that reads all of them, 4 batches of 256 rows, 2 layers — 4 forwards in the
    mean pass and 4 x 8 in the CE pass. The operator stops at the first forward. One CE
    batch is 8 forwards; anything past that is the evaluation ignoring the stop. Only
    forwards made while the evaluation is recorded as running are counted (training and
    the held-out collection never reach the LM head: LayerCapture stops the forward)."""
    world = _on_the_fly_world(
        monkeypatch, tmp_path, rows_per_dataset=2_000,
        hp={"total_steps": 4, "holdout_fraction": 0.2, "holdout_eval_tokens": 256,
            "evaluate_ce_delta": True, "evaluation_token_budget": 12_800},
    )
    forwards = []

    def hook(module, inputs, output):
        evaluation = world.training.evaluation or {}
        if evaluation.get("status") != "running":
            return
        forwards.append(1)
        if stop_during_evaluation and len(forwards) == 1:
            world.training.status = "cancelled"

    handle = world.model.lm_head.register_forward_hook(hook)
    try:
        world.run()
    finally:
        handle.remove()

    evaluation = world.training.evaluation
    assert evaluation is not None and evaluation.get("status") != "failed", evaluation
    if not stop_during_evaluation:
        assert evaluation["status"] == "completed", evaluation
        assert len(forwards) > 8, f"precondition: the evaluation made only {len(forwards)} forwards"
    else:
        after = len(forwards) - 1
        assert after <= 8, (
            f"the evaluation made {after} more forwards after the operator's Stop "
            f"(status recorded: {evaluation.get('status')!r})"
        )
