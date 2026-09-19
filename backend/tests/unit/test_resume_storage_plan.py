"""A resume builds the run's own activation storage: the plan is saved, reused when it fits, re-planned loudly when not.

Review round 1 of the SAE training remediation, findings R1D-1 and R1D-2 (HIGH),
2026-09-15. A rolling-buffer run and an on-the-fly run could not be resumed on real
hardware: the resumed process planned its buffer again from the free memory it
measured, built its source with slightly different quotas, and the source refused
the checkpoint's position. The fix (services/activation_plan.py) saves the plan in
training_state.pt; a resume reuses it when it still fits this process's capacities
(the ones the planner is given, with the planner's margins), and otherwise
re-plans, continues the saved pass under the new quotas without repeating a row,
logs that the resume is not bit-identical with the old and new sizes and quotas,
and records that in every later checkpoint.

THE REPRODUCTIONS below are copied from reviewer R1-D's
`tests/unit/test_r1d_seam_reproductions.py` (branch `review/r1-d-seams`), with the
harness they need: `train_sae_task` and `resume_training_task` end to end on the
CPU over real extractions, the real rolling buffer, and the real on-the-fly source
over a tiny transformers Llama. Faked: the database session, GPU placement, the
memory readings (set per process by each test), model/dataset loading, WebSocket
emits. The tests after them are this fix's own.

MUTATION CONTROLS (2026-09-15; one line broken at a time, this file,
test_activation_plan.py and test_training_resume_equivalence.py run, the edit
confirmed landed, bytes restored, sha256 verified and `git diff` checked clean).
All 17 KILLED on the final code (13666d7b):
  M1  the plan is not saved (activation_plan=None)            -> 18 failed, incl. R1D-1, R1D-2, equivalence x2
  M2  the plan is not reused, cached (fresh mode/size)        -> 3 failed: equivalence x2 (plan == saved_plan),
        [only-a-whole-pool-fits]. SURVIVED the first run: the saved quotas alone kept the
        batches identical. The report's plan == saved_plan assertion was added; re-run: red.
  M3  the plan is not reused, on the fly                      -> the on-the-fly reuse test. SURVIVED the first
        run for the same reason; same assertion added; re-run: red.
  M4a the saved quotas are not reused, cached (allocate again) -> quotas-kept test [cached]
  M4b the saved quotas are not reused, on the fly              -> quotas-kept test [on_the_fly]
        (without the changed-allocator test these two were equivalent mutants)
  M5  the fits check is inverted                              -> 23 failed
  M6  the fallback does not record (history not extended)     -> 14 failed, incl. every re-plan test
  M7  the fallback records itself as bit-identical            -> 6 failed
  M8  rolling buffer: no advance past the interrupted buffer  -> 8 failed: both no-repeat tests (task and unit)
  M9  on the fly: no advance past the interrupted buffer's rows -> 4 failed
  M10 on the fly: the interrupted buffer's permutation not redrawn -> the same-quotas continuation test
  M11 a cycled pool may switch to a fixed pool on re-plan     -> 2 failed, incl. [only-a-whole-pool-fits]
  M12 the re-planned continuation is not wired (exact load)   -> 11 failed, incl. R1D-1, R1D-2
  M13 restored optimizer state not credited, cached reading   -> the credit test [credited]
  M14 restored state not credited, on-the-fly reading         -> the AST pin (that branch needs a card)
  M15 an on-the-fly resume estimates its sources again        -> the no-re-estimate test
  M16 the credit is applied off a card too                    -> 5 failed, incl. equivalence [rolling]
NEGATIVE CONTROL for R1D-1/R1D-2 themselves: M1 and M12 (reverting to re-planning,
and to exact loading) turn both reproductions red.
MERGE CONTROLS (integration, 2026-09-15, after merging onto R1-A/R1-B/R1-C; the
on-the-fly re-plan test run, bytes restored, sha256 verified):
  MF-1 the resumed source is prefilled (prefill=True)       -> 3 plans, not 2
  MF-2 the saved buffer is re-planned under the new quotas  -> RuntimeError, the resume refused
The harness session is test_training_resume_equivalence's, which carries R1-A's
metric discard (`delete` and `>` filters); this file's own copy could do neither.
"""

import json
import logging
from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models.activation_extraction import ActivationExtraction
from src.models.checkpoint import Checkpoint
from src.models.dataset import Dataset
from src.models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from src.models.model import Model
from src.models.training import Training
from src.services.checkpoint_service import load_training_state, save_training_state
from src.services.gpu_placement import Placement
from tests.unit.test_training_resume_equivalence import _Session

# ── the harness ──────────────────────────────────────────────────────────────
# The session is test_training_resume_equivalence's: one fake of the table's
# unique key and of the resume's metric discard (R1-A), not a copy per file.


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

        # THE MEMORY READINGS, per process.
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

    # Added for this fix's tests.
    def step_dir(self, step):
        return self.data_dir / "trainings" / self.training.id / "checkpoints" / f"checkpoint_{step}"

    def state(self, step):
        return load_training_state(self.step_dir(step))


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
    monkeypatch.setattr(hf_datasets, "load_from_disk", by_name)
    monkeypatch.setattr(
        training_tasks, "load_model_from_hf",
        lambda **kw: (model, SimpleNamespace(pad_token_id=0, eos_token_id=1, vocab_size=VOCAB), model.config, {}),
    )
    world.model, world.corpora = model, corpora
    return world


# ── R1D-1 / R1D-2: the reproductions, from R1-D ─────────────────────────────


@pytest.mark.parametrize(
    "free_memory_change",
    [0, 64 * 1024],
    ids=["CONTROL-identical-free-memory", "R1D-1-64KiB-less-free-memory"],
)
def test_a_rolling_buffer_run_resumes_when_the_new_process_reads_different_free_gpu_memory(
    monkeypatch, tmp_path, free_memory_change
):
    """R1D-1 (HIGH). 128,000 tokens per layer; the card's free memory leaves a buffer of
    ~80,000, so the pool is cycled (gpu_rolling). The run pauses after its step-10
    checkpoint; the resumed process reads 64 KiB less free memory — less than one CUDA
    allocator block — and nothing else changes."""
    world = _cached_world(monkeypatch, tmp_path, rows=4_000, seq=16)
    world.gpu_free = _gpu_free_for(80_000)
    world.pause(11)
    assert world.plans[0][0] == "gpu_rolling", world.plans  # precondition: the pool is cycled

    world.gpu_free -= free_memory_change
    try:
        outcome = world.resume()
    except ValueError as exc:
        pytest.fail(
            f"the resume was REFUSED because the resumed process read {free_memory_change:,} bytes less "
            f"free GPU memory: {exc} (plans: first run {world.plans[0]}, resumed run {world.plans[1]})"
        )
    assert world.plans[1][0] == "gpu_rolling", world.plans
    assert outcome["status"] == "completed", outcome


@pytest.mark.parametrize(
    "ram_change",
    [0, 1024**2],
    ids=["CONTROL-identical-available-ram", "R1D-2-1MiB-less-available-ram"],
)
def test_an_on_the_fly_run_resumes_when_the_new_process_reads_different_available_ram(
    monkeypatch, tmp_path, ram_change
):
    """R1D-2 (HIGH). ~60,000 real tokens; available RAM holds a 55,000-token buffer per
    layer (cpu_rolling). Paused after the step-10 checkpoint; the resumed process sees
    1 MiB less MemAvailable, which on a live host moves by more than that every second."""
    world = _on_the_fly_world(monkeypatch, tmp_path, rows_per_dataset=2_000)
    world.ram_bytes = 55_000 * HIDDEN * 4 * len(KEYS) * 2
    world.pause(11)
    assert world.plans[0][0] == "cpu_rolling", world.plans  # precondition: the corpus is cycled

    world.ram_bytes -= ram_change
    try:
        outcome = world.resume()
    except ValueError as exc:
        pytest.fail(
            f"the on-the-fly resume was REFUSED because the resumed process read {ram_change:,} bytes less "
            f"available RAM: {exc} (plans: first run {world.plans[0]}, resumed run {world.plans[1]})"
        )
    assert outcome["status"] == "completed", outcome


# ── this fix's tests ────────────────────────────────────────────────────────


def _no_post_run_evaluation(monkeypatch, world):
    """The post-run evaluation is not what these tests exercise, and review round 1's
    reviewer C is changing it; whatever it records does not touch the source."""
    monkeypatch.setattr(world.tasks, "run_post_run_evaluation", lambda task, **kwargs: {"status": "skipped"})


ORIGIN_ROWS, ORIGIN_SEQ, ORIGIN_BATCH = 6_000, 16, 2_048


def _origin_world(monkeypatch, tmp_path):
    """Two extractions whose every activation ENCODES its origin: [source, row / 4096,
    position / 16, ...] (exact in float32), with a recorded mask so padding and a held-out
    split exist. Batch 2,048, so each process serves tens of thousands of tokens."""
    extractions = []
    for source, name in enumerate(("a", "b")):
        folder = tmp_path / name
        folder.mkdir(parents=True)
        rows, seq = ORIGIN_ROWS, ORIGIN_SEQ
        acts = np.zeros((rows, seq, D), dtype=np.float32)
        acts[..., 0] = source
        acts[..., 1] = (np.arange(rows, dtype=np.float32) / 4096.0)[:, None]
        acts[..., 2] = (np.arange(seq, dtype=np.float32) / 16.0)[None, :]
        acts[..., 3:] = np.random.default_rng(source + 1).normal(size=(rows, seq, D - 3)).astype(np.float32)
        np.save(folder / "layer_0_residual.npy", acts)
        mask = np.ones((rows, seq), dtype=bool)
        mask[::7, 12:] = False
        np.save(folder / "attention_mask.npy", mask)
        (folder / "metadata.json").write_text(json.dumps({
            "num_samples_processed": rows, "layer_indices": [0], "hook_types": ["residual"],
        }))
        extractions.append(SimpleNamespace(id=f"ext_{name}", status="completed", output_path=str(folder),
                                           dataset_id=f"ds_{name}"))
    training = _training_row(
        "train_plan_origins",
        {
            "hidden_dim": D, "latent_dim": LATENT, "batch_size": ORIGIN_BATCH, "learning_rate": 1e-3,
            "total_steps": 20, "checkpoint_interval": 5, "log_interval": 1, "seed": 7,
            "training_layers": [0], "hook_types": ["residual"],
            "architecture_type": "standard_saelens", "l1_alpha": 1e-3, "warmup_steps": 0,
            "sparsity_warmup_steps": 0, "resample_dead_neurons": False, "evaluate_ce_delta": False,
            "grad_clip_norm": None, "holdout_fraction": 0.1, "holdout_eval_tokens": 512,
        },
        extraction_ids=[e.id for e in extractions],
    )
    rows_by_model = {
        Model: [SimpleNamespace(id="m_x", repo_id="org/tiny", quantization="FP16", file_path=None,
                                params_count=None, architecture_config=None, architecture="llama")],
        ActivationExtraction: extractions,
    }
    world = _World(monkeypatch, tmp_path, training, rows_by_model)
    _no_post_run_evaluation(monkeypatch, world)
    return world


def _origin_gpu_free_for(tokens_per_layer):
    from src.services import holdout_evaluation
    from src.workers.training_tasks import sae_buffer_budget

    reserved = holdout_evaluation.holdout_eval_peak_bytes(
        holdout_evaluation.DEFAULT_HOLDOUT_EVAL_CHUNK_TOKENS, D, LATENT
    )
    held_back = -sae_buffer_budget(
        0, num_keys=1, hidden_dim=D, latent_dim=LATENT, batch_size=ORIGIN_BATCH, reserved_bytes=reserved
    )["available"]
    return held_back + int(np.ceil(tokens_per_layer * D * 4 / 0.9)) + 16


def _origins(batch):
    values = batch.double()
    return list(zip(
        values[:, 0].round().long().tolist(),
        (values[:, 1] * 4096).round().long().tolist(),
        (values[:, 2] * 16).round().long().tolist(),
    ))


def _record_draws_and_held_out(monkeypatch, world):
    """Every batch the rolling buffer serves, and every held-out split, by process."""
    from src.services import activation_buffer, activation_mask

    world.process = None
    world.draws = defaultdict(list)
    world.held = defaultdict(set)
    real_next = activation_buffer.RollingActivationBuffer.next_batch

    def next_batch(buf, batch_size):
        out = real_next(buf, batch_size)
        world.draws[world.process].append(_origins(out[(0, "residual")]))
        return out

    real_split = activation_mask.split_documents

    def split(flat, seq_len, *args, **kwargs):
        train, held = real_split(flat, seq_len, *args, **kwargs)
        source = sum(1 for _ in world.held_calls[world.process])
        world.held_calls[world.process].append(1)
        world.held[world.process] |= {(source, int(f) // seq_len, int(f) % seq_len) for f in np.asarray(held)}
        return train, held

    world.held_calls = defaultdict(list)
    monkeypatch.setattr(activation_buffer.RollingActivationBuffer, "next_batch", next_batch)
    monkeypatch.setattr(activation_mask, "split_documents", split)


@pytest.mark.parametrize(
    "change", ["gpu-holds-less", "moves-to-ram", "only-a-whole-pool-fits"],
)
def test_a_resume_whose_saved_plan_no_longer_fits_replans_repeats_no_row_and_records_it(
    monkeypatch, tmp_path, caplog, change
):
    """The fallback. The run plans a 55,000-token gpu_rolling buffer and pauses after its
    step-10 checkpoint; the resumed process can no longer hold that:

      gpu-holds-less           the card holds 52,000 tokens: gpu_rolling 52,000
      moves-to-ram             the card holds no useful buffer, RAM holds 52,000: cpu_rolling
      only-a-whole-pool-fits   the card holds nothing, RAM holds the whole pool: the planner
                               says cpu_all, and the resume keeps CYCLING (cpu_rolling at the
                               pool's size), because a fixed pool samples with replacement

    Required: the run completes; no activation is served twice over the run's effective
    history (the first process's steps 0-10, the resumed process's 11-19 — under one pass
    by construction in the first two cases); no held-out position is ever served and the
    split is the same in both processes; a WARNING names both buffer sizes and both quota
    lists; and the step-15 checkpoint records the resume as not bit-identical, in
    training_state.pt and on its rows."""
    world = _origin_world(monkeypatch, tmp_path)
    _record_draws_and_held_out(monkeypatch, world)
    caplog.set_level(logging.INFO, logger=world.tasks.logger.name)

    world.gpu_free = _origin_gpu_free_for(55_000)
    world.process = 1
    world.pause(11)
    assert world.plans[0] == ("gpu_rolling", 55_000), world.plans  # precondition
    saved = world.state(10)
    assert saved["activation_plan"]["mode"] == "gpu_rolling"
    assert saved["activation_plan"]["buffer_tokens"] == 55_000
    assert saved["activation_plan"]["quotas"] == saved["activation_source"]["quotas"], (
        "the checkpoint's plan does not describe the source it saved the position of"
    )
    total = sum(saved["activation_plan"]["source_tokens"])

    world.process = 2
    if change == "gpu-holds-less":
        world.gpu_free = _origin_gpu_free_for(52_000)
        expected_mode, expected_tokens = "gpu_rolling", 52_000
    elif change == "moves-to-ram":
        world.gpu_free = 0
        world.ram_bytes = 52_000 * D * 4 * 2
        expected_mode, expected_tokens = "cpu_rolling", 52_000
    else:
        world.gpu_free = 0
        expected_mode, expected_tokens = "cpu_rolling", total
    caplog.clear()
    outcome = world.resume()

    assert outcome["status"] == "completed", outcome
    if change == "only-a-whole-pool-fits":
        assert world.plans[1] == ("cpu_all", total), world.plans  # precondition: the planner said load it whole
    assert len(world.draws[1]) == 11 and len(world.draws[2]) == 9, {k: len(v) for k, v in world.draws.items()}

    # NO HELD-OUT POSITION IS TRAINED ON, and the split did not move.
    assert world.held[1] and world.held[1] == world.held[2]
    served = [origin for process in (1, 2) for batch in world.draws[process] for origin in batch]
    assert not set(served) & world.held[1], "a held-out position was served after the re-plan"
    if change != "only-a-whole-pool-fits":
        # 55,000 + 52,000 tokens of buffers from a pool this size: within one pass.
        assert 55_000 + 52_000 < total
        repeats = len(served) - len(set(served))
        assert repeats == 0, f"{repeats} activations were served twice across the resume"

    # THE LOG: not bit-identical, both sizes, both quota lists.
    state15 = world.state(15)
    new_plan = state15["activation_plan"]
    assert (new_plan["mode"], new_plan["buffer_tokens"]) == (expected_mode, expected_tokens), new_plan
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    replan = [m for m in warnings if "RESUME IS NOT BIT-IDENTICAL" in m and "Re-planned" in m]
    assert len(replan) == 1, warnings
    for fragment in (f"{55_000:,}", f"{expected_tokens:,}", str(saved["activation_plan"]["quotas"]),
                     str(new_plan["quotas"])):
        assert fragment in replan[0], (fragment, replan[0])

    # THE RECORD, in the next checkpoint's state and on its rows.
    skipped = saved["activation_source"]["size"] - saved["activation_source"]["position"]
    assert state15["resume_history"] == [{
        "checkpoint_step": 10,
        "checkpoint_id": world.dispatched[-1]["checkpoint_id"],
        "activation_plan": "replanned",
        "bit_identical": False,
        "tokens_skipped": skipped,
        "source_restarted": False,
        "saved_plan": {"mode": "gpu_rolling", "buffer_tokens": 55_000, "quotas": saved["activation_plan"]["quotas"]},
        "plan": {"mode": expected_mode, "buffer_tokens": expected_tokens, "quotas": new_plan["quotas"]},
    }], state15["resume_history"]
    assert skipped > 0
    rows15 = [c for c in world.store[Checkpoint] if c.step == 15]
    assert rows15 and all(c.extra_metadata["resume_history"] == state15["resume_history"] for c in rows15)


@pytest.mark.parametrize("memory", ["same-memory", "less-memory"])
def test_a_checkpoint_without_a_plan_still_resumes_as_before_with_a_warning(monkeypatch, tmp_path, caplog, memory):
    """A training_state.pt written before plans were saved. It re-plans from this
    process's memory, warns, and — exactly as before this fix — its position loads when
    that plan matches and is refused when it does not."""
    world = _cached_world(monkeypatch, tmp_path, rows=4_000, seq=16)
    _no_post_run_evaluation(monkeypatch, world)
    world.gpu_free = _gpu_free_for(80_000)
    world.pause(11)
    states = sorted(world.data_dir.rglob("training_state.pt"))
    assert states, "precondition: the run wrote training state"
    for path in states:
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state.pop("activation_plan") is not None, "precondition: the state had a plan to remove"
        state.pop("format"), state.pop("version")
        save_training_state(path.parent, state)

    caplog.set_level(logging.INFO, logger=world.tasks.logger.name)
    if memory == "less-memory":
        world.gpu_free -= 64 * 1024
        with pytest.raises(ValueError, match="quotas"):
            world.resume()
        assert "records no activation storage plan" in caplog.text
        return

    outcome = world.resume()
    assert outcome["status"] == "completed", outcome
    assert "records no activation storage plan" in caplog.text
    report = world.state(15)["resume_history"]
    assert [(r["activation_plan"], r["bit_identical"]) for r in report] == [("not_recorded", True)], report


@pytest.mark.parametrize("path", ["cached", "on_the_fly"])
def test_a_reused_plan_keeps_its_saved_quotas_even_when_the_allocator_would_now_split_differently(
    monkeypatch, tmp_path, path
):
    """The saved quotas are the ones the source's position was recorded under. Memory is
    identical, so the plan is reused; the resumed build's allocator splits the same buffer
    one token differently (a changed dataset_mixture between pause and resume). The saved
    quotas must win, or the position is refused. Without this, reading the quotas from the
    plan is indistinguishable from allocating again."""
    if path == "cached":
        world = _cached_world(monkeypatch, tmp_path, rows=4_000, seq=16)
        world.gpu_free = _gpu_free_for(80_000)
        world.pause(11)
        record_step = 15
    else:
        world = _on_the_fly_world(monkeypatch, tmp_path, rows_per_dataset=2_000)
        world.ram_bytes = 55_000 * HIDDEN * 4 * len(KEYS) * 2
        world.pause(6)
        record_step = 10
    _no_post_run_evaluation(monkeypatch, world)
    real_allocate = world.tasks.dataset_mixture.allocate_tokens

    def shifted(available, total, weights=None):
        allocation = list(real_allocate(available, total, weights))
        allocation[0] -= 1
        allocation[1] += 1
        return allocation

    monkeypatch.setattr(world.tasks.dataset_mixture, "allocate_tokens", shifted)
    assert world.resume()["status"] == "completed"
    assert world.plans[1] == world.plans[0], "precondition: the memory readings are identical"
    history = world.state(record_step)["resume_history"]
    assert [(r["activation_plan"], r["bit_identical"]) for r in history] == [("reused", True)], history


@pytest.mark.parametrize(
    "restored_on_card", [True, False], ids=["credited", "CONTROL-nothing-restored-to-credit"],
)
def test_a_resume_credits_the_optimizer_state_it_already_restored_to_the_card(
    monkeypatch, tmp_path, restored_on_card
):
    """Reviewer B: on a card the resumed process ALWAYS reads less free memory, because
    restore_training_state has put the Adam moments there before the reading, and the
    budget still subtracts them as pending. Uncredited, every GPU resume re-plans. Here
    the reading is 1 MiB lower by exactly the restored state; the CPU run stands in for
    the card through `restored_state_bytes`. The CONTROL is the same reading with nothing
    restored to credit: it must re-plan, so the credit is what makes the difference."""
    world = _cached_world(monkeypatch, tmp_path, rows=4_000, seq=16)
    _no_post_run_evaluation(monkeypatch, world)
    world.gpu_free = _gpu_free_for(80_000)
    world.pause(11)

    restored = 1024**2
    world.gpu_free -= restored
    monkeypatch.setattr(
        world.tasks.activation_plan, "restored_state_bytes",
        lambda optimizers, models, device: restored if restored_on_card else 0,
    )
    assert world.resume()["status"] == "completed"
    history = world.state(15)["resume_history"]
    expected = ("reused", True) if restored_on_card else ("replanned", False)
    assert [(r["activation_plan"], r["bit_identical"]) for r in history] == [expected], history


def test_every_free_memory_reading_that_sizes_a_buffer_credits_the_restored_state():
    """AST, because the on-the-fly GPU budget only runs on a card (its branch is gated on
    `device.type == "cuda"`, and a CPU-only suite cannot enter it). The cached reading's
    credit is driven above; this pins that BOTH readings are corrected, by the same call,
    right after they are taken."""
    import ast
    import inspect

    from src.workers import training_tasks

    tree = ast.parse(inspect.getsource(training_tasks))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_sae_task")
    readings = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "mem_get_info"]
    credits = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "attr", None) == "free_bytes_for_planning"
        and [t.id for t in n.targets if isinstance(t, ast.Name)] == ["gpu_free"]
        and n.value.args and isinstance(n.value.args[0], ast.Name) and n.value.args[0].id == "gpu_free"
    ]
    assert len(readings) == 2, "precondition: the task reads free GPU memory at its two buffer budgets"
    assert len(credits) == len(readings), f"{len(credits)} credits for {len(readings)} readings"
    for reading in readings:
        assert any(0 < credit.lineno - reading.lineno < 12 for credit in credits), (
            f"the reading at line {reading.lineno} reaches its budget uncredited"
        )


def test_an_on_the_fly_resume_allocates_from_the_runs_own_estimates_not_new_ones(monkeypatch, tmp_path):
    """Reviewer B: the per-source real-token counts on this path are ESTIMATES from a
    sample of rows, and a change to the estimator (64 rows -> 1,024 is pending) between
    pause and resume would move the quotas. The resumed run must not estimate again."""
    world = _on_the_fly_world(monkeypatch, tmp_path, rows_per_dataset=2_000)
    _no_post_run_evaluation(monkeypatch, world)
    world.ram_bytes = 55_000 * HIDDEN * 4 * len(KEYS) * 2
    world.pause(6)

    calls = []
    real_estimate = world.tasks.model_activation_source.estimate_real_tokens

    def changed_estimator(source, seed, index):
        calls.append(index)
        return real_estimate(source, seed, index) + 1_000

    monkeypatch.setattr(world.tasks.model_activation_source, "estimate_real_tokens", changed_estimator)
    assert world.resume()["status"] == "completed"
    assert calls == [], "the resumed run estimated the sources again"
    history = world.state(10)["resume_history"]
    assert [(r["activation_plan"], r["bit_identical"]) for r in history] == [("reused", True)], history


def test_a_whole_pool_run_that_no_longer_fits_restarts_as_a_rolling_buffer_and_says_so(
    monkeypatch, tmp_path, caplog
):
    """A MODE CHANGE (reviewer B). The run loaded its 128,000-token pool whole on the card
    (gpu_all, the fixed-pool sampler); the resumed card holds 100,000, so the pool is now
    cycled through a rolling buffer. A fixed pool draws WITH replacement, so there is no
    pass or row position to continue and nothing it served was promised not to repeat:
    the rolling buffer starts its first pass, and the resume is recorded as a restart,
    not bit-identical."""
    world = _cached_world(monkeypatch, tmp_path, rows=4_000, seq=16)
    _no_post_run_evaluation(monkeypatch, world)
    world.gpu_free = _gpu_free_for(150_000)
    world.pause(11)
    assert world.plans[0] == ("gpu_all", 128_000), world.plans  # precondition
    assert world.state(10)["activation_source"]["kind"] == "fixed_activation_pool"

    world.gpu_free = _gpu_free_for(100_000)
    caplog.set_level(logging.INFO, logger=world.tasks.logger.name)
    assert world.resume()["status"] == "completed"

    assert world.plans[1][0] == "gpu_rolling", world.plans
    state15 = world.state(15)
    assert state15["activation_source"]["kind"] == "rolling_activation_buffer"
    report = state15["resume_history"][-1]
    assert (report["activation_plan"], report["bit_identical"], report["source_restarted"]) == (
        "replanned", False, True,
    ), report
    assert "fixed pool had no pass to continue" in caplog.text


def _record_on_the_fly(monkeypatch, world_holder):
    from src.services import model_activation_source as MAS

    draws, planned = defaultdict(list), defaultdict(list)
    real_next = MAS.ModelActivationSource.next_batch
    real_plan = MAS.ModelActivationSource._plan

    def next_batch(source, batch_size):
        out = real_next(source, batch_size)
        draws[world_holder["phase"]].append({key: value.clone() for key, value in out.items()})
        return out

    def plan(source):
        result = real_plan(source)
        planned[world_holder["phase"]].append({(p.source, p.row) for p in result[0]})
        return result

    monkeypatch.setattr(MAS.ModelActivationSource, "next_batch", next_batch)
    monkeypatch.setattr(MAS.ModelActivationSource, "_plan", plan)
    return draws, planned


def test_an_on_the_fly_resume_that_can_hold_more_reuses_the_saved_plan_and_continues_exactly(
    monkeypatch, tmp_path, caplog
):
    """The resumed process has TWICE the RAM, so a fresh plan would load the corpus whole
    (a different mode and different quotas). The saved plan fits, so it is reused, and the
    batches after the checkpoint are exactly the uninterrupted run's."""
    phase = {"phase": None}
    draws, _ = _record_on_the_fly(monkeypatch, phase)
    ram = 55_000 * HIDDEN * 4 * len(KEYS) * 2

    phase["phase"] = "straight"
    straight = _on_the_fly_world(monkeypatch, tmp_path / "straight", rows_per_dataset=2_000)
    _no_post_run_evaluation(monkeypatch, straight)
    straight.ram_bytes = ram
    assert straight.run()["status"] == "completed"
    assert len(draws["straight"]) == 12

    phase["phase"] = "paused"
    resumed = _on_the_fly_world(monkeypatch, tmp_path / "resumed", rows_per_dataset=2_000)
    _no_post_run_evaluation(monkeypatch, resumed)
    resumed.ram_bytes = ram
    # At the step after the step-5 checkpoint: a later pause re-logs steps the resumed run
    # logs again, which is review finding R1D-3 (reviewer A's), not this one.
    resumed.pause(6)
    resumed.ram_bytes = 2 * ram
    phase["phase"] = "resumed"
    caplog.set_level(logging.INFO, logger=resumed.tasks.logger.name)
    assert resumed.resume()["status"] == "completed"

    assert resumed.plans[0][0] == "cpu_rolling" and resumed.plans[1] != resumed.plans[0], (
        f"precondition: the resumed process would plan differently: {resumed.plans}"
    )
    assert len(draws["resumed"]) == 6
    for step, (a, b) in enumerate(zip(draws["straight"][6:], draws["resumed"], strict=True), start=6):
        for key in a:
            assert torch.equal(a[key], b[key]), f"step {step} {key}: the resumed batch differs"
    assert "Resume reuses the checkpoint's activation storage plan" in caplog.text
    history = resumed.state(10)["resume_history"]
    assert [(r["activation_plan"], r["bit_identical"], r["tokens_skipped"]) for r in history] == [
        ("reused", True, 0)
    ], history
    # The mode and size were reused as well as the quotas (control M3: with the quotas
    # alone reused, every batch above still agrees).
    assert history[0]["plan"] == history[0]["saved_plan"], history[0]


def test_an_on_the_fly_resume_whose_plan_no_longer_fits_continues_the_pass_and_records_it(
    monkeypatch, tmp_path, caplog
):
    """R1D-2's scenario, with what the fallback must do. The resumed source plans the
    interrupted buffer's rows again under the saved quotas (so its cycles stand after them)
    and fills its first buffer from rows that buffer did not take."""
    phase = {"phase": None}
    _, planned = _record_on_the_fly(monkeypatch, phase)
    world = _on_the_fly_world(monkeypatch, tmp_path, rows_per_dataset=5_000)
    _no_post_run_evaluation(monkeypatch, world)
    world.ram_bytes = 55_000 * HIDDEN * 4 * len(KEYS) * 2
    phase["phase"] = 1
    # Paused right after the step-5 checkpoint (a later pause is R1D-3), so the resumed
    # run writes step 10 with its record.
    world.pause(6)
    saved = world.state(5)
    assert world.plans[0] == ("cpu_rolling", 55_000) and len(planned[1]) == 1, (world.plans, len(planned[1]))

    world.ram_bytes -= 1024**2
    phase["phase"] = 2
    caplog.set_level(logging.INFO, logger=world.tasks.logger.name)
    assert world.resume()["status"] == "completed"

    served_first, served_second = planned[1][0], planned[2][-1]
    # Two plans, not three: a resumed source is built without a prefill (review R1-B), so
    # the constructor fills nothing and the saved buffer's re-plan comes first.
    assert len(planned[2]) == 2, "expected: the re-planned saved buffer, then the new buffer, and no prefill"
    assert planned[2][0] == served_first, "the saved buffer was not planned again under its own quotas"
    assert not served_first & served_second, (
        f"{len(served_first & served_second)} rows of the interrupted buffer were taken again in the same pass"
    )
    assert "RESUME IS NOT BIT-IDENTICAL" in caplog.text
    history = world.state(10)["resume_history"]
    assert len(history) == 1, history
    report = history[0]
    assert (report["activation_plan"], report["bit_identical"]) == ("replanned", False), history
    assert report["tokens_skipped"] == saved["activation_source"]["size"] - saved["activation_source"]["position"]
    assert report["saved_plan"]["quotas"] == saved["activation_plan"]["quotas"] != report["plan"]["quotas"]
