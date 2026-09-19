"""R2D-12, ported by review round 3 (R3-A, 2026-09-15): a resumed run equals an uninterrupted one THROUGH the post-run evaluation.

R1-D wrote this guard (`06c1110a:backend/tests/unit/test_r1d_seam_reproductions.py:812`) and it was never
merged: only R1-D's R1D-4..7 reproductions reached the integration branch. This is the guard, on the merged
harnesses: `test_r1d_evaluation_seams._World` over the shared `_Session` (the metric key with the hook type,
comparison operators), on both paths.

A run paused mid-buffer and resumed in a "new process" (every object rebuilt, the global RNGs scrambled)
must equal the uninterrupted run all the way through the post-run evaluation: the final weights, the
training and held-out metric rows from the resume point on, every resample with its step, latents and
source rows (on BOTH sides of the resume point), and the evaluation document.

WHAT CHANGED SINCE R1-D, AND HOW THE PORT SAYS SO.
- R1D-1/R1D-2 are fixed: the checkpoint records the storage plan, and a resume reuses it. The port asserts
  the resume took the saved plan bit-identically (`resume_report`), rather than relying on readings held
  identical by construction.
- R1-D L8 is fixed: a pause writes its own checkpoint at the step it stops after. R1-D's pause at 550 found
  the checkpoint at 500 and repeated 50 steps; here it checkpoints step 549 and continues at 550.
- R1D-3 is fixed: metric rows logged after the checkpoint are discarded, so the pause may land anywhere.
- R3-A: the run is COMPLETED before its evaluation, which runs on the COMPLETED row.

MUTATION CONTROLS (R1-D M1-M4 on the on-the-fly path; re-run by R3-A on both paths, review record):
M1 b_dec re-initialised on resume, M2 JumpReLU thresholds re-calibrated on resume, M3 the source's
position not restored, M4 the RNG not restored.
"""

from collections import Counter, defaultdict
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from src.models.activation_extraction import ActivationExtraction
from src.models.model import Model
from src.models.training_metric import TrainingMetric
from src.workers import training_tasks as _training_tasks
from tests.unit.test_r1d_evaluation_seams import (
    BATCH,
    HIDDEN,
    KEYS,
    VOCAB,
    WIDTH,
    _World,
    _on_the_fly_world,
    _tiny_llama,
    _tokenized,
    _training_row,
)

_REAL_RESAMPLE = _training_tasks.resample_dead_latents
_REAL_RESAMPLE_DUE = _training_tasks.resample_due
_REAL_POST_RUN_EVALUATION = _training_tasks.run_post_run_evaluation
_REAL_RESUME_REPORT = _training_tasks.activation_plan.resume_report

E2E_STOP = 550
E2E_BUFFER_TOKENS = 55_000
E2E_HP = {
    "total_steps": 1_000, "checkpoint_interval": 500, "log_interval": 50,
    "warmup_steps": 20, "lr_decay_steps": 200,
    "resample_dead_neurons": True, "resample_interval": 100, "dead_neuron_threshold": 20,
    "holdout_fraction": 0.2, "holdout_eval_tokens": 256,
    "evaluate_ce_delta": True, "evaluation_token_budget": 1_600,
    "dataset_weights": [3.0, 1.0],
    # A penalty strong enough that latents keep dying, so resamples happen on BOTH sides of the
    # resume point (at 1e-3 none ever died on this data: R1-D's first version asserted equivalence
    # over zero resamples).
    "sparsity_coeff": 1.0,
}
_VOLATILE = {"started_at", "completed_at", "updated_at", "task_id", "requested_at"}


class _Recorder:
    """Resamples (with their step), refills, resumes and the final SAE weights, by phase. Wraps the REAL functions."""

    def __init__(self, monkeypatch):
        from src.services import activation_buffer
        from src.services import model_activation_source as MAS

        self.phase = None
        self.resamples = defaultdict(list)
        self.weights = {}
        self.refills = Counter()
        self.resumes = defaultdict(list)
        self._due_step = None

        def due(step, **kwargs):
            self._due_step = step
            return _REAL_RESAMPLE_DUE(step, **kwargs)

        def resample(model, optimizer, x, dead, **kwargs):
            result = _REAL_RESAMPLE(model, optimizer, x, dead, **kwargs)
            self.resamples[self.phase].append(
                (self._due_step, result.latents.tolist(), result.source_rows.tolist())
            )
            return result

        def evaluate(task, **kwargs):
            self.weights[self.phase] = {
                key: {name: value.detach().clone() for name, value in sae.state_dict().items()}
                for key, sae in kwargs["models"].items()
            }
            return _REAL_POST_RUN_EVALUATION(task, **kwargs)

        def resume_report(**kwargs):
            report = _REAL_RESUME_REPORT(**kwargs)
            self.resumes[self.phase].append(report)
            return report

        monkeypatch.setattr(_training_tasks, "resample_due", due)
        monkeypatch.setattr(_training_tasks, "resample_dead_latents", resample)
        monkeypatch.setattr(_training_tasks, "run_post_run_evaluation", evaluate)
        monkeypatch.setattr(_training_tasks.activation_plan, "resume_report", resume_report)
        for cls in (activation_buffer.RollingActivationBuffer, MAS.ModelActivationSource):
            real_refill = cls.refill

            def refill(source, _real=real_refill):
                self.refills[self.phase] += 1
                return _real(source)

            monkeypatch.setattr(cls, "refill", refill)


def _cached_eval_world(monkeypatch, tmp_path):
    """Two extractions of layers 1 and 2 at the tiny Llama's width, each with a recorded mask (the held-out
    split needs one) and a tokenization longer than what it read, so the post-run evaluation loads the
    model and runs on rows the extraction never read."""
    import datasets as hf_datasets

    from src.services import holdout_evaluation
    from src.workers.training_tasks import sae_buffer_budget

    model = _tiny_llama()
    corpora = {"tok_a": _tokenized(4_000, 1), "tok_b": _tokenized(4_000, 2)}
    rows = 3_200
    extractions = []
    for i, name in enumerate(("a", "b")):
        folder = tmp_path / f"ext_{name}"
        folder.mkdir(parents=True)
        gen = np.random.default_rng(i + 11)
        for layer in (1, 2):
            np.save(folder / f"layer_{layer}_residual.npy",
                    gen.normal(size=(rows, WIDTH, HIDDEN)).astype(np.float32))
        mask = np.ones((rows, WIDTH), dtype=bool)
        mask[::7, 10:] = False
        np.save(folder / "attention_mask.npy", mask)
        (folder / "metadata.json").write_text(json.dumps({
            "num_samples_processed": rows, "max_samples": rows, "layer_indices": [1, 2],
            "hook_types": ["residual"], "seq_len": WIDTH, "dataset_path": f"/tok/tok_{name}",
        }))
        extractions.append(SimpleNamespace(id=f"ext_{name}", status="completed", output_path=str(folder),
                                           dataset_id=f"ds_{name}"))
    training = _training_row(
        "train_r2d12_cached",
        {
            "hidden_dim": HIDDEN, "latent_dim": 64, "batch_size": BATCH, "learning_rate": 1e-3, "seed": 9,
            "training_layers": [1, 2], "hook_types": ["residual"], "architecture_type": "jumprelu",
            "sparsity_warmup_steps": 0, "grad_clip_norm": None, **E2E_HP,
        },
        model_id="m_tiny", extraction_ids=[e.id for e in extractions],
    )
    rows_by_model = {
        Model: [SimpleNamespace(id="m_tiny", repo_id="org/tiny-llama", quantization="FP16", file_path=None,
                                params_count=None, architecture="llama",
                                architecture_config={"hidden_size": HIDDEN})],
        ActivationExtraction: extractions,
    }
    world = _World(monkeypatch, tmp_path, training, rows_by_model)
    reserved = holdout_evaluation.holdout_eval_peak_bytes(
        holdout_evaluation.DEFAULT_HOLDOUT_EVAL_CHUNK_TOKENS, HIDDEN, 64
    )
    held_back = -sae_buffer_budget(
        0, num_keys=2, hidden_dim=HIDDEN, latent_dim=64, batch_size=BATCH, reserved_bytes=reserved
    )["available"]
    world.gpu_free = held_back + int(np.ceil(E2E_BUFFER_TOKENS * HIDDEN * 4 * 2 / 0.9)) + 1024

    def by_name(path):
        return corpora[str(path).rstrip("/").rsplit("/", 1)[-1]]

    monkeypatch.setattr(hf_datasets, "load_from_disk", by_name)
    monkeypatch.setattr(
        world.tasks, "load_model_from_hf",
        lambda **kw: (model, SimpleNamespace(pad_token_id=0, eos_token_id=1, vocab_size=VOCAB), model.config, {}),
    )
    world.model = model
    return world


def _scramble_global_rngs():
    import random

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    torch.rand(37)


def _metric_rows(world, *, held_out, from_step):
    rows = {}
    for m in world.store.get(TrainingMetric, []):
        if m.layer_idx is None or (m.layer_idx < 0) != held_out or m.step < from_step:
            continue
        rows[(m.step, m.layer_idx, m.hook_type)] = (
            m.loss, m.fvu, m.fvu_centred, m.l0_mean, m.l0_sparsity, m.dead_neurons,
        )
    return rows


@pytest.mark.parametrize("path", ["cached-gpu_rolling", "on_the_fly-cpu_rolling"])
def test_a_run_resumed_mid_buffer_equals_an_uninterrupted_one_through_the_post_run_evaluation(
    monkeypatch, tmp_path, path
):
    recorder = _Recorder(monkeypatch)

    def build(sub):
        if path.startswith("cached"):
            return _cached_eval_world(monkeypatch, tmp_path / sub)
        world = _on_the_fly_world(monkeypatch, tmp_path / sub, rows_per_dataset=2_500, hp=E2E_HP)
        world.ram_bytes = E2E_BUFFER_TOKENS * HIDDEN * 4 * len(KEYS) * 2
        return world

    recorder.phase = "straight"
    straight = build("straight")
    assert straight.run()["status"] == "completed"
    straight_plans = list(straight.plans)

    recorder.phase = "resumed"
    resumed = build("resumed")
    resumed.pause(E2E_STOP)
    _scramble_global_rngs()
    assert resumed.resume()["status"] == "completed"

    # Preconditions: the pool is cycled on both sides, the uninterrupted run refills after the resume
    # point, resamples happen on both sides of it, and the resume took the saved plan exactly.
    mode = path.split("-", 1)[1]
    assert [p[0] for p in straight_plans] == [mode], straight_plans
    assert resumed.plans and all(p[0] == mode for p in resumed.plans), resumed.plans
    assert recorder.refills["straight"] >= 2, recorder.refills
    performed = [(step, latents) for step, latents, _ in recorder.resamples["straight"] if latents]
    assert any(step < E2E_STOP for step, _ in performed), performed
    assert any(step > E2E_STOP for step, _ in performed), (
        f"no resample after the resume point, so the RNG restore is untested: {performed}"
    )
    [report] = recorder.resumes["resumed"]
    assert report["checkpoint_step"] == E2E_STOP - 1, report
    assert report["activation_plan"] == "reused" and report["bit_identical"] is True, report

    assert recorder.resamples["resumed"] == recorder.resamples["straight"]
    assert recorder.weights["straight"].keys() == recorder.weights["resumed"].keys()
    for key, expected in recorder.weights["straight"].items():
        for name, value in expected.items():
            assert torch.equal(value, recorder.weights["resumed"][key][name]), f"{key} {name} differs"
    for held_out in (False, True):
        a = _metric_rows(straight, held_out=held_out, from_step=E2E_STOP)
        b = _metric_rows(resumed, held_out=held_out, from_step=E2E_STOP)
        assert a, f"precondition: no {'held-out' if held_out else 'training'} rows after the resume point"
        assert a == b, (held_out, sorted(set(a.items()) ^ set(b.items()))[:4])

    doc_a, doc_b = straight.training.evaluation, resumed.training.evaluation
    assert doc_a["status"] == "completed", doc_a
    assert doc_a.get("blocks_evaluated", doc_a.get("tokens")), f"precondition: nothing was evaluated: {doc_a}"
    assert {k: v for k, v in doc_a.items() if k not in _VOLATILE} == {
        k: v for k, v in doc_b.items() if k not in _VOLATILE
    }
