"""The step loop's gradient state across an OOM, and resampling per SAE (review round 2, R2-C).

Driven through the real task with the resume-equivalence harness: `train_sae_task` end
to end on the CPU over real `.npy` extractions, real SAEs, real Adam, the real rolling
buffer and fixed pool.

R2C-3 — AN OOM THAT SKIPPED A WINDOW'S FIRST STEP RE-APPLIED THE PREVIOUS UPDATE.
Gradients were zeroed only at an accumulation window's first step, after the draw.
An OOM before that point (a refill; or, with several SAEs, the first SAE's step
before the others were zeroed) skipped the step, and the next step is mid-window:
its backward accumulated onto the gradient the previous window's update had already
applied, and the update closing the window applied it again. Fully on the CPU;
under a GradScaler divided by the new scale. Gradients are now released as soon as
the optimizer has applied them. The test checks the accumulation invariant at every
backward, so it also pins the window-start zero_grad, still needed when an OOM falls
between a window's last backward and its update. Red on 7649ddbd (step 9's backward
found the gradient the step-7 update had applied).

R2C-4 — EACH SAE IS RESAMPLED FROM ITS OWN BATCH, TRACKER AND OPTIMIZER. No test ran
resampling with more than one SAE (the multi-hook run turns it off), so the loop could
hand every SAE the first key's activations, dead mask or optimizer with the suite
green. Here four SAEs (two layers x two hooks) read activations at four scales a
decade apart, and each has its OWN latents killed, so a shared tracker, batch or
optimizer is visible. A test finding: the loop is correct today.

MUTATION CONTROLS: see the R2-C record.
"""

import json
from types import SimpleNamespace

import numpy as np
import torch

from src.workers.training_tasks import grad_accum_steps_for
from tests.unit import test_training_resume_equivalence as E


def _oom():
    return RuntimeError("CUDA out of memory. Tried to allocate 4.97 GiB")


def test_every_accumulation_window_accumulates_only_its_own_micro_batches(monkeypatch, tmp_path):
    """Across an OOM in the draw at a window's first step and one in the SAE step before an update.

    At every backward, the gradient already on the parameters must be exactly what this
    window's earlier backwards added: never a gradient an update already applied (R2C-3,
    step 8's draw OOM), and never one left by a window that closed without an update (the
    window-start zero_grad, step 15's OOM between the backward and the update).
    """
    base = E.CONFIGS["rolling-jumprelu-accum"]
    # No resampling here: a resample clears slices of a window's gradient on purpose.
    config = {**base, "hp": {**base["hp"], "resample_dead_neurons": False}}
    harness = E._Harness(monkeypatch, tmp_path, config)
    tasks = E._training_tasks
    accum = grad_accum_steps_for(config["hp"]["batch_size"])
    assert accum == 4, "precondition: the fixture accumulates"
    draw_oom_step, update_oom_step = 2 * accum, 4 * accum - 1  # a window's first step; a window's last

    now = {"step": None}
    draws = []
    real_draw = tasks.draw_cached_batch

    def draw(stream, cached, keys, batch_size, num_samples, device):
        now["step"] = len(draws)  # one draw per step; a skipped step is not retried
        draws.append(batch_size)
        if now["step"] == draw_oom_step:
            raise _oom()
        return real_draw(stream, cached, keys, batch_size, num_samples, device)

    projected = []
    real_project = tasks.project_decoder_gradients

    def project(model):  # JumpReLU, between a window's last backward and its update
        projected.append(now["step"])
        if now["step"] == update_oom_step:
            raise _oom()
        return real_project(model)

    def grads():
        return [
            torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
            for p in harness.models[-1].parameters()
        ]

    records = []
    real_backward = torch.Tensor.backward

    def backward(self, *args, **kwargs):
        if now["step"] is None:
            return real_backward(self, *args, **kwargs)
        before = grads()
        result = real_backward(self, *args, **kwargs)
        records.append((now["step"], before, grads()))
        return result

    monkeypatch.setattr(tasks, "draw_cached_batch", draw)
    monkeypatch.setattr(tasks, "project_decoder_gradients", project)
    monkeypatch.setattr(torch.Tensor, "backward", backward)

    assert harness.run()["status"] == "completed"
    stepped = [step for step, _, _ in records]
    assert draw_oom_step not in stepped and update_oom_step in stepped, "precondition: both OOMs happened"
    assert update_oom_step in projected

    accumulated = {}
    for step, before, after in records:
        window = step // accum
        expected = accumulated.get(window) or [torch.zeros_like(b) for b in before]
        for b, e in zip(before, expected):
            assert torch.allclose(b, e, atol=1e-6), (
                f"step {step}: the gradient present before its backward is not this window's own "
                f"(excess norm {float((b - e).norm()):.3g})"
            )
        accumulated[window] = [e + (a - b) for e, a, b in zip(expected, after, before)]
    assert len({step // accum for step in stepped}) == E.TOTAL // accum


LAYERS = [0, 1]
HOOKS = ["residual", "mlp"]
SAES = [(layer, hook) for layer in LAYERS for hook in HOOKS]  # the task's own key order
SCALES = {key: 10.0 ** i for i, key in enumerate(SAES)}
KILLED = {key: list(range(4 * i, 4 * i + 4)) for i, key in enumerate(SAES)}

MULTI = dict(
    gpu_tokens=(0, 0),
    source_kind="fixed_activation_pool",
    kill_latents=[],
    hp=dict(
        architecture_type="standard_saelens", batch_size=64, learning_rate=1e-3, l1_alpha=1e-3,
        warmup_steps=2, lr_decay_steps=12, sparsity_warmup_steps=3,
        training_layers=LAYERS, hook_types=HOOKS,
        resample_dead_neurons=True, resample_interval=5, dead_neuron_threshold=2,
    ),
    resamples_after_checkpoint=False,
)


def _scaled_extraction(root, name, seed):
    """E._extraction with one file per (layer, hook), each at its own scale."""
    folder = root / name
    folder.mkdir(parents=True)
    gen = np.random.default_rng(seed)
    for key in SAES:
        centres = gen.normal(size=(3, E.D))
        labels = gen.integers(0, 3, size=(E.ROWS, E.SEQ))
        acts = SCALES[key] * (centres[labels] + 0.3 * gen.normal(size=(E.ROWS, E.SEQ, E.D)))
        np.save(folder / f"layer_{key[0]}_{key[1]}.npy", acts.astype(np.float32))
    (folder / "metadata.json").write_text(json.dumps(
        {"num_samples_processed": E.ROWS, "layer_indices": LAYERS, "hook_types": HOOKS}
    ))
    return SimpleNamespace(id=f"ext_m_{name}", status="completed", output_path=str(folder), dataset_id=f"ds_{name}")


def test_each_sae_is_resampled_from_its_own_batch_tracker_and_optimizer(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "_extraction", _scaled_extraction)
    harness = E._Harness(monkeypatch, tmp_path, MULTI)
    tasks = harness.tasks

    inner_create = tasks.create_sae
    created = []

    def create(**kwargs):
        model = inner_create(**kwargs)
        key = SAES[len(created)]
        with torch.no_grad():
            model.encoder.bias[KILLED[key]] = -1e6  # this SAE's own dead latents
        created.append(model)
        return model

    inner_resample = tasks.resample_dead_latents
    calls = []

    def resample(model, optimizer, x, dead, *args, **kwargs):
        result = inner_resample(model, optimizer, x, dead, *args, **kwargs)
        calls.append((model, optimizer, float(x.detach().norm(dim=1).mean()), result.latents.tolist()))
        return result

    monkeypatch.setattr(tasks, "create_sae", create)
    monkeypatch.setattr(tasks, "resample_dead_latents", resample)

    assert harness.run()["status"] == "completed"
    assert len(created) == len(SAES)

    first = {}
    for model, optimizer, row_norm, latents in calls:
        key = SAES[next(i for i, m in enumerate(created) if m is model)]
        assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in model.parameters()}, (
            f"{key}: resampled with another SAE's optimizer"
        )
        # Token norms are ~3x the key's scale; the neighbouring keys are 10x away.
        assert 1.0 < row_norm / SCALES[key] < 6.0, f"{key}: resampled from another SAE's batch ({row_norm:.3g})"
        first.setdefault(key, latents)

    assert set(first) == set(SAES), f"not every SAE was resampled: {sorted(first)}"
    for key, latents in first.items():
        assert set(KILLED[key]) <= set(latents), f"{key}: its own dead latents were not resampled ({latents})"
