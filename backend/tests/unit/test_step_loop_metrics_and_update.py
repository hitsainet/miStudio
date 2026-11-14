"""What the step loop logs is what the step computed, and the update does what it is configured to (review R2-C).

Round 2 mutated every load-bearing line of the step body. Twelve survived the whole loop
suite, because every loop test compared a run with another run of the same code (resume
equivalence) or checked a single field. Each line is pinned here against an independent
computation from the step's own forward, recorded as the task ran:

  H4   the per-SAE `loss` undoes the accumulation division (`loss.item() * k`)
  H10  the CPU branch divides the loss by k before backward (the logged loss showed x k)
  H9   `l0_sparsity` is a FRACTION of latents, not a count per token
  H8   the activity EMA decays by `1 - batch/50,000` per step (the window R1-A's A14 shows in the UI)
  H18  a latent is inactive below an EMA of 0.01
  H5   the sparsity warmup scales the penalty by min(1, step / sparsity_warmup_steps) ...
  H17  ... for L1 frameworks (`l1_alpha`) ...
  H16  ... and for JumpReLU (`sparsity_coeff`)
  H3   JumpReLU decoder columns are renormalised after every update
  H3b  Standard / Skip decoder columns are renormalised after every update
  H1   `grad_clip_norm` clips the gradient the optimizer applies (CPU branch)
  H6   the OOM retry count resets after a successful step: separate OOMs do not add up to a failure

Each was re-run as a negative control against these tests (the R2-C record lists them).

Not controllable here: the GradScaler branch's clip, projection and normalisation (CUDA only).

MUTATION CONTROLS (2026-09-16, acceptance item 10 — `grad_norm` was never written):
  G1 the log site passes grad_norm=None       -> the_row_records_the_norm_the_clip_measured
  G2 the CPU clip site discards the return    -> the_row_records_the_norm_the_clip_measured
     (its dict entry is never set, so the row logs None again — the original defect)
"""

import pytest
import torch

from src.ml.sparse_autoencoder import JumpReLUSAE
from src.models.training_metric import TrainingMetric
from src.workers.training_tasks import grad_accum_steps_for
from tests.unit import test_training_resume_equivalence as E

#: The activity EMA's window and threshold, as R1-A's A14 documents them to the operator.
EMA_WINDOW_TOKENS = 50_000
INACTIVE_BELOW = 0.01

ACCUM = {
    **E.CONFIGS["rolling-jumprelu-accum"],
    "hp": {**E.CONFIGS["rolling-jumprelu-accum"]["hp"], "sparsity_warmup_steps": 4},
}
#: A batch large enough that the activity EMA forgets within the run (decay 0.6 per step),
#: and one latent that fires until step 5 and never again, so "inactive" moves mid-run.
BIG_BATCH = dict(
    gpu_tokens=(0, 0),
    source_kind="fixed_activation_pool",
    kill_latents=[],
    hp=dict(
        architecture_type="standard_saelens", batch_size=20_000, learning_rate=1e-3, l1_alpha=1e-3,
        warmup_steps=2, lr_decay_steps=12, sparsity_warmup_steps=3, resample_dead_neurons=False,
    ),
    resamples_after_checkpoint=False,
)
SILENCED_AT, SILENCED = 5, 3


def _coefficient(model):
    """The coefficient the loss reads. JumpReLU also carries an `l1_alpha` alias the loss never uses."""
    return model.sparsity_coeff if isinstance(model, JumpReLUSAE) else model.l1_alpha


def _decoder_columns(model):
    weight = model.W_dec if hasattr(model, "W_dec") and model.W_dec is not None else model.decoder.weight
    return weight.detach().norm(dim=0)


def _record(monkeypatch, harness, *, silence=None):
    """Wrap the draw (to know the step) and every SAE's training forward (to see what it computed)."""
    tasks = harness.tasks
    now = {"step": None}
    seen = []
    real_draw = tasks.draw_cached_batch

    def draw(stream, cached, keys, batch_size, num_samples, device):
        now["step"] = len(seen_steps)
        seen_steps.append(now["step"])
        if silence and now["step"] == silence[0]:
            with torch.no_grad():
                harness.models[-1].encoder.bias[silence[1]] = -1e6
        return real_draw(stream, cached, keys, batch_size, num_samples, device)

    seen_steps = []
    inner_create = tasks.create_sae

    def create(**kwargs):
        model = inner_create(**kwargs)
        real_forward = model.forward

        def forward(*args, **kw):
            training = kw.get("return_loss", True) and now["step"] is not None
            if training:
                coefficient = float(_coefficient(model))
                columns = _decoder_columns(model).clone()
            out = real_forward(*args, **kw)
            if training:
                x_hat, z, losses = out
                seen.append(dict(step=now["step"], loss=float(losses["loss"]), z=z.detach().clone(),
                                 coefficient=coefficient, columns=columns))
            return out

        model.forward = forward
        return model

    monkeypatch.setattr(tasks, "draw_cached_batch", draw)
    monkeypatch.setattr(tasks, "create_sae", create)
    return seen


@pytest.mark.parametrize("config", [ACCUM, BIG_BATCH], ids=["jumprelu-accum", "standard-big-batch"])
def test_the_per_sae_row_is_what_the_steps_forward_computed(monkeypatch, tmp_path, config):
    hp = config["hp"]
    harness = E._Harness(monkeypatch, tmp_path, config)
    silence = (SILENCED_AT, SILENCED) if config is BIG_BATCH else None
    seen = _record(monkeypatch, harness, silence=silence)

    assert harness.run()["status"] == "completed"
    assert [s["step"] for s in seen] == list(range(E.TOTAL)), "one training forward per step"
    rows = {m.step: m for m in harness.store[TrainingMetric] if m.layer_idx == 0}
    assert sorted(rows) == list(range(E.TOTAL))

    ema = None
    counts = []
    for s in seen:
        step, z, row = s["step"], s["z"], rows[s["step"]]
        # H4 / H10: the loss the forward computed, not the accumulation-divided one.
        assert row.loss == pytest.approx(s["loss"], rel=1e-6), (step, row.loss, s["loss"])
        # H9: a fraction of latents.
        assert row.l0_sparsity == pytest.approx(float((z != 0).float().mean()), rel=1e-6), step
        # H8 / H18: the activity EMA, recomputed from the forward's latents.
        fired = (z > 0).any(dim=0).float()
        decay = max(0.0, 1.0 - z.shape[0] / EMA_WINDOW_TOKENS)
        ema = fired if ema is None else ema * decay + fired
        counts.append(int((ema < INACTIVE_BELOW).sum()))
        assert row.dead_neurons == counts[-1], (step, row.dead_neurons, counts[-1])
    if config is BIG_BATCH:
        assert len(set(counts)) > 1, f"precondition: the inactive count never moved ({counts})"

    # H5 / H16 / H17: the penalty coefficient each forward ran with follows the sparsity warmup.
    warmup = hp["sparsity_warmup_steps"]
    base = hp["sparsity_coeff"] if hp["architecture_type"] == "jumprelu" else hp["l1_alpha"]
    assert warmup > 0
    for s in seen:
        assert s["coefficient"] == pytest.approx(base * min(1.0, s["step"] / warmup), rel=1e-6), s["step"]

    # H3 / H3b: after the first update, every forward sees unit decoder columns.
    first_update = grad_accum_steps_for(hp["batch_size"])
    for s in seen:
        if s["step"] >= first_update:
            assert torch.allclose(s["columns"], torch.ones_like(s["columns"]), atol=1e-6), (
                s["step"], float((s["columns"] - 1).abs().max())
            )


def test_grad_clip_norm_bounds_the_gradient_the_optimizer_applies(monkeypatch, tmp_path):
    clip = 1e-6
    config = {**BIG_BATCH, "hp": {**BIG_BATCH["hp"], "batch_size": 64, "grad_clip_norm": clip}}
    harness = E._Harness(monkeypatch, tmp_path, config)
    norms = []
    real_step = torch.optim.Adam.step

    def step(self, *args, **kwargs):
        grads = [p.grad for group in self.param_groups for p in group["params"] if p.grad is not None]
        norms.append(float(torch.norm(torch.stack([g.norm() for g in grads]))))
        return real_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    assert harness.run()["status"] == "completed"
    assert len(norms) == E.TOTAL
    assert all(n <= clip * (1 + 1e-3) for n in norms), max(norms)
    assert max(norms) > 0.5 * clip, "precondition: the gradients were large enough to clip"


def test_the_row_records_the_norm_the_clip_measured(monkeypatch, tmp_path):
    """`grad_norm` is the PRE-clip total norm `clip_grad_norm_` returns (G1, G2).

    Acceptance item 10. The column, the schema field and `log_metric`'s parameter
    all existed while BOTH clip sites called `clip_grad_norm_` as a bare statement
    and discarded its return — so every row this project has ever written stored
    NULL, and data-model.md documented a field nothing populates.
    """
    clip = 1e-6
    config = {**BIG_BATCH, "hp": {**BIG_BATCH["hp"], "batch_size": 64, "grad_clip_norm": clip}}
    harness = E._Harness(monkeypatch, tmp_path, config)
    returned = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm, *args, **kwargs):
        total = real_clip(parameters, max_norm, *args, **kwargs)
        returned.append(float(total))
        return total

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    assert harness.run()["status"] == "completed"
    assert len(returned) == E.TOTAL, f"one clip per update step: {len(returned)}"

    rows = {m.step: m for m in harness.store[TrainingMetric] if m.layer_idx == 0}
    logged = [rows[s].grad_norm for s in sorted(rows)]
    assert all(v is not None for v in logged), logged
    assert logged == pytest.approx(returned, rel=1e-6)
    # PRE-clip, so the row shows whether the clip BIT: a value above the bound was
    # clipped down to it. A post-clip norm would read <= clip on every row and say
    # nothing.
    assert max(logged) > clip, (max(logged), clip)


def test_no_clipping_configured_logs_no_norm(monkeypatch, tmp_path):
    """None, never a fabricated value: nothing clipped, so there is no norm to report."""
    hp = {k: v for k, v in BIG_BATCH["hp"].items() if k != "grad_clip_norm"}
    harness = E._Harness(monkeypatch, tmp_path, {**BIG_BATCH, "hp": {**hp, "batch_size": 64}})
    assert harness.run()["status"] == "completed"
    rows = [m for m in harness.store[TrainingMetric] if m.layer_idx == 0]
    assert rows, "precondition: the run logged per-SAE rows"
    assert all(m.grad_norm is None for m in rows), [m.grad_norm for m in rows]


def test_separate_ooms_over_a_run_do_not_add_up_to_a_failure(monkeypatch, tmp_path):
    config = E.CONFIGS["rolling-jumprelu-accum"]
    harness = E._Harness(monkeypatch, tmp_path, config)
    real_draw = E._training_tasks.draw_cached_batch
    calls, raised = [], []

    def draw(stream, cached, keys, batch_size, num_samples, device):
        calls.append(1)
        if len(calls) in (4, 10, 16):  # three OOMs, each followed by successful steps
            raised.append(len(calls))
            raise RuntimeError("CUDA out of memory. Tried to allocate 4.97 GiB")
        return real_draw(stream, cached, keys, batch_size, num_samples, device)

    monkeypatch.setattr(E._training_tasks, "draw_cached_batch", draw)
    assert harness.run()["status"] == "completed"
    assert raised == [4, 10, 16]
