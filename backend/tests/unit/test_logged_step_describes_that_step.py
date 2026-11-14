"""A logged row describes the step it is logged under (review round 2, R2-C, 2026-09-15).

Driven through the real task with the resume-equivalence harness: `train_sae_task`
end to end on the CPU over real `.npy` extractions, real SAEs, real Adam, the real
LambdaLR, the real rolling buffer and fixed pool, and the real held-out evaluation.

L3 — THE LOGGED LEARNING RATE WAS THE NEXT STEP'S. The log block read
`get_last_lr()` after the step's `scheduler.step()`. Without accumulation, step 0
logged the warmup factor of step 1 while its update ran at factor 0, and the last
step logged 0 while its update ran at 1/decay. With accumulation the step that ends
a window logged the NEXT window's rate. Every row (aggregate, per SAE, the progress
row and event) now carries the rate this step's gradient is applied with: the rate
of the optimizer update that closes the step's accumulation window. The fixtures
have a warmup AND a decay, and the pins sit on the boundaries: with warmup 0 and no
decay every factor is 1 and the two readings cannot be told apart.

L1 — AT A RESAMPLE STEP THE HELD-OUT ROW SCORED THE RESAMPLE. The resample ran before
the log block, so the held-out evaluation at a step that was both a resample step
and a log step scored freshly re-initialised latents (encoder rows at 0.2x the alive
norm, zeroed moments) under the step's number. Resample intervals are multiples of
the usual log intervals, so that was every resample. The in-sample numbers were never
affected: they are read off the step's own forward, before either. The held-out row
now scores the model the step trained, and the resample shows at the next log.

Both tests were red on the code before the fix (7649ddbd).

MUTATION CONTROLS: see the R2-C record and the table at the end of this docstring.
"""

import pytest
import torch

from src.models.training_metric import TrainingMetric
from src.services import holdout_evaluation
from src.services.lr_schedule import lr_multiplier
from src.workers.training_tasks import grad_accum_steps_for
from tests.unit import test_training_resume_equivalence as E

#: The multiplier each pinned step's row must carry, per configuration: steps on the
#: warmup and decay boundaries, where the rate applied and the next step's rate differ.
PINNED = {
    # No accumulation; warmup 2, decay over the final 12 of 20 steps.
    "pool-standard-mid-decay": {0: 0.0, 1: 0.5, 2: 1.0, 7: 1.0, 8: 1.0, 9: 11 / 12, 19: 1 / 12},
    # Windows of 4; warmup 2, decay 4. A window reads its factor at its first step.
    "rolling-jumprelu-accum": {0: 0.0, 3: 0.0, 4: 1.0, 15: 1.0, 16: 1.0, 19: 1.0},
}


def _rows(harness, kind):
    rows = harness.store.get(TrainingMetric, [])
    if kind == "aggregate":
        return [m for m in rows if m.layer_idx is None]
    if kind == "per_sae":
        return [m for m in rows if m.layer_idx is not None and m.layer_idx >= 0]
    return [m for m in rows if m.layer_idx is not None and m.layer_idx < 0]


def _snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


@pytest.mark.parametrize("name", list(E.CONFIGS))
def test_every_row_logs_the_learning_rate_its_step_is_applied_with(monkeypatch, tmp_path, name):
    config = E.CONFIGS[name]
    hp = config["hp"]
    harness = E._Harness(monkeypatch, tmp_path, config)
    progress = []
    real_progress = harness.tasks.train_sae_task.update_training_progress

    def recording_progress(**kwargs):
        progress.append((kwargs["step"], kwargs.get("learning_rate")))
        return real_progress(**kwargs)

    monkeypatch.setattr(harness.tasks.train_sae_task, "update_training_progress", recording_progress)
    assert harness.run()["status"] == "completed"

    accum = grad_accum_steps_for(hp["batch_size"])
    schedule = dict(total_steps=E.TOTAL, warmup_steps=hp["warmup_steps"], decay_steps=hp["lr_decay_steps"])
    applied = harness.applied_lrs()
    assert len(applied) == E.TOTAL // accum, "precondition: every window closed with an update"

    # The fixture can tell the two readings apart at the pinned steps.
    next_steps = {s: lr_multiplier((s // accum + 1) * accum, **schedule) for s in PINNED[name]}
    assert any(next_steps[s] != factor for s, factor in PINNED[name].items()), "fixture trap: no boundary"

    series = {
        "aggregate": {m.step: m.learning_rate for m in _rows(harness, "aggregate")},
        "per_sae": {m.step: m.learning_rate for m in _rows(harness, "per_sae")},
        "progress": dict(progress),
    }
    for kind, logged in series.items():
        assert sorted(logged) == list(range(E.TOTAL)), (kind, sorted(logged))
        for step, lr in logged.items():
            window = step // accum
            # What the optimizer really applied for this step's window...
            assert lr == pytest.approx(applied[window], abs=1e-12), (kind, step, lr, applied[window])
            # ...which is the configured curve, read at the window's first step.
            assert lr == pytest.approx(
                hp["learning_rate"] * lr_multiplier(window * accum, **schedule), abs=1e-12
            ), (kind, step)
        for step, factor in PINNED[name].items():
            assert logged[step] == pytest.approx(hp["learning_rate"] * factor, abs=1e-12), (kind, step)


@pytest.mark.parametrize("name", list(E.CONFIGS))
def test_the_held_out_row_at_a_resample_step_scores_the_model_that_step_trained(monkeypatch, tmp_path, name):
    base = E.CONFIGS[name]
    config = {**base, "hp": {**base["hp"], "holdout_fraction": 0.34}}
    harness = E._Harness(monkeypatch, tmp_path, config)
    tasks = harness.tasks

    due = {"step": None}
    inner_due = tasks.resample_due

    def gate(step, **kwargs):
        answer = inner_due(step, **kwargs)
        if answer:
            due["step"] = step
        return answer

    events = []
    inner_resample = tasks.resample_dead_latents

    def resample(model, *args, **kwargs):
        before = _snapshot(model)
        result = inner_resample(model, *args, **kwargs)
        events.append((due["step"], result.latents.tolist(), before, _snapshot(model)))
        return result

    evaluations = []
    real_evaluate = holdout_evaluation.evaluate_holdout

    def evaluate(model, *args, **kwargs):
        result = real_evaluate(model, *args, **kwargs)
        evaluations.append((_snapshot(model), result))
        return result

    monkeypatch.setattr(tasks, "resample_due", gate)
    monkeypatch.setattr(tasks, "resample_dead_latents", resample)
    monkeypatch.setattr(holdout_evaluation, "evaluate_holdout", evaluate)

    assert harness.run()["status"] == "completed"

    # Each held-out row is the evaluation made for it, in order.
    held_rows = _rows(harness, "held_out")
    assert held_rows and len(held_rows) == len(evaluations), (len(held_rows), len(evaluations))
    scored = {}
    for row, (state, result) in zip(held_rows, evaluations):
        assert row.fvu == pytest.approx(result["fvu"])
        scored[row.step] = state

    performed = [(step, latents, before, after) for step, latents, before, after in events if latents]
    assert performed, "precondition: nothing was resampled"
    for step, latents, before, after in performed:
        assert step % harness.hp["log_interval"] == 0 and step in scored, "precondition: a resample on a log step"
        encoder = "W_enc" if "W_enc" in before else "encoder.weight"
        rows = torch.tensor(latents)
        assert not torch.equal(before[encoder][rows], after[encoder][rows]), "precondition: the rows changed"

        at_step = scored[step]
        assert all(torch.equal(at_step[k], before[k]) for k in before), (
            f"step {step}: the held-out row scored the resampled latents, not the model the step trained"
        )
        # The resample is not hidden: the next log scores a model that has it.
        following = scored[step + 1][encoder][rows]
        assert (following - after[encoder][rows]).norm() < (following - before[encoder][rows]).norm(), step
