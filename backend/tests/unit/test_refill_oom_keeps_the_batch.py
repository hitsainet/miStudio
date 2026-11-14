"""An OOM while the activation buffer refills does not change the optimisation batch (R1-D L5).

Review round 1, 2026-09-15. The training loop's OOM handler caught every CUDA OOM in
the step — including one raised while `draw_cached_batch` refilled the rolling
buffer, a ~15 GB read unrelated to the SAE's batch — halved `batch_size`, and
PERSISTED the halved value to the row. A transient refill failure therefore changed
the run's optimisation batch for the rest of training and for every resume. A
refill that fails is retried by the buffer on the same rows; the batch it serves
must be the configured one.

Driven through the real task (`train_sae_task` over real `.npy` extractions, the
real rolling buffer) with the resume-equivalence harness. The OOM is injected at
the phase it would come from: the draw, or a call inside the SAE step.

The first and third tests were red on the code before the fix (batch 16 -> 8; no
refill-specific failure).

MUTATION CONTROLS (R1-A, 2026-09-15; applied alone, restored, sha256 verified):
  L5a the refill branch removed (`if oom_phase == "draw":` -> False) -> refill tests red
  L5b the draw never marked as the draw phase                       -> refill tests red
"""

import pytest

from tests.unit import test_training_resume_equivalence as E

BATCH = E.CONFIGS["rolling-jumprelu-accum"]["hp"]["batch_size"]


def _oom():
    return RuntimeError("CUDA out of memory. Tried to allocate 4.97 GiB")


def test_an_oom_during_a_refill_retries_with_the_same_batch(monkeypatch, tmp_path):
    harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["rolling-jumprelu-accum"])
    real_draw = E._training_tasks.draw_cached_batch
    sizes, failed = [], []

    def draw(stream, cached, keys, batch_size, num_samples, device):
        sizes.append(batch_size)
        if len(sizes) == 7 and not failed:
            failed.append(batch_size)
            raise _oom()
        return real_draw(stream, cached, keys, batch_size, num_samples, device)

    monkeypatch.setattr(E._training_tasks, "draw_cached_batch", draw)

    assert harness.run()["status"] == "completed"
    assert failed == [BATCH], "precondition: the injected refill OOM happened"
    assert set(sizes) == {BATCH}, sizes
    assert harness.training.hyperparameters["batch_size"] == BATCH


def test_an_oom_in_the_sae_step_still_halves_the_batch(monkeypatch, tmp_path):
    """The control: the handler's halving is kept for the OOM it was written for."""
    harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["rolling-jumprelu-accum"])
    density = E._training_tasks.feature_density
    real_update = density.update_firing_rate
    calls = []

    def update(previous, z):
        calls.append(1)
        if len(calls) == 7:
            raise _oom()
        return real_update(previous, z)

    monkeypatch.setattr(density, "update_firing_rate", update)

    assert harness.run()["status"] == "completed"
    assert harness.training.hyperparameters["batch_size"] < BATCH


def test_a_refill_that_keeps_running_out_of_memory_fails_the_run_saying_so(monkeypatch, tmp_path):
    harness = E._Harness(monkeypatch, tmp_path, E.CONFIGS["rolling-jumprelu-accum"])
    real_draw = E._training_tasks.draw_cached_batch
    sizes = []

    def draw(stream, cached, keys, batch_size, num_samples, device):
        sizes.append(batch_size)
        if len(sizes) > 3:
            raise _oom()
        return real_draw(stream, cached, keys, batch_size, num_samples, device)

    monkeypatch.setattr(E._training_tasks, "draw_cached_batch", draw)

    with pytest.raises(RuntimeError, match="reading the activation buffer"):
        harness.run()
    assert set(sizes) == {BATCH}, sizes
    assert harness.training.hyperparameters["batch_size"] == BATCH
