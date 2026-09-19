"""Resume and the activation storage plan, review round 2 (reviewer R2-A, 2026-09-15).

Driven through the real `train_sae_task` and `resume_training_task` with the harness in
test_resume_storage_plan.py (real extractions, the real rolling buffer and fixed-pool
sampler, the real on-the-fly source over a tiny transformers Llama; the database, GPU
placement and the memory readings faked per process).

WHAT EACH TEST PINS
* A run resumed twice records each resume once, carried through a PAUSE checkpoint of the
  resumed run (every test before this resumed once, so a history that was never carried
  from the checkpoint passed them all).
* A whole pool moved between the card and the host resumes bit-identically, and its log
  says so. It used to WARN "RESUME IS NOT BIT-IDENTICAL ... every batch after the
  checkpoint differs" while the resume report beside it recorded bit_identical: True.
* An on-the-fly resume marks the training's host RAM reserve allocated only once its
  buffer holds data. A resumed source is built without a prefill (review R1-B), and the
  reserve used to be marked allocated before the replay filled it: another job's host RAM
  guard stopped counting this job's 24 GiB for the whole replay.
* A re-planned run over UNEQUAL sources with dataset_weights repeats no token, serves no
  held-out token, and keeps its weights in the re-planned quotas. The task-level re-plan
  tests before this had two equal sources and no weights, so every quota list was split
  evenly by construction.
* A legacy weights-only resume is recorded in the resume history; it restarted the
  optimizer and the data position, and later checkpoints claimed no resume at all.
* An on-the-fly checkpoint without a plan whose quotas no longer match fails before any
  batch is drawn, with the row FAILED and no metric row discarded.

MUTATION CONTROLS: recorded in
.claude/context/sessions/review_sae_remediation_R2_A_2026-09-15.md (one line broken at a
time, the listed tests run, bytes restored, sha256 verified, `git diff` checked).
"""

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models.activation_extraction import ActivationExtraction
from src.models.checkpoint import Checkpoint
from src.models.model import Model
from src.models.training_metric import TrainingMetric
from src.services import dataset_mixture
from tests.unit.test_resume_storage_plan import (
    D,
    HIDDEN,
    KEYS,
    LATENT,
    ORIGIN_BATCH,
    ORIGIN_SEQ,
    _World,
    _cached_world,
    _gpu_free_for,
    _no_post_run_evaluation,
    _on_the_fly_world,
    _origin_gpu_free_for,
    _origins,
    _record_draws_and_held_out,
    _training_row,
)


def _resume_until(world, stop_at):
    """One resume through `resume_training_task`, paused again at ``stop_at`` (None: run on)."""
    world.dispatched.clear()
    world.tasks.resume_training_task.run(world.training.id)
    assert len(world.dispatched) == 1, world.dispatched
    kwargs = world.dispatched[-1]
    world.stop_at = stop_at
    try:
        return world.run(start_step=kwargs["start_step"], checkpoint_id=kwargs["checkpoint_id"]), kwargs
    finally:
        world.stop_at = None


def _outline(history):
    return [(r["checkpoint_step"], r["activation_plan"], r["bit_identical"]) for r in history]


# ── the history ─────────────────────────────────────────────────────────────


def test_a_run_resumed_twice_records_each_resume_once_through_a_pause_checkpoint(monkeypatch, tmp_path):
    """Paused after its step-10 checkpoint; resumed on a card that holds less (re-planned)
    and paused again at 18, OFF the checkpoint interval, so step 17 is a pause checkpoint of
    the resumed run; resumed again with the same memory (the step-17 plan is reused).

    Step 17's state and rows carry the first resume; step 25's carry both, in order, each
    once, with the checkpoint each resume came from. The pause checkpoint also carries the
    resumed run's own plan and position, or the second resume could not reuse it."""
    world = _cached_world(monkeypatch, tmp_path, rows=4_000, seq=16, hp={"total_steps": 30})
    _no_post_run_evaluation(monkeypatch, world)
    world.gpu_free = _gpu_free_for(80_000)
    world.pause(11)

    world.gpu_free = _gpu_free_for(70_000)
    outcome, first = _resume_until(world, stop_at=18)
    assert outcome["status"] == "paused", outcome
    state17 = world.state(17)
    assert _outline(state17["resume_history"]) == [(10, "replanned", False)]
    assert state17["resume_history"][0]["checkpoint_id"] == first["checkpoint_id"]
    assert (state17["activation_plan"]["mode"], state17["activation_plan"]["buffer_tokens"]) == world.plans[1]
    assert state17["activation_source"]["quotas"] == state17["activation_plan"]["quotas"]
    rows17 = [c for c in world.store[Checkpoint] if c.step == 17]
    assert rows17 and all(c.extra_metadata["resume_history"] == state17["resume_history"] for c in rows17)

    outcome, second = _resume_until(world, stop_at=None)
    assert outcome["status"] == "completed", outcome
    assert second["start_step"] == 18
    state25 = world.state(25)
    assert _outline(state25["resume_history"]) == [(10, "replanned", False), (17, "reused", True)]
    assert state25["resume_history"][0] == state17["resume_history"][0]
    assert state25["resume_history"][1]["checkpoint_id"] == second["checkpoint_id"]
    rows25 = [c for c in world.store[Checkpoint] if c.step == 25]
    assert rows25 and all(c.extra_metadata["resume_history"] == state25["resume_history"] for c in rows25)


def test_a_legacy_weights_only_resume_is_recorded_in_the_resume_history(monkeypatch, tmp_path):
    """A checkpoint with SAE weights and no training_state.pt (every checkpoint written
    before this remediation). The resume restarts Adam, the warmup and the data position; a
    rolling buffer starts its pass again. Every later checkpoint must say so."""
    world = _cached_world(monkeypatch, tmp_path, rows=4_000, seq=16)
    _no_post_run_evaluation(monkeypatch, world)
    world.gpu_free = _gpu_free_for(80_000)
    world.pause(11)
    removed = [path for path in world.data_dir.rglob("training_state.pt")]
    assert removed, "precondition: the run wrote training state to remove"
    for path in removed:
        path.unlink()

    assert world.resume()["status"] == "completed"
    history = world.state(15)["resume_history"]
    assert [
        (r["checkpoint_step"], r["activation_plan"], r["bit_identical"], r["source_restarted"]) for r in history
    ] == [(10, "weights_only", False, True)], history
    assert history[0]["checkpoint_id"] == world.dispatched[-1]["checkpoint_id"]
    rows15 = [c for c in world.store[Checkpoint] if c.step == 15]
    assert rows15 and all(c.extra_metadata["resume_history"] == history for c in rows15)


# ── a whole pool that moved ─────────────────────────────────────────────────


def _record_pool_draws(monkeypatch, phase):
    from src.services import activation_buffer

    draws = {}
    real_next = activation_buffer.FixedPoolSampler.next_batch

    def next_batch(sampler, batch_size):
        out = real_next(sampler, batch_size)
        draws.setdefault(phase["phase"], []).append({key: value.clone() for key, value in out.items()})
        return out

    monkeypatch.setattr(activation_buffer.FixedPoolSampler, "next_batch", next_batch)
    return draws


def test_a_whole_pool_moved_off_the_card_resumes_bit_identically_and_the_log_says_so(monkeypatch, tmp_path, caplog):
    """The run loads its 128,000-token pool whole on the card (gpu_all); the resumed card
    holds nothing, the host holds the pool, so it is re-planned as cpu_all. The same pool
    drawn by the same sampler generator: every batch after the checkpoint equals the
    uninterrupted run's, the report says bit_identical, and no log line says otherwise."""
    phase = {"phase": "straight"}
    draws = _record_pool_draws(monkeypatch, phase)
    straight = _cached_world(monkeypatch, tmp_path / "straight", rows=4_000, seq=16)
    _no_post_run_evaluation(monkeypatch, straight)
    straight.gpu_free = _gpu_free_for(150_000)
    assert straight.run()["status"] == "completed"

    phase["phase"] = "paused"
    world = _cached_world(monkeypatch, tmp_path / "resumed", rows=4_000, seq=16)
    _no_post_run_evaluation(monkeypatch, world)
    world.gpu_free = _gpu_free_for(150_000)
    world.pause(11)
    world.gpu_free = 0
    phase["phase"] = "resumed"
    caplog.set_level(logging.INFO, logger=world.tasks.logger.name)
    assert world.resume()["status"] == "completed"

    assert world.plans[0] == ("gpu_all", 128_000) and world.plans[1] == ("cpu_all", 128_000), world.plans
    history = world.state(15)["resume_history"]
    assert _outline(history) == [(10, "replanned", True)], history
    assert len(draws["resumed"]) == 9
    for step, (a, b) in enumerate(zip(draws["straight"][11:], draws["resumed"], strict=True), start=11):
        for key in a:
            assert torch.equal(a[key].cpu(), b[key].cpu()), f"step {step} {key}: the resumed batch differs"
    claims = [r.getMessage() for r in caplog.records if "NOT BIT-IDENTICAL" in r.getMessage()]
    assert claims == [], claims
    moved = [r.getMessage() for r in caplog.records
             if r.levelno >= logging.WARNING and "continue exactly" in r.getMessage()]
    assert len(moved) == 1 and "gpu_all" in moved[0] and "cpu_all" in moved[0], moved


# ── the host RAM reserve ────────────────────────────────────────────────────


@pytest.mark.parametrize("path", ["cached", "on_the_fly"])
def test_the_host_reserve_is_marked_allocated_once_per_process_after_the_buffer_holds_data(
    monkeypatch, tmp_path, path
):
    """`host_reserve_allocated` tells other jobs' host RAM guards that this training's
    reserve is now inside MemAvailable. It must run once per process, and only when the
    buffer it stands for exists. The on-the-fly resume builds its source empty and fills it
    in the restore; the cached rolling buffer fills in its constructor (the control)."""
    from src.services import activation_buffer
    from src.services import gpu_job_claim
    from src.services import model_activation_source as MAS

    sources, marked = [], []
    for cls in (activation_buffer.RollingActivationBuffer, MAS.ModelActivationSource):
        real_init = cls.__init__

        def init(self, *args, _real=real_init, **kwargs):
            _real(self, *args, **kwargs)
            sources.append(self)

        monkeypatch.setattr(cls, "__init__", init)
    monkeypatch.setattr(gpu_job_claim, "host_reserve_allocated", lambda: marked.append(int(sources[-1].size)))

    if path == "cached":
        world = _cached_world(monkeypatch, tmp_path, rows=4_000, seq=16)
        world.gpu_free = _gpu_free_for(80_000)
    else:
        world = _on_the_fly_world(monkeypatch, tmp_path, rows_per_dataset=2_000)
        world.ram_bytes = 55_000 * HIDDEN * 4 * len(KEYS) * 2
    _no_post_run_evaluation(monkeypatch, world)
    world.pause(6)
    assert len(marked) == 1 and marked[0] > 0, marked
    assert world.resume()["status"] == "completed"
    assert len(sources) == 2 and len(marked) == 2, (len(sources), marked)
    assert marked[1] > 0, f"the reserve was marked allocated while the resumed buffer held {marked[1]} tokens"


# ── uniqueness over unequal, weighted sources ───────────────────────────────

WEIGHTS = [0.8, 0.2]


def _unequal_origin_world(monkeypatch, tmp_path, *, rows=(16_000, 9_000), total_steps=60):
    """test_resume_storage_plan's origin-encoding extractions, of UNEQUAL sizes, trained with
    dataset_weights far from their availability (the larger source would supply ~64%
    unweighted; it is asked for 80%)."""
    extractions = []
    for source, (name, n) in enumerate(zip(("a", "b"), rows)):
        folder = tmp_path / name
        folder.mkdir(parents=True)
        acts = np.zeros((n, ORIGIN_SEQ, D), dtype=np.float32)
        acts[..., 0] = source
        acts[..., 1] = (np.arange(n, dtype=np.float32) / 4096.0)[:, None]
        acts[..., 2] = (np.arange(ORIGIN_SEQ, dtype=np.float32) / 16.0)[None, :]
        acts[..., 3:] = np.random.default_rng(source + 1).normal(size=(n, ORIGIN_SEQ, D - 3)).astype(np.float32)
        np.save(folder / "layer_0_residual.npy", acts)
        mask = np.ones((n, ORIGIN_SEQ), dtype=bool)
        mask[::7, 12:] = False
        np.save(folder / "attention_mask.npy", mask)
        (folder / "metadata.json").write_text(json.dumps({
            "num_samples_processed": n, "layer_indices": [0], "hook_types": ["residual"],
        }))
        extractions.append(SimpleNamespace(id=f"ext_{name}", status="completed", output_path=str(folder),
                                           dataset_id=f"ds_{name}"))
    training = _training_row(
        "train_plan_unequal",
        {
            "hidden_dim": D, "latent_dim": LATENT, "batch_size": ORIGIN_BATCH, "learning_rate": 1e-3,
            "total_steps": total_steps, "checkpoint_interval": 5, "log_interval": 1, "seed": 7,
            "training_layers": [0], "hook_types": ["residual"],
            "architecture_type": "standard_saelens", "l1_alpha": 1e-3, "warmup_steps": 0,
            "sparsity_warmup_steps": 0, "resample_dead_neurons": False, "evaluate_ce_delta": False,
            "grad_clip_norm": None, "holdout_fraction": 0.1, "holdout_eval_tokens": 512,
            "dataset_weights": list(WEIGHTS),
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


def test_a_replanned_run_over_unequal_weighted_sources_repeats_nothing_and_keeps_its_weights(monkeypatch, tmp_path):
    """Planned at 55,000 tokens per layer, paused after the step-10 checkpoint, resumed on a
    card that holds 52,000, then 49 more steps: the re-planned buffer is refilled twice.

    Required, over the run's effective history (the first process's steps 0-10, the
    resumed process's 11-59): no token served twice (each source is taken well inside one
    pass, checked below); no held-out position served, and the split unchanged; the
    re-planned quotas are the WEIGHTED allocation of the new buffer, not the availability
    split; and the resumed process's batches supply source b at its weight."""
    world = _unequal_origin_world(monkeypatch, tmp_path)
    _record_draws_and_held_out(monkeypatch, world)
    world.gpu_free = _origin_gpu_free_for(55_000)
    world.process = 1
    world.pause(11)
    assert world.plans[0] == ("gpu_rolling", 55_000), world.plans
    saved = world.state(10)
    counts = saved["activation_plan"]["source_tokens"]
    saved_quotas = saved["activation_plan"]["quotas"]
    assert counts[0] != counts[1], "precondition: unequal sources"
    assert saved_quotas == dataset_mixture.allocate_tokens(counts, 55_000, WEIGHTS)
    unweighted = dataset_mixture.allocate_tokens(counts, 52_000, None)

    world.process = 2
    world.gpu_free = _origin_gpu_free_for(52_000)
    assert world.resume()["status"] == "completed"

    state = world.state(55)
    new_quotas = state["activation_plan"]["quotas"]
    assert _outline(state["resume_history"]) == [(10, "replanned", False)]
    assert new_quotas == dataset_mixture.allocate_tokens(counts, 52_000, WEIGHTS), new_quotas
    assert new_quotas != saved_quotas and abs(unweighted[1] - new_quotas[1]) > 5_000, (
        "precondition: the weighted and unweighted splits are far apart", unweighted, new_quotas,
    )
    refills_after = state["activation_source"]["refills"] - saved["activation_source"]["refills"]
    assert refills_after >= 2, f"precondition: the re-planned buffer was refilled {refills_after} times"
    assert len(world.draws[1]) == 11 and len(world.draws[2]) == 49, {k: len(v) for k, v in world.draws.items()}
    # Every source is taken well inside one pass, so a repeat cannot be a legitimate new pass.
    for source in (0, 1):
        taken = saved_quotas[source] + (refills_after + 1) * new_quotas[source]
        assert taken < counts[source], (source, taken, counts[source])

    assert world.held[1] and world.held[1] == world.held[2]
    served = [origin for process in (1, 2) for batch in world.draws[process] for origin in batch]
    assert not set(served) & world.held[1], "a held-out position was served"
    repeats = len(served) - len(set(served))
    assert repeats == 0, f"{repeats} activations were served twice across the re-planned resume"
    resumed = [origin for batch in world.draws[2] for origin in batch]
    share_b = sum(1 for origin in resumed if origin[0] == 1) / len(resumed)
    assert abs(share_b - WEIGHTS[1]) < 0.02, share_b


# ── a refusal ───────────────────────────────────────────────────────────────


def test_an_on_the_fly_checkpoint_without_a_plan_whose_quotas_moved_fails_before_drawing_or_discarding(
    monkeypatch, tmp_path, caplog
):
    """A resumed on-the-fly source is built empty (no prefill). When the position cannot be
    loaded, the run must fail there: no batch drawn from an empty source, the row FAILED
    with the reason, and the metric rows after the checkpoint still in place (the discard
    comes after the restore)."""
    from src.services import model_activation_source as MAS
    from src.services.checkpoint_service import save_training_state

    world = _on_the_fly_world(monkeypatch, tmp_path, rows_per_dataset=2_000)
    _no_post_run_evaluation(monkeypatch, world)
    world.ram_bytes = 55_000 * HIDDEN * 4 * len(KEYS) * 2
    world.pause(8)
    for path in sorted(world.data_dir.rglob("training_state.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state.pop("activation_plan") is not None, "precondition: the state had a plan to remove"
        state.pop("format"), state.pop("version")
        save_training_state(path.parent, state)
    metrics_before = len(world.store.get(TrainingMetric, []))
    assert any(m.step > 5 for m in world.store.get(TrainingMetric, [])), "precondition: steps logged past the checkpoint"

    drawn = []
    real_next = MAS.ModelActivationSource.next_batch
    monkeypatch.setattr(MAS.ModelActivationSource, "next_batch",
                        lambda source, n: drawn.append(n) or real_next(source, n))
    world.ram_bytes -= 1024**2
    caplog.set_level(logging.INFO, logger=world.tasks.logger.name)
    with pytest.raises(ValueError, match="quotas"):
        world.resume()
    assert "records no activation storage plan" in caplog.text
    assert drawn == [], "the resumed source served a batch"
    assert world.training.status == "failed" and "quotas" in (world.training.error_message or "")
    assert len(world.store.get(TrainingMetric, [])) == metrics_before, "a refused resume discarded metric rows"


# ── the reading a plan is judged against ────────────────────────────────────


def test_the_free_memory_reading_and_the_restored_state_credit_are_logged(monkeypatch, caplog):
    """The strictness decision (a saved plan is reused only if this process can hold it)
    needs the two processes' readings side by side on the card. Each reading is logged
    with the credit applied to it."""
    from src.services import activation_plan as AP

    monkeypatch.setattr(AP, "restored_state_bytes", lambda optimizers, models, device: 700)
    caplog.set_level(logging.INFO, logger=AP.logger.name)
    assert AP.free_bytes_for_planning(
        123_456, resuming=True, optimizers={}, models={}, device=torch.device("cpu"),
    ) == 124_156
    assert AP.free_bytes_for_planning(
        123_456, resuming=False, optimizers={}, models={}, device=torch.device("cpu"),
    ) == 123_456
    lines = [r.getMessage() for r in caplog.records if "Free memory reading" in r.getMessage()]
    assert len(lines) == 2, lines
    assert "123,456" in lines[0] and "700" in lines[0] and "resum" in lines[0]
    assert "123,456" in lines[1] and "fresh" in lines[1]
