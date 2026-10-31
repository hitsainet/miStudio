"""Probe training runs where the run's model is, and layer selection uses every core (2026-10-22).

Operator: "the trainings don't seem to be using the gpu. Cpu utilization is high, though." Measured
on `pmr_2bbbf15c1a89`: training 845 s and layer selection 302 s of a 1,408 s run, both on the CPU,
while the card held the model at 2.2 GB. The run passed `torch.device("cpu")` to every `train_rule`
call, and `select_layers` fitted its 32 cells one after another.

NEGATIVE CONTROLS (recorded in 0xcc/reviews/probe_training_device_2026-10-22.md):
  D1  `device=_device` dropped from the run's `train_rule` call   → the wiring test
  D2  `training_device` returns the model's card unconditionally  → the does-not-fit test
  D3  the headroom term dropped from `training_device`            → the headroom test
  D4  `select_layers` default workers pinned to 1                 → the default-workers test
  D5  `pool.map` replaced by unordered completion                 → the identical-grid test
  D6  the query drawn from the global stream or on the device    → the seeded-cpu-draw test
  D6b the query started at zero again                            → the not-zero test
  D7  `train_rule` builds its own query, or drops the seed       → the starts-from-that-query test
  D8  the run's fit outside `deterministic_kernels`               → the wiring test
  D9  `deterministic_kernels` never restores the flag             → the restore tests
  D10 the CUBLAS_WORKSPACE_CONFIG line dropped from the image     → the image test
"""
import types
import uuid

import numpy as np
import pytest
import torch

from src.services import probe_monitor_trainer as trainer
from src.services.probe_monitor_trainer import (
    DEVICE_HEADROOM_BYTES,
    select_layers,
    selection_workers,
    training_device,
)

GiB = 1024 ** 3
CARD = torch.device("cuda", 0)  # a device OBJECT needs no card; nothing here allocates on it


# ── the decision ─────────────────────────────────────────────────────────────────────────────


def test_a_model_on_the_cpu_trains_on_the_cpu():
    device, why = training_device(None, GiB, 20 * GiB)
    assert device.type == "cpu"
    assert "not on a CUDA device" in why


def test_unreadable_free_memory_trains_on_the_cpu():
    device, why = training_device(CARD, GiB, None)
    assert device.type == "cpu"
    assert "could not be read" in why


def test_batches_that_fit_train_on_the_models_card():
    device, why = training_device(CARD, 3 * GiB, 20 * GiB)
    assert device == CARD
    assert "fit on the model's card" in why


def test_batches_that_do_not_fit_train_on_the_cpu_and_say_the_numbers():
    device, why = training_device(CARD, 16 * GiB, 10 * GiB)
    assert device.type == "cpu"
    assert "10.00 GiB free" in why


def test_the_headroom_is_part_of_the_fit():
    # 8 GiB of batches + 2 GiB of quarter + the headroom = 12 GiB; 11 GiB free is too little
    # even though the batches alone would fit.
    batches = 8 * GiB
    needed = batches + batches // 4 + DEVICE_HEADROOM_BYTES
    assert training_device(CARD, batches, needed - 1)[0].type == "cpu"
    assert training_device(CARD, batches, needed)[0] == CARD


# ── layer selection ──────────────────────────────────────────────────────────────────────────


def _pooled(rng, n=240, d=24, layers=6):
    labels = (rng.random(n) < 0.5).astype(int)
    direction = rng.standard_normal(d)

    def matrix(scale):
        x = rng.standard_normal((n, d)).astype(np.float32)
        return x + np.outer(labels - 0.5, direction).astype(np.float32) * scale

    pooled = types.SimpleNamespace(
        mean={layer: matrix(0.2 + 0.15 * layer) for layer in range(layers)},
        last={layer: matrix(0.1) for layer in range(layers)},
    )
    order = rng.permutation(n)
    return pooled, labels, order[60:], order[:60]


def test_layer_selection_parallel_is_identical_to_sequential():
    pooled, labels, train, val = _pooled(np.random.default_rng(7))
    sequential = select_layers(pooled, labels, train, val, top_n=2, workers=1)
    parallel = select_layers(pooled, labels, train, val, top_n=2, workers=4)
    assert [(g.pooling, g.layer, g.val_auroc) for g in parallel.grid] == [
        (g.pooling, g.layer, g.val_auroc) for g in sequential.grid
    ]
    assert parallel.chosen == sequential.chosen
    assert parallel.margin == sequential.margin


def test_layer_selection_defaults_to_one_worker_per_cell_up_to_the_cores(monkeypatch):
    asked = []
    real = trainer.selection_workers

    def spy(cells):
        asked.append(cells)
        return real(cells)

    monkeypatch.setattr(trainer, "selection_workers", spy)
    pooled, labels, train, val = _pooled(np.random.default_rng(3))
    select_layers(pooled, labels, train, val)
    assert asked == [12]  # 6 layers x 2 poolings
    assert selection_workers(12) == min(12, len(__import__("os").sched_getaffinity(0)))
    assert selection_workers(12) > 1 or len(__import__("os").sched_getaffinity(0)) == 1


# ── the wiring: the run's decision reaches every train_rule call ───────────────────────────────

pytest.importorskip("datasets")

from tests.unit.test_probe_monitor_pipeline import (  # noqa: E402,F401 - fixtures by import
    _loader_that_records_its_placement,
    arrow_dataset,
    monkeypatch_module,
    pipeline_db,
    tiny_lm,
    tiny_tokenizer,
)


@pytest.fixture(scope="module")
def device_run(pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer, tmp_path_factory,
               monkeypatch_module):
    from src.core import config as config_module
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun
    from src.services import probe_monitor_run as run_module

    model_row = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny-random", status=ModelStatus.READY,
                      file_path=str(arrow_dataset), architecture="LlamaForCausalLM",
                      params_count=1_000)
    dataset_row = Dataset(id=uuid.uuid4(), name="probe-corpus", source="local",
                          status=DatasetStatus.READY, raw_path=str(arrow_dataset), extra_metadata={})
    pipeline_db.add_all([model_row, dataset_row])
    pipeline_db.commit()
    mapping = {"high": "positive", "low": "negative"}
    train = ProbeMonitorDataset(name="train", dataset_id=dataset_row.id, input_column="inputs",
                                label_column="stakes", label_mapping=mapping, pair_column="pair",
                                role="train", counts={})
    held = ProbeMonitorDataset(name="unseen", dataset_id=dataset_row.id, input_column="inputs",
                               label_column="stakes", label_mapping=mapping, role="eval",
                               distribution="out_of_distribution", counts={})
    pipeline_db.add_all([train, held])
    pipeline_db.commit()
    run = ProbeMonitorRun(
        model_id=model_row.id, train_dataset_id=train.id, eval_dataset_ids=[held.id],
        config={"layers": [1, 2], "rules": ["mean", "max"], "scope": "all", "max_length": 64,
                "val_fraction": 0.25, "seed": 1337, "top_n_layers": 2, "target_fpr": 0.1},
        status="running", environment={},
    )
    pipeline_db.add(run)
    pipeline_db.commit()

    monkeypatch_module.setattr(config_module.settings, "data_dir",
                               tmp_path_factory.mktemp("device_artifacts"), raising=False)
    decisions, calls = [], []
    real_decide, real_train = trainer.training_device, trainer.train_rule

    def decide(model_device, batch_bytes, free_bytes, **kwargs):
        device, why = real_decide(model_device, batch_bytes, free_bytes, **kwargs)
        device = torch.device(device.type)  # a fresh object, so identity names THIS decision
        decisions.append({"model_device": model_device, "batch_bytes": batch_bytes,
                          "free_bytes": free_bytes, "device": device})
        return device, why

    def train(*args, **kwargs):
        calls.append(kwargs.get("device"))
        return real_train(*args, **kwargs)

    kernels = []
    real_kernels = trainer.deterministic_kernels

    def kernels_spy(device):
        kernels.append(device)
        return real_kernels(device)

    monkeypatch_module.setattr(trainer, "training_device", decide)
    monkeypatch_module.setattr(trainer, "train_rule", train)
    monkeypatch_module.setattr(trainer, "deterministic_kernels", kernels_spy)
    run_module.execute_probe_run(
        pipeline_db, run.id,
        model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer),
    )
    pipeline_db.refresh(run)
    return run, decisions, calls, kernels


def test_every_train_rule_call_gets_the_layers_decided_device(device_run):
    run, decisions, calls, kernels = device_run
    assert run.status == "completed"
    assert kernels == calls  # every fit runs inside the deterministic-kernel context for its device
    assert len(decisions) == 2  # one decision per chosen layer
    assert len(calls) == 4      # 2 layers x 2 rules
    for layer_index, decision in enumerate(decisions):
        for call in calls[2 * layer_index: 2 * layer_index + 2]:
            assert call is decision["device"]


def test_the_decision_is_asked_about_the_real_batches(device_run):
    _run, decisions, _calls, _kernels = device_run
    for decision in decisions:
        assert decision["model_device"] is None  # the tiny model is on the CPU
        assert decision["free_bytes"] is None
        assert decision["batch_bytes"] > 0


def test_the_run_records_where_each_layer_trained_and_why(device_run):
    run, decisions, _calls, _kernels = device_run
    recorded = run.environment["training_device"]
    assert len(recorded) == len(decisions)
    for entry, decision in zip(recorded.values(), decisions):
        assert entry["device"] == "cpu"
        assert entry["reason"] == "the model is not on a CUDA device"
        assert entry["batch_bytes"] == decision["batch_bytes"]


# ── attention starts as mean pooling, and a card fit is deterministic ───────────────────────────


def test_the_attention_query_is_a_seeded_cpu_draw_that_leaves_global_state_alone():
    torch.manual_seed(7)
    before = torch.get_rng_state()
    query = trainer.attention_query_init(16, torch.device("cpu"), 1337)
    assert torch.equal(torch.get_rng_state(), before)  # its own generator, not the global stream
    torch.manual_seed(1337)
    assert torch.equal(query, torch.randn(16) * 0.02)  # the draw the CPU-trained probes started from
    assert not torch.equal(query, trainer.attention_query_init(16, torch.device("cpu"), 1338))


def test_the_attention_query_is_not_zero():
    # Zero was tried on 2026-10-22 and attention stopped at epoch 1 on real data.
    assert trainer.attention_query_init(16, torch.device("cpu"), 1337).abs().sum() > 0


def test_train_rule_starts_attention_from_that_query_on_its_device(monkeypatch):
    calls = []
    real = trainer.attention_query_init

    def spy(d_model, device, seed):
        calls.append((d_model, device, seed))
        return real(d_model, device, seed)

    monkeypatch.setattr(trainer, "attention_query_init", spy)
    rng = np.random.default_rng(0)
    rows = [rng.standard_normal((4, 8)).astype(np.float32) for _ in range(12)]
    labels = [i % 2 for i in range(12)]
    device = torch.device("cpu")
    trainer.train_rule("attention", rows[:8], labels[:8], rows[8:], labels[8:],
                       epochs=2, patience=1, device=device, seed=41)
    assert calls == [(8, device, 41)]


def test_deterministic_kernels_switch_on_for_a_card_and_restore():
    assert not torch.are_deterministic_algorithms_enabled()
    with trainer.deterministic_kernels(CARD):
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.are_deterministic_algorithms_enabled()


def test_deterministic_kernels_restore_after_a_failing_fit():
    with pytest.raises(RuntimeError):
        with trainer.deterministic_kernels(CARD):
            raise RuntimeError("fit failed")
    assert not torch.are_deterministic_algorithms_enabled()


def test_deterministic_kernels_leave_a_cpu_fit_alone():
    with trainer.deterministic_kernels(torch.device("cpu")):
        assert not torch.are_deterministic_algorithms_enabled()


def test_the_image_sets_the_cublas_workspace_for_deterministic_reductions():
    from pathlib import Path

    dockerfile = (Path(__file__).resolve().parents[2] / "Dockerfile").read_text()
    env_lines = [line.strip() for line in dockerfile.splitlines()]
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in env_lines
