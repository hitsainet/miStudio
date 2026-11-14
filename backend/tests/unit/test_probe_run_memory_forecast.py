"""A probe run says it cannot finish before it spends GPU hours (ENH-006 D116 G10).

`_standardised_batches` refused at the TRAINING stage, after pooled and token capture had already run,
against a fixed 48 GiB. Everything the refusal needs is known at the rendering stage, and a row's token
count needs no GPU at all. So:

* `forecast_probe_training_run` / `POST /probe-monitors/runs/forecast`: tokenise on the CPU, name the figure;
* `start_probe_training_run`: the same forecast on its first call, a refusal or an answer carrying it;
* `execute_probe_run`: the same arithmetic on its own rendered rows, before pooled capture;
* the ceiling is what the host has at the time, not a constant.

The pipeline test runs the REAL `execute_probe_run` and holds the forecast to what the run then measured.

Controls are recorded in `0xcc/reviews/mcp_gaps_controls_2026-11-02.md`.
"""
from __future__ import annotations

import asyncio
import gc
import json
import uuid
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import torch
from mcp import Client
from sqlalchemy import func, select

from src.mcp_server.client import BackendError
from src.mcp_server.config import MCPSettings
from src.mcp_server.server import build_server
from src.models.dataset import Dataset, DatasetStatus
from src.models.model import Model, ModelStatus
from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun
from src.services import probe_monitor_trainer as trainer
from src.services import probe_run_memory as memory
from src.services.probe_monitor_trainer import (
    DEFAULT_BATCH_BYTES,
    ProbeTrainingTooLarge,
    _standardisation,
    _standardised_batches,
    forecast_batches,
    host_ram_usable_bytes,
)
from src.services.resource_config import ResourceConfig
from tests.unit.test_probe_dataset_manifest import arrow_dict, dataset_id  # noqa: F401 - fixtures
from tests.unit.test_probe_monitor_pipeline import (  # noqa: F401 - fixtures by import
    _loader_that_records_its_placement,
    arrow_dataset,
    monkeypatch_module,
    pipeline_db,
    tiny_lm,
    tiny_tokenizer,
)

GIB = 1024 ** 3
FORECAST = "/probe-monitors/runs/forecast"


def plan(**overrides):
    arguments = dict(train_widths=[10] * 8, val_widths=[10] * 2, d_model=16, swept_layers=3, top_n_layers=1,
                     storage_bytes=2, based_on="a test", available_bytes=100 * GIB)
    return memory.forecast_memory(**{**arguments, **overrides})


class TestTheArithmetic:
    @pytest.mark.parametrize("storage_bytes,room_mib,held", [(4, None, 4), (2, None, 4), (2, 110, 2), (4, 110, 4)])
    def test_batches_are_train_and_validation_together_at_the_held_width(self, storage_bytes, room_mib, held):
        """The refusal it replaces read each set alone, so a plan whose sum was over the ceiling could pass.
        Counted at the HELD width (ENH-006 G25): float32 whenever it fits, a 16-bit model's float16 only when
        float32 does not. 9,000 slots x 4,096 dims are 144 MiB at 4 B and 72 MiB at 2 B; a room of 110 MiB fits
        only the second. A float32 model has no narrower width."""
        allowances = memory.RAM_TRANSIENT_BYTES + memory.RAM_WORKER_BASELINE_BYTES
        available = (100 * GIB if room_mib is None else
                     int((allowances + room_mib * 2 ** 20) / (1 - ResourceConfig.RAM_SAFETY_MARGIN)) + 1)
        answer = plan(train_widths=[100] * 80, val_widths=[50] * 20, d_model=4096, swept_layers=1,
                      storage_bytes=storage_bytes, available_bytes=available)
        expected = (forecast_batches([100] * 80, 4096, element_bytes=held)["bytes"]
                    + forecast_batches([50] * 20, 4096, element_bytes=held)["bytes"])
        assert expected == (100 * 80 + 50 * 20) * 4096 * held
        assert answer["batches_bytes"] == expected and answer["batches_element_bytes"] == held
        assert answer["float32_batches_bytes"] == (100 * 80 + 50 * 20) * 4096 * 4
        assert answer["padded_slots"] == 100 * 80 + 50 * 20
        assert f"x 4096 dims x {held} B" in answer["arithmetic"]
        assert answer["fits"] is (held == 2 or room_mib is None)

    def test_pooled_is_rows_by_width_by_two_poolings_by_layers_swept(self):
        answer = plan(train_widths=[1] * 8, val_widths=[1] * 2, d_model=100, swept_layers=7)
        assert answer["pooled_bytes"] == 10 * 100 * 4 * 2 * 7

    def test_needed_takes_the_LARGER_of_pooled_and_batches_not_the_sum(self):
        """`execute_probe_run` drops the pooled vectors once the layers are chosen."""
        allowances = memory.RAM_TRANSIENT_BYTES + memory.RAM_WORKER_BASELINE_BYTES
        pooled_wins = plan(train_widths=[1] * 80, val_widths=[1] * 20, d_model=1024, swept_layers=50)
        assert pooled_wins["pooled_bytes"] > pooled_wins["batches_bytes"]
        assert pooled_wins["needed_bytes"] == pooled_wins["pooled_bytes"] + allowances
        batches_win = plan(train_widths=[900] * 80, val_widths=[900] * 20, d_model=1024, swept_layers=1)
        assert batches_win["batches_bytes"] > batches_win["pooled_bytes"]
        assert batches_win["needed_bytes"] == batches_win["batches_bytes"] + allowances

    def test_batches_on_the_card_are_not_in_host_memory(self):
        off = plan(train_widths=[900] * 80, val_widths=[900] * 20, d_model=1024, swept_layers=1)
        on = plan(train_widths=[900] * 80, val_widths=[900] * 20, d_model=1024, swept_layers=1,
                  card=torch.device("cuda", 0), card_free=64 * GIB)
        assert on["batches_on_card"] is True and off["batches_on_card"] is False
        assert on["needed_bytes"] == on["pooled_bytes"] + memory.RAM_TRANSIENT_BYTES + memory.RAM_WORKER_BASELINE_BYTES
        assert on["needed_bytes"] < off["needed_bytes"]

    def test_more_layers_cost_disk_not_required_ram(self):
        """One layer's batches are held at a time, and a token file is file-backed."""
        answer = plan(train_widths=[100] * 80, val_widths=[100] * 20, d_model=4096, storage_bytes=2, top_n_layers=5)
        file_bytes = 100 * 100 * 4096 * 2
        assert answer["token_file_per_layer_bytes"] == file_bytes
        assert sorted(answer["by_top_n"]) == ["1", "2", "3", "5"]
        for n, row in answer["by_top_n"].items():
            assert row["needed_gib"] == answer["needed_gib"]
            assert row["token_files_gib"] == round(int(n) * file_bytes / GIB, 2)
            assert row["resident_if_nothing_is_evicted_gib"] == round(
                (answer["needed_bytes"] + int(n) * file_bytes) / GIB, 2)

    def test_the_verdict_is_exactly_needed_against_usable(self):
        needed = plan()["needed_bytes"]
        exactly = int(needed / (1 - ResourceConfig.RAM_SAFETY_MARGIN)) + 8
        assert plan(available_bytes=exactly)["fits"] is True
        assert plan(available_bytes=int(needed * 0.9))["fits"] is False

    def test_a_refusal_names_the_figure_what_it_rests_on_and_what_to_change(self):
        answer = plan(available_bytes=GIB, based_on="every one of 10 rows")
        assert answer["fits"] is False
        for fragment in (f"{answer['needed_gib']} GiB", f"{answer['usable_gib']} GiB", "every one of 10 rows",
                         "max_length", "provision"):
            assert fragment in answer["message"], fragment

    def test_other_jobs_reserves_come_off_usable(self):
        answer = plan(available_bytes=100 * GIB, reserved_bytes=24 * GIB)
        assert answer["usable_bytes"] == host_ram_usable_bytes(100 * GIB, 24 * GIB)
        assert answer["usable_bytes"] == int(100 * GIB * (1 - ResourceConfig.RAM_SAFETY_MARGIN)) - 24 * GIB


class TestTheWorkerItRunsInIsCredited:
    """ENH-006 G14. A run executes in a GPU worker and reuses the heap it already holds; MemAvailable
    excludes that heap, so the run needs only `needed - held` more from the host."""

    def test_held_memory_comes_off_what_the_run_needs(self):
        refused = plan(available_bytes=GIB)
        assert refused["fits"] is False
        held = refused["needed_bytes"] - refused["usable_bytes"]
        answer = plan(available_bytes=GIB, held_bytes=held)
        assert answer["fits"] is True
        assert answer["additional_bytes"] == refused["needed_bytes"] - held == answer["usable_bytes"]
        assert answer["worker_held_bytes"] == held
        assert plan(available_bytes=GIB, held_bytes=held - 1)["fits"] is False

    def test_a_worker_holding_more_than_the_run_needs_needs_nothing_more(self):
        answer = plan(available_bytes=0, held_bytes=10**12)
        assert answer["additional_bytes"] == 0 and answer["fits"] is True

    @pytest.mark.parametrize("held", [None, -5 * GIB, 0])
    def test_nothing_or_nonsense_held_credits_nothing(self, held):
        answer = plan(held_bytes=held)
        assert answer["worker_held_bytes"] == 0
        assert answer["additional_bytes"] == answer["needed_bytes"]

    def test_the_message_says_what_the_worker_holds_and_what_more_it_needs(self):
        answer = plan(held_bytes=3 * GIB)
        assert "already holds 3.0 GiB" in answer["arithmetic"]
        assert f"needs {answer['additional_gib']} GiB more" in answer["arithmetic"]
        assert f"{answer['additional_gib']} GiB beyond what its worker already holds" in answer["message"]


class TestTheCeilingIsWhatTheHostHas:
    def test_it_is_the_guards_own_rule(self):
        """`gpu_job_claim.host_ram_refusal`: what is available, less the margin and others' reserves."""
        assert host_ram_usable_bytes(80 * GIB, 0) == 60 * GIB
        assert host_ram_usable_bytes(80 * GIB, 10 * GIB) == 50 * GIB

    def test_it_reads_the_host_when_not_told(self, monkeypatch):
        monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=40 * GIB))
        assert host_ram_usable_bytes() == 30 * GIB

    def _rows(self):
        return [np.ones((4, 8), dtype=np.float32) for _ in range(4)]

    def test_the_build_refuses_at_what_is_free_NOW(self, monkeypatch):
        rows = self._rows()
        mean, std = _standardisation(rows)
        monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=1))
        with pytest.raises(ProbeTrainingTooLarge) as caught:
            _standardised_batches(rows, mean, std, torch.device("cpu"))
        assert "GiB" in str(caught.value) and "padded slots" in str(caught.value)

    def test_the_build_goes_ahead_when_it_is_free(self, monkeypatch):
        rows = self._rows()
        mean, std = _standardisation(rows)
        monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=GIB))
        assert _standardised_batches(rows, mean, std, torch.device("cpu"))

    def test_batches_bound_for_a_card_are_not_held_to_host_memory(self, monkeypatch):
        """They are not going to live there, and `training_device` has already judged the card."""
        monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=1))
        assert trainer._ram_ceiling(torch.device("cuda", 0), None) is None
        assert trainer._ram_ceiling(torch.device("cpu"), None) == host_ram_usable_bytes(1)
        assert trainer._ram_ceiling(torch.device("cuda", 0), 7) == 7


class TestTheRowsAreViewsOfTheFile:
    """Why a token file is not counted as required RAM: the run's rows must stay views of the memmap."""

    def test_the_runs_own_expression_copies_nothing(self, tmp_path):
        path = tmp_path / "tokens.f16"
        written = np.memmap(path, dtype=np.float16, mode="w+", shape=(100, 8))
        written.flush()
        del written
        mapped = np.memmap(path, dtype=np.float16, mode="r", shape=(100, 8))
        row = np.asarray(mapped[10:20])
        assert np.shares_memory(row, mapped)


class TestAggregateRowsIsBounded:
    """It packed every validation negative to the longest one in a single tensor, the same shape as the
    154 GB allocation `plan_length_buckets` exists to end, on the path with no calibration set."""

    def _head(self, d):
        return SimpleNamespace(token_scores=lambda x: x.sum(-1), attention_logits=lambda x: x.sum(-1))

    def test_no_tensor_exceeds_the_batch_budget_and_every_row_keeps_its_score(self, monkeypatch):
        from src.services import probe_monitor_run as run_module

        d_model = 512
        widths = [2] * 2000 + [2000]
        rows = [np.full((w, d_model), float(i % 5), dtype=np.float32) for i, w in enumerate(widths)]
        real_zeros = torch.zeros
        biggest = []

        def spy(shape, **kwargs):
            if len(shape) == 3:
                biggest.append(shape[0] * shape[1] * shape[2] * 4)
            return real_zeros(shape, **kwargs)

        monkeypatch.setattr(torch, "zeros", spy)
        scores = run_module._aggregate_rows(self._head(d_model), "mean", {}, rows)
        monkeypatch.setattr(torch, "zeros", real_zeros)

        assert biggest and max(biggest) <= max(DEFAULT_BATCH_BYTES, 2000 * d_model * 4)
        assert len(biggest) > 1
        assert scores == pytest.approx([float(i % 5) * d_model for i in range(len(widths))])


# ── the run itself ────────────────────────────────────────────────────────────────────────────


def _run_row(db, arrow_dataset, config):
    model_row = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny-random", status=ModelStatus.READY,
                      file_path=str(arrow_dataset), architecture="LlamaForCausalLM", params_count=1_000)
    dataset_row = Dataset(id=uuid.uuid4(), name="probe-corpus", source="local", status=DatasetStatus.READY,
                          raw_path=str(arrow_dataset), extra_metadata={})
    db.add_all([model_row, dataset_row])
    db.commit()
    train = ProbeMonitorDataset(name="train", dataset_id=dataset_row.id, input_column="inputs",
                                label_column="stakes", label_mapping={"high": "positive", "low": "negative"},
                                pair_column="pair", role="train", counts={})
    db.add(train)
    db.commit()
    run = ProbeMonitorRun(model_id=model_row.id, train_dataset_id=train.id, eval_dataset_ids=[], config=config,
                          status="running", environment={})
    db.add(run)
    db.commit()
    return run


CONFIG = {"layers": [1, 2], "rules": ["mean"], "scope": "all", "max_length": 64, "val_fraction": 0.25,
          "seed": 1337, "top_n_layers": 2, "target_fpr": 0.1}


@pytest.fixture(scope="module")
def measured(pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer, tmp_path_factory, monkeypatch_module):
    """The REAL `execute_probe_run`, with a spy on whether the pooled vectors are gone by token capture."""
    from src.core import config as config_module
    from src.services import probe_monitor_run as run_module
    from src.services import probe_monitor_capture as capture_module

    monkeypatch_module.setattr(config_module.settings, "data_dir", tmp_path_factory.mktemp("memory"), raising=False)
    refs, alive_at_capture, captures = [], [], []
    real_pooled, real_tokens = capture_module.capture_pooled, capture_module.capture_tokens

    def pooled_spy(*args, **kwargs):
        result = real_pooled(*args, **kwargs)
        refs.append(weakref.ref(result))
        return result

    def tokens_spy(*args, **kwargs):
        gc.collect()
        alive_at_capture.append(any(ref() is not None for ref in refs))
        capture = real_tokens(*args, **kwargs)
        captures.append(capture)
        return capture

    monkeypatch_module.setattr(capture_module, "capture_pooled", pooled_spy)
    monkeypatch_module.setattr(capture_module, "capture_tokens", tokens_spy)
    run = _run_row(pipeline_db, arrow_dataset, CONFIG)
    run_module.execute_probe_run(pipeline_db, run.id,
                                 model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
    pipeline_db.refresh(run)
    monkeypatch_module.setattr(capture_module, "capture_pooled", real_pooled)
    monkeypatch_module.setattr(capture_module, "capture_tokens", real_tokens)
    return run, alive_at_capture, captures


class TestTheRunChecksItself:
    def test_the_forecast_is_recorded_and_it_is_what_the_run_then_measured(self, measured):
        """The forecast and the training stage must agree to the byte, or the forecast is a guess."""
        run, _alive, captures = measured
        assert run.status == "completed"
        forecast = run.environment["memory_forecast"]
        assert forecast["checked"] is True and forecast["fits"] is True
        assert forecast["real_tokens"] == int(captures[0].offsets[-1])
        for entry in run.environment["training_device"].values():
            assert entry["batch_bytes"] == forecast["batches_bytes"]
        assert forecast["d_model"] == captures[0].d_model
        assert forecast["swept_layers"] == 2 and forecast["top_n_layers"] == 2

    def test_the_token_file_figure_is_the_file_on_disk(self, measured):
        run, _alive, captures = measured
        forecast = run.environment["memory_forecast"]
        for capture in captures:
            assert capture.path.stat().st_size == forecast["token_file_per_layer_bytes"]

    def test_the_pooled_vectors_are_gone_before_token_capture(self, measured):
        """They sat in RAM to the end of the run: 0.8 GiB per swept layer on a 26,000-row Llama run."""
        _run, alive, _captures = measured
        assert alive and not any(alive)

    def test_a_run_that_cannot_finish_stops_BEFORE_pooled_capture(
        self, pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer, tmp_path, monkeypatch
    ):
        from src.core import config as config_module
        from src.services import probe_monitor_capture as capture_module
        from src.services import probe_monitor_run as run_module

        monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
        monkeypatch.setattr(memory, "host_ram_usable_bytes", lambda *a, **k: 0)
        captured = []
        monkeypatch.setattr(capture_module, "capture_pooled", lambda *a, **k: captured.append(1))
        run = _run_row(pipeline_db, arrow_dataset, CONFIG)
        with pytest.raises(ProbeTrainingTooLarge) as caught:
            run_module.execute_probe_run(pipeline_db, run.id,
                                         model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
        assert captured == [], "the capture ran before the refusal: the hours were spent"
        assert "host memory" in str(caught.value) and "every one of" in str(caught.value)
        pipeline_db.refresh(run)
        assert run.environment["memory_forecast"]["fits"] is False

    def test_a_forecast_that_cannot_run_says_so_and_does_not_fail_the_run(self):
        answer = memory.forecast_for_run(None, SimpleNamespace(config=SimpleNamespace()), [], [], [],
                                         scope="all", swept_layers=1, top_n_layers=1, storage_bytes=2)
        assert answer == {"checked": False, "fits": None,
                          "reason": "the loaded model's config states no hidden_size"}
        broken = memory.forecast_for_run(None, SimpleNamespace(config=SimpleNamespace(hidden_size=8)), None, [], [],
                                         scope="all", swept_layers=1, top_n_layers=1, storage_bytes=2)
        assert broken["checked"] is False and broken["fits"] is None and "raised" in broken["reason"]


class TestTheBatchesAreAHostReserve:
    """Between the forecast and the allocation are hours of capture, when another job's guard could not see them."""

    def test_the_run_states_its_figure_then_says_it_has_allocated(
        self, pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer, tmp_path, monkeypatch  # noqa: F811
    ):
        from src.core import config as config_module
        from src.services import gpu_job_claim
        from src.services import probe_monitor_run as run_module

        monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
        calls = []
        monkeypatch.setattr(gpu_job_claim, "declare_host_reserve", lambda mb: calls.append(("declare", mb)))
        monkeypatch.setattr(gpu_job_claim, "host_reserve_allocated", lambda: calls.append(("allocated", None)))
        run = _run_row(pipeline_db, arrow_dataset, CONFIG)
        run_module.execute_probe_run(pipeline_db, run.id,
                                     model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
        pipeline_db.refresh(run)
        mb = run.environment["memory_forecast"]["batches_bytes"] // 2 ** 20
        # Once at the forecast, then a declare and an allocated around each of the two layers' builds.
        assert calls == [("declare", mb), ("declare", mb), ("allocated", None), ("declare", mb), ("allocated", None)]

    def test_it_reaches_the_lease_rows_inside_a_claim(self, monkeypatch):
        from src.services import gpu_job_claim

        written = []
        monkeypatch.setattr(gpu_job_claim.gpu_leases, "set_host_reserve", lambda db, holder, mb: written.append((holder, mb)))
        import contextlib

        @contextlib.contextmanager
        def session():
            yield None

        ctx = SimpleNamespace(held=("gpu-a",), holder="probe:1", session=session, host_reserve_mb=0)
        monkeypatch.setattr(gpu_job_claim, "current_claim", lambda: ctx)
        gpu_job_claim.declare_host_reserve(2048)
        assert written == [("probe:1", 2048)] and ctx.host_reserve_mb == 2048

    def test_outside_a_claim_it_is_a_no_op(self, monkeypatch):
        from src.services import gpu_job_claim

        monkeypatch.setattr(gpu_job_claim, "current_claim", lambda: None)
        monkeypatch.setattr(gpu_job_claim.gpu_leases, "set_host_reserve", MagicMock(side_effect=AssertionError))
        gpu_job_claim.declare_host_reserve(1)


class TestWhatTheCheckIsHanded:
    """Each of these survived a control: the forecast's own arithmetic was pinned and the lines CALLING it
    with the host's state were not, the shape every survivor in this estate has had."""

    def _rendered_as(self, monkeypatch, offsets):
        monkeypatch.setattr("src.services.probe_monitor_capture.count_scored_tokens",
                            lambda rendered, scope: (offsets[-1], np.asarray(offsets)))

    def _model(self):
        return SimpleNamespace(config=SimpleNamespace(hidden_size=8))

    def _forecast(self):
        return memory.forecast_for_run(None, self._model(), [], [0], [1], scope="all", swept_layers=1,
                                       top_n_layers=1, storage_bytes=2)

    def test_the_runs_check_subtracts_other_jobs_reserves(self, monkeypatch):
        self._rendered_as(monkeypatch, [0, 4, 8])
        monkeypatch.setattr(memory, "_card_state", lambda model: (None, None))
        monkeypatch.setattr(memory, "other_reserves_bytes", lambda db: 3 * GIB)
        assert self._forecast()["reserved_gib"] == 3.0

    def test_the_runs_check_credits_the_worker_it_runs_in(self, monkeypatch):
        from src.services import gpu_worker_memory

        self._rendered_as(monkeypatch, [0, 4, 8])
        monkeypatch.setattr(memory, "_card_state", lambda model: (None, None))
        monkeypatch.setattr(memory, "other_reserves_bytes", lambda db: 0)
        asked = []
        monkeypatch.setattr(gpu_worker_memory, "anon_bytes", lambda pid=None: asked.append(pid) or 7 * GIB)
        answer = self._forecast()
        assert asked == [None], "the run's check must read its OWN process"
        assert answer["worker_held_gib"] == 7.0

    def test_the_runs_check_hands_the_card_state_to_the_decision(self, monkeypatch):
        self._rendered_as(monkeypatch, [0, 4, 8])
        monkeypatch.setattr(memory, "other_reserves_bytes", lambda db: 0)
        card = torch.device("cuda", 0)
        for state, on_card in (((card, 64 * GIB), True), ((card, 0), False), ((None, None), False)):
            monkeypatch.setattr(memory, "_card_state", lambda model, _s=state: _s)
            assert self._forecast()["batches_on_card"] is on_card

    def test_the_card_state_is_the_models_device_and_its_free_memory(self, monkeypatch):
        card = torch.device("cuda", 0)
        monkeypatch.setattr("src.services.gpu_stage_memory.cuda_device_of", lambda model: card)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (9 * GIB, 24 * GIB))
        assert memory._card_state(self._model()) == (card, 9 * GIB)
        monkeypatch.setattr("src.services.gpu_stage_memory.cuda_device_of", lambda model: None)
        assert memory._card_state(self._model()) == (None, None)

    @pytest.mark.parametrize("free_gib,expected", [(20, True), (1, False)])
    def test_the_card_decision_is_training_devices_on_the_real_figure(self, free_gib, expected):
        # 8 GiB of batches: 2 ** 33 B / (4096 dims x 4 B) slots, or the same slots at 2 B for 4 GiB.
        width = 2 ** 33 // (4096 * 4) // 100
        answer = plan(train_widths=[width] * 80, val_widths=[width] * 20, d_model=4096, swept_layers=1,
                      storage_bytes=4, card=torch.device("cuda", 0), card_free=free_gib * GIB)
        assert answer["batches_on_card"] is expected

    @pytest.mark.parametrize("storage_bytes,expected", [(4, False), (2, True)])
    def test_the_card_decision_counts_the_held_width(self, storage_bytes, expected):
        """8 GiB of slots at 4 B needs 12 GiB with headroom; at 2 B it is 4 GiB and needs 7. 9 GiB free: a
        16-bit model's batches go on the card held raw, a float32 model's stay in host memory."""
        width = 2 ** 33 // (4096 * 4) // 100
        answer = plan(train_widths=[width] * 80, val_widths=[width] * 20, d_model=4096, swept_layers=1,
                      storage_bytes=storage_bytes, card=torch.device("cuda", 0), card_free=9 * GIB)
        assert answer["batches_on_card"] is expected
        if expected:
            assert answer["batches_element_bytes"] == 2
            assert answer["batches_bytes"] * 2 == answer["float32_batches_bytes"]

    def test_a_model_on_no_card_keeps_its_batches_in_host_memory(self):
        assert plan(card=None, card_free=None)["batches_on_card"] is False

    def test_other_reserves_add_up_the_guards_way(self, monkeypatch):
        """A holder with no recorded reserve counts its kind's, as `host_ram_refusal`'s caller does."""
        from src.services.gpu_job_claim import HOST_RESERVE_MB_BY_KIND

        monkeypatch.setattr("src.services.gpu_leases.host_reserves",
                            lambda db: {"probe:a": 100, "training:b": None, "steering:c": None, "training:d": 0})
        assert memory.other_reserves_bytes(None) == (100 + HOST_RESERVE_MB_BY_KIND["training"]) * 2 ** 20

    def _sixteen_bit_run(self, pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer, tmp_path, monkeypatch,  # noqa: F811
                         room=None):
        """The REAL run with float16 storage; `room` (a fraction of the float32 batches) squeezes both the
        forecast's room and the training stage's to it. Returns the run, the captures and every batch dtype built."""
        from src.core import config as config_module
        from src.services import gpu_worker_memory
        from src.services import probe_monitor_capture as capture_module
        from src.services import probe_monitor_run as run_module

        monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
        monkeypatch.setattr(run_module, "storage_dtype_of_load", lambda model: "float16")
        captures, dtypes = [], []
        real, real_build = capture_module.capture_tokens, trainer._standardised_batches

        def spy(*args, **kwargs):
            captures.append(real(*args, **kwargs))
            return captures[-1]

        def build_spy(*args, **kwargs):
            batches = real_build(*args, **kwargs)
            dtypes.extend(x.dtype for x, _m, _i in batches)
            return batches

        monkeypatch.setattr(capture_module, "capture_tokens", spy)
        monkeypatch.setattr(trainer, "_standardised_batches", build_spy)
        if room is not None:
            float32_bytes = room["float32_batches_bytes"]
            fraction = room["fraction"]
            allowances = memory.RAM_TRANSIENT_BYTES + memory.RAM_WORKER_BASELINE_BYTES
            monkeypatch.setattr(gpu_worker_memory, "anon_bytes", lambda pid=None: 0)
            monkeypatch.setattr(memory, "other_reserves_bytes", lambda db, exclude_holder=None: 0)
            monkeypatch.setattr(memory, "host_ram_usable_bytes",
                                lambda available=None, reserved=0: allowances + int(fraction * float32_bytes))
            monkeypatch.setattr(memory, "batch_room_now", lambda db, available_bytes=None: int(fraction * float32_bytes))
        run = _run_row(pipeline_db, arrow_dataset, CONFIG)
        run_module.execute_probe_run(pipeline_db, run.id,
                                     model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
        pipeline_db.refresh(run)
        return run, captures, dtypes

    def test_a_16_bit_run_is_held_to_its_own_token_file_and_widens_when_float32_fits(
        self, pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer, tmp_path, monkeypatch  # noqa: F811
    ):
        """The fixture model is float32, so the other run cannot tell 4 B from a real run's 2 B. ENH-006 G25: with
        room for float32 the 16-bit run widens once (the pre-G24 path), and the forecast, the per-layer record and
        the batches built all say 4 B."""
        run, captures, dtypes = self._sixteen_bit_run(pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer,
                                                      tmp_path, monkeypatch)
        forecast = run.environment["memory_forecast"]
        assert captures and forecast["token_file_per_layer_bytes"] == forecast["real_tokens"] * forecast["d_model"] * 2
        for capture in captures:
            assert capture.path.stat().st_size == forecast["token_file_per_layer_bytes"]
        assert run.status == "completed"
        assert forecast["storage_bytes"] == 2 and forecast["batches_element_bytes"] == 4
        assert forecast["batches_bytes"] == forecast["padded_slots"] * forecast["d_model"] * 4
        recorded = run.environment["training_device"]
        assert recorded and all(entry["batch_bytes"] == forecast["batches_bytes"] and entry["element_bytes"] == 4
                                for entry in recorded.values())
        assert dtypes and set(dtypes) == {torch.float32}

    def test_a_16_bit_run_holds_raw_when_float32_does_not_fit(
        self, pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer, tmp_path, monkeypatch  # noqa: F811
    ):
        """Room for three quarters of the float32 batches: float16 (half) fits, float32 does not. The forecast,
        the reservation, the per-layer record and the batches built must all be the 2 B width."""
        from src.services import gpu_job_claim

        first, _c, _d = self._sixteen_bit_run(pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer, tmp_path,
                                              monkeypatch)
        sized = first.environment["memory_forecast"]
        assert sized["pooled_bytes"] <= 0.75 * sized["float32_batches_bytes"], "the fixture cannot make the case"
        declared = []
        monkeypatch.setattr(gpu_job_claim, "declare_host_reserve", lambda mb: declared.append(mb))
        run, _captures, dtypes = self._sixteen_bit_run(
            pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer, tmp_path, monkeypatch,
            room={"float32_batches_bytes": sized["float32_batches_bytes"], "fraction": 0.75},
        )
        forecast = run.environment["memory_forecast"]
        assert run.status == "completed" and forecast["fits"] is True
        assert forecast["batches_element_bytes"] == 2
        assert forecast["batches_bytes"] == forecast["padded_slots"] * forecast["d_model"] * 2
        recorded = run.environment["training_device"]
        assert recorded and all(entry["batch_bytes"] == forecast["batches_bytes"] and entry["element_bytes"] == 2
                                for entry in recorded.values())
        assert declared == [forecast["batches_bytes"] // 2 ** 20] * (1 + len(recorded))
        assert dtypes and set(dtypes) == {torch.float16}


# ── the preflight route, for real ─────────────────────────────────────────────────────────────


@pytest.fixture
async def preflight_rows(async_session, dataset_id, tiny_tokenizer, monkeypatch):  # noqa: F811
    from src.services import probe_monitor_run as run_module

    monkeypatch.setattr(run_module, "_load_tokenizer_for_run", lambda row: tiny_tokenizer)
    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY, file_path="/nowhere",
                  architecture="LlamaForCausalLM", params_count=1_000,
                  architecture_config={"hidden_size": 128, "num_hidden_layers": 11})
    view = ProbeMonitorDataset(name="train", dataset_id=uuid.UUID(dataset_id), split="test", input_column="text",
                               label_column="label", label_mapping={"yes": "positive", "no": "negative"},
                               role="train", counts={})
    async_session.add_all([model, view])
    await async_session.commit()
    return model.id, view.id


def _expected_tokens(tokenizer, arrow_path, split="test"):
    from datasets import load_from_disk

    total = 0
    for text in load_from_disk(str(arrow_path))[split]["text"]:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                 add_generation_prompt=True)
        total += len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    return total


class TestThePreflightRoute:
    @pytest.mark.asyncio
    async def test_it_tokenises_every_row_and_creates_nothing(
        self, client, async_session, preflight_rows, tiny_tokenizer, arrow_dict  # noqa: F811
    ):
        model_id, view_id = preflight_rows
        with patch("src.api.v1.endpoints.probe_monitors.gpu_delay") as dispatch:
            response = await client.post(f"/api/v1{FORECAST}", json={
                "model_id": model_id, "train_dataset_id": view_id,
                "config": {"scope": "all", "max_length": 64, "stride": 5, "top_n_layers": 2}})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["checked"] is True, body
        assert body["real_tokens"] == _expected_tokens(tiny_tokenizer, arrow_dict)
        assert body["rows"] == 80
        assert body["d_model"] == 128
        # layers 10, 5, 0 on a depth of 11 at stride 5: read from the row's config, not assumed.
        assert body["swept_layers"] == 3
        assert body["top_n_layers"] == 2
        assert "every one of 80 rows" in body["based_on"] and "CPU" in body["based_on"]
        dispatch.assert_not_called()
        assert (await async_session.execute(select(func.count()).select_from(ProbeMonitorRun))).scalar_one() == 0

    @pytest.mark.asyncio
    async def test_it_subtracts_other_jobs_reserves(self, client, preflight_rows, monkeypatch):
        monkeypatch.setattr(memory, "other_reserves_bytes", lambda db: 5 * GIB)
        model_id, view_id = preflight_rows
        body = (await client.post(f"/api/v1{FORECAST}", json={"model_id": model_id, "train_dataset_id": view_id})).json()
        assert body["checked"] is True and body["reserved_gib"] == 5.0

    @pytest.mark.asyncio
    async def test_it_credits_the_worker_the_supervisor_published(self, client, preflight_rows, monkeypatch, tmp_path):
        import json as _json
        import time as _time

        from src.services import gpu_worker_memory as W

        now = _time.time()
        for card, gib in (("GPU-aaa", 9), ("GPU-bbb", 3)):
            W.entry_path(tmp_path, card).write_text(_json.dumps({"card": card, "pid": 1, "anon_bytes": gib * GIB, "at": now}))
        monkeypatch.setenv(W.MEMORY_DIR_ENV, str(tmp_path))
        model_id, view_id = preflight_rows
        named = (await client.post(f"/api/v1{FORECAST}", json={
            "model_id": model_id, "train_dataset_id": view_id, "gpu": "GPU-aaa"})).json()
        auto = (await client.post(f"/api/v1{FORECAST}", json={"model_id": model_id, "train_dataset_id": view_id})).json()
        assert named["checked"] is True and named["worker_held_gib"] == 9.0
        assert auto["worker_held_gib"] == 3.0, "a job any card may take is credited the least any worker holds"

    @pytest.mark.asyncio
    async def test_too_large_is_an_answer_naming_the_figure(self, client, preflight_rows, monkeypatch):
        monkeypatch.setattr(memory, "host_ram_usable_bytes", lambda *a, **k: 0)
        model_id, view_id = preflight_rows
        response = await client.post(f"/api/v1{FORECAST}", json={"model_id": model_id, "train_dataset_id": view_id})
        assert response.status_code == 200
        assert response.json()["fits"] is False and "GiB" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_a_model_with_no_recorded_width_is_unchecked_not_assumed(self, client, async_session, preflight_rows):
        model_id, view_id = preflight_rows
        (await async_session.get(Model, model_id)).architecture_config = {}
        await async_session.commit()
        body = (await client.post(f"/api/v1{FORECAST}", json={"model_id": model_id, "train_dataset_id": view_id})).json()
        assert body["checked"] is False and body["fits"] is None and "hidden_size" in body["reason"]

    @pytest.mark.asyncio
    async def test_it_refuses_what_the_run_refuses(self, client, async_session, preflight_rows):
        model_id, view_id = preflight_rows
        assert (await client.post(f"/api/v1{FORECAST}", json={"model_id": "nope", "train_dataset_id": view_id})).status_code == 404
        assert (await client.post(f"/api/v1{FORECAST}", json={"model_id": model_id, "train_dataset_id": "nope"})).status_code == 404
        (await async_session.get(ProbeMonitorDataset, view_id)).role = "eval"
        await async_session.commit()
        assert (await client.post(f"/api/v1{FORECAST}", json={"model_id": model_id, "train_dataset_id": view_id})).status_code == 422

    def test_the_route_is_in_the_served_app(self):
        from src.main import app

        assert "post" in app.openapi()["paths"][f"/api/v1{FORECAST}"]


# ── the tools ────────────────────────────────────────────────────────────────────────────────


class _Studio:
    def __init__(self, forecast=None, fail=None):
        self.forecast, self.fail = forecast, fail
        self.calls = []

    async def post(self, path, json_body=None, **params):
        self.calls.append((path, json_body, params))
        if path == FORECAST:
            if self.fail is not None:
                raise self.fail
            return self.forecast
        if path == "/mcp/approvals":
            return {"id": "apr_1", "status": "pending"}
        return {"id": "pmr_9", "task_id": "t_9", "status": "queued"}


def _start(studio, *, approval=True, **extra):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import probe_monitors

    mcp = MCPServer("t")
    probe_monitors.register(mcp, studio, MCPSettings(auth_token="x" * 32, probe_gpu_approval=approval))
    arguments = {"model_id": "m_1", "train_dataset_id": "pmd_1", **extra}
    return asyncio.run(mcp._tool_manager._tools["start_probe_training_run"].fn(**arguments))  # noqa: SLF001


FITS = {"checked": True, "fits": True, "needed_gib": 12.0, "message": "fits"}
TOO_LARGE = {"checked": True, "fits": False, "needed_gib": 99.0, "message": "this run would need 99.0 GiB"}
PAYLOAD = {"model_id": "m_1", "train_dataset_id": "pmd_1", "eval_dataset_ids": [], "gpu": "auto"}


class TestStartingARun:
    def test_too_large_refuses_before_an_approval_is_filed(self):
        studio = _Studio(TOO_LARGE)
        result = _start(studio)
        assert result["refused"] == "probe_training_too_large"
        assert result["reason"] == TOO_LARGE["message"] and result["memory_forecast"] == TOO_LARGE
        assert [c[0] for c in studio.calls] == [FORECAST]

    def test_the_forecast_goes_with_the_approval_request_for_the_exact_payload(self):
        studio = _Studio(FITS)
        result = _start(studio)
        assert result["status"] == "pending_approval" and result["memory_forecast"] == FITS
        assert [c[0] for c in studio.calls] == [FORECAST, "/mcp/approvals"]
        assert studio.calls[0][1] == PAYLOAD
        filed = studio.calls[1][1]
        # ENH-006 G15: the time forecast rides beside the payload, never in it.
        assert {k: v for k, v in filed.items() if k != "forecast"} == {
            "tool_name": "start_probe_training_run", "payload": PAYLOAD}

    def test_tokenising_gets_longer_than_the_clients_default(self):
        studio = _Studio(FITS)
        _start(studio)
        from src.mcp_server.tools.probe_monitors import FORECAST_TIMEOUT_S

        assert studio.calls[0][2] == {"timeout": FORECAST_TIMEOUT_S} and FORECAST_TIMEOUT_S > 60

    def test_the_approved_recall_does_not_tokenise_again(self):
        """The operator approved this payload, and the run checks itself when it starts."""
        studio = _Studio(TOO_LARGE)
        result = _start(studio, approval_request_id="apr_1")
        assert result["id"] == "pmr_9"
        assert FORECAST not in [c[0] for c in studio.calls]

    def test_with_approval_off_it_still_refuses(self):
        studio = _Studio(TOO_LARGE)
        assert _start(studio, approval=False)["refused"] == "probe_training_too_large"
        assert "/probe-monitors/runs" not in [c[0] for c in studio.calls]

    def test_with_approval_off_the_answer_carries_it(self):
        studio = _Studio(FITS)
        result = _start(studio, approval=False)
        assert result["id"] == "pmr_9" and result["memory_forecast"] == FITS

    def test_a_forecast_that_failed_is_said_and_does_not_stop_the_request(self):
        """The run still checks itself; the point is that nobody reads silence as "it fits"."""
        studio = _Studio(fail=BackendError(0, "unreachable"))
        result = _start(studio)
        assert result["status"] == "pending_approval"
        assert result["memory_forecast"]["checked"] is False and "unreachable" in result["memory_forecast"]["reason"]

    def test_an_unchecked_forecast_does_not_refuse(self):
        studio = _Studio({"checked": False, "fits": None, "reason": "no width"})
        assert _start(studio)["status"] == "pending_approval"


class TestTheToolIsLive:
    def _built(self, categories):
        return build_server(MCPSettings(tool_categories=categories, allow_anonymous=True, _env_file=None), stdio=True)

    def test_it_is_on_the_server_the_deployed_manifest_builds(self):
        from tests.unit.test_reachability import TestTheDEPLOYMENTEnablesWhatTheCodeRegisters as T

        mcp, _ = self._built(",".join(sorted(T()._deployed_categories())))
        assert "forecast_probe_training_run" in {t.name for t in asyncio.run(mcp.list_tools())}

    def test_the_tool_sends_its_body_once_to_the_route_with_the_long_timeout(self):
        mcp, studio = self._built("probe_monitors")
        seen = []

        def respond(request):
            seen.append((request.method, request.url.path, json.loads(request.content),
                         request.extensions["timeout"]["read"]))
            return httpx.Response(200, json=FITS)

        studio._http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://test/api/v1")  # noqa: SLF001

        async def call():
            async with Client(mcp) as agent:
                return await agent.call_tool("forecast_probe_training_run", {
                    "model_id": "m_1", "train_dataset_id": "pmd_1", "config": {"top_n_layers": 2}})

        result = asyncio.run(call())
        assert not result.is_error
        from src.mcp_server.tools.probe_monitors import FORECAST_TIMEOUT_S

        assert seen == [("POST", f"/api/v1{FORECAST}",
                         {"model_id": "m_1", "train_dataset_id": "pmd_1", "config": {"top_n_layers": 2}},
                         FORECAST_TIMEOUT_S)]

    def test_the_howto_and_the_instructions_name_it(self):
        from src.mcp_server.tools import howto

        assert "`forecast_probe_training_run`" in howto.TOPICS["behaviour_detectors"]
        mcp, _ = self._built("probe_monitors")
        assert "forecast_probe_training_run" in mcp.instructions
