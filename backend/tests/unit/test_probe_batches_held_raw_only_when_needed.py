"""A 16-bit model's batches are held raw ONLY when float32 does not fit (ENH-006 G25, the open half).

G24 held every 16-bit model's probe batches as their stored float16 and standardised each at use. It fit arm A
in memory and cost its training stage: 9,184 s against arm B's 1,848 s on the node; after `879b8dc64` the
widening still cost 2.3-6x float32's epoch on the workstation. Most runs fit at float32, so they now widen and
standardise once (the pre-G24 path), and only a run that would not fit pays the per-use cost.

One function decides (`batch_holding`), and every place that counts the batches asks it:
- `execute_probe_run`, per chosen layer: the device, the width, the record, the reservation, the build;
- `forecast_memory` (the run's own check and `forecast_probe_training_run`): the batches figure, the card, the
  verdict, and so the reservation the run declares from it;
- the time forecast (G15): the card it expects and the width its training rate is taken at;
- `train_rule` building its own batches (the SAE variant).

What this file pins, with every expected figure written out here rather than taken from the code under test:
- the rule's comparisons, at their boundaries, on the card and in host memory;
- each caller hands the rule its own figures and uses its answer (payload and count);
- the values are the same at either width: the batches against a formula written here in numpy, and the
  weights bit for bit (`torch.equal`) under every rule.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.services import probe_monitor_trainer as trainer
from src.services import probe_run_memory as memory
from src.services import probe_run_time_forecast as tf
from src.services.probe_monitor_trainer import (
    DEVICE_HEADROOM_BYTES,
    _standardisation,
    _standardised_batches,
    batch_holding,
    ready_batch,
    standardisation_terms,
    train_rule,
)
from src.services.resource_config import ResourceConfig
from tests.unit.test_probe_monitor_pipeline import (  # noqa: F401 - fixtures by import
    _loader_that_records_its_placement,
    arrow_dataset,
    monkeypatch_module,
    pipeline_db,
    tiny_lm,
    tiny_tokenizer,
)

GIB = 1024 ** 3
MIB = 1024 ** 2
CARD = torch.device("cuda", 0)
CPU = torch.device("cpu")
D_MODEL = 24
SMALL = D_MODEL * 4 * 120  # several length buckets at test scale
RULES = ["mean", "max", "last", "softmax", "attention", "rolling_mean_max"]


def card_needs(batch_bytes):
    """The card rule, written out: the batches, a quarter again, and the fixed headroom."""
    return batch_bytes + batch_bytes // 4 + DEVICE_HEADROOM_BYTES


def decide(storage, f32, *, card=None, free=None, room=None):
    return batch_holding(storage_bytes=storage, float32_bytes=f32, card=card, card_free=free, host_room=room)


# ── the rule ──────────────────────────────────────────────────────────────────────────────────


class TestTheRuleInHostMemory:
    F32 = 8 * GIB

    def test_float32_that_fits_exactly_is_widened(self):
        held = decide(2, self.F32, room=self.F32)
        assert (held.device.type, held.element_bytes, held.batch_bytes) == ("cpu", 4, self.F32)
        assert held.hold_raw is False

    def test_one_byte_short_holds_raw_at_half(self):
        held = decide(2, self.F32, room=self.F32 - 1)
        assert (held.device.type, held.element_bytes, held.batch_bytes) == ("cpu", 2, self.F32 // 2)
        assert held.hold_raw is True and "does not fit" in held.width_reason

    def test_raw_that_fits_exactly_and_one_byte_short(self):
        assert decide(2, self.F32, room=self.F32 // 2).element_bytes == 2
        short = decide(2, self.F32, room=self.F32 // 2 - 1)
        assert short.element_bytes == 2 and "no width fits" in short.width_reason

    @pytest.mark.parametrize("room", [None, 0, 4 * GIB, 64 * GIB])
    def test_a_float32_model_has_one_width(self, room):
        assert decide(4, self.F32, room=room).element_bytes == 4

    def test_an_unknown_room_holds_a_16_bit_model_raw(self):
        held = decide(2, self.F32, room=None)
        assert held.element_bytes == 2 and "not known" in held.width_reason

    def test_the_record_names_the_width(self):
        record = decide(2, self.F32, room=self.F32).as_record()
        assert record == {"element_bytes": 4, "held": "float32, standardised once",
                          "held_reason": record["held_reason"], "float32_batch_bytes": self.F32}
        assert decide(2, self.F32, room=0).as_record()["held"] == "float16 as stored, standardised at every use"


class TestTheRuleOnTheCard:
    F32 = 8 * GIB

    def test_float32_on_the_card_when_it_fits(self):
        held = decide(2, self.F32, card=CARD, free=card_needs(self.F32), room=0)
        assert (held.device, held.element_bytes) == (CARD, 4)

    def test_raw_on_the_card_before_float32_in_host_memory(self):
        """A per-use widening on a card is one kernel over device memory; a CPU epoch is the slow path."""
        held = decide(2, self.F32, card=CARD, free=card_needs(self.F32) - 1, room=64 * GIB)
        assert (held.device, held.element_bytes) == (CARD, 2)

    def test_neither_width_on_the_card_falls_to_host_memory(self):
        free = card_needs(self.F32 // 2) - 1
        assert decide(2, self.F32, card=CARD, free=free, room=64 * GIB).device.type == "cpu"
        assert decide(2, self.F32, card=CARD, free=free, room=64 * GIB).element_bytes == 4
        assert decide(2, self.F32, card=CARD, free=free, room=self.F32 - 1).element_bytes == 2

    def test_a_float32_model_never_tries_the_card_at_half(self):
        free = card_needs(self.F32) - 1
        assert decide(4, self.F32, card=CARD, free=free, room=64 * GIB).device.type == "cpu"

    def test_an_unreadable_card_is_host_memory(self):
        held = decide(2, self.F32, card=CARD, free=None, room=64 * GIB)
        assert held.device.type == "cpu" and "could not be read" in held.device_reason


# ── the values do not depend on the width ─────────────────────────────────────────────────────


def make_rows(n=64, seed=5):
    """float16 rows of varied length, a channel offset that makes standardising matter, a constant channel."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=D_MODEL)
    rows, labels = [], []
    for i in range(n):
        y = i % 2
        block = rng.normal(3.0, 2.0, size=(int(rng.integers(4, 50)), D_MODEL)) + (0.5 * direction if y else 0.0)
        block[:, 9] = -2.0
        rows.append(block.astype(np.float16))
        labels.append(y)
    return rows, labels


@pytest.fixture
def data16():
    rows, labels = make_rows()
    split = int(len(rows) * 0.75)
    return rows[:split], labels[:split], rows[split:], labels[split:]


def by_formula(rows, indices, mean, std):
    """One batch standardised by hand: float16 widened exactly, minus the mean, over the std (a zero std
    zeroes the channel), padding zero. Written here, not taken from `standardise_batch`."""
    mean = mean.numpy().astype(np.float32)
    std = std.numpy().astype(np.float32)
    degenerate = np.abs(std) <= 1e-6
    divisor = np.where(degenerate, np.float32(1.0), std).astype(np.float32)
    width = max(rows[i].shape[0] for i in indices)
    out = np.zeros((len(indices), width, D_MODEL), dtype=np.float32)
    for position, i in enumerate(indices):
        values = (rows[i].astype(np.float32) - mean) / divisor
        values[:, degenerate] = 0.0
        out[position, : rows[i].shape[0]] = values
    return out


class TestTheValues:
    @pytest.mark.parametrize("hold_raw", [True, False])
    def test_each_width_gives_the_formulas_numbers(self, data16, hold_raw):
        tr = data16[0]
        mean, std = _standardisation(tr)
        std = std.clone()
        std[9] = 0.0  # `_standardisation` floors a constant channel to 1.0; zero reaches the zeroing branch
        terms = standardisation_terms(mean, std, CPU)
        batches = _standardised_batches(tr, mean, std, CPU, budget_bytes=SMALL, hold_raw=hold_raw)
        assert len(batches) > 2
        assert {x.dtype for x, _m, _i in batches} == {torch.float16 if hold_raw else torch.float32}
        for x, mask, indices in batches:
            got = ready_batch(x, mask, terms, torch.empty(x.numel(), dtype=torch.float32)).numpy()
            assert np.array_equal(got, by_formula(tr, indices, mean, std))

    @pytest.mark.parametrize("rule", RULES)
    def test_the_weights_are_bit_identical_at_either_width(self, data16, rule):
        tr, trl, va, val = data16
        raw = train_rule(rule, tr, trl, va, val, seed=1337, epochs=20, batch_bytes=SMALL, hold_raw=True)
        wide = train_rule(rule, tr, trl, va, val, seed=1337, epochs=20, batch_bytes=SMALL, hold_raw=False)
        assert torch.equal(raw.weight, wide.weight) and raw.bias == wide.bias
        assert raw.val_auroc == wide.val_auroc and raw.best_epoch == wide.best_epoch
        assert [h["loss"] for h in raw.history] == [h["loss"] for h in wide.history]
        if rule == "attention":
            assert torch.equal(raw.attention_query, wide.attention_query)

    def test_float32_rows_cannot_be_held_raw(self, data16):
        tr = [r.astype(np.float32) for r in data16[0]]
        mean, std = _standardisation(tr)
        with pytest.raises(ValueError, match="only float16 rows"):
            _standardised_batches(tr, mean, std, CPU, budget_bytes=SMALL, hold_raw=True)


# ── every caller asks the rule and uses its answer ───────────────────────────────────────────


def _spy_rule(monkeypatch):
    calls = []
    real = trainer.batch_holding

    def spy(**kwargs):
        answer = real(**kwargs)
        calls.append((kwargs, answer))
        return answer

    monkeypatch.setattr(trainer, "batch_holding", spy)
    return calls


class TestTheForecastAsksTheRule:
    def plan(self, **overrides):
        arguments = {"train_widths": [100] * 80, "val_widths": [50] * 20, "d_model": 4096, "swept_layers": 1,
                     "top_n_layers": 1, "storage_bytes": 2, "based_on": "a test", "available_bytes": 100 * GIB}
        return memory.forecast_memory(**{**arguments, **overrides})

    def test_it_hands_the_rule_its_float32_figure_and_its_own_room(self, monkeypatch):
        calls = _spy_rule(monkeypatch)
        answer = self.plan(reserved_bytes=3 * GIB, held_bytes=5 * GIB, card=CARD, card_free=7 * GIB)
        assert len(calls) == 1
        kwargs, holding = calls[0]
        usable = int(100 * GIB * (1 - ResourceConfig.RAM_SAFETY_MARGIN)) - 3 * GIB
        assert kwargs == {"storage_bytes": 2, "float32_bytes": 9000 * 4096 * 4, "card": CARD, "card_free": 7 * GIB,
                          "host_room": usable + 5 * GIB - 2 * GIB - 4 * GIB}
        assert answer["batches_bytes"] == holding.batch_bytes and answer["batch_room_bytes"] == kwargs["host_room"]

    def test_the_answer_is_the_rules_whichever_width_it_picks(self, monkeypatch):
        for width in (2, 4):
            monkeypatch.setattr(trainer, "batch_holding", lambda _w=width, **k: trainer.BatchHolding(
                CPU, _w, k["float32_bytes"], "card", "test"))
            answer = self.plan()
            assert answer["batches_element_bytes"] == width
            assert answer["batches_bytes"] == 9000 * 4096 * width
            assert answer["needed_bytes"] == 9000 * 4096 * width + 6 * GIB
            assert f"x 4096 dims x {width} B" in answer["arithmetic"]

    def test_float32_fits_means_the_forecast_at_4_bytes_fits(self):
        """Room for exactly the float32 batches: widened, and the verdict at 4 B is `fits`."""
        allowances = memory.RAM_TRANSIENT_BYTES + memory.RAM_WORKER_BASELINE_BYTES
        f32 = 9000 * 4096 * 4
        margin = 1 - ResourceConfig.RAM_SAFETY_MARGIN
        for extra, width, fits in ((0, 4, True), (-(f32 // 4), 2, True)):
            available = int(-(-(allowances + f32 + extra) // margin))
            # the smallest available memory whose usable figure reaches the target
            while int(available * margin) < allowances + f32 + extra:
                available += 1
            answer = self.plan(available_bytes=available)
            assert answer["batches_element_bytes"] == width and answer["fits"] is fits


class TestTheRunAsksTheRulePerLayer:
    """The run's per-layer wiring, payload and count, on the REAL `execute_probe_run`."""

    def test_the_run_asks_once_for_the_forecast_and_once_per_layer_and_builds_at_its_answer(
        self, pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer, tmp_path, monkeypatch  # noqa: F811
    ):
        from src.core import config as config_module
        from src.services import probe_monitor_run as run_module
        from tests.unit.test_probe_run_memory_forecast import CONFIG, _run_row

        monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
        monkeypatch.setattr(run_module, "storage_dtype_of_load", lambda model: "float16")
        monkeypatch.setattr(memory, "batch_room_now", lambda db, available_bytes=None: 0)  # the run's own room
        calls = _spy_rule(monkeypatch)
        builds = []
        real_build = trainer._standardised_batches

        def build_spy(rows, mean, std, device, **kwargs):
            builds.append(kwargs.get("hold_raw"))
            return real_build(rows, mean, std, device, **kwargs)

        monkeypatch.setattr(trainer, "_standardised_batches", build_spy)
        run = _run_row(pipeline_db, arrow_dataset, CONFIG)
        run_module.execute_probe_run(pipeline_db, run.id,
                                     model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
        pipeline_db.refresh(run)
        assert run.status == "completed"
        forecast = run.environment["memory_forecast"]
        recorded = run.environment["training_device"]
        assert len(calls) == 1 + len(recorded) == 3
        # The forecast's: its own room, ample here, so widened.
        assert calls[0][0]["float32_bytes"] == forecast["float32_batches_bytes"] and calls[0][1].element_bytes == 4
        # Each layer's: the live room (0 here), so raw; the float32 figure is the layer's real batches at 4 B.
        for (kwargs, holding), entry in zip(calls[1:], recorded.values(), strict=True):
            assert kwargs["storage_bytes"] == 2 and kwargs["host_room"] == 0 and kwargs["card"] is None
            assert kwargs["float32_bytes"] == forecast["padded_slots"] * forecast["d_model"] * 4
            assert holding.element_bytes == 2 and entry["element_bytes"] == 2
            assert entry["batch_bytes"] == kwargs["float32_bytes"] // 2
        # Train and validation, per layer, built at the layer's answer.
        assert builds == [True, True] * len(recorded)


class TestTheRoomAtTheTrainingStage:
    def test_it_is_live_usable_less_other_jobs_reserves_less_the_transient(self, monkeypatch):
        from src.services import gpu_job_claim

        monkeypatch.setattr(gpu_job_claim, "current_claim", lambda: SimpleNamespace(holder="probe:me"))
        monkeypatch.setattr("src.services.gpu_leases.host_reserves",
                            lambda db: {"probe:me": 10 * 1024, "training:other": 3 * 1024})
        room = memory.batch_room_now(None, available_bytes=100 * GIB)
        usable = int(100 * GIB * (1 - ResourceConfig.RAM_SAFETY_MARGIN))
        assert room == usable - 3 * GIB - 2 * GIB, "its own reserve is for these batches and is not subtracted"

    def test_outside_a_claim_every_reserve_counts(self, monkeypatch):
        from src.services import gpu_job_claim

        monkeypatch.setattr(gpu_job_claim, "current_claim", lambda: None)
        monkeypatch.setattr("src.services.gpu_leases.host_reserves", lambda db: {"probe:x": 1024})
        usable = int(100 * GIB * (1 - ResourceConfig.RAM_SAFETY_MARGIN))
        assert memory.batch_room_now(None, available_bytes=100 * GIB) == usable - GIB - 2 * GIB


class TestStandaloneTraining:
    """`train_rule` building its own batches (the SAE variant) asks the rule with host memory as its room."""

    def test_it_asks_the_rule_and_builds_at_its_answer(self, data16, monkeypatch):
        tr, trl, va, val = data16
        calls = _spy_rule(monkeypatch)
        monkeypatch.setattr(trainer, "host_ram_usable_bytes", lambda *a, **k: 2 * GIB + 1)
        dtypes = []
        real_build = trainer._standardised_batches
        monkeypatch.setattr(trainer, "_standardised_batches", lambda *a, **k: (
            lambda b: dtypes.extend(x.dtype for x, _m, _i in b) or b)(real_build(*a, **k)))
        train_rule("mean", tr, trl, va, val, seed=1337, epochs=2, batch_bytes=SMALL)
        assert len(calls) == 1
        kwargs, holding = calls[0]
        assert kwargs["storage_bytes"] == 2 and kwargs["host_room"] == 1 and kwargs["card"] is None
        assert holding.element_bytes == 2 and set(dtypes) == {torch.float16}

    def test_with_room_it_widens(self, data16):
        tr, trl, va, val = data16
        assert trainer.standalone_hold_raw(tr, va, CPU, budget_bytes=SMALL) is False
        assert trainer.standalone_hold_raw([r.astype(np.float32) for r in tr], va, CPU) is False


# ── the time forecast decides as the run does ──────────────────────────────────────────────────


def _runs(*entries):
    return [{"environment": {"training_device": {"8": entry}}} for entry in entries]


class TestTheTimeForecast:
    F32 = 8 * GIB

    def test_the_card_takes_a_16_bit_runs_batches_raw_when_float32_does_not_fit(self):
        free = card_needs(self.F32) - 1
        expected = tf.expected_training_device(self.F32, _runs({"device": "cuda:0", "free_bytes": free}),
                                               storage_bytes=2)
        assert expected["expected"] == "gpu" and expected["element_bytes"] == 2
        assert expected["batch_bytes"] == self.F32 // 2
        float32_model = tf.expected_training_device(self.F32, _runs({"device": "cuda:0", "free_bytes": free}))
        assert float32_model["expected"] == "cpu" and float32_model["element_bytes"] == 4

    def test_in_host_memory_the_room_decides_the_width(self):
        runs = _runs({"device": "cpu", "free_bytes": GIB})
        assert tf.expected_training_device(self.F32, runs, storage_bytes=2,
                                           host_room_bytes=self.F32)["element_bytes"] == 4
        assert tf.expected_training_device(self.F32, runs, storage_bytes=2,
                                           host_room_bytes=self.F32 - 1)["element_bytes"] == 2

    def test_training_rates_are_taken_at_the_expected_width(self):
        assert tf.held_width_of_run(_runs({"element_bytes": 2})[0]) == 2
        assert tf.held_width_of_run(_runs({"device": "cpu"})[0]) is None
        mixed = {"environment": {"training_device": {"1": {"element_bytes": 2}, "2": {"element_bytes": 4}}}}
        assert tf.held_width_of_run(mixed) is None

    def test_the_request_hands_the_float32_figure_the_width_and_the_room(self, monkeypatch):
        seen = []
        monkeypatch.setattr(tf, "forecast_time", lambda **kwargs: seen.append(kwargs) or {})
        tf.time_forecast_from_db(MagicMock(), {"model_id": "m", "config": {}}, {
            "checked": True, "rows": 8, "real_tokens": 40, "swept_layers": 1,
            "batches_bytes": 111, "float32_batches_bytes": 222, "storage_bytes": 2, "batch_room_bytes": 333})
        assert len(seen) == 1
        assert (seen[0]["batch_bytes"], seen[0]["storage_bytes"], seen[0]["host_room_bytes"]) == (222, 2, 333)



def _past(run_id, seconds, entry, minutes_ago):
    return {
        "id": run_id, "model_id": "m", "status": "completed", "config": {"scope": "all", "rules": ["mean"]},
        "completed_at": f"2026-11-09T10:{60 - minutes_ago:02d}:00", "eval_dataset_ids": [],
        "calibration_dataset_id": None, "layer_selection": {"chosen": [8]},
        "environment": {"render": {"rendered": 10, "scored_tokens": 1000}, "layers_swept": [8],
                        "stage_seconds": {"training": seconds}, "training_device": {"8": entry}},
    }


class TestTheTrainingRateIsTakenAtTheWidth:
    """A 16-bit run's held-raw epoch is 2.3-6x a widened one's: a rate measured at the other width is not used."""

    UNITS = tf.WorkUnits(rows=10, scored_tokens=1000, tokens=1000, swept_layers=1, chosen_layers=1, probes=1,
                         calibration_rows=0, eval_rows=0)
    CPU_ENTRY = {"device": "cpu", "free_bytes": GIB}

    def forecast(self, history, room):
        return tf.forecast_time(units=self.UNITS, history=history, rows_by_view={}, model_id="m", scope="all",
                                batch_bytes=8 * GIB, storage_bytes=2, host_room_bytes=room)

    def test_a_widened_forecast_uses_only_widened_rates(self):
        history = [_past("raw", 500.0, {**self.CPU_ENTRY, "element_bytes": 2}, 1),
                   _past("wide", 100.0, {**self.CPU_ENTRY, "element_bytes": 4}, 2)]
        wide = self.forecast(history, room=64 * GIB)
        assert wide["training_device"]["element_bytes"] == 4
        assert [r["run_id"] for r in wide["stages"]["training"]["rates"]] == ["wide"]
        raw = self.forecast(history, room=GIB)
        assert raw["training_device"]["element_bytes"] == 2
        assert [r["run_id"] for r in raw["stages"]["training"]["rates"]] == ["raw"]
        assert not wide["caveats"] and not raw["caveats"]

    def test_a_rate_of_unrecorded_width_is_used_and_said(self):
        history = [_past("old", 300.0, dict(self.CPU_ENTRY), 1)]
        answer = self.forecast(history, room=64 * GIB)
        assert [r["run_id"] for r in answer["stages"]["training"]["rates"]] == ["old"]
        assert any("did not record the width" in caveat for caveat in answer["caveats"])
        assert "did not record their batch width" in answer["summary"]
        assert "Evaluation sets not seen before" not in answer["summary"]


class TestTheStartReservationIsTheHeldWidth:
    """M14 survived the first controls: the pipeline fixture's batches are kilobytes, so both widths declared 0 MB.
    Here the forecast is GiB-sized, and the run is stopped right after it reserves (at pooled capture)."""

    def test_the_run_reserves_the_forecasts_held_bytes_not_its_float32_figure(
        self, pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer, tmp_path, monkeypatch  # noqa: F811
    ):
        from src.core import config as config_module
        from src.services import gpu_job_claim
        from src.services import probe_monitor_capture as capture_module
        from src.services import probe_monitor_run as run_module
        from tests.unit.test_probe_run_memory_forecast import CONFIG, _run_row

        class Stop(Exception):
            pass

        monkeypatch.setattr(config_module.settings, "data_dir", tmp_path, raising=False)
        monkeypatch.setattr(memory, "forecast_for_run", lambda *a, **k: {
            "checked": True, "fits": True, "batches_on_card": False, "batches_element_bytes": 2,
            "batches_bytes": 3 * GIB, "float32_batches_bytes": 6 * GIB})
        declared = []
        monkeypatch.setattr(gpu_job_claim, "declare_host_reserve", lambda mb: declared.append(mb))
        monkeypatch.setattr(capture_module, "capture_pooled", lambda *a, **k: (_ for _ in ()).throw(Stop()))
        run = _run_row(pipeline_db, arrow_dataset, CONFIG)
        with pytest.raises(Stop):
            run_module.execute_probe_run(pipeline_db, run.id,
                                         model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
        assert declared == [3 * 1024]
