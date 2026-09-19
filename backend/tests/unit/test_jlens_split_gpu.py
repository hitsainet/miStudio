"""J-lens jobs on a model SPLIT across GPUs (0xcc/plans/Multi-GPU-Plan.md, Phase 2).

A model no single card holds loads with the placement's own `device_map` and its
GPU-only `max_memory` — passed through as the placement spells them, never
restated here, because the core chooses the map strategy (it moved from "auto"
to "sequential" when "auto" was found to rebalance the budgets onto disk). Every
place that assumed one card had to learn otherwise:

  * the registry's cache key named ONE device, so a split copy looked like a
    single-card copy on its first card — a readout naming that card reused a
    model spread over two, and the release emptied one card of the two;
  * the load took `placement.device`, the split's first card, so the split was
    never loaded as one;
  * a readout that kept its model could reuse only a one-card copy;
  * the task row recorded only the first card;
  * input ids went to the first PARAMETER's device, which on a split model need
    not be the embedding's card.

Asserted on CPU. Splits are fake placements over torch devices that are never
touched (`cuda:0`, `cuda:1`); "another device" is `meta`, the only second device
a CPU-only machine has. Every fixture disagrees with the defect it guards: the
split's FIRST card is cuda:1 while its key sorts cuda:0 first, and the model's
first parameter is on `meta` while its embedding is on the CPU.

MUTATION CONTROLS (2026-09-14; each red, then restored and verified by sha256):
  M1  cache key back to one device (device_spec(devices[:1]))
        -> TestASplitLoad::test_a_split_loads_across_its_cards_within_their_budgets,
           TestTheKeyNamesEveryCard::test_a_resident_split_reports_every_card
  M2  primary from_pretrained without max_memory= -> the two TestASplitLoad load tests
  M3  fallback loader without max_memory= -> test_the_fallback_loader_is_held_to_the_same_cards
  M4  device_map = str(placement.device) -> the split-load and fallback tests
  M5  off-GPU refusal disabled -> test_a_split_accelerate_put_on_disk_is_refused_and_every_card_emptied
  M6  _drop empties only the first card -> all three TestASplitCopyIsReleasedFromEveryCard tests
  M7  _drop does not detach dispatch hooks -> test_clearing_a_split_copy_empties_every_card_and_unhooks_it
  M8  plan reuses a resident split for a named card of it
        -> TestTheReadoutPlanForASplit::test_naming_one_card_of_the_split_frees_it_first (x3),
           TestTheReadoutWorkerOnASplit::test_naming_one_card_of_a_resident_split_...
  M9  Auto never reuses a resident split -> test_auto_reuses_the_split (x3), the worker reuse test
  M10 ReadoutService.capture_device back to the first parameter's device
        -> test_the_readout_sends_ids_to_the_embeddings_card_not_the_first_parameters
  M11 _capture_residuals no longer moves the ids -> test_the_capture_moves_the_ids_it_is_handed
  M12 fitter.device back to the first parameter -> test_the_fitter_sends_ids_to_the_embeddings_card
  M14 readout places with allow_shard=False -> test_a_fresh_readout_may_split_and_says_how_big_the_model_is
        (and test_jlens_gpu_placement ...may_split_and_how_big_its_model_is[readout])
  M16 place_on_card records no gpu_uuids -> the fresh-readout test, test_a_split_placement_writes_every_card
  M17 reuse_card records no gpu_uuids -> the worker reuse test, test_reusing_a_split_writes_every_card
  M18 record_gpu does not write gpu_uuids -> both TestTheRowNamesEveryCard split tests
  M19 readout loads with capture_device=placement.device -> every TestTheReadoutWorkerOnASplit test
  M20 readout places with required_mb=None -> the fresh-readout test (and the gpu_placement [readout] case)
  M22 a budget-less split is loaded anyway -> test_a_split_without_budgets_is_never_loaded
  M23 idle release does not see a split as on a GPU -> test_placing_any_job_frees_an_idle_split_copy
  M24 estimate ignores the checkpoint dtype -> test_a_float32_checkpoint_counts_four_bytes_whatever_its_row_says
  M25 estimate ignores bitsandbytes for quantized rows -> test_a_quantized_row_counts_what_bitsandbytes_holds
  (M13 lives in test_jlens_intervention_task.py, M15 in test_jlens_gpu_placement.py,
   M21 in test_jlens_reachable_mcp.py.)

REVIEW ROUND 2 (2026-09-14): a Q4 row is sized from its architecture_config, not its
packed count (`base_model_budget.params_for_sizing`). Control Q7 (estimate_weights_mb
passes no architecture_config) -> TestTheWeightsEstimate::test_a_four_bit_row..., red.
The split load's own controls (mapping before loading, no out-of-memory reload) are
P1-P5 in test_every_split_load_is_mapped_first.py; M2 and M3 re-run there, red.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.services import gpu_placement
from src.services.gpu_placement import GpuCard, Placement

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
THIRD_UUID = "GPU-3a3a3a3a-3a3a-3a3a-3a3a-3a3a3a3a3a3a"

TI = GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000)
RTX = GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000)
THIRD = GpuCard(index=2, uuid=THIRD_UUID, name="NVIDIA GeForce RTX 3060", total_mb=12_288, free_mb=12_000)
CARDS = [TI, RTX]

TI_DEVICE = torch.device("cuda", 0)
RTX_DEVICE = torch.device("cuda", 1)

#: Qwen2.5-14B-scale: fits neither card alone at bf16, fits both together.
PARAMS = 14_770_000_000
#: An FP16 row whose checkpoint dtype cannot be read counts 2 bytes a parameter.
WEIGHTS_MB = PARAMS * 2 / (1024 * 1024)


def _split() -> Placement:
    """What `place_job` returns for this model on Auto: the 3090 first (most free)."""
    return Placement(
        card=RTX,
        device=RTX_DEVICE,
        cards=(RTX, TI),
        devices=(RTX_DEVICE, TI_DEVICE),
        max_memory_mb={1: 21_976, 0: 9_976},
    )


SPLIT_MAX_MEMORY = {1: "21976MiB", 0: "9976MiB"}


def _record(**fields):
    base = dict(
        id="m_1",
        repo_id="org/model",
        file_path="models/m_1",
        quantization="FP16",
        params_count=PARAMS,
    )
    base.update(fields)
    return types.SimpleNamespace(**base)


# ── The registry ───────────────────────────────────────────────────────────


class _Loaded(torch.nn.Module):
    """What `from_pretrained` hands back: accelerate's map, and an lm_head for W_U."""

    def __init__(self, hf_device_map):
        super().__init__()
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.hf_device_map = dict(hf_device_map)


@contextmanager
def _loading(tmp_path, hf_device_map, primary_fails=False):
    """The registry's real `load_for_readout`, with only the weights faked.

    `seen.primary` / `seen.fallback` are the kwargs each loader received;
    `seen.emptied` the cards each release emptied.
    """
    from src.services import jlens_model_registry as registry

    seen = types.SimpleNamespace(primary=[], fallback=[], emptied=[])

    def primary(repo_id, **kwargs):
        seen.primary.append(kwargs)
        if primary_fails:
            raise RuntimeError("native-dtype load failed")
        return _Loaded(hf_device_map)

    def fallback(**kwargs):
        seen.fallback.append(kwargs)
        return _Loaded(hf_device_map), object(), None, {}

    with patch.object(registry, "_CACHE", registry._SingleEntryCache()), patch.object(
        type(registry.settings), "resolve_data_path", lambda self, raw: tmp_path
    ), patch("transformers.AutoModelForCausalLM.from_pretrained", primary), patch(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: object()
    ), patch("src.ml.model_loader.load_model_from_hf", fallback), patch(
        "src.ml.layer_discovery.discover_transformer_structure",
        lambda model: types.SimpleNamespace(num_layers=2),
    ), patch(
        "src.services.analysis_service.resolve_snapshot_dir", lambda *a: None
    ), patch("torch.cuda.is_available", lambda: True), patch(
        "src.ml.model_devices.empty_cache_on",
        lambda devices: seen.emptied.append(sorted(str(d) for d in devices)),
    ):
        yield registry, seen


class TestASplitLoad:
    GPU_ONLY_MAP = {"model.embed_tokens": 1, "model.layers.1": 0, "lm_head": 0}

    def test_a_split_loads_across_its_cards_within_their_budgets(self, tmp_path):
        with _loading(tmp_path, self.GPU_ONLY_MAP) as (registry, seen):
            loaded = registry.load_for_readout(_record(), placement=_split())

        assert len(seen.primary) == 1 and seen.fallback == []
        assert seen.primary[0]["device_map"] == _split().device_map
        assert seen.primary[0]["max_memory"] == SPLIT_MAX_MEMORY
        assert loaded.key == "m_1:FP16@cuda:0+cuda:1"

    def test_a_single_card_loads_on_that_card_with_no_budget(self, tmp_path):
        with _loading(tmp_path, {"": 1}) as (registry, seen):
            loaded = registry.load_for_readout(
                _record(), placement=Placement(card=RTX, device=RTX_DEVICE)
            )

        assert seen.primary[0]["device_map"] == "cuda:1"
        assert seen.primary[0]["max_memory"] is None
        assert loaded.key == "m_1:FP16@cuda:1"

    def test_the_fallback_loader_is_held_to_the_same_cards(self, tmp_path):
        with _loading(tmp_path, self.GPU_ONLY_MAP, primary_fails=True) as (registry, seen):
            registry.load_for_readout(_record(), placement=_split())

        assert len(seen.fallback) == 1
        assert seen.fallback[0]["device_map"] == _split().device_map
        assert seen.fallback[0]["max_memory"] == SPLIT_MAX_MEMORY

    def test_a_split_accelerate_put_on_disk_is_refused_and_every_card_emptied(self, tmp_path):
        """accelerate keeps "disk" as a last resort with no "cpu" budget, so a
        model the cards cannot hold still LOADS and reads layers from disk."""
        on_disk = {"model.embed_tokens": 1, "model.layers.1": 0, "lm_head": "disk"}
        with _loading(tmp_path, on_disk) as (registry, seen):
            with pytest.raises(registry.ModelNotAvailable, match="disk"):
                registry.load_for_readout(_record(), placement=_split())
            assert registry.loaded_model_key() is None, "a refused load was cached"

        assert seen.fallback == [], "the refusal was taken for a failed load and retried"
        assert seen.emptied == [["cuda:0", "cuda:1"]]

    def test_a_split_without_budgets_is_never_loaded(self, tmp_path):
        """A placement that REUSES a resident split names its cards and carries
        no budgets; loading from it would fill the cards unbounded."""
        reused = Placement(card=RTX, device=RTX_DEVICE, cards=(RTX, TI), devices=(RTX_DEVICE, TI_DEVICE))
        with _loading(tmp_path, self.GPU_ONLY_MAP) as (registry, seen):
            with pytest.raises(RuntimeError, match="no per-card budget"):
                registry.load_for_readout(_record(), placement=reused)

        assert seen.primary == [] and seen.fallback == []


class TestTheWeightsEstimate:
    """What Auto is told the model needs. Without it Auto never splits; with the
    row's LABEL instead of the precision the load uses, it splits a model that
    fits or refuses one that would."""

    MiB = 1024 * 1024

    @staticmethod
    @contextmanager
    def _checkpoint(tmp_path, dtype):
        import json

        from src.services import jlens_model_registry as registry

        (tmp_path / "model.safetensors").write_bytes(b"")
        (tmp_path / "config.json").write_text(json.dumps({"dtype": dtype}))
        with patch.object(type(registry.settings), "resolve_data_path", lambda self, raw: tmp_path):
            yield registry

    def test_a_quantized_row_counts_what_bitsandbytes_holds(self):
        from src.services.jlens_model_registry import estimate_weights_mb

        assert estimate_weights_mb(_record(quantization="Q4")) == pytest.approx(PARAMS * 0.6 / self.MiB)

    def test_a_four_bit_row_is_sized_from_its_architecture_not_its_packed_count(self):
        """Review round 2. A Q4 row's params_count is the PACKED count (bitsandbytes
        stores two values a byte, so the loaded model reports about half its
        parameters; aaa233de). Sized from it, a J-lens job asked Auto for half the
        model's weights. The row's architecture_config says what the model is:
        32 x (4*4096^2 + 3*4096*14336) + 2*128256*4096 = 8,835,301,376, by hand."""
        from src.services.jlens_model_registry import estimate_weights_mb

        arch = {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 128_256,
                "intermediate_size": 14_336}
        record = _record(quantization="Q4", params_count=4_940_000_000, architecture_config=arch)
        assert estimate_weights_mb(record) == pytest.approx(8_835_301_376 * 0.6 / self.MiB)

        eight_bit = _record(quantization="Q8", params_count=4_940_000_000, architecture_config=arch)
        assert estimate_weights_mb(eight_bit) == pytest.approx(4_940_000_000 * 1.1 / self.MiB)

    def test_a_float32_checkpoint_counts_four_bytes_whatever_its_row_says(self, tmp_path):
        with self._checkpoint(tmp_path, "float32") as registry:
            estimate = registry.estimate_weights_mb(_record(quantization="FP16"))
        assert estimate == pytest.approx(PARAMS * 4 / self.MiB)

    def test_a_bf16_checkpoint_counts_two_bytes_whatever_its_row_says(self, tmp_path):
        with self._checkpoint(tmp_path, "bfloat16") as registry:
            estimate = registry.estimate_weights_mb(_record(quantization="FP32"))
        assert estimate == pytest.approx(PARAMS * 2 / self.MiB)

    def test_an_unknown_size_is_none_so_auto_behaves_as_before(self):
        from src.services.jlens_model_registry import estimate_weights_mb

        assert estimate_weights_mb(_record(params_count=None)) is None
        assert estimate_weights_mb(MagicMock()) is None


class TestTheKeyNamesEveryCard:
    def test_the_same_cards_in_either_order_are_one_copy(self):
        from src.services.jlens_model_registry import device_spec, parse_device_spec

        spec = device_spec((RTX_DEVICE, TI_DEVICE))
        assert spec == device_spec(("cuda:0", "cuda:1")) == "cuda:0+cuda:1"
        assert parse_device_spec(spec) == ("cuda:0", "cuda:1")
        assert device_spec((RTX_DEVICE,)) == "cuda:1"

    def test_a_resident_split_reports_every_card(self):
        from src.services import jlens_model_registry as registry

        keys = []
        with patch.object(registry._CACHE, "get_or_load", lambda key, loader: keys.append(key)):
            registry.load_for_readout(_record(), placement=_split())
        assert keys == ["m_1:FP16@cuda:0+cuda:1"]

        saved = registry._CACHE._entry
        registry._CACHE._entry = types.SimpleNamespace(key=keys[0])
        try:
            assert registry.resident_device_for(_record()) == "cuda:0+cuda:1"
        finally:
            registry._CACHE._entry = saved


class TestASplitCopyIsReleasedFromEveryCard:
    @contextmanager
    def _spies(self):
        emptied, unhooked = [], []
        with patch("torch.cuda.is_available", lambda: True), patch(
            "src.ml.model_devices.empty_cache_on",
            lambda devices: emptied.append(sorted(str(d) for d in devices)),
        ), patch(
            "src.ml.model_devices.detach_dispatch_hooks", lambda model: unhooked.append(model)
        ):
            yield emptied, unhooked

    def test_clearing_a_split_copy_empties_every_card_and_unhooks_it(self):
        from src.services import jlens_model_registry as registry

        model = torch.nn.Linear(2, 2)
        cache = registry._SingleEntryCache()
        cache._entry = types.SimpleNamespace(key="m_1:FP16@cuda:0+cuda:1", model=model)
        with self._spies() as (emptied, unhooked):
            cache.clear()

        assert cache.loaded_key is None
        assert emptied == [["cuda:0", "cuda:1"]]
        assert unhooked == [model]

    def test_evicting_a_split_copy_empties_every_card(self):
        from src.services import jlens_model_registry as registry

        cache = registry._SingleEntryCache()
        cache._entry = types.SimpleNamespace(key="m_1:FP16@cuda:0+cuda:1", model=torch.nn.Linear(2, 2))
        with self._spies() as (emptied, _unhooked):
            cache.get_or_load("m_2:FP16@cpu", lambda: types.SimpleNamespace(key="m_2:FP16@cpu"))

        assert emptied == [["cuda:0", "cuda:1"]]
        assert cache.loaded_key == "m_2:FP16@cpu"

    def test_placing_any_job_frees_an_idle_split_copy(self):
        from src.services import jlens_model_registry as registry

        saved = registry._CACHE._entry
        registry._CACHE._entry = types.SimpleNamespace(key="m_1:FP16@cuda:0+cuda:1")
        try:
            with self._spies() as (emptied, _unhooked):
                assert registry.release_idle_gpu_copy() is True
            assert registry._CACHE.loaded_key is None
        finally:
            registry._CACHE._entry = saved
        assert emptied == [["cuda:0", "cuda:1"]]


# ── The readout's reuse decision ───────────────────────────────────────────


SPLIT_SPEC = "cuda:0+cuda:1"
HELD = (TI, RTX)


class TestTheReadoutPlanForASplit:
    """`plan_readout` with a copy left loaded across both cards."""

    @staticmethod
    def _plan(*args, **kwargs):
        from src.workers.jlens_readout_tasks import plan_readout

        return plan_readout(*args, **kwargs)

    @pytest.mark.parametrize("requested", ["auto", None, ""])
    def test_auto_reuses_the_split(self, requested):
        plan = self._plan(SPLIT_SPEC, HELD, requested, cards=CARDS)
        assert (plan.reuse, plan.release_first) == (True, False)

    @pytest.mark.parametrize("named", [TI_UUID, RTX_UUID, RTX_UUID.lower()])
    def test_naming_one_card_of_the_split_frees_it_first(self, named):
        """A named card is honoured or refused, never swapped for a split."""
        plan = self._plan(SPLIT_SPEC, HELD, named, cards=CARDS)
        assert (plan.reuse, plan.release_first) == (False, True)

    def test_all_reuses_a_split_over_every_card(self):
        plan = self._plan(SPLIT_SPEC, HELD, "all", cards=CARDS)
        assert (plan.reuse, plan.release_first) == (True, False)

    def test_all_does_not_reuse_a_split_that_misses_a_card(self):
        plan = self._plan(SPLIT_SPEC, HELD, "all", cards=CARDS + [THIRD])
        assert (plan.reuse, plan.release_first) == (False, True)

    def test_all_does_not_reuse_a_single_card_copy(self):
        plan = self._plan("cuda:1", RTX, "all", cards=CARDS)
        assert (plan.reuse, plan.release_first) == (False, True)

    def test_a_split_with_a_card_nvml_cannot_name_is_freed_then_placed(self):
        plan = self._plan(SPLIT_SPEC, (TI, None), "auto", cards=CARDS)
        assert (plan.reuse, plan.release_first) == (False, True)


@contextmanager
def _readout_task(*, resident=None, held=None, place=None, inventory=CARDS):
    """The readout task with the database, NVML and the model faked.

    `events` is the order the task meets the GPU; `placed` is
    `(required_mb, allow_shard)` for each placement.
    """
    from src.workers.jlens_readout_tasks import compute_readout

    events, placed = [], []
    record = _record(file_path=None)
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = record

    @contextmanager
    def fake_db():
        yield db

    def fake_place_job(requested, required_mb=None, allow_shard=False):
        events.append(("place", requested))
        placed.append((required_mb, allow_shard))
        return place(requested) if place is not None else _split()

    def fake_record_gpu(task_id, gpu_uuid, gpu_uuids=None, attempts=10):
        events.append(("record", gpu_uuid, gpu_uuids))
        return True

    def fake_load(model_record, capture_device="cpu", placement=None):
        events.append(("load", placement))
        return object()

    compute_readout.push_request(id="t-split")
    try:
        with patch("src.core.database.get_sync_db", fake_db), patch(
            "src.services.gpu_placement.place_job", fake_place_job
        ), patch("src.workers.jlens_progress.record_gpu", fake_record_gpu), patch(
            "src.services.jlens_model_registry.load_for_readout", fake_load
        ), patch(
            "src.services.jlens_model_registry.clear_cache", lambda: events.append(("release",))
        ), patch(
            "src.services.jlens_model_registry.resident_device_for", lambda record: resident
        ), patch(
            "src.services.gpu_placement.card_for_device",
            lambda device, cards=None: (held or {}).get(str(device)),
        ), patch(
            "src.services.gpu_placement.list_cards", lambda: list(inventory)
        ), patch(
            "src.services.gpu_placement.make_current",
            lambda device: events.append(("current", torch.device(device))),
        ), patch(
            "src.workers.jlens_readout_tasks._read_out",
            lambda self, **kwargs: {"meta": {"kind": "meta"}, "tokens": []},
        ), patch("src.workers.jlens_progress.update_row", MagicMock()), patch.object(
            compute_readout, "update_state", MagicMock()
        ):
            yield compute_readout, events, placed
    finally:
        compute_readout.pop_request()


class TestTheReadoutWorkerOnASplit:
    HELD_BY_DEVICE = {"cuda:0": TI, "cuda:1": RTX}

    def test_a_fresh_readout_may_split_and_says_how_big_the_model_is(self):
        with _readout_task() as (task, events, placed):
            task.run(model_id="m_1", prompt="hello", gpu_request="all")

        # The weights plus the 2 GiB of activation headroom place_on_card adds.
        assert placed == [(pytest.approx(WEIGHTS_MB + 2 * 1024), True)]
        assert events == [
            ("place", "all"),
            ("record", RTX_UUID, [RTX_UUID, TI_UUID]),
            ("load", _split()),
        ]

    def test_auto_reuses_a_split_left_loaded_and_records_both_cards(self):
        with _readout_task(resident=SPLIT_SPEC, held=self.HELD_BY_DEVICE) as (task, events, placed):
            task.run(model_id="m_1", prompt="hello", gpu_request="auto")

        assert placed == [], "Auto placed afresh instead of reusing the split left loaded"
        assert events[:2] == [("current", TI_DEVICE), ("record", TI_UUID, [TI_UUID, RTX_UUID])]
        kind, placement = events[2]
        assert kind == "load"
        assert placement.is_shard
        assert placement.all_devices == (TI_DEVICE, RTX_DEVICE)
        assert placement.uuids == [TI_UUID, RTX_UUID]
        assert len(events) == 3, "the kept split was released although the caller kept it"

    def test_naming_one_card_of_a_resident_split_frees_it_then_places_on_that_card(self):
        on_ti = Placement(card=TI, device=TI_DEVICE)
        with _readout_task(
            resident=SPLIT_SPEC, held=self.HELD_BY_DEVICE, place=lambda requested: on_ti
        ) as (task, events, placed):
            task.run(model_id="m_1", prompt="hello", gpu_request=TI_UUID)

        assert events == [
            ("release",),
            ("place", TI_UUID),
            ("record", TI_UUID, None),
            ("load", on_ti),
        ]

    def test_unload_after_frees_a_split(self):
        with _readout_task() as (task, events, _placed):
            task.run(model_id="m_1", prompt="hello", gpu_request="auto", unload_after=True)

        assert events[-1] == ("release",)


# ── The row ────────────────────────────────────────────────────────────────


@pytest.fixture
def task_rows(monkeypatch):
    """A real `task_queue` table, in memory, behind the sync session the helpers use."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from src.models.task_queue import TaskQueue

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    TaskQueue.__table__.create(engine)
    session_factory = sessionmaker(bind=engine)

    @contextmanager
    def fake_sync_db():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("src.core.database.get_sync_db", fake_sync_db)
    yield session_factory
    engine.dispose()


def _row(session_factory, task_id):
    from src.models.task_queue import TaskQueue

    session = session_factory()
    try:
        return session.query(TaskQueue).filter_by(task_id=task_id).one()
    finally:
        session.close()


class TestTheRowNamesEveryCard:
    def test_a_split_placement_writes_every_card(self, task_rows, monkeypatch):
        from src.workers import jlens_progress

        jlens_progress.open_row(jlens_progress.READOUT, "m_1", "t-1")
        monkeypatch.setattr(
            gpu_placement, "place_job", lambda requested, required_mb=None, allow_shard=False: _split()
        )

        jlens_progress.place_on_card("t-1", "all", required_mb=WEIGHTS_MB, allow_shard=True)

        row = _row(task_rows, "t-1")
        assert row.gpu_uuid == RTX_UUID
        assert row.gpu_uuids == [RTX_UUID, TI_UUID]

    def test_one_card_leaves_gpu_uuids_empty(self, task_rows, monkeypatch):
        from src.workers import jlens_progress

        jlens_progress.open_row(jlens_progress.READOUT, "m_1", "t-1")
        monkeypatch.setattr(
            gpu_placement,
            "place_job",
            lambda requested, required_mb=None, allow_shard=False: Placement(card=RTX, device=RTX_DEVICE),
        )

        jlens_progress.place_on_card("t-1", "auto", required_mb=WEIGHTS_MB, allow_shard=True)

        row = _row(task_rows, "t-1")
        assert (row.gpu_uuid, row.gpu_uuids) == (RTX_UUID, None)

    def test_reusing_a_split_writes_every_card(self, task_rows):
        from src.workers import jlens_progress

        jlens_progress.open_row(jlens_progress.READOUT, "m_1", "t-1")

        jlens_progress.reuse_card("t-1", (TI, RTX), (TI_DEVICE, RTX_DEVICE), "auto")

        row = _row(task_rows, "t-1")
        assert (row.gpu_uuid, row.gpu_uuids) == (TI_UUID, [TI_UUID, RTX_UUID])


# ── Inputs go to the embedding's card ──────────────────────────────────────


class _Pass(torch.nn.Module):
    def forward(self, x):
        return x


class _EmbeddingNotFirst(torch.nn.Module):
    """A split model's shape on a CPU-only machine.

    The FIRST registered parameter (`lm_head`) is on `meta`, the embedding on
    the CPU — so "the first parameter's device" and "the embedding's device"
    disagree, as they do on a split model whose checkpoint registers another
    module first. Ids sent to `meta` fail in the embedding lookup.
    """

    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(4, 8, bias=False, device="meta")
        self.embed = torch.nn.Embedding(8, 4)
        self.layers = torch.nn.ModuleList([_Pass(), _Pass()])
        self.seen = []

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None):
        self.seen.append(input_ids.device)
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return types.SimpleNamespace(logits=None)


class _Recorder(torch.nn.Module):
    """Records where its ids arrived and runs its layers on a CPU tensor."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Pass(), _Pass()])
        self.seen = []

    def forward(self, input_ids=None):
        self.seen.append(input_ids.device)
        hidden = torch.zeros(1, int(input_ids.shape[-1]), 4)
        for layer in self.layers:
            hidden = layer(hidden)
        return None


def _service(model, **kwargs):
    from src.services.jlens_readout_service import ReadoutService

    return ReadoutService(
        model=model,
        tokenizer=None,
        structure=types.SimpleNamespace(num_layers=2, layers_module=model.layers),
        unembedding=torch.randn(8, 4),
        model_name="org/model",
        **kwargs,
    )


class TestInputsGoToTheEmbeddingsCard:
    def test_the_readout_sends_ids_to_the_embeddings_card_not_the_first_parameters(self):
        model = _EmbeddingNotFirst()
        assert next(model.parameters()).device.type == "meta", "precondition: the fixture must disagree"

        service = _service(model)
        captured = service._capture_residuals(torch.tensor([[1, 2, 3]]), [0, 1])

        assert torch.device(service.capture_device) == torch.device("cpu")
        assert model.seen == [torch.device("cpu")]
        assert sorted(captured.by_layer) == [0, 1]

    def test_the_capture_moves_the_ids_it_is_handed(self):
        """The band report hands the capture the tokenizer's CPU ids unmoved."""
        model = _Recorder()
        service = _service(model, capture_device="meta")

        service._capture_residuals(torch.tensor([[1, 2, 3]]), [0])

        assert model.seen == [torch.device("meta")]

    def test_the_fitter_sends_ids_to_the_embeddings_card(self):
        from src.ml.jlens_fitter import JacobianFitter

        model = _EmbeddingNotFirst()
        fitter = JacobianFitter(model, None, types.SimpleNamespace(num_layers=2, layers_module=model.layers))

        assert fitter.device == torch.device("cpu")
