"""Activation extraction runs on a model split across GPUs when no single card can hold it.

Multi-GPU Phase 2 (0xcc/plans/Multi-GPU-Plan.md). The node has a 12 GB RTX 3080 Ti
(index 0) and a 24 GB RTX 3090 (index 1). Extraction was written for one card:
the whole model forced onto one device, inputs moved to `model.device`, memory
measured and released on one card, and the job placed without a size — so Auto
could never split, and a model larger than every card died out of memory on the
3090. Now the worker sizes the job from the model's config and places it with
`allow_shard=True`; a split loads through `load_model_from_hf` within a GPU-only
budget; inputs follow the input embedding; every card is cleaned; and the row
records every card.

Nothing here needs a GPU. A "split" is faked with devices that genuinely differ:
the model whose `model.device` is `meta` while its embedding is on the CPU, and
placements whose two cards are two different CUDA indices recorded by a stubbed
`torch.cuda`. A fixture where every tensor sat on one device would agree with the
defect by construction.

MUTATION CONTROLS (each applied to the named line, run red, restored
byte-identically and checked by sha256; 2026-09-14):
  M1  model_tasks  extraction `place_job(..., allow_shard=True)` -> no allow_shard
      -> test_a_model_no_card_can_hold_is_split_and_every_card_recorded_before_the_load,
         test_all_splits_even_a_model_that_fits_one_card
  M2  activation_service  inputs `.to(inputs_on)` -> `.to(model.device)`
      -> test_inputs_go_to_the_embedding_card_not_model_device
  M3  activation_service  `_cleanup_model` empties `cuda[:1]` only
      -> test_cleanup_empties_and_measures_every_card
  M4  activation_service  split load without `max_memory=`
      -> test_a_split_loads_through_the_split_loader_with_its_gpu_only_budget
  M5  model_tasks  `record_gpu_uuid(_db, extraction_id, placement.uuid)` (no gpu_uuids)
      -> test_a_model_no_card_can_hold_is_split_and_every_card_recorded_before_the_load
  M6  model_tasks  extraction placed unsized (`required_mb=None`)
      -> test_a_model_no_card_can_hold_is_split_and_every_card_recorded_before_the_load
  M7  activation_service  `detach_dispatch_hooks(model)` removed
      -> test_cleanup_detaches_dispatch_hooks_before_moving_the_model
  M8  activation_service  finally `_release_gpu_memory(devices[:1])`
      -> test_a_failed_split_load_releases_every_card
  M9  activation_service  success-path `_cleanup_model(model, devices[:1])`
      -> test_the_service_cleans_every_card_of_its_placement
  M10 model_tasks  pre-extraction cleanup over `placed_cards[:1]`
      -> test_a_model_no_card_can_hold_is_split_and_every_card_recorded_before_the_load
  M11 model_tasks  extraction `finally` over `placement.all_devices[:1]`
      -> test_a_model_no_card_can_hold_is_split_and_every_card_recorded_before_the_load
  M12 model_tasks  `create_extraction(...)` without `gpu_uuids=`
      -> test_a_row_the_worker_creates_records_every_card
  M13 activation_service  `load_format_for` returns the row's quantization unchanged
      -> test_an_fp32_row_splits_at_the_fp16_a_single_card_uses,
         test_an_fp32_row_is_sized_at_the_fp16_it_loads_at
  M14 activation_service  `_release_gpu_memory` empties `devices[:1]`
      -> test_a_failed_split_load_releases_every_card
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from src.models.model import QuantizationFormat
from src.services import activation_service as service_module
from src.services import gpu_placement
from src.services.activation_service import ActivationExtractionError, ActivationService
from src.services.gpu_placement import GpuCard, Placement

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"

TI_CARD = GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000)
RTX_CARD = GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000)
CARDS = [TI_CARD, RTX_CARD]

CUDA_0 = torch.device("cuda", 0)
CUDA_1 = torch.device("cuda", 1)

#: What `place_job` builds for a split over the two cards, most free first.
SPLIT = Placement(
    card=RTX_CARD,
    device=CUDA_1,
    cards=(RTX_CARD, TI_CARD),
    devices=(CUDA_1, CUDA_0),
    max_memory_mb={1: 21_976, 0: 9_976},
)


def _config(layers: int) -> SimpleNamespace:
    """A decoder config `estimate_parameter_count` can size.

    56 layers is ~24,164 MB at FP16 with headroom: more than the 3090's 23,000 MB
    free, less than both cards' budgets (21,976 + 9,976). Two layers is ~3,320 MB.
    """
    return SimpleNamespace(hidden_size=4096, num_hidden_layers=layers, vocab_size=32_000, intermediate_size=11_008)


@pytest.fixture
def cuda(monkeypatch):
    """torch.cuda on a card-less machine: two cards, and a record of what named which.

    `empty_cache` records the device context it ran inside, so a test can tell
    WHICH card's cache was emptied, not merely that something was.
    """
    state = {"current": None}
    calls: list = []

    def recorder(name, result):
        def call(device=None, *args, **kwargs):
            calls.append((name, device))
            return result
        return call

    for name, result in [("memory_allocated", 0), ("memory_reserved", 0), ("synchronize", None)]:
        monkeypatch.setattr(torch.cuda, name, recorder(name, result))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("empty_cache", state["current"])))

    @contextmanager
    def device(target):
        previous, state["current"] = state["current"], torch.device(target)
        calls.append(("device", torch.device(target)))
        try:
            yield
        finally:
            state["current"] = previous

    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    uuids = {0: TI_UUID, 1: RTX_UUID}
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid=uuids[index]))
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "fake")
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)
    return calls


def _named(calls, name):
    return [device for call, device in calls if call == name]


# ── The load ───────────────────────────────────────────────────────────────

class _LoadedSplit:
    dtype = torch.float16
    hf_device_map = {"model.embed_tokens": 1, "model.layers.0": 1, "model.layers.1": 0, "lm_head": 0}


@pytest.fixture
def split_loader(monkeypatch, cuda):
    """`load_model_from_hf` as the extraction service sees it; `from_pretrained` must not be reached."""
    import transformers

    seen: list = []

    def load_model_from_hf(**kwargs):
        seen.append(kwargs)
        return _LoadedSplit(), object(), object(), {}

    def forced_single_card(*args, **kwargs):
        raise AssertionError("a split was loaded with the single-card from_pretrained call")

    monkeypatch.setattr(service_module, "load_model_from_hf", load_model_from_hf)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", forced_single_card)
    return seen


class TestTheSplitLoad:
    def test_a_split_loads_through_the_split_loader_with_its_gpu_only_budget(self, split_loader, tmp_path):
        service = ActivationService.__new__(ActivationService)
        service._load_model(str(tmp_path), QuantizationFormat.FP16, placement=SPLIT)

        assert len(split_loader) == 1
        kwargs = split_loader[0]
        assert kwargs["repo_id"] == str(tmp_path)
        # The placement's own map, whatever the core names it; never one card's.
        assert kwargs["device_map"] == SPLIT.device_map != str(SPLIT.device)
        assert kwargs["max_memory"] == SPLIT.max_memory == {1: "21976MiB", 0: "9976MiB"}
        assert "cpu" not in kwargs["max_memory"], "a split budget may never offer the CPU"
        assert kwargs["local_files_only"] is True
        assert kwargs["quant_format"] == QuantizationFormat.FP16

    def test_an_fp32_row_splits_at_the_fp16_a_single_card_uses(self, split_loader, tmp_path):
        """The single-card load always passes torch_dtype=float16, so FP32 rows were fp16 all along."""
        service = ActivationService.__new__(ActivationService)
        service._load_model(str(tmp_path), QuantizationFormat.FP32, placement=SPLIT)

        assert split_loader[0]["quant_format"] == QuantizationFormat.FP16

    def test_a_quantized_row_keeps_its_quantization_when_split(self, split_loader, tmp_path):
        service = ActivationService.__new__(ActivationService)
        service._load_model(str(tmp_path), QuantizationFormat.Q4, placement=SPLIT)

        assert split_loader[0]["quant_format"] == QuantizationFormat.Q4

    def test_a_single_card_is_still_forced_whole_onto_its_card(self, monkeypatch, cuda, tmp_path):
        import transformers

        loaded = {}

        class _Model:
            device = CUDA_1
            dtype = torch.float16

        def from_pretrained(path, **kwargs):
            loaded.update(kwargs)
            return _Model()

        monkeypatch.setattr(service_module, "load_model_from_hf", lambda **k: pytest.fail("split loader used for one card"))
        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", from_pretrained)
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: object())

        service = ActivationService.__new__(ActivationService)
        service._load_model(str(tmp_path), QuantizationFormat.FP16, placement=Placement(card=RTX_CARD, device=CUDA_1))

        assert loaded["device_map"] == {"": CUDA_1}
        assert "max_memory" not in loaded


# ── The forward pass ───────────────────────────────────────────────────────

def test_inputs_go_to_the_embedding_card_not_model_device(tmp_path):
    """A real forward pass on CPU, through `_run_extraction` itself.

    `model.device` is `meta` here while every weight, the embedding included, is
    on the CPU — the shape of a split whose first parameter is not the
    embedding. Inputs sent to `model.device` land on meta and the lookup fails.
    """
    from datasets import Dataset
    from transformers import LlamaConfig, LlamaForCausalLM

    from src.ml.forward_hooks import HookType

    class _FirstParameterElsewhere(LlamaForCausalLM):
        @property
        def device(self):
            return torch.device("meta")

    torch.manual_seed(0)
    model = _FirstParameterElsewhere(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
    )).eval()
    received = []
    model.register_forward_pre_hook(
        lambda module, args, kwargs: received.append(
            (args[0].device if args else kwargs["input_ids"].device, kwargs["attention_mask"].device)
        ),
        with_kwargs=True,
    )
    dataset = Dataset.from_dict({
        "input_ids": [list(range(1, 9)), list(range(9, 17))],
        "attention_mask": [[1] * 8, [1] * 8],
    })

    service = ActivationService.__new__(ActivationService)
    activations = service._run_extraction(
        model, SimpleNamespace(pad_token_id=0, eos_token_id=0), dataset, "llama",
        [1], [HookType.RESIDUAL], 2, 2, None, None, None, output_dir=tmp_path,
    )

    assert received == [(torch.device("cpu"), torch.device("cpu"))], (
        f"inputs went to {received}, not the embedding's device"
    )
    assert [array.shape for array in activations.values()] == [(2, 8, 16)]


# ── Cleanup ────────────────────────────────────────────────────────────────

class _DispatchedModel(torch.nn.Module):
    """A model carrying an accelerate hook, as a split load leaves one."""

    def __init__(self):
        super().__init__()
        from accelerate.hooks import ModelHook, add_hook_to_module

        self.layer = torch.nn.Linear(2, 2)
        add_hook_to_module(self.layer, ModelHook())
        self.hooked_when_moved = None

    def cpu(self):
        self.hooked_when_moved = hasattr(self.layer, "_hf_hook")
        return self


class TestCleanupCoversEveryCard:
    def test_cleanup_empties_and_measures_every_card(self, cuda):
        service = ActivationService.__new__(ActivationService)

        report = service._cleanup_model(_DispatchedModel(), SPLIT.all_devices)

        assert set(_named(cuda, "empty_cache")) == {CUDA_1, CUDA_0}, (
            f"the cache was emptied on {_named(cuda, 'empty_cache')}; a split's other card kept its share"
        )
        assert set(_named(cuda, "synchronize")) == {CUDA_1, CUDA_0}
        assert set(_named(cuda, "memory_reserved")) == {CUDA_1, CUDA_0}, "a card was never measured"
        assert set(report["devices"]) == {"cuda:1", "cuda:0"}

    def test_cleanup_detaches_dispatch_hooks_before_moving_the_model(self, cuda):
        model = _DispatchedModel()
        service = ActivationService.__new__(ActivationService)

        service._cleanup_model(model, SPLIT.all_devices)

        assert model.hooked_when_moved is False, (
            "model.cpu() ran with accelerate's hooks still attached"
        )

    def test_a_card_that_kept_its_memory_is_named_even_when_the_other_released(self, monkeypatch, caplog):
        service = ActivationService.__new__(ActivationService)
        after = {"cuda:1": {"allocated": 0.0, "reserved": 0.3}, "cuda:0": {"allocated": 0.0, "reserved": 6.99}}
        monkeypatch.setattr(ActivationService, "_gpu_memory", staticmethod(lambda device: after[str(device)]))
        before = {"cuda:1": {"allocated": 15.0, "reserved": 16.0}, "cuda:0": {"allocated": 7.0, "reserved": 8.0}}

        with caplog.at_level("INFO"):
            report = service._report_cleanup(SPLIT.all_devices, before)

        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1 and warnings[0].startswith("cuda:0 did not give the memory back")
        assert report["released"] == pytest.approx(16.0 - 0.3 + 8.0 - 6.99)
        assert report["after"]["reserved"] == pytest.approx(7.29)


def _extract(service, placement):
    return service.extract_activations(
        model_id="m_1", model_path="/m", architecture="llama",
        quantization=QuantizationFormat.FP16, dataset_path="/d",
        layer_indices=[1], hook_types=["residual"], max_samples=2,
        placement=placement,
    )


class TestTheServiceCleansItsPlacement:
    def test_a_failed_split_load_releases_every_card(self, monkeypatch, cuda, tmp_path):
        """No model handle: the finally's release must still reach both cards."""
        def load_that_died(self, model_path, quantization, placement):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB on the second card")

        monkeypatch.setattr(ActivationService, "_load_model", load_that_died)
        monkeypatch.setattr(ActivationService, "_extraction_dir", lambda self, extraction_id: tmp_path)
        service = ActivationService.__new__(ActivationService)

        with pytest.raises(ActivationExtractionError):
            _extract(service, SPLIT)

        assert set(_named(cuda, "empty_cache")) == {CUDA_1, CUDA_0}, (
            f"a split load that died released only {_named(cuda, 'empty_cache')}"
        )

    def test_the_service_cleans_every_card_of_its_placement(self, monkeypatch, tmp_path):
        cleaned, released = [], []
        model = SimpleNamespace(eval=lambda: None)

        monkeypatch.setattr(ActivationService, "_load_model", lambda self, path, q, placement: (model, object()))
        monkeypatch.setattr(ActivationService, "_extraction_dir", lambda self, extraction_id: tmp_path)
        monkeypatch.setattr(ActivationService, "_load_dataset", lambda self, path, n: [0, 1])
        monkeypatch.setattr(ActivationService, "_run_extraction", lambda self, *a, **k: {})
        monkeypatch.setattr(ActivationService, "_save_activations", lambda self, output_dir, activations: [])
        monkeypatch.setattr(ActivationService, "_calculate_statistics", lambda self, activations, on_progress=None: {})
        monkeypatch.setattr(ActivationService, "_log_gpu_memory", lambda self, stage, devices: None)
        monkeypatch.setattr(ActivationService, "_cleanup_model", lambda self, m, devices: cleaned.append((m, devices)))
        monkeypatch.setattr(ActivationService, "_release_gpu_memory", lambda self, devices: released.append(devices))
        service = ActivationService.__new__(ActivationService)

        _extract(service, SPLIT)

        assert cleaned == [(model, (CUDA_1, CUDA_0))], f"the model was cleaned on {cleaned}"
        assert released == [(CUDA_1, CUDA_0)]


# ── The worker ─────────────────────────────────────────────────────────────

class _Query:
    def __init__(self, result):
        self._result = result

    def __getattr__(self, name):
        return lambda *args, **kwargs: self

    def first(self):
        return self._result


class _SyncDB:
    """Answers queries by model class name; each commit snapshots the row's card columns."""

    def __init__(self, **rows):
        self.rows = rows
        self.committed = []

    def query(self, model):
        return _Query(self.rows.get(model.__name__))

    def commit(self):
        row = self.rows.get("ActivationExtraction")
        self.committed.append(None if row is None else (row.gpu_uuid, row.gpu_uuids))

    def add(self, obj):
        pass

    def refresh(self, obj):
        pass

    def rollback(self):
        pass


@pytest.fixture
def worker(monkeypatch, tmp_path, cuda):
    import transformers

    from src.models.dataset import DatasetStatus
    from src.models.model import ModelStatus
    from src.services.extraction_db_service import ExtractionDatabaseService as Rows
    from src.workers import model_tasks
    from src.workers.base_task import DatabaseTask

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    row = SimpleNamespace(id="ext_1", gpu_uuid=None, gpu_uuids=None, gpu_request=None)
    model_row = SimpleNamespace(
        status=ModelStatus.READY, file_path=str(model_dir), architecture="llama",
        quantization=QuantizationFormat.FP16,
    )
    db = _SyncDB(
        Model=model_row,
        Dataset=SimpleNamespace(status=DatasetStatus.READY, tokenizations=[object()]),
        ActivationExtraction=row,
    )
    h = SimpleNamespace(
        row=row, model_row=model_row, db=db, cuda=cuda, loads=[], created=[], placed=[],
        configs=[], layers=56,
    )

    @contextmanager
    def session():
        yield db

    def config_from_pretrained(source, **kwargs):
        h.configs.append((source, kwargs))
        return _config(h.layers)

    real_place_job = model_tasks.place_job

    def place_job(*args, **kwargs):
        h.placed.append((args, kwargs))
        return real_place_job(*args, **kwargs)

    def create_extraction(**kwargs):
        h.created.append(kwargs)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config_from_pretrained)
    monkeypatch.setattr(model_tasks, "place_job", place_job)
    monkeypatch.setattr(DatabaseTask, "get_db", lambda self: session())
    monkeypatch.setattr(Rows, "get_extraction", staticmethod(lambda _db, extraction_id: row))
    monkeypatch.setattr(Rows, "create_extraction", staticmethod(create_extraction))
    monkeypatch.setattr(Rows, "update_progress", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(Rows, "mark_completed", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(Rows, "mark_failed", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(
        model_tasks, "select_tokenization_for_model",
        lambda *args: SimpleNamespace(tokenized_path=str(dataset_dir), max_length=2048),
    )
    monkeypatch.setattr(model_tasks, "emit_extraction_progress", lambda **kwargs: None)
    monkeypatch.setattr(model_tasks, "emit_extraction_failed", lambda **kwargs: None)
    monkeypatch.setattr(
        model_tasks, "cancel_checker",
        lambda *args, **kwargs: SimpleNamespace(poll_now=lambda: False, reason=None),
    )
    monkeypatch.setattr(model_tasks, "build_extraction_progress_callback", lambda *args: None)
    monkeypatch.setattr(model_tasks, "build_merge_heartbeat", lambda *args: None)
    monkeypatch.setattr(model_tasks, "build_statistics_heartbeat", lambda *args: None)

    class _ActivationService:
        def extract_activations(self, **kwargs):
            h.loads.append({
                "placement": kwargs["placement"],
                "row": (row.gpu_uuid, row.gpu_uuids),
                "committed": list(db.committed),
                "cuda_before_load": list(cuda),
            })
            return {"num_samples": 1, "saved_files": [], "statistics": {}}

    monkeypatch.setattr(model_tasks, "ActivationService", _ActivationService)

    def run(**overrides):
        task_kwargs = dict(
            model_id="m_1", dataset_id="ds_1", layer_indices=[11], hook_types=["residual"],
            max_samples=10, batch_size=8, extraction_id="ext_1",
        )
        task_kwargs.update(overrides)
        return model_tasks.extract_activations.run(**task_kwargs)

    h.run = run
    return h


class TestTheWorkerPlacesASizedJobThatMaySplit:
    def test_a_model_no_card_can_hold_is_split_and_every_card_recorded_before_the_load(self, worker):
        worker.layers = 56
        worker.run(gpu_request="auto")

        assert len(worker.loads) == 1, "the extraction never reached the load"
        load = worker.loads[0]
        placement = load["placement"]
        assert placement.is_shard, f"a ~24 GB model was placed on {placement.describe()}"
        assert placement.all_devices == (CUDA_1, CUDA_0)
        assert placement.max_memory == {1: "21976MiB", 0: "9976MiB"}

        assert load["row"] == (RTX_UUID, [RTX_UUID, TI_UUID]), "the row did not name every card at load time"
        assert (RTX_UUID, [RTX_UUID, TI_UUID]) in load["committed"], "the cards were assigned but never committed"

        before = load["cuda_before_load"]
        assert set(_named(before, "empty_cache")) == {CUDA_1, CUDA_0}, "a card was not cleared before the load"
        assert set(_named(before, "memory_reserved")) == {CUDA_1, CUDA_0}
        # Once before the load and once in the finally, on each card.
        emptied = _named(worker.cuda, "empty_cache")
        assert sorted(map(str, emptied)) == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"], (
            f"the task's release covered {emptied}"
        )

    def test_the_size_is_read_from_the_models_own_files(self, worker):
        worker.run(gpu_request="auto")

        (source, kwargs), = worker.configs
        assert source == worker.model_row.file_path
        assert kwargs == {"local_files_only": True}
        (args, place_kwargs), = worker.placed
        assert args == ("auto",)
        assert place_kwargs["allow_shard"] is True
        assert place_kwargs["required_mb"] == pytest.approx(11_595_153_408 * 2 / 1024**2 + 2048)

    def test_a_model_that_fits_one_card_is_placed_on_one_card(self, worker):
        worker.layers = 2
        worker.run(gpu_request="auto")

        placement = worker.loads[0]["placement"]
        assert not placement.is_shard
        assert placement.device == CUDA_1
        assert worker.loads[0]["row"] == (RTX_UUID, None)

    def test_all_splits_even_a_model_that_fits_one_card(self, worker):
        worker.layers = 2
        worker.run(gpu_request="all")

        placement = worker.loads[0]["placement"]
        assert placement.is_shard and placement.all_devices == (CUDA_1, CUDA_0)
        assert worker.loads[0]["row"] == (RTX_UUID, [RTX_UUID, TI_UUID])

    def test_an_fp32_row_is_sized_at_the_fp16_it_loads_at(self, worker):
        worker.layers = 2
        worker.model_row.quantization = QuantizationFormat.FP32
        worker.run(gpu_request="auto")

        (_, place_kwargs), = worker.placed
        params = 2 * (4 * 4096 * 4096 + 3 * 4096 * 11_008) + 2 * 32_000 * 4096
        assert place_kwargs["required_mb"] == pytest.approx(params * 2 / 1024**2 + 2048)

    def test_a_row_the_worker_creates_records_every_card(self, worker):
        worker.db.rows["ActivationExtraction"] = None
        worker.run(gpu_request="auto", extraction_id=None)

        (created,) = worker.created
        assert created["gpu_uuid"] == RTX_UUID
        assert created["gpu_uuids"] == [RTX_UUID, TI_UUID]
