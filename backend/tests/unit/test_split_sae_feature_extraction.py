"""SAE feature extraction on a model no single card holds: sized, split, recorded and cleaned on every card.

Multi-GPU Phase 2 (0xcc/plans/Multi-GPU-Plan.md). `extract_features_for_sae`
keeps the SAE on the placement's first card and lets the base model span the
placement's cards. That needs, and these tests pin:

* a placement SIZED with the model and the SAE, and allowed to split — Auto
  cannot choose a split without a size;
* both cards recorded (`gpu_uuid` first, `gpu_uuids` all) before anything loads;
* the model loaded with the placement's `device_map` and a budget that leaves
  the SAE's share free on the SAE's card;
* input ids on the embedding's card, not the SAE's;
* memory measured, and the cleanup run, on every card of the split, and a split
  model freed in place rather than copied into host memory.

Nothing here needs a GPU: placement, CUDA, the loader and the database are faked.

MUTATION CONTROLS (2026-09-14; each applied alone, this file +
test_extraction_runs_on_the_chosen_gpu.py run, source restored and checked by
sha256). All went red:
  E1  allow_shard=False                                  -> it_is_placed_..._allowed_to_split, unknown_size
  E2  placed with required_mb=None                       -> it_is_placed_with_the_model_and_the_sae_and_allowed_to_split
  E3  only gpu_uuid recorded, gpu_uuids dropped          -> every_card_is_recorded_before_anything_loads
  E4  finally cleans placement.device (first card only)  -> memory_is_measured_and_released_on_every_card (+ Phase 1 test)
  E5  _cuda_devices_to_clean keeps the first of a split  -> a_cleanup_given_every_card_touches_every_card
  E6  memory synchronized/measured on the first card     -> memory_is_measured_and_released_on_every_card
  E7  loader max_memory=placement.max_memory (no carve)  -> the_sae_stays_on_the_first_card_and_the_model_loads_beside_it
  E8  loader device_map=str(device)                      -> the_sae_stays_on_the_first_card_and_the_model_loads_beside_it
  E9  batch input ids back on `device`                   -> every_batch_goes_to_the_embedding_card
  E10 calibration input ids back on `device`             -> every_batch_goes_to_the_embedding_card
  E11 split model copied to the CPU after sampling       -> a_split_model_is_released_in_place_after_sampling
  E12 cleanup copies a split model to the CPU            -> a_split_model_is_freed_in_place
  E13 model_input_device = device                        -> every_batch_goes_to_the_embedding_card

REVIEW ROUND 2, PLACEMENT + LOADER (2026-09-14). An out-of-memory MODEL LOAD (the loader's
OutOfMemoryError since f6124e22) went through the batch-OOM diagnostics, which replaced its
message with re-tokenize/halve-the-batch advice and cut the loader's figures and remedy to
200 characters. Red on c8fccbcd: test_a_model_load_that_runs_out_of_memory_keeps_the_loaders_message.
  X1  a load OOM treated as a batch OOM again (the isinstance check removed) -> that test, red
The fixture's Model row gained `name` (the ORM row has one): the diagnostics read it, and
without it the test was red for an AttributeError, not for the defect.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.services import extraction_service as module
from src.services.base_model_budget import SAE_WORKING_RESERVE_MB
from src.services.extraction_service import ExtractionService, cleanup_gpu_memory
from src.services.gpu_placement import GpuCard, Placement

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
CUDA_0 = torch.device("cuda", 0)
CUDA_1 = torch.device("cuda", 1)
MiB = 1024**2

CARDS = [
    GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000),
    GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000),
]
#: Most free first, as place_job orders a split: the SAE's card is torch index 1.
SPLIT = Placement(
    card=CARDS[1], device=CUDA_1, cards=(CARDS[1], CARDS[0]), devices=(CUDA_1, CUDA_0),
    max_memory_mb={1: 21_976, 0: 9_976},
)
ONE = Placement(card=CARDS[1], device=CUDA_1)

#: The SAE the fake loader returns: W_enc [latent=8, hidden=4].
SAE_CARD_MB = (2 * 4 * 8 + 4 + 8) * 4 / MiB + SAE_WORKING_RESERVE_MB


class _Query:
    def __init__(self, result):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def populate_existing(self):
        return self

    def first(self):
        return self._result[0] if isinstance(self._result, list) else self._result

    def all(self):
        return self._result if isinstance(self._result, list) else [self._result]


class _DB:
    """Queries answer by model class name; every commit snapshots the job's recorded cards."""

    def __init__(self, **rows):
        self.rows = rows
        self.committed = []

    def query(self, model):
        return _Query(self.rows.get(model.__name__))

    def commit(self):
        job = self.rows["ExtractionJob"]
        self.committed.append((job.gpu_uuid, job.gpu_uuids))

    def rollback(self):
        pass


@pytest.fixture
def cuda_calls(monkeypatch):
    """torch.cuda on a card-less machine; records the device each call names, in order."""
    calls = []

    def recorder(name, result):
        def call(device=None, *args, **kwargs):
            calls.append((name, device))
            return result
        return call

    for name, result in [
        ("memory_allocated", 0), ("memory_reserved", 0), ("synchronize", None),
        ("mem_get_info", (0, 0)), ("reset_peak_memory_stats", None),
    ]:
        monkeypatch.setattr(torch.cuda, name, recorder(name, result))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("empty_cache", None)))
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    return calls


@pytest.fixture
def to_model_load(monkeypatch, cuda_calls):
    """Drive extract_features_for_sae from placement to the base-model load."""
    from src.workers import model_tasks, websocket_emitter

    class _ModelLoadObserved(Exception):
        pass

    h = SimpleNamespace(
        placements=[], sae_moves=[], loads=[], cleanups=[], cuda=cuda_calls, Observed=_ModelLoadObserved,
    )

    class _FakeSae:
        def load_state_dict(self, state):
            pass

        def to(self, device):
            h.sae_moves.append(device)
            return self

        def eval(self):
            return self

    # `max_length` is on the real DatasetTokenization row and the mixture path
    # compares it across corpora — a stub without it fails the uniformity check
    # before the check can express anything.
    tokenization = SimpleNamespace(
        status=module.TokenizationStatus.READY, tokenized_path="/data/tok", max_length=2048
    )

    def load_model_from_hf(**kwargs):
        h.loads.append({**kwargs, "committed": list(h.db.committed)})
        raise _ModelLoadObserved()

    monkeypatch.setattr(ExtractionService, "update_extraction_status_sync", lambda self, *a, **k: None)
    monkeypatch.setattr(websocket_emitter, "emit_progress", lambda **kwargs: None)
    monkeypatch.setattr(
        module, "cleanup_gpu_memory",
        lambda models_to_cleanup=None, context="", device=None: h.cleanups.append(device),
    )
    monkeypatch.setattr(
        module, "load_sae_auto_detect",
        lambda path, device="cpu": ({"W_enc": torch.zeros(8, 4)}, None, "mistudio"),
    )
    monkeypatch.setattr(module, "create_sae", lambda **kwargs: _FakeSae())
    monkeypatch.setattr(model_tasks, "select_tokenization_for_model", lambda c, m, d: tokenization)
    monkeypatch.setattr(module, "load_from_disk", lambda path: [0, 1, 2])
    monkeypatch.setattr(module, "load_model_from_hf", load_model_from_hf)

    def run(placement, params_count=14_000_000_000, quantization="FP16", architecture_config=None,
            raises=_ModelLoadObserved):
        h.job = SimpleNamespace(
            id="extr_2", external_sae_id="sae_b", status="queued", gpu_request="all",
            gpu_uuid=None, gpu_uuids=None, statistics=None,
        )
        h.db = _DB(
            ExtractionJob=h.job,
            ExternalSAE=SimpleNamespace(
                id="sae_b", local_path="/saes/b", layer=11, name="sae b", model_id="m_1",
                model_name=None, architecture="standard", n_features=8, d_model=4,
            ),
            Model=SimpleNamespace(
                id="m_1", name="org/model", repo_id="org/model", quantization=quantization, file_path=None,
                params_count=params_count, architecture_config=architecture_config,
            ),
            # `name` is NOT NULL on the real Dataset row and the mixture path
            # labels each corpus with it; a stub without it is thinner than the
            # row it stands for.
            Dataset=SimpleNamespace(id="ds_1", name="ds one"),
            DatasetTokenization=[tokenization],
        )

        def place_job(requested="auto", required_mb=None, cards=None, allow_shard=False):
            h.placements.append({"requested": requested, "required_mb": required_mb, "allow_shard": allow_shard})
            return placement

        monkeypatch.setattr(module, "place_job", place_job)
        with pytest.raises(raises):
            ExtractionService(h.db).extract_features_for_sae("sae_b", {"dataset_id": "ds_1"})
        return h

    h.run = run
    return h


class TestASplitExtraction:
    def test_it_is_placed_with_the_model_and_the_sae_and_allowed_to_split(self, to_model_load):
        h = to_model_load.run(SPLIT)

        # 14B at FP16: 28.0 GB of weights plus estimate_model_memory's 20%.
        expected = 33_600_000_000 / MiB + SAE_CARD_MB
        assert h.placements == [{"requested": "all", "required_mb": pytest.approx(expected), "allow_shard": True}]

    def test_a_four_bit_model_is_sized_from_its_architecture_not_its_packed_count(self, to_model_load):
        """A Q4 row's params_count was counted off the quantized model: half of every linear weight."""
        from src.ml.model_loader import QuantizationFormat, estimate_model_memory, estimate_parameter_count

        architecture = {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 128_256, "intermediate_size": 14_336}
        described = estimate_parameter_count(SimpleNamespace(**architecture))
        h = to_model_load.run(SPLIT, params_count=described // 2, quantization="Q4", architecture_config=architecture)

        expected = estimate_model_memory(described, QuantizationFormat.Q4) / MiB + SAE_CARD_MB
        assert h.placements[0]["required_mb"] == pytest.approx(expected)

    def test_every_card_is_recorded_before_anything_loads(self, to_model_load):
        h = to_model_load.run(SPLIT)

        assert (h.job.gpu_uuid, h.job.gpu_uuids) == (RTX_UUID, [RTX_UUID, TI_UUID])
        assert (RTX_UUID, [RTX_UUID, TI_UUID]) in h.loads[0]["committed"]

    def test_the_sae_stays_on_the_first_card_and_the_model_loads_beside_it(self, to_model_load):
        h = to_model_load.run(SPLIT)

        assert h.sae_moves == [CUDA_1]
        load = h.loads[0]
        assert load["device_map"] == SPLIT.device_map
        assert load["max_memory"] == {1: f"{21_976 - math.ceil(SAE_CARD_MB)}MiB", 0: "9976MiB"}, (
            "the model's budget on the SAE's card does not leave the SAE's share free"
        )

    def test_memory_is_measured_and_released_on_every_card(self, to_model_load):
        h = to_model_load.run(SPLIT)

        assert [d for name, d in h.cuda if name == "synchronize"] == [CUDA_1, CUDA_0]
        assert [d for name, d in h.cuda if name == "memory_allocated"] == [CUDA_1, CUDA_0]
        assert h.cleanups == [(CUDA_1, CUDA_0)], "the finally cleaned only some of the split's cards"

    def test_one_card_keeps_its_device_map_and_no_budget(self, to_model_load):
        h = to_model_load.run(ONE)

        assert (h.loads[0]["device_map"], h.loads[0]["max_memory"]) == ("cuda:1", None)
        assert (h.job.gpu_uuid, h.job.gpu_uuids) == (RTX_UUID, None)

    def test_a_model_of_unknown_size_is_placed_without_one(self, to_model_load):
        h = to_model_load.run(ONE, params_count=None)

        assert h.placements == [{"requested": "all", "required_mb": None, "allow_shard": True}]


def _extract_function():
    tree = ast.parse(Path(module.__file__).read_text())
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "extract_features_for_sae"
    )


#: What `load_model_from_hf` raises on an out-of-memory load: torch's own message
#: (longer than the 200 characters the batch-size diagnostics keep) and the remedy.
LOAD_OOM = (
    "Out of memory loading org/model at FP16: CUDA out of memory. Tried to allocate 2.00 GiB. "
    "GPU 1 has a total capacity of 23.56 GiB of which 1.20 GiB is free. Of the allocated memory "
    "21.10 GiB is allocated by PyTorch, and 312.00 MiB is reserved by PyTorch but unallocated. If "
    "reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "to avoid fragmentation. Free memory on the GPU(s), choose another GPU, or use a more aggressive "
    "quantization."
)


def test_a_model_load_that_runs_out_of_memory_keeps_the_loaders_message(to_model_load, monkeypatch):
    """Review round 2. An out-of-memory LOAD is not a batch that ran out: the extraction's
    OOM diagnostics told the user to re-tokenize and halve the batch size, and cut the
    loader's message (card, figures, remedy) to 200 characters."""
    from src.ml.model_loader import OutOfMemoryError

    def out_of_memory(**kwargs):
        raise OutOfMemoryError(LOAD_OOM)

    statuses = []
    monkeypatch.setattr(module, "load_model_from_hf", out_of_memory)
    monkeypatch.setattr(
        ExtractionService, "update_extraction_status_sync",
        lambda self, job_id, status, **kwargs: statuses.append((status, kwargs.get("error_message"))),
    )

    to_model_load.run(SPLIT, raises=OutOfMemoryError)

    assert statuses[-1] == (module.ExtractionStatus.FAILED.value, LOAD_OOM)


def test_every_batch_goes_to_the_embedding_card():
    """The CALLS, by the syntax tree: the sampling loop cannot be driven without a model."""
    function = _extract_function()
    assigned = [
        ast.unparse(node.value) for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "model_input_device" for target in node.targets)
    ]
    assert assigned == ["input_device(base_model)"]

    devices = {
        call.args[0].id: ast.unparse(keyword.value)
        for call in ast.walk(function)
        if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "tensor"
        and call.args and isinstance(call.args[0], ast.Name)
        for keyword in call.keywords if keyword.arg == "device"
    }
    for name in ("cal_padded", "cal_masks", "padded_input_ids", "attention_masks"):
        assert devices[name] == "model_input_device", f"{name} is not put on the embedding's card"


def test_a_split_model_is_released_in_place_after_sampling():
    """`base_model.cpu()` on a split copies every layer into host memory to drop it."""
    function = _extract_function()
    branches = [
        node for node in ast.walk(function)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "placement.is_shard"
    ]
    assert len(branches) == 1
    assert [ast.unparse(s) for s in branches[0].body] == ["detach_dispatch_hooks(base_model)"]
    assert [ast.unparse(s) for s in branches[0].orelse] == ["base_model.cpu()"]


class TestCleanupOfASplit:
    def test_a_cleanup_given_every_card_touches_every_card(self, monkeypatch, cuda_calls):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        cleanup_gpu_memory(context="test", device=(CUDA_1, CUDA_0))

        for call in ("synchronize", "reset_peak_memory_stats"):
            assert [d for name, d in cuda_calls if name == call] == [CUDA_1, CUDA_0], call

    def test_a_split_model_is_freed_in_place(self, monkeypatch, cuda_calls):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        model = _Recorded()
        detached = []
        monkeypatch.setattr(module, "cuda_devices", lambda m: [CUDA_1, CUDA_0])
        monkeypatch.setattr(module, "detach_dispatch_hooks", detached.append)

        cleanup_gpu_memory(models_to_cleanup=[model], context="test", device=(CUDA_1, CUDA_0))

        assert detached == [model]
        assert model.cpu_calls == 0, "a split model was copied into host memory to be freed"

    def test_a_single_card_model_is_still_moved_off_its_card(self, monkeypatch, cuda_calls):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        model = _Recorded()
        monkeypatch.setattr(module, "cuda_devices", lambda m: [CUDA_1])
        monkeypatch.setattr(module, "detach_dispatch_hooks", lambda m: pytest.fail("detached a one-card model"))

        cleanup_gpu_memory(models_to_cleanup=[model], context="test", device=CUDA_1)

        assert model.cpu_calls == 1


class _Recorded(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        return self
