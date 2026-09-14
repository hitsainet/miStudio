"""SAE training runs on the GPU the caller chose, and says which one it used.

Submit, row, worker, card:

* ``POST /api/v1/trainings`` resolves ``gpu`` before a row exists: ``"auto"``,
  or the UUID of the named card (an index becomes the UUID it names now). An
  unknown card is a 400 with no row created and nothing queued.
* The task is dispatched with the training id only. The worker reads
  ``gpu_request`` from the ROW, so a resume (which re-dispatches the id) asks
  for the same card, and a NULL on an older row means auto.
* The worker places the job, records ``gpu_uuid`` before anything is loaded, and
  hands ``placement.device`` to the SAEs and to the base-model loader.
* A ``GpuPlacementError`` fails the job with its message. There is no fallback
  to another card and no second attempt.
* Memory estimates are judged against the card the job would use (the named
  one, or the most free for Auto), never against card 0.

Nothing here needs a GPU: the inventory and ``place_job`` are faked.

MUTATION CONTROLS (2026-09-13; each applied alone, this file run, the source
restored byte-identically and checked by sha256). All 13 went red:
  M1  endpoint resolves inside the try (unknown card -> 500)  -> 2 failed (the 400 test)
  M2  service omits gpu_request from the Training row         -> 4 failed (stores the request)
  M3  worker uses AUTO instead of the row's gpu_request       -> 2 failed (row request, refusal)
  M4  worker device back to torch.device("cuda" if ...)       -> 3 failed + test_no_new_hard_coded_gpu
  M5  base-model load device_map back to "auto"               -> 3 failed (uses the placed card)
  M6  worker skips writing gpu_uuid                           -> 3 failed (uses the placed card)
  M7  worker retries place_job(AUTO) after a refusal          -> 1 failed (refusal fails the job)
  M8  spliced-CE loader back to (path, quantization=...)      -> 1 failed (loader signature)
  M9  spliced-CE loader drops device_map                      -> 1 failed (loader signature)
  M10 gpu_memory_view: Auto takes cards[0]                    -> 3 failed (estimates, system resources)
  M11 after_return: memory_allocated() on the current device  -> 1 failed (every card)
  M12 get_system_resources reads gpu_memory_view(0)           -> 1 failed (Auto's card or the named one)
  M13 TrainingCreate drops the gpu field                      -> 7 failed (schema default, create, 400)

PHASE 2 — a base model no single card holds (2026-09-14). On-the-fly training
may split; training on cached activations never does. Each control applied
alone, this file + test_wiring_reachable.py run, source restored and checked by
sha256. All went red:
  T1  allow_shard=False on the on-the-fly path                -> split test + 3 placement tests
  T2  allow_shard=True for every run (cached may split)       -> cached_..._never_placed_split[ids, id]
  T3  placement sized without the base model                  -> split test, model_that_fits_one_card
  T4  only gpu_uuid recorded, gpu_uuids dropped               -> split test
  T5  loader max_memory=placement.max_memory (no SAE carve)   -> split test
  T6  loader device_map=str(device)                           -> split test
  T7  step input ids back on `device`                         -> every_on_the_fly_batch_goes_to_the_embedding_card
  T8  model_input_device = device                             -> every_on_the_fly_batch_goes_to_the_embedding_card
  T9  padding mask from attention_mask_tensor again           -> test_wiring_reachable::..._applies_the_mask
  T10 helper builds the mask on acts_flat.device              -> its_mask_indexes_activations_held_on_another_device
  T11 memory monitor reads (device,) only                     -> the_step_reports_memory_for_every_card
  T12 gpu_memory_across counts the first card only            -> gpu_memory_is_summed_over_every_card
  T13 spliced-CE ids on the first parameter's device          -> the_spliced_ce_inputs_go_to_the_embedding_card
  T14 the filter is called and its result discarded           -> test_r4_survivors::..._indexes_by_the_mask
      (that guard scraped for `acts_flat = kept`, which moved into the helper;
       it now asserts the ASSIGNMENT by the syntax tree)
  (B1/B3 in base_model_budget also turn the split test red; see test_base_model_budget.py.)

"all" AT SUBMIT (review round 1, 2026-09-14). A run on cached activations never
splits, so the endpoint refuses "all" before a row exists — deciding with the
worker's own `extraction_ids_of` — instead of queueing a job the worker refuses
hours later. On the fly, "all" is stored and dispatched. Each control applied
alone, this file run, source restored and checked by sha256:
  K1  `resolve_gpu_request` loses its refusal branch -> all_on_cached_activations_is_a_400_...[ids, id]
  K3  the endpoint passes can_split=True always      -> the same two cases
  K3b the endpoint passes can_split=False always     -> all_on_the_fly_is_stored_and_dispatched
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch
from sqlalchemy import func, select

from src.models.dataset import Dataset
from src.models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from src.models.model import Model
from src.models.training import Training
from src.services import gpu_placement
from src.services.gpu_placement import GpuCard, GpuPlacementError, Placement

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"

# The node since 2026-09-13: the SMALLER card at index 0. A fixture with the
# bigger or freer card first would agree with "use card 0" by construction.
CARDS = [
    GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000),
    GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000),
]

GiB = 1024**3


# ── Submit ─────────────────────────────────────────────────────────────────

def test_the_schema_auto_is_the_placement_auto():
    """schemas.gpu defines AUTO itself to avoid an import cycle; it must not drift."""
    from src.schemas import gpu as gpu_schema
    from src.schemas.training import TrainingCreate

    assert gpu_schema.AUTO == gpu_placement.AUTO
    assert TrainingCreate.model_fields["gpu"].default == gpu_placement.AUTO


@pytest.fixture
def inventory(monkeypatch):
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)


@pytest.fixture
def dispatched(monkeypatch):
    """Every train_sae_task dispatch the endpoint makes."""
    from src.api.v1.endpoints import trainings as endpoint
    from src.services import training_service

    calls = []

    class _Task:
        def apply_async(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(endpoint, "train_sae_task", _Task())
    monkeypatch.setattr(training_service, "_emit_training_event_sync", lambda **kwargs: None)
    return calls


async def _add_model(session):
    from src.models.model import ModelStatus, QuantizationFormat

    session.add(Model(
        id="m_gputest1", name="tiny", architecture="llama", params_count=1_000,
        quantization=QuantizationFormat.FP16, status=ModelStatus.READY,
    ))
    await session.commit()


def _payload(gpu=None):
    body = {
        "model_id": "m_gputest1",
        "dataset_ids": ["ds_gputest1"],
        "hyperparameters": {
            "hidden_dim": 64, "latent_dim": 256, "learning_rate": 1e-3,
            "batch_size": 64, "total_steps": 10,
        },
    }
    if gpu is not None:
        body["gpu"] = gpu
    return body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sent, stored",
    [("1", RTX_UUID), (TI_UUID.lower(), TI_UUID), ("auto", "auto"), (None, "auto")],
)
async def test_create_stores_the_resolved_request_and_dispatches_only_the_id(
    client, async_session, inventory, dispatched, sent, stored
):
    await _add_model(async_session)

    response = await client.post("/api/v1/trainings", json=_payload(sent))

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["gpu_request"] == stored
    assert body["gpu_uuid"] is None  # nothing has started yet

    row = (await async_session.execute(select(Training).where(Training.id == body["id"]))).scalar_one()
    await async_session.refresh(row)
    assert row.gpu_request == stored

    # The worker reads the request from the row, so the dispatch carries the id.
    assert len(dispatched) == 1
    args, kwargs = dispatched[0]
    assert args == ()
    assert kwargs == {"args": [body["id"]], "task_id": row.celery_task_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("sent", ["GPU-00000000-0000-0000-0000-000000000000", "7"])
async def test_an_unknown_card_is_a_400_with_no_row_and_nothing_queued(
    client, async_session, inventory, dispatched, sent
):
    await _add_model(async_session)

    response = await client.post("/api/v1/trainings", json=_payload(sent))

    assert response.status_code == 400, response.text
    assert "No GPU" in response.text and "RTX 3090" in response.text
    count = (await async_session.execute(select(func.count()).select_from(Training))).scalar_one()
    assert count == 0
    assert dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cached", [{"extraction_ids": ["ext_m_gputest1_1"]}, {"extraction_id": "ext_m_gputest1_1"}],
    ids=["ids", "id"],
)
async def test_all_on_cached_activations_is_a_400_with_no_row_and_nothing_queued(
    client, async_session, inventory, dispatched, cached
):
    """A run on cached activations loads no base model and never splits, so the
    worker refuses "all" — hours later, after the job waited its turn. The
    endpoint knows the extractions at submit, so it refuses there."""
    await _add_model(async_session)

    response = await client.post("/api/v1/trainings", json={**_payload("all"), **cached})

    assert response.status_code == 400, response.text
    assert "cannot run split across GPUs" in response.text
    count = (await async_session.execute(select(func.count()).select_from(Training))).scalar_one()
    assert count == 0
    assert dispatched == []


@pytest.mark.asyncio
async def test_all_on_the_fly_is_stored_and_dispatched(client, async_session, inventory, dispatched):
    """The on-the-fly path loads the model and splits: "all" is honoured."""
    await _add_model(async_session)

    response = await client.post("/api/v1/trainings", json=_payload("all"))

    assert response.status_code == 201, response.text
    assert response.json()["gpu_request"] == "all"
    assert len(dispatched) == 1


# ── Start ──────────────────────────────────────────────────────────────────

class _Stop(Exception):
    """Raised by the fake loader: the test has seen what it needs."""


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter_by(self, **criteria):
        return _Query(r for r in self._rows if all(getattr(r, k, None) == v for k, v in criteria.items()))

    def filter(self, *conditions):
        # One row per model class in these fixtures, so the condition is moot.
        return self

    def order_by(self, *columns):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _Session:
    """A sync session over in-memory rows that remembers what each commit held."""

    def __init__(self, rows, training):
        self._rows = rows
        self._training = training
        self.committed_gpu_uuids = []
        self.committed_gpu_uuid_lists = []

    def query(self, model):
        return _Query(self._rows.get(model, []))

    def commit(self):
        self.committed_gpu_uuids.append(self._training.gpu_uuid)
        self.committed_gpu_uuid_lists.append(getattr(self._training, "gpu_uuids", None))

    def rollback(self):
        pass

    def add(self, obj):
        pass

    def close(self):
        pass


class _SAE(torch.nn.Module):
    """Stands in for create_sae's module and records where it was moved."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.moved_to = []

    def to(self, device):  # noqa: D102 - records instead of moving
        self.moved_to.append(device)
        return self


@pytest.fixture
def worker(monkeypatch, tmp_path):
    """Drive train_sae_task on the on-the-fly path until the base-model load."""
    from src.workers import base_task, training_tasks, websocket_emitter

    training = SimpleNamespace(
        id="train_gpu1", model_id="m_x", status="pending", current_step=0, current_loss=None,
        dataset_id="ds_x", dataset_ids=["ds_x"], extraction_id=None, extraction_ids=None,
        hyperparameters={
            "hidden_dim": 8, "latent_dim": 16, "batch_size": 64, "learning_rate": 1e-3,
            "total_steps": 10, "seed": 7, "training_layers": [0], "hook_types": ["residual"],
            "architecture_type": "standard", "l1_alpha": 1e-3,
        },
        gpu_request=RTX_UUID, gpu_uuid=None, gpu_uuids=None,
        checkpoint_dir=None, error_message=None, error_traceback=None, completed_at=None,
    )
    rows = {
        Training: [training],
        Model: [SimpleNamespace(id="m_x", repo_id="org/tiny", quantization="FP16", file_path=None)],
        Dataset: [SimpleNamespace(id="ds_x")],
        DatasetTokenization: [SimpleNamespace(
            status=TokenizationStatus.READY, tokenized_path=str(tmp_path / "tok"),
            tokenizer_repo_id="org/tiny", vocab_size=100,
        )],
    }
    session = _Session(rows, training)

    @contextmanager
    def get_sync_db():
        yield session

    state = SimpleNamespace(
        training=training, session=session, estimates=[], placements=[], allow_shard=[], saes=[], loads=[],
        placement=Placement(card=CARDS[1], device=torch.device("cuda", 1)),
    )

    def estimate(**kwargs):
        state.estimates.append(kwargs)
        return {"total_gb": 0.5, "total_mb": 512.0, "fits_in_6gb": True, "available_gpu_gb": 20.0}

    def place(requested, required_mb=None, cards=None, allow_shard=False):
        state.placements.append((requested, required_mb))
        state.allow_shard.append(allow_shard)
        if isinstance(state.placement, Exception):
            raise state.placement
        return state.placement

    def create_sae(**kwargs):
        sae = _SAE()
        state.saes.append(sae)
        return sae

    def loader(**kwargs):
        state.loads.append({
            **kwargs,
            "committed_gpu_uuids": list(session.committed_gpu_uuids),
            "committed_gpu_uuid_lists": list(session.committed_gpu_uuid_lists),
        })
        raise _Stop("stopped at the base-model load")

    monkeypatch.setattr(base_task, "get_sync_db", get_sync_db)
    monkeypatch.setattr(training_tasks, "estimate_training_memory", estimate)
    monkeypatch.setattr(training_tasks, "place_job", place)
    monkeypatch.setattr(training_tasks, "create_sae", create_sae)
    monkeypatch.setattr(training_tasks, "load_model_from_hf", loader)
    monkeypatch.setattr(training_tasks, "select_tokenization_for_model", lambda c, m, d: c[0])
    monkeypatch.setattr(training_tasks, "load_from_disk", lambda path: [0, 0, 0, 0])
    monkeypatch.setattr(
        training_tasks.TrainingValidator, "validate_sparsity_config", staticmethod(lambda hp: ([], []))
    )
    monkeypatch.setattr(training_tasks.settings, "data_dir", tmp_path)
    monkeypatch.setattr(websocket_emitter, "emit_training_progress", lambda *a, **k: True)

    state.run = lambda: training_tasks.train_sae_task.run(training.id)
    return state


@pytest.mark.parametrize("on_row, asked", [(RTX_UUID, RTX_UUID), ("auto", "auto"), (None, "auto")])
def test_the_worker_places_the_row_request_and_uses_the_placed_card(worker, on_row, asked):
    worker.training.gpu_request = on_row

    with pytest.raises(_Stop):
        worker.run()

    # The row's request, with the estimate as the memory the card must have.
    assert worker.placements == [(asked, 512.0)]
    assert [e["gpu"] for e in worker.estimates] == [asked]

    # Recorded and COMMITTED before the base model was loaded.
    assert worker.training.gpu_uuid == RTX_UUID
    assert len(worker.loads) == 1
    assert RTX_UUID in worker.loads[0]["committed_gpu_uuids"]

    # The placed device, not a bare "cuda" and not "auto".
    assert [sae.moved_to for sae in worker.saes] == [[torch.device("cuda", 1)]]
    assert worker.loads[0]["device_map"] == "cuda:1"
    # One card: no split budget, and nothing recorded as a split. The job loads
    # a base model, so it was allowed to split had no card fitted.
    assert worker.loads[0]["max_memory"] is None
    assert worker.training.gpu_uuids is None
    assert worker.allow_shard == [True]


def test_a_training_deleted_while_queued_stops_instead_of_crashing(worker, monkeypatch):
    """The row vanishes between placement and recording the card.

    Before 2026-09-14 the worker wrote `gpu_uuid` onto None and died with
    AttributeError; now a deleted row is a stop. See
    test_training_deleted_row_stops.py for the step loop and the hardware finding.
    """
    from src.workers import training_tasks

    placed = training_tasks.place_job

    def place_then_delete(*args, **kwargs):
        placement = placed(*args, **kwargs)
        worker.session._rows[Training] = []
        return placement

    monkeypatch.setattr(training_tasks, "place_job", place_then_delete)

    result = worker.run()

    assert result == {"status": "cancelled", "step": 0, "reason": "deleted"}
    assert worker.loads == [], "a deleted training must not go on to load the base model"


def test_a_placement_refusal_fails_the_job_with_its_message(worker):
    message = (
        "GPU 0 (NVIDIA GeForce RTX 3080 Ti, 11,000 of 12,288 MB free) cannot take this job: "
        "it needs ~20,000 MB. Choose another GPU or Auto."
    )
    worker.training.gpu_request = TI_UUID
    worker.placement = GpuPlacementError(message, requested=TI_UUID, required_mb=20_000, cards=CARDS)

    with pytest.raises(GpuPlacementError):
        worker.run()

    assert worker.placements == [(TI_UUID, 512.0)]  # asked once; no other card tried
    assert worker.training.status == "failed"
    assert worker.training.error_message == message
    assert worker.training.gpu_uuid is None
    assert worker.saes == [] and worker.loads == []


def test_the_spliced_ce_loader_uses_the_loader_signature_and_the_placed_card(monkeypatch, tmp_path):
    """The loader is autospecced, so a keyword it does not accept is a TypeError."""
    from src.ml import layer_discovery, model_loader
    from src.ml.model_loader import QuantizationFormat
    from src.workers import training_tasks

    loaded = SimpleNamespace(eval=lambda: None, device=torch.device("cpu"))
    loader = create_autospec(model_loader.load_model_from_hf, return_value=(loaded, None, None, {}))
    monkeypatch.setattr(training_tasks, "load_model_from_hf", loader)
    monkeypatch.setattr(layer_discovery, "discover_transformer_structure", lambda model: object())

    rows = {
        Training: [SimpleNamespace(id="t1", model_id="m_x", gpu_uuid=None)],
        Model: [SimpleNamespace(id="m_x", repo_id="org/tiny", quantization="Q8", file_path=str(tmp_path))],
    }
    session = _Session(rows, rows[Training][0])

    @contextmanager
    def get_db():
        yield session

    training_tasks._evaluate_spliced_ce(
        SimpleNamespace(get_db=get_db), training_id="t1", hp={}, models={},
        layer_hook_combinations=[], base_model=None, tokenizer=object(), dataset=[],
        architecture=None, device=torch.device("cuda", 1), step=10,
    )

    loader.assert_called_once_with(
        repo_id="org/tiny",
        quant_format=QuantizationFormat.Q8,
        cache_dir=tmp_path,
        device_map="cuda:1",
        local_files_only=True,
    )


def test_after_return_reports_memory_for_every_card(monkeypatch):
    from src.workers import training_tasks

    seen = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: seen.append(("allocated", device)) or 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: seen.append(("reserved", device)) or 0)

    training_tasks.train_sae_task.after_return("SUCCESS", None, "task-1", (), {}, None)

    assert sorted(seen) == [("allocated", 0), ("allocated", 1), ("reserved", 0), ("reserved", 1)]


# ── Estimates ──────────────────────────────────────────────────────────────

@pytest.fixture
def cuda_cards(monkeypatch):
    """Torch and NVML both see CARDS; this process holds 1 GiB on card 0 and 2 GiB on card 1."""
    uuids = [TI_UUID, RTX_UUID]
    reserved = {0: 1 * GiB, 1: 2 * GiB}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda index: SimpleNamespace(
            uuid=uuids[index][len("GPU-"):], name=CARDS[index].name,
            total_memory=CARDS[index].total_mb * 1024**2,
        ),
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: reserved[index])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: reserved[index] // 2)
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)


def _free_gb(card):
    return round(card.free_mb / 1024, 2)


def test_the_training_estimate_is_judged_against_the_card_the_job_would_use(cuda_cards):
    from src.utils.resource_estimation import estimate_training_memory

    size = dict(hidden_dim=4096, latent_dim=78_750, batch_size=1024)
    auto = estimate_training_memory(**size, gpu="auto")
    ti = estimate_training_memory(**size, gpu=TI_UUID)

    # Sized between the two cards, or both cards would give the same verdict.
    assert _free_gb(CARDS[0]) < auto["total_gb"] < _free_gb(CARDS[1])
    assert auto["available_gpu_gb"] == _free_gb(CARDS[1])  # Auto: the most free, not index 0
    assert auto["fits_in_6gb"] is True
    assert ti["available_gpu_gb"] == _free_gb(CARDS[0])
    assert ti["fits_in_6gb"] is False
    assert estimate_training_memory(**size, gpu=torch.device("cuda", 0))["available_gpu_gb"] == _free_gb(CARDS[0])


def test_the_multilayer_estimate_passes_the_card_through(cuda_cards):
    from src.utils.resource_estimation import estimate_multilayer_training_memory

    size = dict(hidden_dim=1024, latent_dim=16_384, batch_size=1024, num_layers=3)
    assert estimate_multilayer_training_memory(**size, gpu=TI_UUID)["available_gpu_gb"] == _free_gb(CARDS[0])
    assert estimate_multilayer_training_memory(**size)["available_gpu_gb"] == _free_gb(CARDS[1])


def test_without_nvml_the_estimate_still_considers_every_card(cuda_cards, monkeypatch):
    from src.utils.resource_estimation import gpu_memory_view

    monkeypatch.setattr(gpu_placement, "list_cards", lambda: [])
    view = gpu_memory_view()
    assert view["index"] == 1
    assert view["free_bytes"] == CARDS[1].total_mb * 1024**2 - 2 * GiB


def test_without_cuda_the_estimate_keeps_its_cpu_fallback(monkeypatch):
    from src.services.resource_config import ResourceConfig
    from src.utils.resource_estimation import available_gpu_gb_for

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert available_gpu_gb_for("auto") == 6.0
    assert ResourceConfig.get_system_resources()["gpu_available"] is False


def test_system_resources_describe_auto_s_card_or_the_one_named(cuda_cards):
    from src.services.resource_config import ResourceConfig

    auto = ResourceConfig.get_system_resources()
    assert (auto["gpu_index"], auto["gpu_uuid"], auto["gpu_name"]) == (1, RTX_UUID, CARDS[1].name)
    assert auto["gpu_memory_available_gb"] == CARDS[1].free_mb / 1024
    assert auto["gpu_memory_reserved_gb"] == 2.0  # this process, on THAT card
    assert auto["gpu_total_memory_gb"] == CARDS[1].total_mb / 1024

    named = ResourceConfig.get_system_resources(TI_UUID)
    assert (named["gpu_index"], named["gpu_memory_reserved_gb"]) == (0, 1.0)
    assert named["gpu_memory_available_gb"] == CARDS[0].free_mb / 1024


@pytest.mark.parametrize("device_type, seconds", [("gpu", 0.8), ("cuda", 0.8), ("cpu", 8.0)])
def test_processing_time_is_a_timing_class_not_a_device(device_type, seconds):
    from src.utils.resource_estimation import estimate_processing_time

    result = estimate_processing_time(
        num_samples=100, num_layers=1, batch_size=10, model_params_count=100_000_000,
        device_type=device_type,
    )
    assert result["seconds_per_batch"] == seconds
    assert estimate_processing_time(
        num_samples=100, num_layers=1, batch_size=10, model_params_count=100_000_000
    )["seconds_per_batch"] == 0.8


# ── Phase 2: a base model no single card holds ─────────────────────────────

MiB = 1024**2

#: A split as place_job builds it: most free first, so the SAEs' card is the
#: 3090 at TORCH index 1. A fixture whose first card were index 0 could not tell
#: "the SAEs' card" from "card 0".
SPLIT = Placement(
    card=CARDS[1], device=torch.device("cuda", 1),
    cards=(CARDS[1], CARDS[0]), devices=(torch.device("cuda", 1), torch.device("cuda", 0)),
    max_memory_mb={1: 21_976, 0: 9_976},
)


def test_on_the_fly_training_is_sized_with_its_model_and_loads_split_beside_the_saes(worker):
    """The on-the-fly path loads a base model, so it is placed with the model's size.

    Without the size, Auto cannot choose a split: it takes the freest card and the
    load fails there. On a split the SAEs (and their optimizer and batches) stay on
    the first card, so the model's budget on that card leaves their share free.
    """
    worker.session._rows[Model][0].params_count = 14_000_000_000
    worker.placement = SPLIT

    with pytest.raises(_Stop):
        worker.run()

    # 14B at FP16 is 28.0 GB of weights; estimate_model_memory adds 20%.
    assert worker.placements == [(RTX_UUID, 512.0 + 33_600_000_000 / MiB)]
    assert worker.allow_shard == [True]

    load = worker.loads[0]
    assert load["device_map"] == SPLIT.device_map
    assert load["max_memory"] == {1: f"{21_976 - 512}MiB", 0: "9976MiB"}, (
        "the model's budget on the SAEs' card does not leave the SAEs' 512 MB free"
    )
    assert [sae.moved_to for sae in worker.saes] == [[torch.device("cuda", 1)]]

    assert (worker.training.gpu_uuid, worker.training.gpu_uuids) == (RTX_UUID, [RTX_UUID, TI_UUID])
    assert [RTX_UUID, TI_UUID] in load["committed_gpu_uuid_lists"], (
        "the split's cards were not committed before the model load"
    )


def test_a_model_that_fits_one_card_is_still_sized_with_it(worker):
    worker.session._rows[Model][0].params_count = 1_200_000_000

    with pytest.raises(_Stop):
        worker.run()

    assert worker.placements == [(RTX_UUID, 512.0 + 2_880_000_000 / MiB)]
    assert (worker.loads[0]["device_map"], worker.loads[0]["max_memory"]) == ("cuda:1", None)


def test_a_four_bit_base_model_is_sized_from_its_architecture_not_its_packed_count(worker):
    """A Q4 row's params_count was counted off the quantized model: half of every linear weight."""
    from src.ml.model_loader import QuantizationFormat, estimate_model_memory, estimate_parameter_count

    architecture = {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 128_256, "intermediate_size": 14_336}
    described = estimate_parameter_count(SimpleNamespace(**architecture))
    row = worker.session._rows[Model][0]
    row.params_count, row.quantization, row.architecture_config = described // 2, "Q4", architecture

    with pytest.raises(_Stop):
        worker.run()

    expected = 512.0 + estimate_model_memory(described, QuantizationFormat.Q4) / MiB
    assert worker.placements == [(RTX_UUID, pytest.approx(expected))]


@pytest.mark.parametrize(
    "cached", [{"extraction_ids": ["ext_1"]}, {"extraction_id": "ext_1"}], ids=["ids", "id"],
)
def test_training_on_cached_activations_is_never_placed_split(worker, cached):
    """It loads no base model: SAEs, optimizer and rolling buffer live on one card.

    The row's model has a size here on purpose — a cached run sized with a model
    it never loads would claim memory it does not use.
    """
    worker.session._rows[Model][0].params_count = 14_000_000_000
    for field, value in cached.items():
        setattr(worker.training, field, value)
    worker.placement = _Stop("placed")

    with pytest.raises(_Stop):
        worker.run()

    assert worker.allow_shard == [False], "a cached-activation training was allowed to split"
    assert worker.placements == [(RTX_UUID, 512.0)]


class TestRealTokenActivations:
    """The on-the-fly step's padding filter, and the device its mask is on."""

    def test_it_keeps_only_real_positions(self):
        from src.workers.training_tasks import real_token_activations

        acts = torch.arange(8.0).reshape(4, 2)
        kept = real_token_activations(acts, [[1, 0], [1, 1]], step=1)
        assert kept.tolist() == [[0.0, 1.0], [4.0, 5.0], [6.0, 7.0]]

    def test_its_mask_indexes_activations_held_on_another_device(self):
        """`meta` stands in for a card the activations are held on."""
        from src.workers.training_tasks import real_token_activations

        acts = torch.empty(4, 2, device="meta")
        kept = real_token_activations(acts, [[1, 0], [1, 1]], step=1)
        assert (tuple(kept.shape), kept.device.type) == ((3, 2), "meta")

    def test_a_mask_on_another_device_cannot_index_the_captured_activations(self):
        """Why the tensor handed to the model is not reused: it is on the model's
        input card while HookManager keeps activations on the CPU."""
        acts = torch.arange(8.0).reshape(4, 2)
        on_the_models_card = torch.tensor([[1, 0], [1, 1]], device="meta").reshape(-1).bool()
        with pytest.raises((RuntimeError, NotImplementedError)):
            acts[on_the_models_card]

    def test_a_mask_that_does_not_match_is_not_applied(self):
        from src.workers.training_tasks import real_token_activations

        acts = torch.arange(8.0).reshape(4, 2)
        assert torch.equal(real_token_activations(acts, [[1, 0, 1]], step=1), acts)

    def test_an_all_padding_batch_trains_on_the_raw_batch(self):
        from src.workers.training_tasks import real_token_activations

        acts = torch.arange(8.0).reshape(4, 2)
        assert torch.equal(real_token_activations(acts, [[0, 0], [0, 0]], step=1), acts)


def test_gpu_memory_is_summed_over_every_card(monkeypatch):
    from src.workers.training_tasks import gpu_memory_across

    allocated = {0: 3 * MiB, 1: 5 * MiB}
    reserved = {0: 7 * MiB, 1: 11 * MiB}
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: allocated[torch.device(d).index])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: reserved[torch.device(d).index])

    devices = (torch.device("cuda", 1), torch.device("cuda", 0), torch.device("cpu"))
    assert gpu_memory_across(devices) == (8.0, 18.0)


def _train_task_tree():
    import ast
    from pathlib import Path

    from src.workers import training_tasks

    tree = ast.parse(Path(training_tasks.__file__).read_text())
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_sae_task"
    )


def _calls_to(tree, name):
    import ast

    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) == name or getattr(node.func, "attr", None) == name)
    ]


def test_every_on_the_fly_batch_goes_to_the_embedding_card():
    """The CALLS, by the syntax tree: the step loop cannot be driven without a model."""
    import ast

    task = _train_task_tree()
    assigned = [
        node.value for node in ast.walk(task)
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "model_input_device" for target in node.targets)
    ]
    assert [ast.unparse(value) for value in assigned] == ["input_device(base_model)"]

    devices = {
        call.args[0].id: ast.unparse(keyword.value)
        for call in _calls_to(task, "tensor")
        if call.args and isinstance(call.args[0], ast.Name)
        for keyword in call.keywords if keyword.arg == "device"
    }
    assert devices["padded_input_ids"] == "model_input_device"
    assert devices["attention_masks"] == "model_input_device"


def test_the_step_reports_memory_for_every_card_of_the_placement():
    import ast

    calls = _calls_to(_train_task_tree(), "gpu_memory_across")
    assert [[ast.unparse(arg) for arg in call.args] for call in calls] == [["placement.all_devices"]]


class _CeRows:
    """A tokenized dataset with only what the CE evaluation reads."""

    column_names = ["input_ids", "attention_mask"]

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def select(self, indices):
        return _CeRows([self._rows[i] for i in indices])

    def __getitem__(self, column):
        return [row[column] for row in self._rows]


def test_the_spliced_ce_inputs_go_to_the_embedding_card(monkeypatch):
    from src.ml import layer_discovery
    from src.services import sae_evaluation
    from src.workers import training_tasks

    class _SplitModel(torch.nn.Module):
        """The first registered parameter is on "another card" (meta); the embedding is not."""

        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 2, device="meta")
            self.embed = torch.nn.Embedding(5, 2)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

        def get_input_embeddings(self):
            return self.embed

    seen = []

    def measure(model, sae, layer, ids, mask):
        seen.append((ids.device, mask.device))
        return dict(ce_baseline=1.0, ce_spliced=1.0, ce_ablated=2.0, ce_delta=0.0, loss_recovered=1.0)

    monkeypatch.setattr(
        layer_discovery, "discover_transformer_structure",
        lambda model: SimpleNamespace(layers_module=model.layers),
    )
    monkeypatch.setattr(sae_evaluation, "spliced_ce_delta", measure)

    training_tasks._evaluate_spliced_ce(
        SimpleNamespace(log_metric=lambda **kwargs: None), training_id="t1", hp={},
        models={(0, "residual"): object()}, layer_hook_combinations=[(0, "residual")],
        base_model=_SplitModel(), tokenizer=object(),
        dataset=_CeRows([{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 0]}]),
        architecture=None, device=torch.device("cuda", 1), step=10,
    )

    assert seen == [(torch.device("cpu"), torch.device("cpu"))], (
        "the CE inputs went to the first parameter's device, not the embedding's"
    )
