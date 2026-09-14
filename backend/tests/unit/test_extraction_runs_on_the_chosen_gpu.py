"""Activation extraction and SAE feature extraction run on the chosen GPU.

The node gained a second card on 2026-09-13 and the new RTX 3080 Ti took
index 0. Both extraction kinds had been written for one card:

  * activation extraction took `gpu_id: int = 0`. The frontend had already
    switched to sending `gpu` ("auto" or a UUID), so the schema dropped it as
    an unknown field and every extraction ran on index 0, the 12 GB card;
  * SAE feature extraction used `device = "cuda"`, the current device (index
    0 again), and measured memory with `memory_allocated(0)`.

Each job now stores what was asked for (`gpu_request`) when it is submitted, is
placed by `gpu_placement.place_job` when it starts, records the card it got
(`gpu_uuid`) before anything loads, and hands that device to the loader.

Everything is driven through the real endpoints and the real worker bodies;
only the edges are faked (NVML, CUDA, the broker, the database). This
workstation has no GPU.

MUTATION CONTROLS (each applied, run red, then restored byte-identically):
  C1  models.py      dispatch `gpu_request="auto"` instead of the resolved value
  C2  models.py      retry resolves `AUTO` instead of the stored request
  C3  model_tasks.py hand the service `torch.device("cuda", 0)`
  C4  model_tasks.py drop the `record_gpu_uuid` call
  C5  model_tasks.py drop `GpuPlacementError` from the do-not-retry condition
  C6  model_tasks.py drop the `Retry` re-raise in the outer handler
  C7  saes.py        single extraction stops passing `gpu_request`
  C8  saes.py        batch extraction stops passing `gpu_request`
  C9  extraction_service.py  place from `config.get("gpu")` instead of the row
  C10 extraction_service.py  load the SAE on `"cuda"`
  C11 extraction_service.py  stop recording `gpu_uuid` on the job row
  C12 features.py    synchronize the current device
  C13 extraction_service.py  batch rows stop storing `gpu_request`
  C14 activation_service.py  `_load_model` is handed `torch.device("cuda", 0)`
  C15 activation_service.py  `from_pretrained(device_map={"": "cuda:0"})`
All fifteen went red against their named test and were restored (2026-09-13).
"""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from fastapi import HTTPException

from src.services import gpu_placement
from src.services.gpu_placement import GpuCard, GpuPlacementError, Placement

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
UNKNOWN_UUID = "GPU-00000000-0000-0000-0000-000000000000"

CARDS = [
    GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000),
    GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000),
]
TI = Placement(card=CARDS[0], device=torch.device("cuda", 0))
RTX = Placement(card=CARDS[1], device=torch.device("cuda", 1))
REFUSAL = f"{CARDS[0].describe()} cannot take this job: it needs ~15,000 MB. Choose another GPU or Auto."


# ── Shared fakes ───────────────────────────────────────────────────────────

@pytest.fixture
def inventory(monkeypatch):
    """The node as NVML reports it: the 3080 Ti at index 0, the 3090 at 1."""
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)


@pytest.fixture
def cuda_calls(monkeypatch):
    """torch.cuda on a card-less machine; records the device each call names."""
    calls = []

    def recorder(name, result):
        def call(device=None, *args, **kwargs):
            calls.append((name, device))
            return result
        return call

    for name, result in [
        ("memory_allocated", 0),
        ("memory_reserved", 0),
        ("synchronize", None),
        ("mem_get_info", (0, 0)),
        ("reset_peak_memory_stats", None),
    ]:
        monkeypatch.setattr(torch.cuda, name, recorder(name, result))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("empty_cache", None)))
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)

    @contextmanager
    def device(target):
        calls.append(("device", target))
        yield

    monkeypatch.setattr(torch.cuda, "device", device)
    return calls


def _placer(result):
    """A fake `place_job`: records each request, then returns or raises `result`."""
    requests = []

    def place_job(requested="auto", required_mb=None, cards=None, allow_shard=False):
        requests.append(requested)
        if isinstance(result, BaseException):
            raise result
        return result

    place_job.requests = requests
    return place_job


class _Query:
    def __init__(self, result):
        self._result = result

    def filter_by(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def options(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def populate_existing(self):
        return self

    def with_for_update(self, **kwargs):
        return self

    def first(self):
        return self._result


class _SyncDB:
    """A sync session whose queries answer by model class name.

    Every commit snapshots each row's `gpu_uuid`, so a test can tell a value
    that was committed from one that was merely assigned.
    """

    def __init__(self, **rows_by_class):
        self.rows = rows_by_class
        self.committed = []

    def query(self, model):
        return _Query(self.rows.get(model.__name__))

    def commit(self):
        self.committed.append({name: getattr(row, "gpu_uuid", None) for name, row in self.rows.items()})

    def add(self, obj):
        pass

    def refresh(self, obj):
        pass

    def rollback(self):
        pass


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _AsyncDB:
    """An AsyncSession that answers `execute` from a queue, in order."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.added = []

    async def execute(self, statement):
        return _Result(self.answers.pop(0))

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass

    async def refresh(self, obj):
        pass


# ── Activation extraction: submit ──────────────────────────────────────────

@pytest.fixture
def models_endpoint(monkeypatch, inventory):
    from src.api.v1.endpoints import models as endpoint
    from src.models.model import ModelStatus
    from src.workers import websocket_emitter

    harness = SimpleNamespace(
        endpoint=endpoint,
        db=MagicMock(name="sync_db"),
        delay=MagicMock(name="extract_activations.delay", return_value=SimpleNamespace(id="celery-1")),
        sessions=0,
    )

    async def get_model(_db, model_id):
        return SimpleNamespace(id=model_id, status=ModelStatus.READY)

    @contextmanager
    def get_sync_db():
        harness.sessions += 1
        yield harness.db

    monkeypatch.setattr(endpoint.ModelService, "get_model", staticmethod(get_model))
    monkeypatch.setattr(endpoint, "get_sync_db", get_sync_db)
    monkeypatch.setattr(endpoint.extract_activations, "delay", harness.delay)
    monkeypatch.setattr(websocket_emitter, "emit_extraction_progress", lambda **kwargs: None)
    return harness


def _activation_request(gpu=None):
    from src.schemas.model import ActivationExtractionRequest

    fields = dict(dataset_id="ds_1", layer_indices=[11], hook_types=["residual"], max_samples=10)
    if gpu is not None:
        fields["gpu"] = gpu
    return ActivationExtractionRequest(**fields)


def _rows_written(db):
    return [call.args[0] for call in db.add.call_args_list]


def _original_extraction(gpu_request):
    # `gpu_id=1` is the legacy index of the 3090. A retry that still read it
    # would answer RTX_UUID, which differs from every expected value below.
    return SimpleNamespace(
        id="ext_old", model_id="m_1", dataset_id="ds_1", layer_indices=[11],
        hook_types=["residual"], batch_size=8, micro_batch_size=None, max_samples=10,
        gpu_request=gpu_request, gpu_id=1,
    )


class TestActivationExtractionIsSubmittedWithItsCard:
    def test_a_named_card_is_stored_as_its_uuid_and_handed_to_the_task(self, models_endpoint):
        h = models_endpoint
        response = asyncio.run(
            h.endpoint.extract_model_activations("m_1", _activation_request("1"), db=None)
        )

        rows = _rows_written(h.db)
        assert len(rows) == 1
        assert rows[0].gpu_request == RTX_UUID, (
            "the row does not record the card that was asked for; index 1 must "
            "be stored as the 3090's UUID, which survives a renumbering"
        )
        assert h.delay.call_count == 1
        sent = h.delay.call_args.kwargs
        assert sent["gpu_request"] == RTX_UUID
        assert "gpu_id" not in sent
        assert sent["extraction_id"] == rows[0].id == response["extraction_id"]

    def test_no_choice_is_stored_and_sent_as_auto(self, models_endpoint):
        h = models_endpoint
        asyncio.run(h.endpoint.extract_model_activations("m_1", _activation_request(), db=None))

        assert [row.gpu_request for row in _rows_written(h.db)] == ["auto"]
        assert h.delay.call_count == 1
        assert h.delay.call_args.kwargs["gpu_request"] == "auto"

    def test_an_unknown_card_is_a_400_and_nothing_is_written_or_queued(self, models_endpoint):
        h = models_endpoint
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                h.endpoint.extract_model_activations("m_1", _activation_request(UNKNOWN_UUID), db=None)
            )

        assert exc.value.status_code == 400
        assert "No GPU" in exc.value.detail
        assert h.db.add.call_count == 0
        assert h.delay.call_count == 0
        assert h.sessions == 0, "a database session was opened for a request that was refused"

    @pytest.mark.parametrize("stored, expected", [(TI_UUID, TI_UUID), (None, "auto")])
    def test_a_retry_reuses_the_stored_request_not_the_legacy_index(self, models_endpoint, stored, expected):
        from src.schemas.model import ExtractionRetryRequest

        h = models_endpoint
        h.db.query.return_value.filter.return_value.first.return_value = _original_extraction(stored)

        asyncio.run(h.endpoint.retry_extraction("m_1", "ext_old", ExtractionRetryRequest(), db=None))

        assert [row.gpu_request for row in _rows_written(h.db)] == [expected]
        assert h.delay.call_count == 1
        assert h.delay.call_args.kwargs["gpu_request"] == expected

    def test_a_retry_whose_card_has_left_the_node_is_a_400(self, models_endpoint):
        from src.schemas.model import ExtractionRetryRequest

        h = models_endpoint
        h.db.query.return_value.filter.return_value.first.return_value = _original_extraction(UNKNOWN_UUID)

        with pytest.raises(HTTPException) as exc:
            asyncio.run(h.endpoint.retry_extraction("m_1", "ext_old", ExtractionRetryRequest(), db=None))

        assert exc.value.status_code == 400
        assert h.db.add.call_count == 0
        assert h.delay.call_count == 0


# ── Activation extraction: the worker ──────────────────────────────────────

@pytest.fixture
def activation_worker(monkeypatch, tmp_path, cuda_calls):
    from celery.exceptions import Retry

    from src.models.dataset import DatasetStatus
    from src.models.model import ModelStatus, QuantizationFormat
    from src.services.extraction_db_service import ExtractionDatabaseService as Rows
    from src.workers import model_tasks
    from src.workers.base_task import DatabaseTask

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    row = SimpleNamespace(id="ext_1", gpu_uuid=None, gpu_request=None)
    db = _SyncDB(
        Model=SimpleNamespace(
            status=ModelStatus.READY, file_path=str(model_dir),
            architecture="lfm2", quantization=QuantizationFormat.FP16,
        ),
        Dataset=SimpleNamespace(status=DatasetStatus.READY, tokenizations=[object()]),
        ActivationExtraction=row,
    )
    h = SimpleNamespace(row=row, db=db, cuda=cuda_calls, failed=[], retries=[], loads=[], service_error=None)

    @contextmanager
    def session():
        yield db

    monkeypatch.setattr(DatabaseTask, "get_db", lambda self: session())
    monkeypatch.setattr(Rows, "get_extraction", staticmethod(lambda _db, extraction_id: row))
    monkeypatch.setattr(Rows, "update_progress", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(Rows, "mark_completed", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(Rows, "mark_failed", staticmethod(lambda **kwargs: h.failed.append(kwargs["error_message"])))
    monkeypatch.setattr(
        model_tasks, "select_tokenization_for_model",
        lambda *args: SimpleNamespace(tokenized_path=str(dataset_dir)),
    )
    monkeypatch.setattr(model_tasks, "emit_extraction_progress", lambda **kwargs: None)
    monkeypatch.setattr(model_tasks, "emit_extraction_failed", lambda **kwargs: None)
    monkeypatch.setattr(
        model_tasks, "cancel_checker",
        lambda *args, **kwargs: SimpleNamespace(poll_now=lambda: False, reason=None),
    )
    monkeypatch.setattr(model_tasks, "build_extraction_progress_callback", lambda *args: None)
    monkeypatch.setattr(model_tasks, "build_statistics_heartbeat", lambda *args: None)

    class _ActivationService:
        def extract_activations(self, **kwargs):
            h.loads.append({
                "kwargs": kwargs,
                "gpu_uuid": row.gpu_uuid,
                "committed": list(db.committed),
            })
            if h.service_error is not None:
                raise h.service_error
            return {"num_samples": 1, "saved_files": [], "statistics": {}}

    monkeypatch.setattr(model_tasks, "ActivationService", _ActivationService)

    def retry(exc=None, kwargs=None, **options):
        # Celery's own order: the new attempt is sent, then Retry is raised.
        h.retries.append(kwargs)
        raise Retry("retry", exc=exc)

    monkeypatch.setattr(model_tasks.extract_activations, "retry", retry)

    def run(place, **overrides):
        monkeypatch.setattr(model_tasks, "place_job", place)
        task_kwargs = dict(
            model_id="m_1", dataset_id="ds_1", layer_indices=[11], hook_types=["residual"],
            max_samples=10, batch_size=8, extraction_id="ext_1",
        )
        task_kwargs.update(overrides)
        return model_tasks.extract_activations.run(**task_kwargs)

    h.run = run
    return h


class TestTheActivationWorkerRunsOnItsCard:
    def test_it_places_from_the_request_records_the_card_and_loads_there(self, activation_worker):
        h = activation_worker
        place = _placer(RTX)

        h.run(place, gpu_request=RTX_UUID)

        assert place.requests == [RTX_UUID]
        assert len(h.loads) == 1
        load = h.loads[0]
        assert load["kwargs"]["placement"].device == torch.device("cuda", 1), (
            "the model was not loaded on the card the job was placed on"
        )
        assert "gpu_id" not in load["kwargs"]
        assert load["gpu_uuid"] == RTX_UUID, "the card was not recorded before the model load"
        assert any(snapshot["ActivationExtraction"] == RTX_UUID for snapshot in load["committed"]), (
            "the card was assigned but never committed before the model load"
        )

    def test_memory_is_read_and_released_on_the_placed_card(self, activation_worker):
        h = activation_worker
        h.run(_placer(RTX), gpu_request=RTX_UUID)

        named = {device for name, device in h.cuda if name in ("memory_allocated", "memory_reserved", "device")}
        assert named == {torch.device("cuda", 1)}, f"memory was read or released on {named}"

    def test_a_message_with_no_request_is_placed_as_auto(self, activation_worker):
        h = activation_worker
        place = _placer(TI)
        h.run(place)

        assert place.requests == ["auto"]
        assert h.row.gpu_uuid == TI_UUID

    def test_a_legacy_gpu_id_is_ignored(self, activation_worker):
        h = activation_worker
        place = _placer(RTX)
        h.run(place, gpu_request=RTX_UUID, gpu_id=0)

        assert place.requests == [RTX_UUID]
        assert h.loads[0]["kwargs"]["placement"].device == torch.device("cuda", 1)

    def test_a_card_that_cannot_take_the_job_fails_it_with_the_reason_and_is_not_retried(self, activation_worker):
        h = activation_worker

        with pytest.raises(GpuPlacementError):
            h.run(_placer(GpuPlacementError(REFUSAL)), gpu_request=TI_UUID)

        assert h.failed == [REFUSAL], "the job was not failed with the placement's message"
        assert h.retries == [], "a refused placement was retried — it cannot succeed on a second attempt"
        assert h.loads == [], "the model was loaded despite the refusal"
        assert h.row.gpu_uuid is None


class TestARetryKeepsWhatWasAskedFor:
    def test_an_oom_retry_is_sent_once_halved_with_the_card_and_the_budget(self, activation_worker):
        from celery.exceptions import Retry

        from src.ml.model_loader import OutOfMemoryError

        h = activation_worker
        h.service_error = OutOfMemoryError("CUDA out of memory. Tried to allocate 4.00 GiB")

        with pytest.raises(Retry):
            h.run(
                _placer(RTX), gpu_request=RTX_UUID, batch_size=8,
                max_seq_length=2048, micro_batch_token_budget=8192,
            )

        assert len(h.retries) == 1, (
            f"{len(h.retries)} retries were queued for one extraction: the Retry "
            "raised by the OOM back-off fell into the generic handler, which sent "
            "another"
        )
        sent = h.retries[0]
        assert sent["batch_size"] == 4
        assert sent["gpu_request"] == RTX_UUID
        assert (sent["max_seq_length"], sent["micro_batch_token_budget"]) == (2048, 8192)
        assert h.failed == [], "an extraction being retried was marked FAILED"

    def test_a_transient_failure_is_retried_with_the_card_and_the_budget(self, activation_worker):
        from celery.exceptions import Retry

        h = activation_worker
        h.service_error = RuntimeError("stale NFS handle")

        with pytest.raises(Retry):
            h.run(
                _placer(RTX), gpu_request=RTX_UUID,
                max_seq_length=2048, micro_batch_token_budget=8192,
            )

        assert len(h.retries) == 1
        sent = h.retries[0]
        assert sent["gpu_request"] == RTX_UUID
        assert (sent["max_seq_length"], sent["micro_batch_token_budget"]) == (2048, 8192)


# ── SAE feature extraction: submit ─────────────────────────────────────────

@pytest.fixture
def sae_endpoints(monkeypatch, inventory):
    from src.api.v1.endpoints import saes as endpoint
    from src.models.external_sae import SAEStatus
    from src.services.extraction_service import ExtractionService
    from src.workers.extraction_tasks import extract_features_from_sae_task

    sae = SimpleNamespace(
        id="sae_a", status=SAEStatus.READY.value, local_path="/saes/a", name="sae a", model_id="m_1",
    )
    harness = SimpleNamespace(
        endpoint=endpoint,
        sae=sae,
        dispatch=MagicMock(name="apply_async", return_value=SimpleNamespace(id="celery-1")),
    )

    async def get_sae(_db, sae_id):
        return SimpleNamespace(**{**vars(sae), "id": sae_id})

    async def no_active_extraction(self, training_id=None, sae_id=None):
        return None

    monkeypatch.setattr(endpoint.SAEManagerService, "get_sae", staticmethod(get_sae))
    monkeypatch.setattr(ExtractionService, "_check_active_extraction", no_active_extraction)
    monkeypatch.setattr(extract_features_from_sae_task, "apply_async", harness.dispatch)
    return harness


class TestSaeExtractionIsSubmittedWithItsCard:
    def test_the_job_row_stores_the_card_and_the_config_does_not(self, sae_endpoints):
        from src.schemas.extraction import ExtractionConfigRequest

        h = sae_endpoints
        dataset = SimpleNamespace(status="ready", name="ds")
        db = _AsyncDB(h.sae, dataset, dataset)

        response = asyncio.run(h.endpoint.start_sae_extraction(
            "sae_a", ExtractionConfigRequest(gpu="1"), dataset_id="ds_1", db=db,
        ))

        assert [job.gpu_request for job in db.added] == [RTX_UUID]
        # The response shows the request too; the card follows once it starts.
        assert response.gpu_request == RTX_UUID
        assert response.gpu_uuid is None
        assert h.dispatch.call_count == 1
        sae_id, config = h.dispatch.call_args.kwargs["args"]
        assert sae_id == "sae_a"
        assert "gpu" not in config, "a second copy of the request in config could disagree with the row"

    def test_an_unknown_card_is_a_400_and_no_job_is_created(self, sae_endpoints):
        from src.schemas.extraction import ExtractionConfigRequest

        h = sae_endpoints
        dataset = SimpleNamespace(status="ready", name="ds")
        db = _AsyncDB(h.sae, dataset, dataset)

        with pytest.raises(HTTPException) as exc:
            asyncio.run(h.endpoint.start_sae_extraction(
                "sae_a", ExtractionConfigRequest(gpu=UNKNOWN_UUID), dataset_id="ds_1", db=db,
            ))

        assert exc.value.status_code == 400
        assert db.added == []
        assert h.dispatch.call_count == 0

    def test_every_job_in_a_batch_stores_the_card(self, sae_endpoints):
        from src.models.dataset import DatasetStatus
        from src.schemas.extraction import BatchExtractionRequest

        h = sae_endpoints
        sae_b = SimpleNamespace(**{**vars(h.sae), "id": "sae_b", "name": "sae b"})
        db = _AsyncDB(SimpleNamespace(status=DatasetStatus.READY, name="ds"), h.sae, sae_b)

        response = asyncio.run(h.endpoint.start_batch_sae_extraction(
            BatchExtractionRequest(sae_ids=["sae_a", "sae_b"], dataset_id="ds_1", gpu=TI_UUID), db=db,
        ))

        assert response.total_created == 2, response.skipped_saes
        assert [job.gpu_request for job in db.added] == [TI_UUID, TI_UUID]
        # Only the first is queued now; the second is chained when it finishes.
        assert h.dispatch.call_count == 1
        assert "gpu" not in h.dispatch.call_args.kwargs["args"][1]

    def test_an_unknown_card_is_a_400_and_the_batch_creates_nothing(self, sae_endpoints):
        from src.models.dataset import DatasetStatus
        from src.schemas.extraction import BatchExtractionRequest

        h = sae_endpoints
        db = _AsyncDB(SimpleNamespace(status=DatasetStatus.READY, name="ds"), h.sae)

        with pytest.raises(HTTPException) as exc:
            asyncio.run(h.endpoint.start_batch_sae_extraction(
                BatchExtractionRequest(sae_ids=["sae_a"], dataset_id="ds_1", gpu=UNKNOWN_UUID), db=db,
            ))

        assert exc.value.status_code == 400
        assert db.added == []
        assert h.dispatch.call_count == 0


# ── SAE feature extraction: the worker and batch chaining ─────────────────

def _job(**fields):
    job = dict(
        id="extr_2", external_sae_id="sae_b", status="queued", gpu_request=None, gpu_uuid=None,
        statistics=None, batch_id=None, batch_position=None, batch_total=None, config={},
        celery_task_id=None,
    )
    job.update(fields)
    return SimpleNamespace(**job)


@pytest.fixture
def sae_worker(monkeypatch, cuda_calls):
    from src.services import extraction_service as module
    from src.services.extraction_service import ExtractionService
    from src.workers import websocket_emitter

    class _LoadObserved(Exception):
        """Raised by the fake SAE loader once the load has been recorded."""

    h = SimpleNamespace(statuses=[], loads=[], cleanups=[], cuda=cuda_calls, LoadObserved=_LoadObserved)

    def record_status(self, extraction_id, status, **kwargs):
        h.statuses.append((extraction_id, status, kwargs.get("error_message")))

    monkeypatch.setattr(ExtractionService, "update_extraction_status_sync", record_status)
    monkeypatch.setattr(websocket_emitter, "emit_progress", lambda **kwargs: None)
    monkeypatch.setattr(
        module, "cleanup_gpu_memory",
        lambda models_to_cleanup=None, context="", device=None: h.cleanups.append(device),
    )

    def run(job, place, config=None):
        db = _SyncDB(
            ExtractionJob=job,
            ExternalSAE=SimpleNamespace(id=job.external_sae_id, local_path="/saes/b", layer=11, name="sae b"),
        )

        def load_sae_auto_detect(path, device="cpu"):
            h.loads.append({"device": device, "gpu_uuid": job.gpu_uuid, "committed": list(db.committed)})
            raise _LoadObserved()

        monkeypatch.setattr(module, "load_sae_auto_detect", load_sae_auto_detect)
        monkeypatch.setattr(module, "place_job", place)
        return ExtractionService(db).extract_features_for_sae(
            job.external_sae_id, config if config is not None else {"dataset_id": "ds_1"},
        )

    h.run = run
    return h


class _QueryWithAll(_Query):
    def all(self):
        result = self._result
        return result if isinstance(result, list) else ([] if result is None else [result])


class _SyncDBWithLists(_SyncDB):
    def query(self, model):
        return _QueryWithAll(self.rows.get(model.__name__))


@pytest.fixture
def sae_worker_to_model_load(monkeypatch, cuda_calls):
    """Drive extract_features_for_sae past the SAE loader to the base-model load.

    The `sae_worker` fixture stops at the SAE loader, so nothing pinned the two
    device hops after it: the SAE module's `.to(device)` and the base model's
    `device_map`. Either could go back to a bare "cuda" (index 0) unseen.
    """
    from src.services import extraction_service as module
    from src.services.extraction_service import ExtractionService
    from src.workers import model_tasks, websocket_emitter

    class _ModelLoadObserved(Exception):
        """Raised by the fake base-model loader once the load has been recorded."""

    h = SimpleNamespace(sae_moves=[], model_loads=[], ModelLoadObserved=_ModelLoadObserved)

    class _FakeSae:
        def load_state_dict(self, state):
            pass

        def to(self, device):
            h.sae_moves.append(device)
            return self

        def eval(self):
            return self

    tokenization = SimpleNamespace(status=module.TokenizationStatus.READY, tokenized_path="/data/tok")

    def load_model_from_hf(**kwargs):
        h.model_loads.append(kwargs)
        raise _ModelLoadObserved()

    monkeypatch.setattr(ExtractionService, "update_extraction_status_sync", lambda self, *a, **k: None)
    monkeypatch.setattr(websocket_emitter, "emit_progress", lambda **kwargs: None)
    monkeypatch.setattr(
        module, "cleanup_gpu_memory", lambda models_to_cleanup=None, context="", device=None: None
    )
    monkeypatch.setattr(
        module, "load_sae_auto_detect",
        lambda path, device="cpu": ({"W_enc": torch.zeros(8, 4)}, None, "mistudio"),
    )
    monkeypatch.setattr(module, "create_sae", lambda **kwargs: _FakeSae())
    monkeypatch.setattr(
        model_tasks, "select_tokenization_for_model",
        lambda candidates, model_id, dataset_id: tokenization,
    )
    monkeypatch.setattr(module, "load_from_disk", lambda path: [0, 1, 2])
    monkeypatch.setattr(module, "load_model_from_hf", load_model_from_hf)

    def run(placement):
        job = _job(gpu_request=placement.uuid)
        db = _SyncDBWithLists(
            ExtractionJob=job,
            ExternalSAE=SimpleNamespace(
                id=job.external_sae_id, local_path="/saes/b", layer=11, name="sae b",
                model_id="m_1", model_name=None, architecture="standard", n_features=8, d_model=4,
            ),
            Model=SimpleNamespace(id="m_1", repo_id="org/model", quantization="FP16", file_path=None),
            Dataset=SimpleNamespace(id="ds_1"),
            DatasetTokenization=[tokenization],
        )
        monkeypatch.setattr(module, "place_job", _placer(placement))
        return ExtractionService(db).extract_features_for_sae(
            job.external_sae_id, {"dataset_id": "ds_1"},
        )

    h.run = run
    return h


class TestTheSaeWorkerLoadsEverythingOnItsCard:
    """Past the SAE loader: the SAE module and the base model follow the placement.

    MUTATION CONTROLS (2026-09-13, main after bd2783ef):
      S1 `sae.to(device)` -> `sae.to("cuda")`                      -> both cases fail
      S2 base model `device_map=str(device)` -> `device_map="auto"` -> both cases fail
    """

    @pytest.mark.parametrize("placement", [TI, RTX], ids=["3080ti", "3090"])
    def test_the_sae_and_the_base_model_land_on_the_placed_card(
        self, sae_worker_to_model_load, placement
    ):
        h = sae_worker_to_model_load

        with pytest.raises(h.ModelLoadObserved):
            h.run(placement)

        assert h.sae_moves == [placement.device], "the SAE module was moved to another card"
        assert len(h.model_loads) == 1
        assert h.model_loads[0]["device_map"] == str(placement.device), (
            "the base model was not loaded onto the placed card"
        )


class TestTheSaeWorkerRunsOnItsCard:
    def test_it_places_from_its_row_records_the_card_and_loads_there(self, sae_worker):
        h = sae_worker
        job = _job(gpu_request=RTX_UUID)
        place = _placer(RTX)

        with pytest.raises(h.LoadObserved):
            h.run(job, place)

        assert place.requests == [RTX_UUID]
        assert len(h.loads) == 1
        assert h.loads[0]["device"] == "cuda:1", "the SAE was not loaded on the placed card"
        assert h.loads[0]["gpu_uuid"] == RTX_UUID, "the card was not recorded before the load"
        assert any(snapshot["ExtractionJob"] == RTX_UUID for snapshot in h.loads[0]["committed"]), (
            "the card was assigned but never committed before the load"
        )
        assert h.cleanups == [(torch.device("cuda", 1),)], "the finally cleaned some other card"

    def test_memory_is_measured_on_the_placed_card(self, sae_worker):
        h = sae_worker
        with pytest.raises(h.LoadObserved):
            h.run(_job(gpu_request=RTX_UUID), _placer(RTX))

        named = {device for name, device in h.cuda if name in ("synchronize", "memory_allocated")}
        assert named == {torch.device("cuda", 1)}, f"memory was measured on {named}"

    def test_a_row_from_before_requests_were_recorded_is_placed_as_auto(self, sae_worker):
        h = sae_worker
        place = _placer(TI)
        with pytest.raises(h.LoadObserved):
            h.run(_job(gpu_request=None), place)

        assert place.requests == ["auto"]

    def test_a_card_that_cannot_take_the_job_fails_it_with_the_reason(self, sae_worker):
        from src.models.extraction_job import ExtractionStatus

        h = sae_worker
        job = _job(gpu_request=TI_UUID)

        with pytest.raises(GpuPlacementError):
            h.run(job, _placer(GpuPlacementError(REFUSAL)))

        assert (job.id, ExtractionStatus.FAILED.value, REFUSAL) in h.statuses
        assert h.loads == []
        assert job.gpu_uuid is None
        assert h.cleanups == [None], "a job that placed nothing named a card to clean"


class TestAChainedBatchJobPlacesWithItsOwnRequest:
    def test_the_next_job_is_placed_from_its_own_row(self, sae_worker, monkeypatch):
        from src.workers import extraction_tasks, nlp_analysis_tasks

        h = sae_worker
        shared_config = {"dataset_id": "ds_1", "top_k_examples": 10}
        # Different requests on purpose: a fixture where both jobs asked for the
        # same card could not tell "its own row" from "the previous job's".
        current = _job(
            id="extr_1", external_sae_id="sae_a", status="completed", batch_id="batch_1",
            batch_position=1, batch_total=2, gpu_request=RTX_UUID, gpu_uuid=RTX_UUID,
            config=shared_config, celery_task_id="celery-1",
        )
        following = _job(
            id="extr_2", external_sae_id="sae_b", batch_id="batch_1", batch_position=2,
            batch_total=2, gpu_request=TI_UUID, config=shared_config,
        )

        sent = []

        class _Task:
            def apply_async(self, args=None, **options):
                sent.append(args)
                return SimpleNamespace(id="celery-2")

        monkeypatch.setattr(extraction_tasks, "extract_features_from_sae_task", _Task())

        nlp_analysis_tasks._start_next_batch_job(_SyncDB(ExtractionJob=following), current)

        assert len(sent) == 1
        sae_id, config = sent[0]
        assert sae_id == "sae_b"
        assert "gpu" not in config

        place = _placer(TI)
        with pytest.raises(h.LoadObserved):
            h.run(following, place, config=config)

        assert place.requests == [TI_UUID], (
            f"the chained job was placed on {place.requests}, not its own stored request"
        )
        assert h.loads[0]["device"] == "cuda:0"
        assert following.gpu_uuid == TI_UUID


# ── Cleanup that is not scoped to one job ──────────────────────────────────

class TestCleanupActsOnTheCardsInUse:
    def test_the_analysis_cleanup_endpoint_cleans_every_card_holding_memory(self, monkeypatch, cuda_calls):
        from src.api.v1.endpoints import features

        reserved = {0: 0, 1: 6 * 1024**3}
        allocated = {0: 0, 1: 4 * 1024**3}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: reserved[index])
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: allocated[index])

        result = asyncio.run(features.cleanup_analysis_gpu())

        assert result["cleaned"] is True, result
        assert result["vram_before_gb"] == 4.0, "the card holding the memory was not measured"
        # Card 0 holds nothing, so it is never synchronized: that would create
        # a CUDA context on it.
        assert [device for name, device in cuda_calls if name == "synchronize"] == [1]
        assert [device for name, device in cuda_calls if name == "device"] == [1]

    def test_a_job_cleanup_touches_only_its_own_card(self, monkeypatch, cuda_calls):
        from src.services.extraction_service import cleanup_gpu_memory

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        cleanup_gpu_memory(context="test", device=torch.device("cuda", 1))

        for call in ("synchronize", "reset_peak_memory_stats", "memory_allocated"):
            named = {device for name, device in cuda_calls if name == call}
            assert named == {torch.device("cuda", 1)}, f"{call} named {named}"

    def test_without_a_device_it_sweeps_the_cards_this_process_holds_memory_on(self, monkeypatch, cuda_calls):
        from src.services.extraction_service import cleanup_gpu_memory

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: {0: 0, 1: 1024**3}[index])

        cleanup_gpu_memory(context="sweep")

        assert [device for name, device in cuda_calls if name == "synchronize"] == [1]
        assert [device for name, device in cuda_calls if name == "reset_peak_memory_stats"] == [1]


# ── The activation service loads where it is told ─────────────────────────

class TestTheActivationServiceLoadsOnTheDeviceItIsGiven:
    """The last two hops: service -> `_load_model` -> `from_pretrained`.

    `_load_model` built `cuda:{gpu_id}` from an index defaulting to 0. The
    worker tests above stop at the service boundary, so these follow the device
    the rest of the way.
    """

    def test_extract_activations_hands_the_device_to_the_loader(self, monkeypatch, tmp_path):
        from src.services.activation_service import ActivationExtractionError, ActivationService

        seen = []

        class _Loaded(Exception):
            pass

        def load_model(self, model_path, quantization, placement):
            seen.append(placement)
            raise _Loaded()

        monkeypatch.setattr(ActivationService, "_load_model", load_model)
        monkeypatch.setattr(ActivationService, "_extraction_dir", lambda self, extraction_id: tmp_path)
        monkeypatch.setattr(ActivationService, "_release_gpu_memory", lambda self, devices: {})
        monkeypatch.setattr(ActivationService, "_log_gpu_memory", lambda self, stage, devices: None)

        service = ActivationService.__new__(ActivationService)
        with pytest.raises(ActivationExtractionError):
            service.extract_activations(
                model_id="m_1", model_path="/m", architecture="lfm2",
                quantization=SimpleNamespace(value="FP16"), dataset_path="/d",
                layer_indices=[11], hook_types=["residual"], max_samples=4,
                placement=RTX,
            )

        assert [placement.device for placement in seen] == [torch.device("cuda", 1)]

    def test_the_whole_model_is_put_on_that_device(self, monkeypatch, tmp_path):
        import transformers

        from src.models.model import QuantizationFormat
        from src.services.activation_service import ActivationService

        loaded = {}

        class _Model:
            device = torch.device("cuda", 1)
            dtype = torch.float16

        def model_from_pretrained(path, **kwargs):
            loaded.update(kwargs)
            return _Model()

        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_from_pretrained)
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: object())
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA GeForce RTX 3090")

        service = ActivationService.__new__(ActivationService)
        service._load_model(str(tmp_path), QuantizationFormat.FP16, placement=RTX)

        assert loaded["device_map"] == {"": torch.device("cuda", 1)}, (
            "from_pretrained was not told to put the whole model on the placed card"
        )

    def test_a_cpu_placement_is_refused_rather_than_loaded(self, monkeypatch, tmp_path):
        """Auto on a machine with no CUDA places on the CPU; extraction needs a GPU."""
        from src.models.model import QuantizationFormat
        from src.services.activation_service import ActivationExtractionError, ActivationService

        service = ActivationService.__new__(ActivationService)
        with pytest.raises(ActivationExtractionError, match="GPU is required"):
            service._load_model(
                str(tmp_path), QuantizationFormat.FP16,
                placement=Placement(card=None, device=torch.device("cpu")),
            )
