"""Model download: the inspection load runs on the GPU the caller chose.

`download_and_load_model` loads every new model once, to read its architecture
for the database. It loaded with `device_map="auto"` — spreading the weights
over every visible card, the 12 GB 3080 Ti at index 0 and whatever card miLLM
serves from included — and then read `memory_reserved()` of whichever card
happened to be current.

Now the request travels with the task (as a kwarg, and in `retry_params` so a
retry asks for the same card), the worker places the job before the load,
stamps the card on the live task_queue row, and hands the placed device to the
loader, whose preflight measures that same card.

MUTATION CONTROLS (each applied, run red, restored byte-identically):
  D1  the download endpoint stops passing `gpu_request` to `.delay`
      -> test_download_passes_the_uuid_the_index_names
  D2  the loader gets `device_map="auto"` again
      -> test_the_card_is_recorded_before_the_load_and_the_load_uses_it
"""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import torch
from fastapi import HTTPException

from src.api.v1.endpoints import models as models_ep
from src.api.v1.endpoints import task_queue as task_queue_ep
from src.core.config import settings
from src.models.model import Model, ModelStatus, QuantizationFormat
from src.models.task_queue import TaskQueue
from src.schemas.model import ModelDownloadRequest, ModelRedownloadRequest
from src.schemas.task_queue import TaskQueueRetryRequest
from src.services import gpu_placement
from src.services.gpu_placement import GpuCard, GpuPlacementError, Placement
from src.workers import model_tasks

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
UNKNOWN = "GPU-00000000-0000-0000-0000-000000000000"

CARDS = [
    GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000),
    GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000),
]

METADATA = {
    "architecture": "llama",
    "params_count": 1_000,
    "architecture_config": {},
    "memory_required_bytes": 1,
    "quantization": "FP16",
}


@pytest.fixture(autouse=True)
def inventory(monkeypatch):
    """The node's two cards, as NVML would list them."""
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)


def _celery_task():
    task = MagicMock()
    task.delay.return_value = SimpleNamespace(id="celery-1")
    return task


# ── Submit ─────────────────────────────────────────────────────────────────

class TestTheDownloadEndpoint:
    def _start(self, request, task, initiate):
        with patch.object(models_ep, "download_and_load_model", task), \
             patch.object(models_ep.ModelService, "get_model_by_name", AsyncMock(return_value=None)), \
             patch.object(models_ep.ModelService, "initiate_model_download", initiate):
            return asyncio.run(models_ep.download_model(request, db=AsyncMock()))

    def test_download_passes_the_uuid_the_index_names(self):
        task = _celery_task()
        initiate = AsyncMock(return_value=SimpleNamespace(id="m_new", celery_task_id=None))
        request = ModelDownloadRequest(repo_id="vendor/model", gpu="1")

        self._start(request, task, initiate)

        assert initiate.await_count == 1
        assert task.delay.call_count == 1
        assert task.delay.call_args.kwargs == {
            "model_id": "m_new",
            "repo_id": "vendor/model",
            "quantization": request.quantization.value,
            "access_token": None,
            "trust_remote_code": False,
            "gpu_request": RTX_UUID,
        }

    def test_auto_is_passed_as_auto_without_reading_the_inventory(self, monkeypatch):
        monkeypatch.setattr(gpu_placement, "list_cards", lambda: pytest.fail("NVML read for auto"))
        task = _celery_task()
        initiate = AsyncMock(return_value=SimpleNamespace(id="m_new", celery_task_id=None))

        self._start(ModelDownloadRequest(repo_id="vendor/model"), task, initiate)

        assert task.delay.call_count == 1
        assert task.delay.call_args.kwargs["gpu_request"] == "auto"

    def test_an_unknown_card_is_a_400_and_nothing_is_created_or_queued(self):
        task = _celery_task()
        initiate = AsyncMock()

        with pytest.raises(HTTPException) as exc:
            self._start(ModelDownloadRequest(repo_id="vendor/model", gpu=UNKNOWN), task, initiate)

        assert exc.value.status_code == 400
        assert "RTX 3090" in exc.value.detail
        assert initiate.await_count == 0
        assert task.delay.call_count == 0


class TestTheRedownloadEndpoint:
    @staticmethod
    def _model(file_path=None):
        return SimpleNamespace(
            id="m_1", status=ModelStatus.READY, repo_id="vendor/model",
            quantization=QuantizationFormat.FP16, file_path=file_path,
            quantized_path=None, progress=100.0, error_message=None, celery_task_id=None,
        )

    def test_an_unknown_card_is_refused_before_the_existing_files_are_deleted(self, tmp_path):
        """Re-download deletes the model it replaces. A bad card must not cost the
        user the model they already had, so the refusal comes first."""
        existing = tmp_path / "m_1"
        existing.mkdir()
        task = _celery_task()
        rmtree = MagicMock()

        with patch.object(models_ep.ModelService, "get_model", AsyncMock(return_value=self._model(str(existing)))), \
             patch.object(type(settings), "resolve_deletable_path", lambda self, stored: existing), \
             patch("shutil.rmtree", rmtree), \
             patch.object(models_ep, "download_and_load_model", task):
            with pytest.raises(HTTPException) as exc:
                asyncio.run(models_ep.redownload_model(
                    "m_1", ModelRedownloadRequest(quantization=QuantizationFormat.Q8, gpu=UNKNOWN),
                    db=AsyncMock(),
                ))

        assert exc.value.status_code == 400
        assert rmtree.call_count == 0
        assert task.delay.call_count == 0

    def test_a_named_card_travels_with_the_redownload(self):
        task = _celery_task()

        with patch.object(models_ep.ModelService, "get_model", AsyncMock(return_value=self._model())), \
             patch.object(models_ep, "download_and_load_model", task):
            asyncio.run(models_ep.redownload_model(
                "m_1", ModelRedownloadRequest(quantization=QuantizationFormat.Q8, gpu="0"),
                db=AsyncMock(),
            ))

        assert task.delay.call_count == 1
        assert task.delay.call_args.kwargs["gpu_request"] == TI_UUID


class TestARetryAsksForTheSameCard:
    def test_the_retry_dispatch_carries_the_request_from_retry_params(self):
        row = SimpleNamespace(
            id="tq_1", task_type="download", entity_type="model", entity_id="m_1",
            status="failed", retry_count=1, task_id=None,
            retry_params={"repo_id": "vendor/model", "quantization": "FP16",
                          "trust_remote_code": False, "gpu_request": RTX_UUID},
        )
        task = _celery_task()

        with patch.object(task_queue_ep.TaskQueueService, "get_task_by_id", AsyncMock(return_value=row)), \
             patch.object(task_queue_ep.TaskQueueService, "increment_retry_count", AsyncMock()), \
             patch.object(task_queue_ep.AppSettingService, "get_decrypted_value", AsyncMock(return_value=None)), \
             patch.object(model_tasks, "download_and_load_model", task):
            asyncio.run(task_queue_ep.retry_task("tq_1", TaskQueueRetryRequest(), db=AsyncMock()))

        assert task.delay.call_count == 1
        assert task.delay.call_args.kwargs["gpu_request"] == RTX_UUID


# ── Start ──────────────────────────────────────────────────────────────────

class _Query:
    """The few query shapes the download task issues."""

    def __init__(self, first=None, rows=()):
        self._first, self._rows = first, list(rows)

    def filter_by(self, **_):
        return self

    def filter(self, *_):
        return self

    def first(self):
        # A live retry row answers `.first()` as well, which is how the failure
        # handler finds it.
        if self._first is not None:
            return self._first
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


def _drive(place, load, tmp_path, *, gpu_request=RTX_UUID, live_rows=(), sizing=None):
    """Run the real task body; everything outside it is stubbed.

    `sizing` stands in for `required_mb_for_load`, which reads the model's
    config from the Hub; unsized by default, so no test reaches the network.
    """
    if sizing is None:
        sizing = MagicMock(return_value=None)
    model_row = SimpleNamespace(status=None, progress=None, error_message=None)
    added = []
    db = MagicMock()
    db.add.side_effect = added.append
    db.query.side_effect = lambda cls: (
        _Query(first=model_row) if cls is Model else _Query(rows=live_rows)
    )
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=db)
    ctx.__exit__ = MagicMock(return_value=False)

    outcome = SimpleNamespace(model_row=model_row, added=added, error=None)
    with patch("src.workers.base_task.DatabaseTask.get_db", return_value=ctx), \
         patch("src.services.gpu_placement.place_job", place), \
         patch.object(model_tasks, "required_mb_for_load", sizing), \
         patch("src.workers.base_task.mark_task_queue_entries_completed", MagicMock()), \
         patch("src.ml.transformers_compat.patch_transformers_compatibility", MagicMock()), \
         patch.object(model_tasks, "load_model_from_hf", load), \
         patch.object(model_tasks, "send_progress_update", MagicMock()), \
         patch.object(model_tasks, "DownloadProgressMonitor", MagicMock()), \
         patch.object(model_tasks, "clear_cancel_request", MagicMock()), \
         patch.object(model_tasks, "cancel_checker", MagicMock()), \
         patch.object(model_tasks, "settings", SimpleNamespace(models_dir=tmp_path)):
        try:
            model_tasks.download_and_load_model.run(
                "m_1", "vendor/model", "FP16", gpu_request=gpu_request,
            )
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome.error = exc
    return outcome


class TestTheWorkerRunsOnThePlacedCard:
    def test_the_card_is_recorded_before_the_load_and_the_load_uses_it(self, tmp_path):
        live_retry_row = SimpleNamespace(gpu_uuid=None)
        place = MagicMock(return_value=Placement(card=CARDS[1], device=torch.device("cuda", 1)))
        seen = {}

        def load(**kwargs):
            seen["kwargs"] = kwargs
            seen["uuid_at_load"] = live_retry_row.gpu_uuid
            return MagicMock(), MagicMock(), MagicMock(), METADATA

        outcome = _drive(place, load, tmp_path, live_rows=[live_retry_row])

        assert outcome.error is None, outcome.error
        assert place.call_args_list == [call(RTX_UUID, required_mb=None, allow_shard=True)]
        assert seen["kwargs"]["device_map"] == "cuda:1"
        assert seen["kwargs"]["max_memory"] is None, "a single-card load was given a split budget"
        assert seen["uuid_at_load"] == RTX_UUID, "the card was not recorded before the load"
        assert outcome.model_row.status == ModelStatus.READY

    def test_a_refused_card_fails_the_download_with_its_message(self, tmp_path):
        message = (
            "GPU 1 (NVIDIA GeForce RTX 3090, 900 of 24,576 MB free) cannot take this "
            "job: it needs ~4,000 MB. Choose another GPU or Auto."
        )
        place = MagicMock(side_effect=GpuPlacementError(message))
        load = MagicMock()

        outcome = _drive(place, load, tmp_path)

        assert isinstance(outcome.error, GpuPlacementError)
        assert load.call_count == 0, "a refused card must not be swapped for another"
        assert outcome.model_row.status == ModelStatus.ERROR
        assert outcome.model_row.error_message == message

        retry_rows = [obj for obj in outcome.added if isinstance(obj, TaskQueue)]
        assert len(retry_rows) == 1
        assert retry_rows[0].retry_params["gpu_request"] == RTX_UUID
        assert retry_rows[0].gpu_uuid is None, "it never reached a card"

    def test_the_download_task_does_not_retry_itself(self):
        """A refused card is a decision, not a transient fault: no retry loop."""
        assert model_tasks.download_and_load_model.max_retries == 0


# ── Phase 2: a model larger than any one card ──────────────────────────────

def _too_big_for_one_card():
    """~24,164 MB at FP16 with headroom: more than the 3090's 23,000 MB free,
    less than the two cards' split budgets (21,976 + 9,976 MB)."""
    return SimpleNamespace(hidden_size=4096, num_hidden_layers=56, vocab_size=32_000, intermediate_size=11_008)


@pytest.fixture
def two_cards(monkeypatch):
    """torch.cuda with the node's two cards. Returns the device context each cache release ran in."""
    state = {"current": None}
    emptied = []

    @contextlib.contextmanager
    def device(target):
        previous, state["current"] = state["current"], torch.device(target)
        try:
            yield
        finally:
            state["current"] = previous

    uuids = {0: TI_UUID, 1: RTX_UUID}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid=uuids[index]))
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(state["current"]))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 0)
    return emptied


@pytest.fixture
def big_config(monkeypatch):
    """The Hub config the sizing reads, recorded; no network."""
    import transformers

    read = []

    def from_pretrained(source, **kwargs):
        read.append((source, kwargs))
        return _too_big_for_one_card()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)
    return read


class _HookedModel(torch.nn.Module):
    """What a split load returns: modules carrying accelerate hooks."""

    def __init__(self):
        super().__init__()
        from accelerate.hooks import ModelHook, add_hook_to_module

        self.layer = torch.nn.Linear(2, 2)
        add_hook_to_module(self.layer, ModelHook())


SPLIT_CARDS = {torch.device("cuda", 1), torch.device("cuda", 0)}


class TestAModelLargerThanAnyCardCanBeRegistered:
    """The download's inspection load is sized, and split only when no single card holds it.

    Driven through the REAL `place_job` and the REAL `required_mb_for_load`;
    only NVML, CUDA, the Hub config and the weights load are faked.

    MUTATION CONTROLS (each applied, run red, restored byte-identically; 2026-09-14):
      D3  `place_job(..., allow_shard=True)` -> no allow_shard
          -> test_the_inspection_load_is_split_within_a_gpu_only_budget
      D4  `device_map=placement.device_map` -> `str(placement.device)`
          -> test_the_inspection_load_is_split_within_a_gpu_only_budget
      D5  `max_memory=placement.max_memory` removed
          -> test_the_inspection_load_is_split_within_a_gpu_only_budget
      D6  the finally empties `placement.all_devices[:1]` only
          -> test_the_inspection_load_is_split_within_a_gpu_only_budget,
             test_a_split_that_fails_records_every_card_on_its_new_retry_row
      D7  `row.gpu_uuids = gpu_uuids` removed from record_download_card
          -> test_the_inspection_load_is_split_within_a_gpu_only_budget
      D8  the new failure TaskQueue row without `gpu_uuids=`
          -> test_a_split_that_fails_records_every_card_on_its_new_retry_row
      D9  `existing_entry.gpu_uuids = ...` removed
          -> test_a_retry_refused_at_placement_does_not_keep_the_previous_splits_cards
          (SURVIVED first against a retry that failed AFTER the load: the row had
          been stamped with the same cards before the load, so the fixture agreed
          with the defect. Rewritten; re-run red.)
      D10 `detach_dispatch_hooks(model_obj)` removed from the finally
          -> test_the_inspection_load_is_split_within_a_gpu_only_budget
      D11 sizing drops the headroom (`+ _ACTIVATION_HEADROOM_GB * 1024`)
          -> test_the_size_is_exactly_what_the_loader_preflight_accepts
      D12 the download placed unsized (`required_mb=None`)
          -> test_the_inspection_load_is_split_within_a_gpu_only_budget
    """

    def test_the_inspection_load_is_split_within_a_gpu_only_budget(self, two_cards, big_config, tmp_path):
        live_retry_row = SimpleNamespace(gpu_uuid=None, gpu_uuids=None)
        loaded = _HookedModel()
        seen = {}

        def load(**kwargs):
            seen["kwargs"] = kwargs
            seen["row_at_load"] = (live_retry_row.gpu_uuid, live_retry_row.gpu_uuids)
            return loaded, MagicMock(), MagicMock(), METADATA

        outcome = _drive(
            gpu_placement.place_job, load, tmp_path, gpu_request="auto",
            live_rows=[live_retry_row], sizing=model_tasks.required_mb_for_load,
        )

        assert outcome.error is None, outcome.error
        assert big_config == [(
            "vendor/model",
            {"cache_dir": str(tmp_path / "raw" / "m_1"), "trust_remote_code": False, "token": None},
        )]
        expected = Placement(
            card=CARDS[1], device=torch.device("cuda", 1), cards=(CARDS[1], CARDS[0]),
            devices=(torch.device("cuda", 1), torch.device("cuda", 0)), max_memory_mb={1: 21_976, 0: 9_976},
        )
        assert seen["kwargs"]["device_map"] == expected.device_map != "cuda:1", (
            "a model no card can hold was loaded onto one card"
        )
        assert seen["kwargs"]["max_memory"] == expected.max_memory == {1: "21976MiB", 0: "9976MiB"}
        assert seen["row_at_load"] == (RTX_UUID, [RTX_UUID, TI_UUID]), (
            "the retry row did not name every card before the load"
        )
        assert set(two_cards) == SPLIT_CARDS, f"the release emptied {two_cards}, not every card of the split"
        assert not hasattr(loaded.layer, "_hf_hook"), "the split model was dropped with its dispatch hooks attached"
        assert outcome.model_row.status == ModelStatus.READY

    def test_a_split_that_fails_records_every_card_on_its_new_retry_row(self, two_cards, big_config, tmp_path):
        outcome = _drive(
            gpu_placement.place_job, MagicMock(side_effect=RuntimeError("load failed on the second card")),
            tmp_path, gpu_request="auto", sizing=model_tasks.required_mb_for_load,
        )

        assert isinstance(outcome.error, RuntimeError)
        retry_rows = [obj for obj in outcome.added if isinstance(obj, TaskQueue)]
        assert len(retry_rows) == 1
        assert (retry_rows[0].gpu_uuid, retry_rows[0].gpu_uuids) == (RTX_UUID, [RTX_UUID, TI_UUID])
        assert set(two_cards) == SPLIT_CARDS, "a failed split load was released on one card"

    def test_a_retry_refused_at_placement_does_not_keep_the_previous_splits_cards(self, tmp_path):
        """The failure handler's write is the ONLY writer when the attempt never reached a card.

        A retry that did reach a card has its row stamped before the load, so a
        fixture that fails after the load cannot tell whether the handler writes
        `gpu_uuids` at all — the first version of this test passed with that line
        deleted. Here the live row still carries the cards of an earlier attempt
        that split, and this attempt is refused before any card is used.
        """
        live_retry_row = SimpleNamespace(
            id="tq_1", gpu_uuid=RTX_UUID, gpu_uuids=[RTX_UUID, TI_UUID], status="running",
            retry_count=1, error_message=None, task_id=None,
        )
        refusal = GpuPlacementError("All 2 GPUs together can hold ~31,952 MB of a split model")
        load = MagicMock()

        outcome = _drive(
            MagicMock(side_effect=refusal), load, tmp_path, gpu_request="all", live_rows=[live_retry_row],
        )

        assert outcome.error is refusal
        assert load.call_count == 0
        assert live_retry_row.status == "failed", "the harness did not reach the retry-row branch"
        assert (live_retry_row.gpu_uuid, live_retry_row.gpu_uuids) == (None, None), (
            "a refused retry still claims the two cards a previous attempt was split across"
        )

    def test_the_size_is_exactly_what_the_loader_preflight_accepts(self, monkeypatch, big_config):
        """One arithmetic, two places: a card the placement accepts, the preflight accepts."""
        import math

        from src.ml.model_loader import estimate_parameter_count
        from src.services import resource_config

        required_mb = model_tasks.required_mb_for_load("vendor/model", "Q8")
        params = estimate_parameter_count(_too_big_for_one_card())

        def preflight_with(free_bytes):
            monkeypatch.setattr(
                resource_config, "_usable_gpu_memory", lambda device: (free_bytes, free_bytes, "cuda:1")
            )
            resource_config.preflight_gpu_capacity(
                params_count=params, quantization="Q8", device="cuda:1", model_name="vendor/model"
            )

        preflight_with(math.ceil(required_mb * 1024**2))
        with pytest.raises(resource_config.VRAMInsufficientError):
            preflight_with(math.floor((required_mb - 1) * 1024**2))

    def test_an_unreadable_config_leaves_the_job_unsized(self, monkeypatch):
        import transformers

        def missing(source, **kwargs):
            raise OSError("vendor/model does not appear to have a file named config.json")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", missing)

        assert model_tasks.required_mb_for_load("vendor/model", "FP16") is None
