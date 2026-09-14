"""Logit lens: every load and tensor move uses the card the caller chose.

`LogitLensService` was a process singleton that fixed `self._device = "cuda"`
at construction and loaded models with `device_map="auto"` — the weights spread
over every visible card while the unembedding went to whichever card was
current. The same instance serves every request in the API process.

Now the device is a required argument of every computation and the model cache
is keyed by (model, device). Where the lens runs decides how the card is chosen:

* API process (the synchronous dashboard route; the export BackgroundTask):
  `logit_lens_device` resolves the card and returns a device. It never makes a
  card current — that is per-process state every request shares.
* Celery workers (the queued dashboard task; the local Neuronpedia push):
  `place_job`, before any work. A refusal fails the job with its message.

MUTATION CONTROLS (each applied, run red, restored byte-identically):
  LL1  `_load_model` loads with device_map="auto" again
       -> test_the_model_loads_onto_one_card_cached_per_card
  LL2  `logit_lens_device` makes the card current with torch.cuda.set_device
       -> test_the_lens_runs_on_the_named_card_and_nothing_is_made_current
  LL3  `execute_export` swallows GpuPlacementError as a skipped stage
       -> test_a_refused_card_fails_the_export_instead_of_skipping_the_lens

"all" NEVER REACHES THE LENS (multi-GPU Phase 2, review round 1, 2026-09-14).
The resolver read "all" as a card name ("No GPU 'all' on this node", a 409 on
the dashboard route); the export, the queued dashboard task and the push
accepted it and failed later. Each route now refuses it at submit. Controls
(each alone, run red, restored byte-identically and checked by sha256):
  K1  `resolve_gpu_request` loses its refusal branch -> the dashboard, export and push cases
  K2  `logit_lens_device` resolves with resolve_card again
       -> TestAllIsRefusedWhenTheLensIsSubmitted::test_the_api_process_resolver_refuses_all_...
       (recorded before `resolve_card` itself refused "all", bdce4b6c; not re-run since)
  K5  export route stops refusing "all"              -> ...::test_the_export_is_a_400_and_no_export_starts
  K6  dashboard route stops refusing it              -> ...::test_the_dashboard_route_is_a_400_... (both)
  K7  push route stops refusing it                   -> ...::test_the_push_is_a_400_and_no_row_is_written
The routes refuse "all" only when the logit lens runs (`include_logit_lens`,
`compute_dashboard_data`, both default True): without it they do no GPU work;
see test_all_is_refused_where_a_job_cannot_split.py.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from fastapi import HTTPException

from src.api.v1.endpoints import neuronpedia as np_ep
from src.models.external_sae import SAEStatus
from src.models.neuronpedia_export import ExportStatus
from src.schemas.neuronpedia import ComputeDashboardDataRequest, NeuronpediaExportRequest
from src.services import gpu_placement
from src.services import logit_lens_service as logit_lens_module
from src.services import neuronpedia_export_service as export_module
from src.services import neuronpedia_local_service as local_module
from src.services.gpu_placement import GpuCard, GpuPlacementError, Placement
from src.services.logit_lens_service import LogitLensService, logit_lens_device
from src.services.neuronpedia_export_service import ExportConfig, NeuronpediaExportService
from src.services.neuronpedia_local_service import LocalPushConfig
from src.workers import neuronpedia_push_tasks as push_module
from src.workers.neuronpedia_tasks import compute_dashboard_data_task

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
UNKNOWN = "GPU-00000000-0000-0000-0000-000000000000"
CUDA_0 = torch.device("cuda", 0)
CUDA_1 = torch.device("cuda", 1)

CARDS = [
    GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000),
    GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000),
]

REFUSED = (
    "GPU 1 (NVIDIA GeForce RTX 3090, 900 of 24,576 MB free) cannot take this job: "
    "it needs ~4,000 MB. Choose another GPU or Auto."
)


@pytest.fixture(autouse=True)
def inventory(monkeypatch):
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)


@pytest.fixture
def cuda(monkeypatch):
    """This process sees both cards; returns every device made current."""
    made_current = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda index: SimpleNamespace(uuid=[TI_UUID, RTX_UUID][index][len("GPU-"):]),
    )
    monkeypatch.setattr(torch.cuda, "set_device", made_current.append)
    return made_current


@pytest.fixture
def restore_event_loop():
    """The Celery task installs and closes its own loop; do not leave it current."""
    yield
    asyncio.set_event_loop(None)


def _celery_task():
    task = MagicMock()
    task.delay.return_value = SimpleNamespace(id="celery-1")
    return task


def _lens_service():
    service = MagicMock()
    service.compute_logit_lens_for_sae = AsyncMock(return_value={0: "r0", 1: "r1"})
    service.save_logit_lens_results = AsyncMock()
    return service


# ── API process: the synchronous dashboard route ───────────────────────────

def _small(gpu="auto", **fields):
    return ComputeDashboardDataRequest(
        sae_id="sae_1", feature_indices=[0, 1],
        include_histograms=False, include_top_tokens=False, gpu=gpu, **fields,
    )


def _route(request, service, task):
    with patch("src.services.logit_lens_service.get_logit_lens_service", return_value=service), \
         patch.object(np_ep, "compute_dashboard_data_task", task):
        return asyncio.run(np_ep.compute_dashboard_data(request, background_tasks=MagicMock(), db=AsyncMock()))


class TestTheDashboardRouteInTheApiProcess:
    def test_the_lens_runs_on_the_named_card_and_nothing_is_made_current(self, cuda):
        service, task = _lens_service(), _celery_task()

        response = _route(_small(gpu="0"), service, task)

        assert service.compute_logit_lens_for_sae.await_count == 1
        assert service.compute_logit_lens_for_sae.await_args.kwargs["device"] == CUDA_0
        assert cuda == [], "the API process made a card current for every request it serves"
        assert task.delay.call_count == 0
        assert response.status == "completed"

    def test_auto_takes_the_card_with_the_most_free_memory(self, cuda):
        service = _lens_service()

        _route(_small(), service, _celery_task())

        assert service.compute_logit_lens_for_sae.await_args.kwargs["device"] == CUDA_1
        assert cuda == []

    def test_an_unknown_card_is_a_400_and_nothing_runs(self, cuda):
        service, task = _lens_service(), _celery_task()

        with pytest.raises(HTTPException) as exc:
            _route(_small(gpu=UNKNOWN), service, task)

        assert exc.value.status_code == 400
        assert service.compute_logit_lens_for_sae.await_count == 0
        assert task.delay.call_count == 0

    def test_a_named_card_this_process_cannot_use_is_a_409_with_the_reason(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        service = _lens_service()

        with pytest.raises(HTTPException) as exc:
            _route(_small(gpu="0"), service, _celery_task())

        assert exc.value.status_code == 409
        assert "CUDA is not available" in exc.value.detail
        assert service.compute_logit_lens_for_sae.await_count == 0

    def test_without_the_lens_no_card_is_resolved(self, monkeypatch):
        monkeypatch.setattr(
            logit_lens_module, "logit_lens_device", lambda requested: pytest.fail("resolved a card for no lens"),
        )
        response = _route(_small(include_logit_lens=False), _lens_service(), _celery_task())
        assert response.status == "completed"

    def test_a_large_request_queues_the_resolved_card(self, cuda):
        service, task = _lens_service(), _celery_task()

        _route(ComputeDashboardDataRequest(sae_id="sae_1", gpu="1"), service, task)

        assert task.delay.call_count == 1
        assert task.delay.call_args.kwargs["gpu_request"] == RTX_UUID
        assert service.compute_logit_lens_for_sae.await_count == 0


class TestTheApiProcessResolver:
    def test_it_returns_the_named_cards_device_without_making_it_current(self, cuda):
        assert logit_lens_device(RTX_UUID) == CUDA_1
        assert logit_lens_device(None) == CUDA_1  # an old job with no request: auto
        assert cuda == []

    def test_without_cuda_auto_runs_on_the_cpu_and_a_named_card_is_refused(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert logit_lens_device("auto") == torch.device("cpu")
        with pytest.raises(GpuPlacementError, match="CUDA is not available"):
            logit_lens_device(TI_UUID)


# ── "all": the logit lens never splits ─────────────────────────────────────

SPLIT_REFUSED = "cannot run split across GPUs"


class TestAllIsRefusedWhenTheLensIsSubmitted:
    """The lens runs on one device, so "all" can never be honoured. It used to be
    accepted everywhere a card could be named: the API-process resolver then
    reported it as a card that did not exist ("No GPU 'all' on this node"), and
    the export, the queued dashboard task and the push accepted the job and
    failed it later."""

    def test_the_api_process_resolver_refuses_all_as_a_split_not_as_a_missing_card(self, cuda):
        with pytest.raises(GpuPlacementError, match=SPLIT_REFUSED):
            logit_lens_device("all")
        assert cuda == []

    @pytest.mark.parametrize("request_", [
        _small(gpu="all"),
        ComputeDashboardDataRequest(sae_id="sae_1", gpu="all"),
    ], ids=["synchronous", "queued"])
    def test_the_dashboard_route_is_a_400_and_nothing_runs(self, cuda, request_):
        service, task = _lens_service(), _celery_task()

        with pytest.raises(HTTPException) as exc:
            _route(request_, service, task)

        assert exc.value.status_code == 400
        assert SPLIT_REFUSED in exc.value.detail
        assert service.compute_logit_lens_for_sae.await_count == 0
        assert task.delay.call_count == 0

    def test_the_export_is_a_400_and_no_export_starts(self):
        service = MagicMock()
        service.start_export = AsyncMock()
        background = MagicMock()

        with patch.object(np_ep, "get_neuronpedia_export_service", return_value=service):
            with pytest.raises(HTTPException) as exc:
                asyncio.run(np_ep.start_export(
                    NeuronpediaExportRequest(sae_id="sae_1", gpu="all"), background, db=AsyncMock(),
                ))

        assert exc.value.status_code == 400
        assert SPLIT_REFUSED in exc.value.detail
        assert service.start_export.await_count == 0
        assert background.add_task.call_count == 0

    def test_the_push_is_a_400_and_no_row_is_written(self, monkeypatch):
        monkeypatch.setattr(np_ep.settings, "neuronpedia_local_db_url", "postgresql://neuronpedia")
        db = MagicMock()
        db.get = AsyncMock(return_value=SimpleNamespace(n_features=4))
        db.add = MagicMock()
        task = _celery_task()

        with patch.object(push_module, "push_to_neuronpedia_local_task", task):
            with pytest.raises(HTTPException) as exc:
                asyncio.run(np_ep.push_to_local_neuronpedia(sae_id="sae_1", gpu="all", db=db))

        assert exc.value.status_code == 400
        assert SPLIT_REFUSED in exc.value.detail
        assert db.add.call_count == 0
        assert task.delay.call_count == 0


# ── Celery worker: the queued dashboard task ───────────────────────────────

def _session_factory():
    @asynccontextmanager
    async def session():
        yield AsyncMock()

    return session


class TestTheQueuedDashboardTask:
    def _run(self, place, service, **fields):
        with patch("src.services.gpu_placement.place_job", place), \
             patch("src.core.database.AsyncSessionLocal", _session_factory()), \
             patch("src.services.logit_lens_service.get_logit_lens_service", return_value=service):
            return compute_dashboard_data_task.run(
                sae_id="sae_1", feature_indices=[0, 1],
                include_histograms=False, include_top_tokens=False, **fields,
            )

    def test_the_placed_device_reaches_the_lens_and_the_card_is_reported(self, restore_event_loop):
        place = MagicMock(return_value=Placement(card=CARDS[1], device=CUDA_1))
        service = _lens_service()

        result = self._run(place, service, gpu_request=RTX_UUID)

        assert [c.args for c in place.call_args_list] == [(RTX_UUID,)]
        assert service.compute_logit_lens_for_sae.await_count == 1
        assert service.compute_logit_lens_for_sae.await_args.kwargs["device"] == CUDA_1
        assert result["gpu_uuid"] == RTX_UUID

    def test_a_refused_card_fails_the_task_with_its_message(self, restore_event_loop):
        place = MagicMock(side_effect=GpuPlacementError(REFUSED))
        service = _lens_service()

        with pytest.raises(GpuPlacementError, match="Choose another GPU or Auto"):
            self._run(place, service, gpu_request=RTX_UUID)

        assert service.compute_logit_lens_for_sae.await_count == 0

    def test_without_the_lens_nothing_is_placed(self, restore_event_loop):
        place = MagicMock()

        self._run(place, _lens_service(), include_logit_lens=False, gpu_request=RTX_UUID)

        assert place.call_count == 0


# ── The service itself ─────────────────────────────────────────────────────

class TestTheServiceUsesTheCallersDevice:
    def test_every_load_and_tensor_lands_on_the_callers_device(self, tmp_path, monkeypatch):
        """On the `meta` device, so a hard-coded "cuda" or "cpu" anywhere shows up
        as a tensor on the wrong device rather than passing by coincidence."""
        meta = torch.device("meta")
        service = LogitLensService()
        sae = SimpleNamespace(
            status=SAEStatus.READY.value, local_path=str(tmp_path), model_id=None,
            model_name="org/base", n_features=4, training_id=None,
        )
        db = MagicMock()
        db.get = AsyncMock(return_value=sae)

        sae_devices = []

        def load_sae(path, device):
            sae_devices.append(device)
            return {"W_dec": torch.randn(3, 4)}, None, None

        monkeypatch.setattr(logit_lens_module, "load_sae_auto_detect", load_sae)
        model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.randn(10, 3)))
        service._load_model = AsyncMock(return_value=(model, MagicMock()))
        seen = {}

        async def batch(W_dec, W_U, tokenizer, indices, k):
            seen["devices"] = (W_dec.device, W_U.device)
            return {}

        service._compute_batch_logit_lens = batch

        asyncio.run(service.compute_logit_lens_for_sae(db, "sae_1", [0, 1], force_recompute=True, device=meta))

        assert sae_devices == ["meta"]
        assert service._load_model.await_args.args == ("org/base", meta)
        assert seen["devices"] == (meta, meta)

    def test_the_device_is_required(self):
        with pytest.raises(TypeError):
            LogitLensService().compute_logit_lens_for_sae(MagicMock(), "sae_1")

    def test_the_model_loads_onto_one_card_cached_per_card(self, monkeypatch):
        from_pretrained = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
        monkeypatch.setattr(logit_lens_module.AutoTokenizer, "from_pretrained", MagicMock())
        monkeypatch.setattr(logit_lens_module.AutoModelForCausalLM, "from_pretrained", from_pretrained)
        service = LogitLensService()

        asyncio.run(service._load_model("org/base", CUDA_1))
        asyncio.run(service._load_model("org/base", CUDA_1))
        asyncio.run(service._load_model("org/base", CUDA_0))

        assert [c.kwargs["device_map"] for c in from_pretrained.call_args_list] == [{"": CUDA_1}, {"": CUDA_0}]


# ── API process: the export BackgroundTask ─────────────────────────────────

class TestTheExport:
    def test_the_route_resolves_before_the_job_exists_and_carries_the_card(self):
        service = MagicMock()
        service.start_export = AsyncMock(return_value="exp-1")
        background = MagicMock()

        with patch.object(np_ep, "get_neuronpedia_export_service", return_value=service):
            asyncio.run(np_ep.start_export(NeuronpediaExportRequest(sae_id="sae_1", gpu="1"), background, db=AsyncMock()))

        assert service.start_export.await_count == 1
        assert service.start_export.await_args.args[2].gpu == RTX_UUID
        assert background.add_task.call_count == 1

    def test_an_unknown_card_is_a_400_and_no_export_starts(self):
        service = MagicMock()
        service.start_export = AsyncMock()
        background = MagicMock()

        with patch.object(np_ep, "get_neuronpedia_export_service", return_value=service):
            with pytest.raises(HTTPException) as exc:
                asyncio.run(np_ep.start_export(
                    NeuronpediaExportRequest(sae_id="sae_1", gpu=UNKNOWN), background, db=AsyncMock(),
                ))

        assert exc.value.status_code == 400
        assert service.start_export.await_count == 0
        assert background.add_task.call_count == 0

    def test_the_card_is_stored_on_the_job_row_and_read_back(self):
        service = NeuronpediaExportService.__new__(NeuronpediaExportService)
        db = MagicMock()
        db.get = AsyncMock(return_value=SimpleNamespace(status=SAEStatus.READY.value, training_id=None))
        db.execute = AsyncMock(return_value=MagicMock())
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()

        asyncio.run(service.start_export(db, "sae_1", ExportConfig(gpu=RTX_UUID)))

        assert db.add.call_count == 1
        stored = db.add.call_args.args[0].config
        assert stored["gpu"] == RTX_UUID
        assert service._parse_config(stored).gpu == RTX_UUID
        assert service._parse_config({}).gpu is None  # an older row: auto

    def test_the_lens_stage_runs_on_the_stored_card(self):
        service = NeuronpediaExportService.__new__(NeuronpediaExportService)
        lens = _lens_service()
        requested = []

        with patch.object(export_module, "get_logit_lens_service", return_value=lens), \
             patch.object(logit_lens_module, "logit_lens_device", lambda r: requested.append(r) or CUDA_1):
            # Resolved where the lens's lease resolves it (multi-GPU Phase 3 review round 1, R1-5):
            # in single mode `logit_lens_lease` is exactly `logit_lens_device`.
            asyncio.run(service._compute_logit_lens(
                AsyncMock(), SimpleNamespace(id="sae_1"), [SimpleNamespace(neuron_index=0)],
                ExportConfig(gpu=RTX_UUID),
            ))

        assert requested == [RTX_UUID]
        assert lens.compute_logit_lens_for_sae.await_count == 1
        assert lens.compute_logit_lens_for_sae.await_args.kwargs["device"] == CUDA_1

    def test_a_refused_card_fails_the_export_instead_of_skipping_the_lens(self, tmp_path):
        service = NeuronpediaExportService.__new__(NeuronpediaExportService)
        service._exports_dir = tmp_path
        job = SimpleNamespace(
            id="exp-1", sae_id="sae_1", status="pending", started_at=None,
            config={"include_histograms": False, "include_top_tokens": False,
                    "include_saelens_format": False, "gpu": RTX_UUID},
            feature_count=None, error_message=None, completed_at=None,
        )
        db = MagicMock()
        db.get = AsyncMock(side_effect=[job, SimpleNamespace(id="sae_1", model_name="org/base", model_id=None)])
        db.commit = AsyncMock()
        service._load_features = AsyncMock(return_value=[])
        service._update_stage = AsyncMock()
        service._compute_logit_lens = AsyncMock(side_effect=GpuPlacementError(REFUSED))
        service._generate_json_files = AsyncMock()

        with pytest.raises(GpuPlacementError):
            asyncio.run(service.execute_export(db, "exp-1"))

        assert job.status == ExportStatus.FAILED.value
        assert job.error_message == REFUSED
        assert service._generate_json_files.await_count == 0, "exported without the lens that was asked for"


# ── Celery worker: the local Neuronpedia push ──────────────────────────────

class TestThePush:
    def test_the_route_resolves_before_the_tracking_row_and_queues_the_card(self, monkeypatch):
        monkeypatch.setattr(np_ep.settings, "neuronpedia_local_db_url", "postgresql://neuronpedia")
        db = MagicMock()
        db.get = AsyncMock(return_value=SimpleNamespace(n_features=4))
        db.add = MagicMock()
        db.commit = AsyncMock()
        task = _celery_task()

        with patch.object(push_module, "push_to_neuronpedia_local_task", task):
            asyncio.run(np_ep.push_to_local_neuronpedia(sae_id="sae_1", gpu="1", db=db))

        assert task.delay.call_count == 1
        assert task.delay.call_args.kwargs["gpu_request"] == RTX_UUID

    def test_an_unknown_card_is_a_400_and_no_row_is_written(self, monkeypatch):
        monkeypatch.setattr(np_ep.settings, "neuronpedia_local_db_url", "postgresql://neuronpedia")
        db = MagicMock()
        db.get = AsyncMock(return_value=SimpleNamespace(n_features=4))
        db.add = MagicMock()
        task = _celery_task()

        with patch.object(push_module, "push_to_neuronpedia_local_task", task):
            with pytest.raises(HTTPException) as exc:
                asyncio.run(np_ep.push_to_local_neuronpedia(sae_id="sae_1", gpu=UNKNOWN, db=db))

        assert exc.value.status_code == 400
        assert db.add.call_count == 0
        assert task.delay.call_count == 0

    def _run_task(self, place, **fields):
        session = AsyncMock()
        session.get = AsyncMock(return_value=SimpleNamespace(id="sae_1", n_features=4, name="sae"))

        @asynccontextmanager
        async def factory():
            yield session

        service = MagicMock()
        service.push_sae_to_local = AsyncMock(return_value=SimpleNamespace(
            success=True, neurons_created=4, activations_created=0, explanations_created=0,
            model_id="m", source_id="s", neuronpedia_url="u", error_message=None,
        ))
        service.close = AsyncMock()
        service_cls = MagicMock(return_value=service)

        with patch("src.services.gpu_placement.place_job", place), \
             patch.object(push_module, "AsyncSessionLocal", factory), \
             patch.object(push_module, "NeuronpediaLocalPushService", service_cls), \
             patch.object(push_module, "emit_neuronpedia_push_progress", MagicMock()):
            result = push_module.push_to_neuronpedia_local_task.run(
                push_job_id="push_1", sae_id="sae_1", **fields,
            )
        return result, service_cls, service

    def test_the_placed_device_travels_with_the_push_config(self):
        place = MagicMock(return_value=Placement(card=CARDS[1], device=CUDA_1))

        result, _, service = self._run_task(place, gpu_request=RTX_UUID)

        assert result["success"] is True
        assert [c.args for c in place.call_args_list] == [(RTX_UUID,)]
        assert service.push_sae_to_local.await_args.kwargs["config"].device == CUDA_1

    def test_a_refused_card_fails_the_push_with_its_message(self):
        place = MagicMock(side_effect=GpuPlacementError(REFUSED))

        result, service_cls, _ = self._run_task(place, gpu_request=RTX_UUID)

        assert result == {"success": False, "error": REFUSED}
        assert service_cls.call_count == 0

    def test_a_push_without_dashboard_data_is_never_placed(self):
        place = MagicMock()

        result, _, service = self._run_task(place, compute_dashboard_data=False, gpu_request=RTX_UUID)

        assert place.call_count == 0
        assert service.push_sae_to_local.await_args.kwargs["config"].device is None

    def test_the_dashboard_step_hands_the_placed_device_to_the_lens(self):
        lens = _lens_service()
        lens.compute_logit_lens_for_sae = AsyncMock(return_value={})
        service = local_module.NeuronpediaLocalPushService()

        with patch.object(local_module, "get_logit_lens_service", return_value=lens), \
             patch.object(local_module, "get_histogram_service", MagicMock(side_effect=RuntimeError("not under test"))), \
             patch.object(local_module, "logit_lens_device", lambda r: pytest.fail("resolved a card twice")):
            asyncio.run(service._compute_dashboard_data_if_needed(
                AsyncMock(), SimpleNamespace(id="sae_1", n_features=2),
                LocalPushConfig(feature_indices=[0, 1], device=CUDA_1),
            ))

        assert lens.compute_logit_lens_for_sae.await_count == 1
        assert lens.compute_logit_lens_for_sae.await_args.kwargs["device"] == CUDA_1
