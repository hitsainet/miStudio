"""Local-model labeling runs on the GPU the caller chose; a remote judge never touches one.

`LocalLabelingService` picked `"cuda"` for itself — whichever card CUDA called
current, which became the 12 GB 3080 Ti when a second card was added — and
logged `memory_allocated(0)` whatever card the model was on.

Now only a LOCAL job carries a GPU request. The endpoint resolves it before the
row exists (an unknown card is a 400), the row stores "auto" or a UUID, and the
worker places the job, records the card before the load, and hands the placed
device to the service. The OpenAI and OpenAI-compatible methods run on a remote
judge: they record no request and are never placed.

MUTATION CONTROLS (each applied, run red, restored byte-identically):
  L1  the local branch builds the judge on device="cpu" instead of the placed device
      -> test_the_local_judge_is_placed_recorded_then_loaded_on_that_card
  L2  judge_config stores the raw `gpu` instead of resolving it
      -> test_a_local_job_stores_the_uuid_the_index_names

PHASE 2 — a judge no single card holds (2026-09-14; each applied alone, this file
+ test_split_local_labeling.py + test_local_labeling_context.py run, restored and
checked by sha256). All went red:
  LS1 place_job without allow_shard       -> TestASplitJudge + both worker placement tests
  LS2 place_job with required_mb=None     -> TestASplitJudge + both worker placement tests
  LS3 only gpu_uuid recorded              -> TestASplitJudge
  LS4 the judge built with max_memory=None -> TestASplitJudge
  LS5 the judge built without device_map  -> TestASplitJudge, the_local_judge_is_placed_recorded_then_loaded
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
import torch
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from src.api.v1.endpoints import labeling as labeling_ep
from src.models.extraction_job import ExtractionJob
from src.models.feature import Feature
from src.models.labeling_job import LabelingJob, LabelingStatus
from src.schemas.labeling import LabelingConfigRequest
from src.services import gpu_placement
from src.services import labeling_service as labeling_service_module
from src.services.gpu_placement import GpuCard, GpuPlacementError, Placement
from src.services.labeling_service import LabelingService
from src.services.labeling_trial_service import LabelingTrialService, TrialError
from src.services.local_labeling_service import LocalLabelingService
from src.workers.labeling_tasks import label_features_task

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
UNKNOWN = "GPU-00000000-0000-0000-0000-000000000000"
CUDA_1 = torch.device("cuda", 1)

CARDS = [
    GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000),
    GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000),
]

EXAMPLE = {
    "prefix_tokens": ["In", "the"],
    "prime_token": "garden",
    "suffix_tokens": ["grew"],
    "max_activation": 5.0,
    "sample_index": 0,
}

REFUSED = (
    "GPU 1 (NVIDIA GeForce RTX 3090, 900 of 24,576 MB free) cannot take this job: "
    "it needs ~4,000 MB. Choose another GPU or Auto."
)

#: The judge's size as `local_judge_required_mb` would report it.
JUDGE_MB = 4_321.0


@pytest.fixture(autouse=True)
def inventory(monkeypatch):
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)


# ── Submit ─────────────────────────────────────────────────────────────────

def _response_row(method):
    """A real LabelingJob row the response model can validate."""
    now = datetime.now(timezone.utc)
    job = LabelingJob(
        id="label_1", extraction_job_id="extr_1", labeling_method=method,
        status=LabelingStatus.QUEUED.value, progress=0.0, features_labeled=0,
        total_features=1, statistics={}, created_at=now, updated_at=now,
    )
    for column in LabelingJob.__table__.columns:
        default = column.default
        if getattr(job, column.name, None) is None and default is not None and not callable(default.arg):
            setattr(job, column.name, default.arg)
    return job


def _celery_task():
    task = MagicMock()
    task.delay.return_value = SimpleNamespace(id="celery-1")
    return task


class TestTheLabelingEndpoint:
    def _start(self, config, method="local", endpoint=None):
        service = MagicMock()
        service.start_labeling = AsyncMock(return_value=_response_row(method))
        task = _celery_task()
        with patch.object(labeling_ep, "LabelingService", MagicMock(return_value=service)), \
             patch.object(labeling_ep, "label_features_task", task):
            call_endpoint = endpoint or labeling_ep.start_labeling
            asyncio.run(call_endpoint(config, db=AsyncMock()))
        return service, task

    def test_a_local_job_stores_the_uuid_the_index_names(self):
        service, task = self._start(
            LabelingConfigRequest(extraction_job_id="extr_1", labeling_method="local", gpu="1")
        )

        assert service.start_labeling.await_count == 1
        assert service.start_labeling.await_args.kwargs["config"]["gpu"] == RTX_UUID
        assert task.delay.call_args_list == [call("label_1")]

    def test_a_panel_job_resolves_the_same_way(self):
        service, task = self._start(
            labeling_ep.LabelingPanelRequest(
                extraction_job_id="extr_1", labeling_method="local", gpu="0", feature_ids=["f1"],
            ),
            endpoint=labeling_ep.start_labeling_panel,
        )

        assert service.start_labeling.await_args.kwargs["config"]["gpu"] == TI_UUID
        assert task.delay.call_count == 1

    def test_an_unknown_card_is_a_400_and_nothing_is_created_or_queued(self):
        service = MagicMock()
        service.start_labeling = AsyncMock()
        task = _celery_task()

        with patch.object(labeling_ep, "LabelingService", MagicMock(return_value=service)), \
             patch.object(labeling_ep, "label_features_task", task):
            with pytest.raises(HTTPException) as exc:
                asyncio.run(labeling_ep.start_labeling(
                    LabelingConfigRequest(extraction_job_id="extr_1", labeling_method="local", gpu=UNKNOWN),
                    db=AsyncMock(),
                ))

        assert exc.value.status_code == 400
        assert service.start_labeling.await_count == 0
        assert task.delay.call_count == 0

    @pytest.mark.parametrize("method", ["openai", "openai_compatible"])
    def test_a_remote_judge_ignores_the_field_and_never_reads_the_inventory(self, method, monkeypatch):
        monkeypatch.setattr(gpu_placement, "list_cards", lambda: pytest.fail("NVML read for a remote judge"))

        service, task = self._start(
            LabelingConfigRequest(extraction_job_id="extr_1", labeling_method=method, gpu=UNKNOWN),
            method=method,
        )

        assert "gpu" not in service.start_labeling.await_args.kwargs["config"]
        assert task.delay.call_count == 1


class TestTheResumeSweepEndpoint:
    def _start(self, config, sweep_service):
        db = AsyncMock()
        db.run_sync = AsyncMock(side_effect=lambda fn: fn(MagicMock()))
        step = MagicMock()
        with patch("src.services.labeling_sweep_service.LabelingSweepService", sweep_service), \
             patch.object(labeling_ep, "resume_sweep_step", step), \
             patch.object(labeling_ep, "_sweep_response", lambda sweep: sweep):
            asyncio.run(labeling_ep.start_resume_sweep(
                "extr_1", labeling_ep.ResumeSweepRequest(max_batches=1, config=config), db=db,
            ))
        return step

    def test_the_sweep_freezes_the_resolved_card(self):
        sweep_service = MagicMock()
        sweep_service.return_value.create.return_value = SimpleNamespace(id="sweep_1")

        step = self._start(
            LabelingConfigRequest(extraction_job_id="extr_1", labeling_method="local", gpu="0"),
            sweep_service,
        )

        assert sweep_service.return_value.create.call_count == 1
        assert sweep_service.return_value.create.call_args.kwargs["config"]["gpu"] == TI_UUID
        assert step.delay.call_args_list == [call("sweep_1")]

    def test_an_unknown_card_is_a_400_and_no_sweep_exists(self):
        sweep_service = MagicMock()

        with pytest.raises(HTTPException) as exc:
            self._start(
                LabelingConfigRequest(extraction_job_id="extr_1", labeling_method="local", gpu=UNKNOWN),
                sweep_service,
            )

        assert exc.value.status_code == 400
        assert sweep_service.return_value.create.call_count == 0


class TestTheRowRecordsARequestOnlyForALocalJudge:
    def test_a_local_row_stores_the_resolved_request(self):
        row = LabelingService.build_labeling_job_row(
            job_id="j", extraction_job_id="e", total_features=1,
            config={"labeling_method": "local", "gpu": RTX_UUID},
        )
        assert (row.gpu_request, row.gpu_uuid) == (RTX_UUID, None)

    def test_a_remote_row_stores_none_even_if_the_config_names_a_card(self):
        """A sweep config frozen before the endpoint stripped it must not name a card."""
        row = LabelingService.build_labeling_job_row(
            job_id="j", extraction_job_id="e", total_features=1,
            config={"labeling_method": "openai_compatible", "gpu": RTX_UUID},
        )
        assert row.gpu_request is None


class TestATrialCannotAskForALocalJudge:
    def test_a_local_trial_is_refused_before_any_read_or_write(self):
        db = MagicMock(spec=AsyncSession)
        db.execute = AsyncMock()
        db.add = MagicMock()

        with pytest.raises(TrialError, match="not supported for trials"):
            asyncio.run(LabelingTrialService(db).start_trial("extr_1", ["f1"], {"labeling_method": "local"}))

        assert db.execute.await_count == 0
        assert db.add.call_count == 0

    def test_the_trial_endpoint_answers_422(self):
        db = MagicMock(spec=AsyncSession)
        db.execute = AsyncMock()

        with pytest.raises(HTTPException) as exc:
            asyncio.run(labeling_ep.start_labeling_trial(
                labeling_ep.LabelingTrialRequest(
                    extraction_job_id="extr_1", feature_ids=["f1"], labeling_method="local",
                ),
                db=db,
            ))

        assert exc.value.status_code == 422


# ── Start ──────────────────────────────────────────────────────────────────

class _PassThroughFilter:
    def filter_features_from_examples(self, features, examples, all_examples, verdict_examples=None):
        n = len(features)
        stats = {"features_to_label": n, "total_features": n, "features_skipped": 0, "skip_percentage": 0.0}
        return features, examples, all_examples, stats


def _job(method, **fields):
    return LabelingJob(
        id="label_1", extraction_job_id="extr_1", labeling_method=method,
        status=LabelingStatus.QUEUED.value, progress=0.0, features_labeled=0,
        total_features=1, statistics={"batch_size": 10}, **fields,
    )


def _session(job):
    features = [SimpleNamespace(id="feat_1", neuron_index=0, nlp_analysis=None, category="semantic", name="garden")]
    session = Mock(spec=Session)

    def query(model):
        q = Mock()
        q.filter.return_value = q
        q.order_by.return_value = q
        q.first.return_value = None
        q.all.return_value = []
        if model is LabelingJob:
            q.first.return_value = job
        elif model is ExtractionJob:
            q.first.return_value = SimpleNamespace(id="extr_1")
        elif model is Feature:
            q.all.return_value = features
        return q

    session.query = Mock(side_effect=query)
    return session


def _drive(job, place, local_cls):
    """Run the real `label_features_for_extraction`; storage and judges are stubbed."""
    service = LabelingService(_session(job))
    service._retrieve_top_examples_batch_sync = Mock(return_value={"feat_1": [dict(EXAMPLE)]})
    service._retrieve_bottom_examples_batch_sync = Mock(return_value={})
    service._persist_filtered_out = Mock()
    service._claim_features = Mock()
    service._raise_if_cancelled = Mock()
    service._persist_label_outcome = Mock(return_value=LabelingService.LABEL_STATUS_SUCCEEDED)

    remote = MagicMock(side_effect=RuntimeError("stop: the remote judge was constructed"))
    with patch("src.services.gpu_placement.place_job", place), \
         patch.object(labeling_service_module, "local_judge_required_mb", lambda name: JUDGE_MB), \
         patch("src.utils.token_filter.get_feature_filter", return_value=_PassThroughFilter()), \
         patch.object(labeling_service_module, "LocalLabelingService", local_cls), \
         patch.object(labeling_service_module, "OpenAILabelingService", remote), \
         patch.object(labeling_service_module, "emit_labeling_progress", MagicMock()):
        return service.label_features_for_extraction(job.id)


def _local_judge_class(job, seen):
    def construct(**kwargs):
        seen["kwargs"] = kwargs
        seen["uuid_at_construction"] = job.gpu_uuid
        seen["uuids_at_construction"] = job.gpu_uuids
        judge = MagicMock()
        judge.generate_label.return_value = {"category": "semantic", "specific": "garden", "description": ""}
        return judge

    return MagicMock(side_effect=construct)


class TestTheWorkerPlacesOnlyALocalJudge:
    def test_the_local_judge_is_placed_recorded_then_loaded_on_that_card(self):
        job = _job("local", local_model="org/judge", gpu_request=RTX_UUID)
        place = MagicMock(return_value=Placement(card=CARDS[1], device=CUDA_1))
        seen = {}
        local_cls = _local_judge_class(job, seen)

        _drive(job, place, local_cls)

        assert place.call_args_list == [call(RTX_UUID, required_mb=JUDGE_MB, allow_shard=True)]
        assert local_cls.call_count == 1
        assert seen["kwargs"] == {
            "model_name": "org/judge", "device": CUDA_1, "device_map": "cuda:1", "max_memory": None,
        }
        assert seen["uuid_at_construction"] == RTX_UUID, "the card was not recorded before the load"
        assert job.gpu_uuid == RTX_UUID
        assert job.status == LabelingStatus.COMPLETED.value

    def test_an_old_row_with_no_request_is_placed_as_auto(self):
        job = _job("local", local_model="org/judge")
        place = MagicMock(return_value=Placement(card=CARDS[1], device=CUDA_1))

        _drive(job, place, _local_judge_class(job, {}))

        assert place.call_args_list == [call(None, required_mb=JUDGE_MB, allow_shard=True)]

    def test_a_refused_card_fails_the_job_with_its_message(self):
        job = _job("local", local_model="org/judge", gpu_request=RTX_UUID)
        place = MagicMock(side_effect=GpuPlacementError(REFUSED))
        local_cls = MagicMock()

        with pytest.raises(GpuPlacementError):
            _drive(job, place, local_cls)

        assert job.status == LabelingStatus.FAILED.value
        assert job.error_message == REFUSED
        assert job.gpu_uuid is None
        assert local_cls.call_count == 0, "a refused card must not be swapped for another"

    @pytest.mark.parametrize("method", ["openai", "openai_compatible"])
    def test_a_remote_judge_is_never_placed(self, method):
        # Even a stray request on the row must not place a remote job.
        job = _job(method, gpu_request=RTX_UUID)
        place = MagicMock()
        local_cls = MagicMock()

        with pytest.raises(Exception):
            _drive(job, place, local_cls)

        assert place.call_count == 0
        assert local_cls.call_count == 0
        assert job.gpu_uuid is None
        assert job.status == LabelingStatus.FAILED.value

    def test_a_refused_card_is_not_retried(self):
        assert not issubclass(GpuPlacementError, label_features_task.autoretry_for)


class TestTheLocalJudgeUsesItsDevice:
    def test_it_never_chooses_a_device_itself(self):
        with pytest.raises(TypeError):
            LocalLabelingService(model_name="org/judge")

    def test_the_model_loads_onto_the_placed_card_and_reads_that_cards_memory(self, monkeypatch):
        import src.services.local_labeling_service as module

        from_pretrained = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", MagicMock())
        monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", from_pretrained)
        asked = []
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: asked.append(device) or 0)

        LocalLabelingService(model_name="org/judge", device=CUDA_1).load_model()

        kwargs = from_pretrained.call_args.kwargs
        assert kwargs["device_map"] == {"": CUDA_1}
        assert kwargs["load_in_4bit"] is True
        assert asked == [CUDA_1]

    def test_on_the_cpu_nothing_is_quantised_and_no_gpu_memory_is_read(self, monkeypatch):
        import src.services.local_labeling_service as module

        from_pretrained = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", MagicMock())
        monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", from_pretrained)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: pytest.fail("read GPU memory"))

        LocalLabelingService(model_name="org/judge", device="cpu").load_model()

        kwargs = from_pretrained.call_args.kwargs
        assert kwargs["device_map"] == {"": torch.device("cpu")}
        assert kwargs["load_in_4bit"] is False

    def test_the_prompt_goes_to_the_judges_card(self):
        moved = []

        class _Inputs(dict):
            def to(self, device):
                moved.append(device)
                return self

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.return_value = _Inputs(input_ids=torch.zeros(1, 3, dtype=torch.long))
        tokenizer.decode.return_value = '{"specific": "garden", "category": "semantic", "description": "d"}'
        model = MagicMock()
        model.generate.return_value = torch.zeros(1, 5, dtype=torch.long)

        judge = LocalLabelingService(model_name="org/judge", device=CUDA_1)
        judge.model, judge.tokenizer, judge.is_loaded = model, tokenizer, True
        label = judge.generate_label(examples=[dict(EXAMPLE)], feature_id="feat_1")

        assert moved == [CUDA_1]
        assert label["specific"] == "garden"

    def test_unloading_clears_and_reads_the_judges_card(self, monkeypatch):
        entered, asked = [], []

        class _DeviceContext:
            def __init__(self, device):
                entered.append(device)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(torch.cuda, "device", _DeviceContext)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: asked.append(device) or 0)

        judge = LocalLabelingService(model_name="org/judge", device=CUDA_1)
        judge.model, judge.tokenizer, judge.is_loaded = object(), object(), True
        judge.unload_model()

        assert entered == [CUDA_1]
        assert asked == [CUDA_1]


class TestASplitJudge:
    """Phase 2: a judge no single card holds is placed across cards, every card recorded."""

    def test_the_split_is_recorded_before_the_load_and_its_budget_reaches_the_judge(self):
        cuda_0 = torch.device("cuda", 0)
        split = Placement(
            card=CARDS[1], device=CUDA_1, cards=(CARDS[1], CARDS[0]), devices=(CUDA_1, cuda_0),
            max_memory_mb={1: 21_976, 0: 9_976},
        )
        job = _job("local", local_model="org/judge", gpu_request="all")
        place = MagicMock(return_value=split)
        seen = {}

        _drive(job, place, _local_judge_class(job, seen))

        assert place.call_args_list == [call("all", required_mb=JUDGE_MB, allow_shard=True)]
        assert seen["kwargs"] == {
            "model_name": "org/judge", "device": CUDA_1,
            "device_map": split.device_map, "max_memory": {1: "21976MiB", 0: "9976MiB"},
        }
        assert seen["uuids_at_construction"] == [RTX_UUID, TI_UUID], "the split's cards were not recorded before the load"
        assert (job.gpu_uuid, job.gpu_uuids) == (RTX_UUID, [RTX_UUID, TI_UUID])
