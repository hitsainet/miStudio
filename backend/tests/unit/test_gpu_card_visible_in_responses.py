"""Every GPU job's API response shows the card it asked for and the card it ran on.

Multi-GPU Phase 1 recorded `gpu_request` / `gpu_uuid` on every job row, but found
on 2026-09-13 that only trainings, labeling and steering RETURNED them: J-lens
runs and model downloads (task_queue), SAE feature extractions, activation
extractions and circuit captures wrote the card to a column no response read.
A recorded value nobody can see is the reachability failure this repo's
CLAUDE.md warns about. These tests read each serializer's OUTPUT.

MUTATION CONTROLS (2026-09-13, each red, restored byte-identically):
  V1 task_queue._serialize_task drops "gpu_uuid"               -> task-queue test fails
  V2 circuit_discovery._capture_out drops the gpu pair         -> capture test fails
  V3 ExtractionService.list_extractions drops "gpu_uuid"       -> SAE list test fails
  V4 get_extraction_status_for_sae drops "gpu_request"         -> SAE status test fails
  V5 list_model_extractions drops "gpu_uuid"                   -> activation list test fails
  V6 get_active_extraction drops "gpu_request"                 -> activation active test fails
"""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
NOW = datetime(2026, 9, 13, 21, 0, tzinfo=timezone.utc)


def _find(obj, key, value):
    """The first dict anywhere in `obj` whose `key` equals `value`."""
    if isinstance(obj, dict):
        if obj.get(key) == value:
            return obj
        children = obj.values()
    elif isinstance(obj, (list, tuple)):
        children = obj
    else:
        return None
    for child in children:
        found = _find(child, key, value)
        if found is not None:
            return found
    return None


# ── task_queue: J-lens runs and model downloads ────────────────────────────

def test_a_task_queue_row_reports_the_card_it_ran_on():
    from src.api.v1.endpoints.task_queue import _serialize_task
    from src.schemas.task_queue import TaskQueueData

    task = SimpleNamespace(
        id="tq_1", task_id="celery-1", task_type="jlens_readout", entity_id="m_1",
        entity_type="model", status="completed", progress=100.0, error_message=None,
        retry_params={"gpu": RTX_UUID}, retry_count=0, gpu_uuid=RTX_UUID,
        created_at=NOW, started_at=NOW, completed_at=NOW, updated_at=NOW,
    )

    row = TaskQueueData(**_serialize_task(task, entity_info=None))

    assert row.gpu_uuid == RTX_UUID


# ── circuit captures ───────────────────────────────────────────────────────

def test_a_circuit_capture_reports_its_request_and_card():
    from src.api.v1.endpoints.circuit_discovery import _capture_out

    run = SimpleNamespace(
        id="cap_1", status="completed", progress=100.0, error_message=None, manifest={},
        bytes_total=1, events_total=1, stale=False, gpu_request="auto", gpu_uuid=RTX_UUID,
        created_at=NOW, updated_at=NOW,
    )

    out = _capture_out(run)

    assert (out["gpu_request"], out["gpu_uuid"]) == ("auto", RTX_UUID)


# ── SAE feature extractions ────────────────────────────────────────────────

class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one(self):
        return self.value

    def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        return SimpleNamespace(all=lambda: list(self.value))

    def all(self):
        return list(self.value)


class _Session:
    """An AsyncSession answering `execute` from a queue, in order."""

    def __init__(self, *answers):
        self.answers = list(answers)

    async def execute(self, statement):
        return _Result(self.answers.pop(0))


def _sae_job(**fields):
    job = dict(
        id="extr_1", training_id=None, external_sae_id="sae_a", config={"dataset_id": "ds_1"},
        status="completed", progress=100.0, features_extracted=10, total_features=10,
        statistics={}, layer_index=None, hook_type=None, error_message=None,
        created_at=NOW, updated_at=NOW, completed_at=NOW,
        filter_special=True, filter_single_char=True, filter_punctuation=True,
        filter_numbers=True, filter_fragments=True, filter_stop_words=False,
        context_prefix_tokens=25, context_suffix_tokens=25,
        nlp_status=None, nlp_progress=0.0, nlp_processed_count=0, nlp_error_message=None,
        gpu_request=RTX_UUID, gpu_uuid=RTX_UUID,
    )
    job.update(fields)
    return SimpleNamespace(**job)


SAE = SimpleNamespace(id="sae_a", name="sae a", model_name="LFM2.5-1.2B", model_id="m_1")


def test_the_sae_extraction_list_reports_request_and_card_through_its_schema():
    from src.schemas.extraction import ExtractionStatusResponse
    from src.services.extraction_service import ExtractionService

    session = _Session(1, [_sae_job()], [SAE], [("m_1", "LFM2.5-1.2B")], [("ds_1", "owt")])

    rows, total = asyncio.run(ExtractionService(session).list_extractions())

    assert total == 1
    response = ExtractionStatusResponse(**rows[0])
    assert (response.gpu_request, response.gpu_uuid) == (RTX_UUID, RTX_UUID)


def test_the_sae_extraction_status_reports_request_and_card():
    from src.services.extraction_service import ExtractionService

    session = _Session(_sae_job(gpu_request="auto", gpu_uuid=RTX_UUID), SAE)

    status = asyncio.run(ExtractionService(session).get_extraction_status_for_sae("sae_a"))

    assert (status["gpu_request"], status["gpu_uuid"]) == ("auto", RTX_UUID)


# ── activation extractions ─────────────────────────────────────────────────

def _activation_endpoints(monkeypatch):
    from src.api.v1.endpoints import models as endpoint
    from src.services import activation_service

    extraction = SimpleNamespace(
        id="ext_1", model_id="m_1", dataset_id="ds_1", celery_task_id="celery-1",
        status=SimpleNamespace(value="extracting"), phase="extracting", progress=40.0, samples_processed=4,
        max_samples=10, layer_indices=[11], hook_types=["residual"], batch_size=16,
        gpu_request="auto", gpu_uuid=RTX_UUID, created_at=NOW, updated_at=NOW,
        completed_at=None, error_message=None, statistics={}, saved_files=[],
    )

    class _NoTrainings:
        """The list's deletion-eligibility query: no training uses the extraction."""

        def query(self, *columns):
            return self

        def filter(self, *conditions):
            return self

        def all(self):
            return []

    @contextmanager
    def sync_db():
        yield _NoTrainings()

    monkeypatch.setattr(endpoint.ModelService, "get_model", AsyncMock(return_value=SimpleNamespace(id="m_1", name="LFM2.5-1.2B-Instruct")))
    monkeypatch.setattr(endpoint, "get_sync_db", sync_db)
    monkeypatch.setattr(
        endpoint.ExtractionDatabaseService, "list_extractions_for_model",
        staticmethod(lambda db, model_id, limit=50: [extraction]),
    )
    monkeypatch.setattr(
        endpoint.ExtractionDatabaseService, "get_active_extraction_for_model",
        staticmethod(lambda db, model_id: extraction),
    )
    monkeypatch.setattr(activation_service.ActivationService, "__init__", lambda self: None)
    monkeypatch.setattr(activation_service.ActivationService, "list_extractions", lambda self: [])
    return endpoint


def test_the_activation_extraction_list_reports_request_and_card(monkeypatch):
    endpoint = _activation_endpoints(monkeypatch)

    listed = asyncio.run(endpoint.list_model_extractions("m_1", db=None))

    row = _find(listed, "extraction_id", "ext_1")
    assert row is not None
    assert (row["gpu_request"], row["gpu_uuid"]) == ("auto", RTX_UUID)


def test_the_active_activation_extraction_reports_request_and_card(monkeypatch):
    endpoint = _activation_endpoints(monkeypatch)

    active = asyncio.run(endpoint.get_active_extraction("m_1", db=None))

    assert (active["data"]["gpu_request"], active["data"]["gpu_uuid"]) == ("auto", RTX_UUID)
