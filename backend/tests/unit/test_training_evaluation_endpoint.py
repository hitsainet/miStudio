"""``POST /trainings/{id}/evaluate`` and ``GET /trainings/{id}/evaluation`` (remediation item 6).

Reachability is asserted against the LIVE app — ``app.openapi()['paths']`` —
never ``app.routes`` (FastAPI 0.141 wraps included routers so their routes carry
no ``.path``) and never by importing the endpoint module. The dispatch is
asserted by PAYLOAD and CALL COUNT: a test that only checks "was called" passes
against a call sending the wrong arguments.

MUTATION CONTROLS (2026-09-15; each applied alone, this file run, source restored and
verified by sha256):
  E1 @router.post("/{training_id}/evaluate") deleted     RED  test_both_routes_are_in_the_live_app + the five POST tests
  E2 token_budget dropped from the dispatch payload      RED  test_it_records_pending_and_dispatches_once_with_the_payload
  E3 the pending/running guard disabled                  RED  test_an_evaluation_already_running_is_a_409_unless_forced
  E4 the completed-status guard disabled                 RED  test_a_training_that_is_not_completed_is_a_409
  E5 the task module removed from the worker include     SURVIVED here (the endpoint imports the module, so the test
                                                              process registers it anyway) — a real gap, now guarded by
                                                              test_training_evaluation_task_registered_in_worker.py (E5r RED)
  E6 the task's route removed                            RED  test_audit_mutation_pins::test_every_registered_task_routes_to_its_intended_queue
  E7 @router.get("/{training_id}/evaluation") deleted    RED  test_both_routes_are_in_the_live_app, test_get_returns_the_recorded_evaluation
"""

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from src.models.model import Model
from src.models.training import Training
from src.services import gpu_placement
from src.services.gpu_placement import GpuCard

RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
CARDS = [GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000)]


def test_both_routes_are_in_the_live_app():
    from src.main import app

    paths = app.openapi()["paths"]
    assert "post" in paths.get("/api/v1/trainings/{training_id}/evaluate", {}), (
        "the re-run endpoint is not registered; train_6247e768 cannot be evaluated"
    )
    assert "get" in paths.get("/api/v1/trainings/{training_id}/evaluation", {})


@pytest.fixture
def dispatched(monkeypatch):
    from src.api.v1.endpoints import trainings as endpoint
    from src.workers import training_evaluation_tasks

    calls = []
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)
    monkeypatch.setattr(
        endpoint, "dispatch_gpu_task",
        lambda task, **kwargs: calls.append((task, kwargs)) or SimpleNamespace(id=kwargs.get("task_id")),
    )
    return SimpleNamespace(calls=calls, task=training_evaluation_tasks.evaluate_training_task)


async def _training(session, tmp_path, monkeypatch, status="completed", export=True, evaluation=None,
                    gpu_request="auto"):
    from src.core.config import settings
    from src.models.model import ModelStatus, QuantizationFormat

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    session.add(Model(
        id="m_evaltest", name="tiny", architecture="llama", params_count=1_000,
        quantization=QuantizationFormat.FP16, status=ModelStatus.READY,
    ))
    session.add(Training(
        id="train_evaltest", model_id="m_evaltest", dataset_id="ds_x", dataset_ids=["ds_x"],
        status=status, progress=100.0, current_step=10, total_steps=10,
        hyperparameters={"hidden_dim": 8, "latent_dim": 16}, gpu_request=gpu_request, evaluation=evaluation,
    ))
    await session.commit()
    if export:
        (tmp_path / "trainings" / "train_evaltest" / "community_format").mkdir(parents=True)


async def _stored(session):
    row = (await session.execute(select(Training).where(Training.id == "train_evaltest"))).scalar_one()
    await session.refresh(row)
    return row.evaluation


@pytest.mark.asyncio
async def test_without_a_gpu_the_job_goes_where_the_training_asked_to_run(
    client, async_session, tmp_path, monkeypatch, dispatched
):
    """REVIEW R1-C C23. Every fixture's training requested "auto", which is also the
    fallback, so ignoring the training's own request survived the suite.

    NEGATIVE CONTROL: `gpu or "auto"` in the endpoint -> RED here.
    """
    await _training(async_session, tmp_path, monkeypatch, gpu_request=RTX_UUID)

    response = await client.post("/api/v1/trainings/train_evaltest/evaluate")

    assert response.status_code == 202, response.text
    [(task, kwargs)] = dispatched.calls
    assert kwargs["gpu_request"] == RTX_UUID
    assert kwargs["kwargs"] == {"training_id": "train_evaltest", "gpu_request": RTX_UUID, "token_budget": None}
    assert (await _stored(async_session))["gpu_request"] == RTX_UUID


@pytest.mark.asyncio
async def test_it_records_pending_and_dispatches_once_with_the_payload(
    client, async_session, tmp_path, monkeypatch, dispatched
):
    await _training(async_session, tmp_path, monkeypatch)

    response = await client.post("/api/v1/trainings/train_evaltest/evaluate?token_budget=65536")

    assert response.status_code == 202, response.text
    body = response.json()["data"]
    stored = await _stored(async_session)
    assert stored["status"] == "pending" and stored["trigger"] == "rerun"
    assert stored["task_id"] == body["task_id"]

    assert len(dispatched.calls) == 1
    task, kwargs = dispatched.calls[0]
    assert task is dispatched.task
    assert kwargs == {
        "gpu_request": "auto",
        "kwargs": {"training_id": "train_evaltest", "gpu_request": "auto", "token_budget": 65536},
        "task_id": body["task_id"],
    }


@pytest.mark.asyncio
async def test_an_unknown_training_is_a_404(client, dispatched):
    response = await client.post("/api/v1/trainings/train_nope/evaluate")
    assert response.status_code == 404
    assert dispatched.calls == []


@pytest.mark.asyncio
async def test_a_training_that_is_not_completed_is_a_409(client, async_session, tmp_path, monkeypatch, dispatched):
    await _training(async_session, tmp_path, monkeypatch, status="running")
    response = await client.post("/api/v1/trainings/train_evaltest/evaluate")
    assert response.status_code == 409 and "completed" in response.text
    assert dispatched.calls == [] and await _stored(async_session) is None


@pytest.mark.asyncio
async def test_a_training_with_no_export_is_a_409(client, async_session, tmp_path, monkeypatch, dispatched):
    await _training(async_session, tmp_path, monkeypatch, export=False)
    response = await client.post("/api/v1/trainings/train_evaltest/evaluate")
    assert response.status_code == 409 and "export" in response.text
    assert dispatched.calls == []


@pytest.mark.asyncio
async def test_an_evaluation_already_running_is_a_409_unless_forced(
    client, async_session, tmp_path, monkeypatch, dispatched
):
    await _training(async_session, tmp_path, monkeypatch, evaluation={"status": "running", "task_id": "old"})

    refused = await client.post("/api/v1/trainings/train_evaltest/evaluate")
    assert refused.status_code == 409 and dispatched.calls == []

    forced = await client.post("/api/v1/trainings/train_evaltest/evaluate?force=true")
    assert forced.status_code == 202 and len(dispatched.calls) == 1


@pytest.mark.asyncio
async def test_an_unknown_card_is_a_400_with_nothing_written_or_queued(
    client, async_session, tmp_path, monkeypatch, dispatched
):
    await _training(async_session, tmp_path, monkeypatch)
    response = await client.post(
        "/api/v1/trainings/train_evaltest/evaluate?gpu=GPU-00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 400
    assert dispatched.calls == [] and await _stored(async_session) is None


@pytest.mark.asyncio
async def test_get_returns_the_recorded_evaluation(client, async_session, tmp_path, monkeypatch, dispatched):
    await _training(async_session, tmp_path, monkeypatch, evaluation={"status": "completed", "ce_base": 2.5})
    response = await client.get("/api/v1/trainings/train_evaltest/evaluation")
    assert response.status_code == 200
    assert response.json()["data"] == {"status": "completed", "ce_base": 2.5}

    detail = await client.get("/api/v1/trainings/train_evaltest")
    assert detail.json()["evaluation"] == {"status": "completed", "ce_base": 2.5}
