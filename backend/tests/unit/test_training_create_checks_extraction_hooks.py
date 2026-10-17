"""A training on cached activations is refused at create when an extraction lacks a requested hook (review R1-A, A5).

A training reads layer_{L}_{hook}.npy for every (layer, hook type) it trains. An extraction
records the hook types it captured, and nothing compared the two until the worker loaded the
files: a residual-only extraction behind a residual+mlp request was queued, waited for its
card, and failed at load. The create now answers 422 naming the missing hooks.

MUTATION CONTROLS (one line broken at a time, this file run, bytes restored, sha256 and the
working tree verified clean; the full table is in the A5 record):
  G1 the create's hook comparison removed               -> the service and endpoint refusals
  G2 the endpoint's UnprocessableTrainingRequest -> 422  -> the endpoint test (400, not 422)
"""

import uuid
from unittest.mock import patch

import pytest

from src.models.activation_extraction import ActivationExtraction, ExtractionStatus
from src.models.dataset import Dataset, DatasetStatus
from src.models.model import Model, ModelStatus, QuantizationFormat
from src.schemas.training import TrainingCreate, TrainingHyperparameters
from src.services.training_service import TrainingService, UnprocessableTrainingRequest


async def _seed(async_session, captured_hooks):
    model = Model(
        id="m_hooks", name="tiny", repo_id="org/tiny", status=ModelStatus.READY.value,
        quantization=QuantizationFormat.FP16.value, architecture="llama", params_count=1_000,
    )
    dataset = Dataset(id=uuid.uuid4(), name="corpus", source="HuggingFace", status=DatasetStatus.READY)
    async_session.add_all([model, dataset])
    await async_session.flush()
    async_session.add(ActivationExtraction(
        id="ext_m_hooks", model_id="m_hooks", dataset_id=str(dataset.id), layer_indices=[3],
        hook_types=list(captured_hooks), max_samples=10, status=ExtractionStatus.COMPLETED, progress=100.0,
    ))
    await async_session.commit()
    return str(dataset.id)


def _request(dataset_id, hooks):
    return TrainingCreate(
        model_id="m_hooks",
        dataset_ids=[dataset_id],
        extraction_ids=["ext_m_hooks"],
        hyperparameters=TrainingHyperparameters(
            hidden_dim=8, latent_dim=16, l1_alpha=0.001, learning_rate=0.0003, batch_size=64,
            total_steps=100, training_layers=[3], hook_types=list(hooks),
        ),
    )


@pytest.mark.asyncio
async def test_a_request_for_a_hook_the_extraction_never_captured_is_refused(async_session):
    dataset_id = await _seed(async_session, ["residual"])
    with patch("src.services.training_service._emit_training_event_sync"):
        with pytest.raises(UnprocessableTrainingRequest, match=r"no activations for \['mlp'\]"):
            await TrainingService.create_training(async_session, _request(dataset_id, ["residual", "mlp"]))


@pytest.mark.asyncio
async def test_a_request_the_extraction_can_serve_is_created(async_session):
    dataset_id = await _seed(async_session, ["residual", "mlp"])
    with patch("src.services.training_service._emit_training_event_sync"):
        training = await TrainingService.create_training(async_session, _request(dataset_id, ["residual", "mlp"]))
    assert training.hyperparameters["hook_types"] == ["residual", "mlp"]


@pytest.mark.asyncio
async def test_the_endpoint_answers_422_naming_the_missing_hook(client, async_session):
    dataset_id = await _seed(async_session, ["residual"])
    payload = _request(dataset_id, ["residual", "mlp"]).model_dump(mode="json")
    with patch("src.services.training_service._emit_training_event_sync"):
        response = await client.post("/api/v1/trainings", json=payload)
    assert response.status_code == 422, response.text
    assert "no activations for ['mlp']" in response.json()["detail"], response.text
