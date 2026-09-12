"""
Tests for the three deferred review remediations:

- 003-F2: training_metrics UNIQUE(training_id, step, layer_idx)
- 007-F1: NeuronpediaPushJob ORM model maps the neuronpedia_pushes table
- 002-F3: models.celery_task_id column exists (revoke-on-cancel plumbing)
"""

import pytest
from sqlalchemy.exc import IntegrityError

from src.models.training import Training, TrainingStatus
from src.models.training_metric import TrainingMetric
from src.models.model import Model, ModelStatus, QuantizationFormat
from src.models.neuronpedia_push import NeuronpediaPushJob, NeuronpediaPushStatus


async def _make_training(session) -> str:
    """Create a minimal Training row and return its id."""
    training = Training(
        id="train_uqtest",
        model_id="m_uqtest",
        dataset_id="ds_uqtest",
        dataset_ids=["ds_uqtest"],
        status=TrainingStatus.PENDING.value,
        progress=0.0,
        current_step=0,
        total_steps=100,
        hyperparameters={"total_steps": 100},
    )
    # A model row is needed for the FK (models.id RESTRICT).
    session.add(
        Model(
            id="m_uqtest",
            name="uqtest",
            architecture="gpt2",
            params_count=0,
            quantization=QuantizationFormat.FP32,
            status=ModelStatus.READY,
        )
    )
    session.add(training)
    await session.commit()
    return training.id


@pytest.mark.asyncio
async def test_training_metric_unique_constraint(async_session):
    """003-F2: a duplicate (training_id, step, layer_idx) row must be rejected."""
    tid = await _make_training(async_session)

    async_session.add(TrainingMetric(training_id=tid, step=10, layer_idx=0, loss=1.0))
    await async_session.commit()

    # Same (training_id, step, layer_idx) → IntegrityError
    async_session.add(TrainingMetric(training_id=tid, step=10, layer_idx=0, loss=2.0))
    with pytest.raises(IntegrityError):
        await async_session.commit()
    await async_session.rollback()


@pytest.mark.asyncio
async def test_training_metric_null_layer_is_distinct(async_session):
    """003-F2: aggregated (layer_idx NULL) and per-layer rows coexist at same step."""
    tid = await _make_training(async_session)

    async_session.add(TrainingMetric(training_id=tid, step=5, layer_idx=None, loss=1.0))
    async_session.add(TrainingMetric(training_id=tid, step=5, layer_idx=0, loss=1.1))
    async_session.add(TrainingMetric(training_id=tid, step=5, layer_idx=1, loss=1.2))
    # Two NULL-layer rows at the same step are allowed (NULLs distinct in PG).
    async_session.add(TrainingMetric(training_id=tid, step=5, layer_idx=None, loss=1.3))
    await async_session.commit()  # must not raise


@pytest.mark.asyncio
async def test_neuronpedia_push_job_maps_table(async_session):
    """007-F1: NeuronpediaPushJob round-trips through the neuronpedia_pushes table."""
    job = NeuronpediaPushJob(
        id="push_test_1",
        sae_id="sae_test",
        status=NeuronpediaPushStatus.QUEUED.value,
        progress=0,
        features_pushed=0,
        total_features=42,
    )
    async_session.add(job)
    await async_session.commit()

    fetched = await async_session.get(NeuronpediaPushJob, "push_test_1")
    assert fetched is not None
    assert fetched.total_features == 42
    assert fetched.status == "queued"


def test_model_has_celery_task_id_column():
    """002-F3: the celery_task_id column exists on the Model mapping."""
    assert "celery_task_id" in {c.name for c in Model.__table__.columns}
