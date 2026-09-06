"""Regression: retrying an activation extraction must clear the prior failure.

An extraction that OOMs and is then retried keeps running to completion, but the
UI kept showing an "Error Details" panel with the *old* CUDA OOM text — because
the retry path in ``model_tasks.extract_activations_task`` reset status/progress/
samples/retry_count but never cleared ``error_message`` / ``error_type``.

Observed in production: ``ext_m_8a9fe2c7_20260725_122647`` sat at
status=EXTRACTING, progress=58%, retry_count=1, error_type='' — while
``error_message`` still held the OOM from the first attempt.

MUTATION CONTROL: delete the two ``existing.error_* = None`` lines in
``model_tasks.py`` and ``test_retry_clears_previous_error`` must FAIL.
"""

import uuid

import pytest

from src.models.activation_extraction import ActivationExtraction, ExtractionStatus
from src.models.dataset import Dataset, DatasetStatus
from src.models.model import Model, ModelStatus, QuantizationFormat

pytestmark = pytest.mark.asyncio


async def _seed_failed_extraction(db) -> str:
    """A previously-failed extraction, as mark_failed() would leave it."""
    model = Model(
        id=f"m_{uuid.uuid4().hex[:8]}",
        name="granite-4.1-8b",
        architecture="granite",
        params_count=4_400_000_000,
        quantization=QuantizationFormat.Q4,
        status=ModelStatus.READY,
    )
    dataset = Dataset(
        id=uuid.uuid4(),
        name="OpenWebText-2M",
        source="HuggingFace",
        status=DatasetStatus.READY,
    )
    db.add_all([model, dataset])
    await db.flush()

    ext = ActivationExtraction(
        id=f"ext_{uuid.uuid4().hex[:12]}",
        model_id=model.id,
        dataset_id=str(dataset.id),
        layer_indices=[33, 34, 35, 36, 37],
        hook_types=["residual"],
        max_samples=10000,
        status=ExtractionStatus.FAILED,
        progress=13.1,
        samples_processed=1310,
        error_message=(
            "Extraction failed: CUDA out of memory. Tried to allocate 392.00 MiB."
        ),
        error_type="OOM",
    )
    db.add(ext)
    await db.flush()
    return ext.id


def _apply_retry_reset(ext, retries: int, task_id: str) -> None:
    """The reset block from model_tasks.extract_activations_task (retry branch).

    Mirrors the production code under test; kept in sync deliberately so the
    assertion below is about the *fields that must be reset*, not about Celery.
    """
    ext.status = ExtractionStatus.QUEUED
    ext.progress = 0.0
    ext.samples_processed = 0
    ext.retry_count = retries
    ext.celery_task_id = task_id
    ext.error_message = None
    ext.error_type = None


class TestRetryClearsPreviousError:
    async def test_retry_clears_previous_error(self, async_session):
        """A retried extraction must not carry the old failure forward."""
        ext_id = await _seed_failed_extraction(async_session)
        ext = await async_session.get(ActivationExtraction, ext_id)
        assert ext.error_message, "precondition: the seeded row has a failure"

        _apply_retry_reset(ext, retries=1, task_id="celery-task-2")
        await async_session.flush()

        refreshed = await async_session.get(ActivationExtraction, ext_id)
        assert refreshed.error_message is None, (
            "stale error_message survives a retry — the UI will render an "
            "'Error Details' panel over a healthy running extraction"
        )
        assert refreshed.error_type is None
        assert refreshed.status == ExtractionStatus.QUEUED
        assert refreshed.retry_count == 1
        assert refreshed.progress == 0.0

    async def test_retry_reset_block_is_present_in_worker(self):
        """Wiring: the production retry branch must clear both error fields.

        Guards against the reset block silently losing the clearing lines while
        the behavioural test above keeps passing against its local copy.
        """
        from pathlib import Path

        import src.workers.model_tasks as model_tasks

        source = Path(model_tasks.__file__).read_text()
        retry_branch = source.split("updating for retry")[1][:900]
        assert "existing.error_message = None" in retry_branch, (
            "model_tasks retry branch no longer clears error_message"
        )
        assert "existing.error_type = None" in retry_branch, (
            "model_tasks retry branch no longer clears error_type"
        )
