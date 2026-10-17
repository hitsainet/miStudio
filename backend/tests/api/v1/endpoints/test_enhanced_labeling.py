"""
Tests for enhanced per-feature labeling API endpoints.

POST /features/{feature_id}/label/enhanced
GET  /features/{feature_id}/label/enhanced/latest
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.feature import Feature
from src.models.enhanced_labeling_job import EnhancedLabelingJob, EnhancedLabelingStatus


pytestmark = pytest.mark.asyncio


# ── shared helpers ────────────────────────────────────────────────────────────

async def _make_extraction_job(session: AsyncSession, job_id: str = "test-ext-001") -> ExtractionJob:
    job = ExtractionJob(
        id=job_id,
        config={"evaluation_samples": 100, "top_k_examples": 20},
        status=ExtractionStatus.COMPLETED,
    )
    session.add(job)
    await session.flush()
    return job


async def _make_feature(
    session: AsyncSession,
    extraction_job_id: str,
    feature_id: str = "feat_test_001",
    neuron_index: int = 42,
) -> Feature:
    feature = Feature(
        id=feature_id,
        extraction_job_id=extraction_job_id,
        neuron_index=neuron_index,
        name="test_feature",
        activation_frequency=0.05,
        interpretability_score=0.8,
        max_activation=5.0,
        label_source="auto",
    )
    session.add(feature)
    await session.flush()
    return feature


# ── POST /features/{id}/label/enhanced ───────────────────────────────────────

class TestStartEnhancedLabeling:
    """Tests for POST /api/v1/features/{feature_id}/label/enhanced."""

    async def test_feature_not_found_returns_404(self, client: AsyncClient):
        response = await client.post("/api/v1/features/does-not-exist/label/enhanced")
        assert response.status_code == 404

    async def test_missing_endpoint_setting_returns_400(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        ext_job = await _make_extraction_job(async_session)
        feature = await _make_feature(async_session, ext_job.id)

        with patch(
            "src.api.v1.endpoints.enhanced_labeling.AppSettingService.get_decrypted_value",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.post(f"/api/v1/features/{feature.id}/label/enhanced")

        assert response.status_code == 400
        assert "openai_compatible_endpoint" in response.json()["detail"]

    async def test_success_queues_job_and_returns_201(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        ext_job = await _make_extraction_job(async_session)
        feature = await _make_feature(async_session, ext_job.id)

        fake_celery = MagicMock()
        fake_celery.id = "celery-task-xyz"

        with (
            patch(
                "src.api.v1.endpoints.enhanced_labeling.AppSettingService.get_decrypted_value",
                new_callable=AsyncMock,
                side_effect=[
                    "openai_compatible",   # enhanced_labeling_method
                    "http://llm.local/v1",  # openai_compatible_endpoint
                    "my-model",             # openai_compatible_model
                    "4",                    # enhanced_labeling_max_workers
                ],
            ),
            patch(
                "src.workers.enhanced_labeling_tasks.enhanced_label_feature_task.delay",
                return_value=fake_celery,
            ),
        ):
            response = await client.post(f"/api/v1/features/{feature.id}/label/enhanced")

        assert response.status_code == 201
        data = response.json()
        assert data["feature_id"] == feature.id
        assert data["status"] == "queued"
        assert data["method"] == "openai_compatible"
        # MIS-E2E-111: `endpoint` is deliberately NOT in the response. It is
        # the internal LLM server URL, and this schema published it to the
        # browser while its sibling withheld it. These two assertions used to
        # require the leak, which is a test pinning a defect rather than
        # preventing it.
        assert "endpoint" not in data, (
            "the internal LLM endpoint URL is back in the response body"
        )

        assert data["model"] == "my-model"
        assert data["workers"] == 4
        assert data["examples_total"] == 20
        assert data["examples_completed"] == 0
        assert data["celery_task_id"] == "celery-task-xyz"

    async def test_success_openai_method_uses_openai_endpoint(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        ext_job = await _make_extraction_job(async_session, job_id="test-ext-oai")
        feature = await _make_feature(
            async_session, ext_job.id, feature_id="feat_oai_001", neuron_index=101
        )

        fake_celery = MagicMock()
        fake_celery.id = "celery-task-oai"

        with (
            patch(
                "src.api.v1.endpoints.enhanced_labeling.AppSettingService.get_decrypted_value",
                new_callable=AsyncMock,
                side_effect=[
                    "openai",                # enhanced_labeling_method
                    "gpt-4o-mini",           # enhanced_labeling_openai_model
                    "sk-test-key",           # openai_api_key
                    "4",                     # enhanced_labeling_max_workers
                ],
            ),
            patch(
                "src.workers.enhanced_labeling_tasks.enhanced_label_feature_task.delay",
                return_value=fake_celery,
            ),
        ):
            response = await client.post(f"/api/v1/features/{feature.id}/label/enhanced")

        assert response.status_code == 201
        data = response.json()
        assert data["method"] == "openai"
        assert "endpoint" not in data, "see MIS-E2E-111 above"
        assert data["model"] == "gpt-4o-mini"

    async def test_openai_method_without_api_key_returns_400(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        ext_job = await _make_extraction_job(async_session, job_id="test-ext-oai2")
        feature = await _make_feature(
            async_session, ext_job.id, feature_id="feat_oai_002", neuron_index=102
        )

        with patch(
            "src.api.v1.endpoints.enhanced_labeling.AppSettingService.get_decrypted_value",
            new_callable=AsyncMock,
            side_effect=[
                "openai",         # enhanced_labeling_method
                "gpt-4o-mini",    # enhanced_labeling_openai_model
                None,             # openai_api_key (missing)
            ],
        ):
            response = await client.post(f"/api/v1/features/{feature.id}/label/enhanced")

        assert response.status_code == 400
        assert "openai_api_key" in response.json()["detail"]

    async def test_duplicate_active_job_returns_existing(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        ext_job = await _make_extraction_job(async_session)
        feature = await _make_feature(async_session, ext_job.id)

        existing = EnhancedLabelingJob(
            id=f"elj_42_existing",
            feature_id=feature.id,
            status=EnhancedLabelingStatus.RUNNING.value,
            endpoint="http://llm.local/v1",
            model="existing-model",
            workers=4,
            examples_total=20,
        )
        async_session.add(existing)
        await async_session.flush()

        # Should return the existing job without creating a new one or calling task
        response = await client.post(f"/api/v1/features/{feature.id}/label/enhanced")

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == existing.id
        assert data["status"] == "running"
        assert data["model"] == "existing-model"

    async def test_queued_job_also_counts_as_duplicate(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        ext_job = await _make_extraction_job(async_session, job_id="test-ext-002")
        feature = await _make_feature(
            async_session, ext_job.id, feature_id="feat_test_002", neuron_index=99
        )

        queued = EnhancedLabelingJob(
            id="elj_99_queued",
            feature_id=feature.id,
            status=EnhancedLabelingStatus.QUEUED.value,
            endpoint="http://llm.local/v1",
            model="queued-model",
            workers=4,
            examples_total=20,
        )
        async_session.add(queued)
        await async_session.flush()

        response = await client.post(f"/api/v1/features/{feature.id}/label/enhanced")

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == queued.id
        assert data["status"] == "queued"


# ── GET /features/{id}/label/enhanced/latest ─────────────────────────────────

class TestGetLatestEnhancedLabelingJob:
    """Tests for GET /api/v1/features/{feature_id}/label/enhanced/latest."""

    async def test_no_job_returns_null(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        ext_job = await _make_extraction_job(async_session, job_id="test-ext-003")
        feature = await _make_feature(
            async_session, ext_job.id, feature_id="feat_test_003", neuron_index=7
        )

        response = await client.get(f"/api/v1/features/{feature.id}/label/enhanced/latest")

        assert response.status_code == 200
        assert response.json() is None

    async def test_returns_most_recent_job(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        ext_job = await _make_extraction_job(async_session, job_id="test-ext-004")
        feature = await _make_feature(
            async_session, ext_job.id, feature_id="feat_test_004", neuron_index=8
        )

        job = EnhancedLabelingJob(
            id="elj_8_completed",
            feature_id=feature.id,
            status=EnhancedLabelingStatus.COMPLETED.value,
            endpoint="http://llm.local/v1",
            model="synthesis-model",
            workers=4,
            examples_total=20,
            examples_completed=20,
        )
        async_session.add(job)
        await async_session.flush()

        response = await client.get(f"/api/v1/features/{feature.id}/label/enhanced/latest")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job.id
        assert data["feature_id"] == feature.id
        assert data["status"] == "completed"
        assert data["examples_completed"] == 20

    async def test_unknown_feature_returns_null(
        self, client: AsyncClient
    ):
        response = await client.get("/api/v1/features/unknown-feat/label/enhanced/latest")
        assert response.status_code == 200
        assert response.json() is None
