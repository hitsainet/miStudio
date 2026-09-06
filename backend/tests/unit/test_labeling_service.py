"""
Unit tests for LabelingService.

Tests the core labeling service functionality including:
- Starting labeling jobs
- Validating extraction status
- Preventing duplicate labeling
- Listing and filtering labeling jobs
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.labeling_service import LabelingService
from src.models.labeling_job import LabelingJob, LabelingStatus, LabelingMethod
from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.feature import Feature


@pytest.fixture
def mock_async_session():
    """Mock async database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def labeling_service(mock_async_session):
    """Create LabelingService instance with mocked async database."""
    return LabelingService(mock_async_session)


@pytest.fixture
def completed_extraction():
    """Create a completed extraction job."""
    extraction = Mock(spec=ExtractionJob)
    extraction.id = "extr_test_123"
    extraction.status = ExtractionStatus.COMPLETED.value
    return extraction


@pytest.fixture
def mock_features():
    """Create mock features."""
    features = []
    for i in range(10):
        feature = Mock(spec=Feature)
        feature.id = f"feat_{i}"
        feature.neuron_index = i
        features.append(feature)
    return features


class TestStartLabeling:
    """Tests for start_labeling method."""

    @pytest.mark.asyncio
    async def test_start_labeling_success(
        self, labeling_service, mock_async_session, completed_extraction, mock_features
    ):
        """Test successfully starting a labeling job."""
        # Setup mocks for async execution - order matches actual service code
        mock_result_extraction = Mock()
        mock_result_extraction.scalar_one_or_none.return_value = completed_extraction

        mock_result_active = Mock()
        mock_result_active.scalar_one_or_none.return_value = None  # No active labeling

        mock_result_count = Mock()
        mock_result_count.scalar_one.return_value = len(mock_features)  # Uses scalar_one() not scalar()

        # Configure execute to return different results for different queries
        # Order: 1) get extraction, 2) check active labeling, 3) count features
        mock_async_session.execute = AsyncMock(side_effect=[
            mock_result_extraction,  # First call: get extraction
            mock_result_active,      # Second call: check active labeling
            mock_result_count,       # Third call: count features
        ])

        config = {
            "labeling_method": "openai",
            "openai_model": "gpt-4o-mini",
        }

        # Execute
        labeling_job = await labeling_service.start_labeling("extr_test_123", config)

        # Verify
        assert labeling_job.extraction_job_id == "extr_test_123"
        assert labeling_job.status == LabelingStatus.QUEUED.value
        assert labeling_job.labeling_method == "openai"
        assert labeling_job.total_features == len(mock_features)
        mock_async_session.add.assert_called_once()
        mock_async_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_labeling_extraction_not_found(self, labeling_service, mock_async_session):
        """Test error when extraction doesn't exist."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_async_session.execute = AsyncMock(return_value=mock_result)

        config = {"labeling_method": "openai"}

        with pytest.raises(ValueError, match="not found"):
            await labeling_service.start_labeling("nonexistent", config)

    @pytest.mark.asyncio
    async def test_start_labeling_extraction_not_completed(
        self, labeling_service, mock_async_session, completed_extraction
    ):
        """Test error when extraction is not completed."""
        completed_extraction.status = ExtractionStatus.EXTRACTING.value
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = completed_extraction
        mock_async_session.execute = AsyncMock(return_value=mock_result)

        config = {"labeling_method": "openai"}

        with pytest.raises(ValueError, match="must be completed"):
            await labeling_service.start_labeling("extr_test_123", config)

    @pytest.mark.asyncio
    async def test_start_labeling_active_labeling_exists(
        self, labeling_service, mock_async_session, completed_extraction
    ):
        """Test error when active labeling already exists."""
        # Mock active labeling job
        active_labeling = Mock(spec=LabelingJob)
        active_labeling.id = "label_existing"
        active_labeling.status = LabelingStatus.LABELING.value

        mock_result_extraction = Mock()
        mock_result_extraction.scalar_one_or_none.return_value = completed_extraction

        mock_result_count = Mock()
        mock_result_count.scalar.return_value = 10

        mock_result_active = Mock()
        mock_result_active.scalar_one_or_none.return_value = active_labeling

        mock_async_session.execute = AsyncMock(side_effect=[
            mock_result_extraction,
            mock_result_count,
            mock_result_active,
        ])

        config = {"labeling_method": "openai"}

        with pytest.raises(ValueError, match="already has an active labeling"):
            await labeling_service.start_labeling("extr_test_123", config)

    @pytest.mark.asyncio
    async def test_start_labeling_no_features(
        self, labeling_service, mock_async_session, completed_extraction
    ):
        """Test error when extraction has no features."""
        mock_result_extraction = Mock()
        mock_result_extraction.scalar_one_or_none.return_value = completed_extraction

        mock_result_active = Mock()
        mock_result_active.scalar_one_or_none.return_value = None

        mock_result_count = Mock()
        mock_result_count.scalar_one.return_value = 0  # Uses scalar_one() not scalar()

        # Order: 1) get extraction, 2) check active labeling, 3) count features
        mock_async_session.execute = AsyncMock(side_effect=[
            mock_result_extraction,
            mock_result_active,
            mock_result_count,
        ])

        config = {"labeling_method": "openai"}

        with pytest.raises(ValueError, match="has no features"):
            await labeling_service.start_labeling("extr_test_123", config)


class TestGetLabelingJob:
    """Tests for get_labeling_job method."""

    @pytest.mark.asyncio
    async def test_get_labeling_job_found(self, labeling_service, mock_async_session):
        """Test getting an existing labeling job."""
        mock_job = Mock(spec=LabelingJob)
        mock_job.id = "label_test_123"

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_async_session.execute = AsyncMock(return_value=mock_result)

        result = await labeling_service.get_labeling_job("label_test_123")

        assert result == mock_job

    @pytest.mark.asyncio
    async def test_get_labeling_job_not_found(self, labeling_service, mock_async_session):
        """Test getting a non-existent labeling job."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_async_session.execute = AsyncMock(return_value=mock_result)

        result = await labeling_service.get_labeling_job("nonexistent")

        assert result is None


class TestListLabelingJobs:
    """Tests for list_labeling_jobs method."""

    @pytest.mark.asyncio
    async def test_list_labeling_jobs_all(self, labeling_service, mock_async_session):
        """Test listing all labeling jobs."""
        mock_jobs = [Mock(spec=LabelingJob) for _ in range(5)]

        mock_result_count = Mock()
        mock_result_count.scalar_one.return_value = 5  # Uses scalar_one() not scalar()

        mock_result_jobs = Mock()
        mock_result_jobs.scalars.return_value.all.return_value = mock_jobs

        mock_async_session.execute = AsyncMock(side_effect=[
            mock_result_count,
            mock_result_jobs,
        ])

        jobs, total = await labeling_service.list_labeling_jobs(limit=50, offset=0)

        assert len(jobs) == 5
        assert total == 5

    @pytest.mark.asyncio
    async def test_list_labeling_jobs_filtered_by_extraction(
        self, labeling_service, mock_async_session
    ):
        """Test listing labeling jobs filtered by extraction ID."""
        mock_jobs = [Mock(spec=LabelingJob) for _ in range(3)]

        mock_result_count = Mock()
        mock_result_count.scalar_one.return_value = 3  # Uses scalar_one() not scalar()

        mock_result_jobs = Mock()
        mock_result_jobs.scalars.return_value.all.return_value = mock_jobs

        mock_async_session.execute = AsyncMock(side_effect=[
            mock_result_count,
            mock_result_jobs,
        ])

        jobs, total = await labeling_service.list_labeling_jobs(
            extraction_job_id="extr_test_123", limit=50, offset=0
        )

        assert len(jobs) == 3
        assert total == 3

    @pytest.mark.asyncio
    async def test_list_labeling_jobs_pagination(self, labeling_service, mock_async_session):
        """Test pagination of labeling jobs."""
        mock_jobs = [Mock(spec=LabelingJob) for _ in range(10)]

        mock_result_count = Mock()
        mock_result_count.scalar_one.return_value = 100  # Uses scalar_one() not scalar()

        mock_result_jobs = Mock()
        mock_result_jobs.scalars.return_value.all.return_value = mock_jobs

        mock_async_session.execute = AsyncMock(side_effect=[
            mock_result_count,
            mock_result_jobs,
        ])

        jobs, total = await labeling_service.list_labeling_jobs(limit=10, offset=20)

        assert len(jobs) == 10
        assert total == 100


class TestCancelLabelingJob:
    """Tests for cancel_labeling_job method."""

    @pytest.mark.asyncio
    async def test_cancel_labeling_job_success(self, labeling_service, mock_async_session):
        """Test successfully cancelling a labeling job."""
        mock_job = Mock(spec=LabelingJob)
        mock_job.id = "label_test_123"
        mock_job.status = LabelingStatus.QUEUED.value
        mock_job.celery_task_id = None  # No Celery task to revoke

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_async_session.execute = AsyncMock(return_value=mock_result)

        # The cancel now goes through `core.cancellation.request_cancel`, which
        # deliberately opens its OWN sync session: the point of the registry is
        # that the word written is the word the worker's checker reads, and a
        # fresh session is what lets it see and be seen across processes. This
        # mock async session therefore cannot observe the write — so the
        # assertion moved from "the mock's attribute changed" to "the request
        # was made, for the right scope and row".
        with patch(
            "src.core.cancellation.request_cancel", return_value=Mock(requested=True)
        ) as request_cancel:
            result = await labeling_service.cancel_labeling_job("label_test_123")

        assert result is True
        request_cancel.assert_called_once()
        args, kwargs = request_cancel.call_args
        assert args[0] == "labeling"
        assert args[1] == "label_test_123"
        # R1-15: the celery id is the ONE case a plain revoke() genuinely
        # handles — a task that has not started. The fixture left it None, so
        # nothing checked it was forwarded at all.
        assert "celery_task_id" in kwargs, (
            "the celery id is not forwarded, so a queued job is never revoked"
        )

    @pytest.mark.asyncio
    async def test_cancel_labeling_job_not_found(self, labeling_service, mock_async_session):
        """Test error when labeling job doesn't exist."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_async_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError, match="not found"):
            await labeling_service.cancel_labeling_job("nonexistent")

    @pytest.mark.asyncio
    async def test_cancel_labeling_job_invalid_status(self, labeling_service, mock_async_session):
        """Test error when labeling job is not cancellable."""
        mock_job = Mock(spec=LabelingJob)
        mock_job.id = "label_test_123"
        mock_job.status = LabelingStatus.COMPLETED.value

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_async_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError, match="Cannot cancel"):
            await labeling_service.cancel_labeling_job("label_test_123")


class TestDeleteLabelingJob:
    """Tests for delete_labeling_job method."""

    @pytest.mark.asyncio
    async def test_delete_labeling_job_success(self, labeling_service, mock_async_session):
        """Test successfully deleting a labeling job."""
        mock_job = Mock(spec=LabelingJob)
        mock_job.id = "label_test_123"
        mock_job.status = LabelingStatus.COMPLETED.value

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_async_session.execute = AsyncMock(return_value=mock_result)

        result = await labeling_service.delete_labeling_job("label_test_123")

        assert result is True
        mock_async_session.delete.assert_awaited_once_with(mock_job)
        mock_async_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_delete_labeling_job_not_found(self, labeling_service, mock_async_session):
        """Test error when labeling job doesn't exist."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_async_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError, match="not found"):
            await labeling_service.delete_labeling_job("nonexistent")

    @pytest.mark.asyncio
    async def test_delete_labeling_job_active(self, labeling_service, mock_async_session):
        """Test deleting an active labeling job (auto-cancels and deletes)."""
        mock_job = Mock(spec=LabelingJob)
        mock_job.id = "label_test_123"
        mock_job.status = LabelingStatus.LABELING.value
        mock_job.celery_task_id = None  # No Celery task to revoke

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_async_session.execute = AsyncMock(return_value=mock_result)

        # The service now auto-cancels and deletes active jobs instead of raising an error
        result = await labeling_service.delete_labeling_job("label_test_123")

        assert result is True
        mock_async_session.delete.assert_awaited_once_with(mock_job)
        mock_async_session.commit.assert_awaited()


class TestLabelFeaturesStatusTransition:
    """Phase 5 regression tests — labeling_job status transition ordering.

    Covers review findings #7 and #8:
    - Extraction job and features must be validated BEFORE the job is set to LABELING.
    - If validation fails, the labeling_job must never reach LABELING status.
    """

    def _make_sync_session(self):
        from sqlalchemy.orm import Session
        session = Mock(spec=Session)
        session.commit = Mock()
        return session

    def _make_labeling_job(self, extraction_job_id="extr_test_123"):
        job = Mock(spec=LabelingJob)
        job.id = "label_test_456"
        job.extraction_job_id = extraction_job_id
        job.status = LabelingStatus.QUEUED.value
        job.updated_at = None
        return job

    def test_extraction_not_found_never_sets_labeling(self):
        """If extraction job is missing, status must stay QUEUED — never LABELING."""
        session = self._make_sync_session()
        labeling_job = self._make_labeling_job()

        def query_side_effect(model):
            mock_q = Mock()
            mock_q.filter.return_value = mock_q
            mock_q.order_by.return_value = mock_q
            if model is LabelingJob:
                mock_q.first.return_value = labeling_job
            elif model is ExtractionJob:
                mock_q.first.return_value = None
            else:
                mock_q.first.return_value = None
                mock_q.all.return_value = []
            return mock_q

        session.query = Mock(side_effect=query_side_effect)
        service = LabelingService(session)

        with pytest.raises(ValueError, match="Extraction job"):
            service.label_features_for_extraction("label_test_456")

        assert labeling_job.status == LabelingStatus.QUEUED.value, (
            f"Status was set to {labeling_job.status} before validation completed"
        )
        session.commit.assert_not_called()

    def test_no_features_never_sets_labeling(self):
        """If no features exist for the extraction, status must stay QUEUED."""
        session = self._make_sync_session()
        labeling_job = self._make_labeling_job()

        extraction_job = Mock(spec=ExtractionJob)
        extraction_job.id = "extr_test_123"
        extraction_job.status = ExtractionStatus.COMPLETED.value

        def query_side_effect(model):
            mock_q = Mock()
            mock_q.filter.return_value = mock_q
            mock_q.order_by.return_value = mock_q
            if model is LabelingJob:
                mock_q.first.return_value = labeling_job
            elif model is ExtractionJob:
                mock_q.first.return_value = extraction_job
            elif model is Feature:
                mock_q.all.return_value = []
                mock_q.first.return_value = None
            return mock_q

        session.query = Mock(side_effect=query_side_effect)
        service = LabelingService(session)

        with pytest.raises(ValueError, match="No features found"):
            service.label_features_for_extraction("label_test_456")

        assert labeling_job.status == LabelingStatus.QUEUED.value
        session.commit.assert_not_called()

    def test_labeling_job_not_found_raises(self):
        """Missing labeling job must raise ValueError immediately."""
        session = self._make_sync_session()

        def query_side_effect(model):
            mock_q = Mock()
            mock_q.filter.return_value = mock_q
            mock_q.first.return_value = None
            return mock_q

        session.query = Mock(side_effect=query_side_effect)
        service = LabelingService(session)

        with pytest.raises(ValueError, match="Labeling job"):
            service.label_features_for_extraction("nonexistent_job")

        session.commit.assert_not_called()

    def test_async_session_raises_assertion(self):
        """Passing an AsyncSession must raise AssertionError immediately."""
        from sqlalchemy.ext.asyncio import AsyncSession as _AsyncSession
        async_session = Mock(spec=_AsyncSession)
        service = LabelingService(async_session)

        with pytest.raises(AssertionError, match="sync SQLAlchemy Session"):
            service.label_features_for_extraction("label_test_456")
