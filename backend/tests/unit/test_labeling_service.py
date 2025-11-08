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
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from src.services.labeling_service import LabelingService
from src.models.labeling_job import LabelingJob, LabelingStatus, LabelingMethod
from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.feature import Feature


@pytest.fixture
def db_session():
    """Mock database session."""
    return Mock(spec=Session)


@pytest.fixture
def labeling_service(db_session):
    """Create LabelingService instance with mocked database."""
    return LabelingService(db_session)


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

    def test_start_labeling_success(
        self, labeling_service, db_session, completed_extraction, mock_features
    ):
        """Test successfully starting a labeling job."""
        # Setup mocks
        db_session.query().filter().first.return_value = completed_extraction
        db_session.query().filter().count.return_value = len(mock_features)
        db_session.query().filter().filter().first.return_value = None  # No active labeling

        config = {
            "labeling_method": "openai",
            "openai_model": "gpt-4o-mini",
        }

        # Execute
        labeling_job = labeling_service.start_labeling("extr_test_123", config)

        # Verify
        assert labeling_job.extraction_job_id == "extr_test_123"
        assert labeling_job.status == LabelingStatus.QUEUED.value
        assert labeling_job.labeling_method == "openai"
        assert labeling_job.total_features == len(mock_features)
        db_session.add.assert_called_once()
        db_session.commit.assert_called_once()

    def test_start_labeling_extraction_not_found(self, labeling_service, db_session):
        """Test error when extraction doesn't exist."""
        db_session.query().filter().first.return_value = None

        config = {"labeling_method": "openai"}

        with pytest.raises(ValueError, match="not found"):
            labeling_service.start_labeling("nonexistent", config)

    def test_start_labeling_extraction_not_completed(
        self, labeling_service, db_session, completed_extraction
    ):
        """Test error when extraction is not completed."""
        completed_extraction.status = ExtractionStatus.EXTRACTING.value
        db_session.query().filter().first.return_value = completed_extraction

        config = {"labeling_method": "openai"}

        with pytest.raises(ValueError, match="must be completed"):
            labeling_service.start_labeling("extr_test_123", config)

    def test_start_labeling_active_labeling_exists(
        self, labeling_service, db_session, completed_extraction
    ):
        """Test error when active labeling already exists."""
        db_session.query().filter().first.return_value = completed_extraction
        db_session.query().filter().count.return_value = 10

        # Mock active labeling job
        active_labeling = Mock(spec=LabelingJob)
        active_labeling.id = "label_existing"
        active_labeling.status = LabelingStatus.LABELING.value
        db_session.query().filter().filter().first.return_value = active_labeling

        config = {"labeling_method": "openai"}

        with pytest.raises(ValueError, match="already has an active labeling"):
            labeling_service.start_labeling("extr_test_123", config)

    def test_start_labeling_no_features(
        self, labeling_service, db_session, completed_extraction
    ):
        """Test error when extraction has no features."""
        db_session.query().filter().first.return_value = completed_extraction
        db_session.query().filter().count.return_value = 0
        db_session.query().filter().filter().first.return_value = None

        config = {"labeling_method": "openai"}

        with pytest.raises(ValueError, match="has no features"):
            labeling_service.start_labeling("extr_test_123", config)


class TestGetLabelingJob:
    """Tests for get_labeling_job method."""

    def test_get_labeling_job_found(self, labeling_service, db_session):
        """Test getting an existing labeling job."""
        mock_job = Mock(spec=LabelingJob)
        mock_job.id = "label_test_123"
        db_session.query().filter().first.return_value = mock_job

        result = labeling_service.get_labeling_job("label_test_123")

        assert result == mock_job
        db_session.query.assert_called_once()

    def test_get_labeling_job_not_found(self, labeling_service, db_session):
        """Test getting a non-existent labeling job."""
        db_session.query().filter().first.return_value = None

        result = labeling_service.get_labeling_job("nonexistent")

        assert result is None


class TestListLabelingJobs:
    """Tests for list_labeling_jobs method."""

    def test_list_labeling_jobs_all(self, labeling_service, db_session):
        """Test listing all labeling jobs."""
        mock_jobs = [Mock(spec=LabelingJob) for _ in range(5)]
        db_session.query().count.return_value = 5
        db_session.query().order_by().limit().offset().all.return_value = mock_jobs

        jobs, total = labeling_service.list_labeling_jobs(limit=50, offset=0)

        assert len(jobs) == 5
        assert total == 5

    def test_list_labeling_jobs_filtered_by_extraction(
        self, labeling_service, db_session
    ):
        """Test listing labeling jobs filtered by extraction ID."""
        mock_jobs = [Mock(spec=LabelingJob) for _ in range(3)]
        db_session.query().filter().count.return_value = 3
        db_session.query().filter().order_by().limit().offset().all.return_value = (
            mock_jobs
        )

        jobs, total = labeling_service.list_labeling_jobs(
            extraction_job_id="extr_test_123", limit=50, offset=0
        )

        assert len(jobs) == 3
        assert total == 3

    def test_list_labeling_jobs_pagination(self, labeling_service, db_session):
        """Test pagination of labeling jobs."""
        mock_jobs = [Mock(spec=LabelingJob) for _ in range(10)]
        db_session.query().count.return_value = 100
        db_session.query().order_by().limit().offset().all.return_value = mock_jobs

        jobs, total = labeling_service.list_labeling_jobs(limit=10, offset=20)

        assert len(jobs) == 10
        assert total == 100


class TestCancelLabelingJob:
    """Tests for cancel_labeling_job method."""

    def test_cancel_labeling_job_success(self, labeling_service, db_session):
        """Test successfully cancelling a labeling job."""
        mock_job = Mock(spec=LabelingJob)
        mock_job.id = "label_test_123"
        mock_job.status = LabelingStatus.QUEUED.value
        db_session.query().filter().first.return_value = mock_job

        result = labeling_service.cancel_labeling_job("label_test_123")

        assert result is True
        assert mock_job.status == LabelingStatus.CANCELLED.value
        db_session.commit.assert_called_once()

    def test_cancel_labeling_job_not_found(self, labeling_service, db_session):
        """Test error when labeling job doesn't exist."""
        db_session.query().filter().first.return_value = None

        with pytest.raises(ValueError, match="not found"):
            labeling_service.cancel_labeling_job("nonexistent")

    def test_cancel_labeling_job_invalid_status(self, labeling_service, db_session):
        """Test error when labeling job is not cancellable."""
        mock_job = Mock(spec=LabelingJob)
        mock_job.id = "label_test_123"
        mock_job.status = LabelingStatus.COMPLETED.value
        db_session.query().filter().first.return_value = mock_job

        with pytest.raises(ValueError, match="Cannot cancel"):
            labeling_service.cancel_labeling_job("label_test_123")


class TestDeleteLabelingJob:
    """Tests for delete_labeling_job method."""

    def test_delete_labeling_job_success(self, labeling_service, db_session):
        """Test successfully deleting a labeling job."""
        mock_job = Mock(spec=LabelingJob)
        mock_job.id = "label_test_123"
        mock_job.status = LabelingStatus.COMPLETED.value
        db_session.query().filter().first.return_value = mock_job

        result = labeling_service.delete_labeling_job("label_test_123")

        assert result is True
        db_session.delete.assert_called_once_with(mock_job)
        db_session.commit.assert_called()

    def test_delete_labeling_job_not_found(self, labeling_service, db_session):
        """Test error when labeling job doesn't exist."""
        db_session.query().filter().first.return_value = None

        with pytest.raises(ValueError, match="not found"):
            labeling_service.delete_labeling_job("nonexistent")

    def test_delete_labeling_job_active(self, labeling_service, db_session):
        """Test error when trying to delete active labeling job."""
        mock_job = Mock(spec=LabelingJob)
        mock_job.id = "label_test_123"
        mock_job.status = LabelingStatus.LABELING.value
        db_session.query().filter().first.return_value = mock_job

        with pytest.raises(ValueError, match="Cannot delete active"):
            labeling_service.delete_labeling_job("label_test_123")
