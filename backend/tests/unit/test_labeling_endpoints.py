"""
Unit tests for labeling API endpoints.

Tests the REST API for labeling operations.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.main import app
from src.models.labeling_job import LabelingStatus


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_labeling_job():
    """Create mock labeling job."""
    return {
        "id": "label_extr_test_20251108_123456",
        "extraction_job_id": "extr_test_123",
        "labeling_method": "openai",
        "openai_model": "gpt-4o-mini",
        "status": LabelingStatus.QUEUED.value,
        "progress": 0.0,
        "features_labeled": 0,
        "total_features": 100,
        "created_at": "2025-01-08T12:00:00Z",
        "updated_at": "2025-01-08T12:00:00Z",
    }


class TestStartLabeling:
    """Tests for POST /api/v1/labeling endpoint."""

    @patch("src.api.v1.endpoints.labeling.LabelingService")
    @patch("src.api.v1.endpoints.labeling.label_features_task")
    def test_start_labeling_success(
        self, mock_task, mock_service_class, client, mock_labeling_job
    ):
        """Test successfully starting a labeling job."""
        # Setup mocks
        mock_service = Mock()
        mock_service.start_labeling.return_value = Mock(**mock_labeling_job)
        mock_service_class.return_value = mock_service
        mock_task.delay.return_value = Mock(id="celery_task_123")

        # Make request
        response = client.post(
            "/api/v1/labeling",
            json={
                "extraction_job_id": "extr_test_123",
                "labeling_method": "openai",
                "openai_model": "gpt-4o-mini",
            },
        )

        # Verify
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == mock_labeling_job["id"]
        assert data["extraction_job_id"] == "extr_test_123"
        assert data["status"] == LabelingStatus.QUEUED.value
        mock_task.delay.assert_called_once()

    @patch("src.api.v1.endpoints.labeling.LabelingService")
    def test_start_labeling_extraction_not_found(self, mock_service_class, client):
        """Test error when extraction doesn't exist."""
        mock_service = Mock()
        mock_service.start_labeling.side_effect = ValueError("Extraction not found")
        mock_service_class.return_value = mock_service

        response = client.post(
            "/api/v1/labeling",
            json={
                "extraction_job_id": "nonexistent",
                "labeling_method": "openai",
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch("src.api.v1.endpoints.labeling.LabelingService")
    def test_start_labeling_active_labeling_exists(self, mock_service_class, client):
        """Test error when active labeling already exists."""
        mock_service = Mock()
        mock_service.start_labeling.side_effect = ValueError(
            "already has an active labeling"
        )
        mock_service_class.return_value = mock_service

        response = client.post(
            "/api/v1/labeling",
            json={
                "extraction_job_id": "extr_test_123",
                "labeling_method": "openai",
            },
        )

        assert response.status_code == 409
        assert "already" in response.json()["detail"].lower()


class TestGetLabelingStatus:
    """Tests for GET /api/v1/labeling/{labeling_job_id} endpoint."""

    @patch("src.api.v1.endpoints.labeling.LabelingService")
    def test_get_labeling_status_success(
        self, mock_service_class, client, mock_labeling_job
    ):
        """Test successfully getting labeling job status."""
        mock_service = Mock()
        mock_service.get_labeling_job.return_value = Mock(**mock_labeling_job)
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/labeling/label_extr_test_20251108_123456")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == mock_labeling_job["id"]
        assert data["status"] == LabelingStatus.QUEUED.value

    @patch("src.api.v1.endpoints.labeling.LabelingService")
    def test_get_labeling_status_not_found(self, mock_service_class, client):
        """Test error when labeling job doesn't exist."""
        mock_service = Mock()
        mock_service.get_labeling_job.return_value = None
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/labeling/nonexistent")

        assert response.status_code == 404


class TestListLabelingJobs:
    """Tests for GET /api/v1/labeling endpoint."""

    @patch("src.api.v1.endpoints.labeling.LabelingService")
    def test_list_labeling_jobs_success(
        self, mock_service_class, client, mock_labeling_job
    ):
        """Test successfully listing labeling jobs."""
        mock_service = Mock()
        mock_jobs = [Mock(**mock_labeling_job) for _ in range(3)]
        mock_service.list_labeling_jobs.return_value = (mock_jobs, 3)
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/labeling")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        assert data["meta"]["total"] == 3

    @patch("src.api.v1.endpoints.labeling.LabelingService")
    def test_list_labeling_jobs_filtered(
        self, mock_service_class, client, mock_labeling_job
    ):
        """Test listing labeling jobs with filters."""
        mock_service = Mock()
        mock_jobs = [Mock(**mock_labeling_job)]
        mock_service.list_labeling_jobs.return_value = (mock_jobs, 1)
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/labeling?extraction_job_id=extr_test_123")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        mock_service.list_labeling_jobs.assert_called_once()


class TestCancelLabeling:
    """Tests for POST /api/v1/labeling/{labeling_job_id}/cancel endpoint."""

    @patch("src.api.v1.endpoints.labeling.LabelingService")
    def test_cancel_labeling_success(self, mock_service_class, client):
        """Test successfully cancelling a labeling job."""
        mock_service = Mock()
        mock_service.cancel_labeling_job.return_value = True
        mock_service_class.return_value = mock_service

        response = client.post("/api/v1/labeling/label_test_123/cancel")

        assert response.status_code == 200
        assert "cancelled successfully" in response.json()["message"].lower()

    @patch("src.api.v1.endpoints.labeling.LabelingService")
    def test_cancel_labeling_not_found(self, mock_service_class, client):
        """Test error when labeling job doesn't exist."""
        mock_service = Mock()
        mock_service.cancel_labeling_job.side_effect = ValueError("not found")
        mock_service_class.return_value = mock_service

        response = client.post("/api/v1/labeling/nonexistent/cancel")

        assert response.status_code == 404


class TestDeleteLabeling:
    """Tests for DELETE /api/v1/labeling/{labeling_job_id} endpoint."""

    @patch("src.api.v1.endpoints.labeling.LabelingService")
    def test_delete_labeling_success(self, mock_service_class, client):
        """Test successfully deleting a labeling job."""
        mock_service = Mock()
        mock_service.delete_labeling_job.return_value = True
        mock_service_class.return_value = mock_service

        response = client.delete("/api/v1/labeling/label_test_123")

        assert response.status_code == 204

    @patch("src.api.v1.endpoints.labeling.LabelingService")
    def test_delete_labeling_not_found(self, mock_service_class, client):
        """Test error when labeling job doesn't exist."""
        mock_service = Mock()
        mock_service.delete_labeling_job.side_effect = ValueError("not found")
        mock_service_class.return_value = mock_service

        response = client.delete("/api/v1/labeling/nonexistent")

        assert response.status_code == 404

    @patch("src.api.v1.endpoints.labeling.LabelingService")
    def test_delete_labeling_active(self, mock_service_class, client):
        """Test error when trying to delete active labeling job."""
        mock_service = Mock()
        mock_service.delete_labeling_job.side_effect = ValueError(
            "Cannot delete active"
        )
        mock_service_class.return_value = mock_service

        response = client.delete("/api/v1/labeling/label_test_123")

        assert response.status_code == 409
