"""
Unit tests for extraction Pydantic schemas.

Tests request validation, field constraints, and response serialization
for feature extraction endpoints.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from src.schemas.extraction import (
    ExtractionConfigRequest,
    ExtractionStatusResponse,
)


class TestExtractionConfigRequest:
    """Test ExtractionConfigRequest validation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExtractionConfigRequest()

        assert config.evaluation_samples == 10000
        assert config.top_k_examples == 100

    def test_valid_custom_values(self):
        """Test valid custom configuration."""
        config = ExtractionConfigRequest(
            evaluation_samples=50000,
            top_k_examples=500
        )

        assert config.evaluation_samples == 50000
        assert config.top_k_examples == 500

    def test_evaluation_samples_minimum_boundary(self):
        """Test evaluation_samples minimum constraint (1000)."""
        # Valid: exactly at minimum
        config = ExtractionConfigRequest(evaluation_samples=1000)
        assert config.evaluation_samples == 1000

        # Invalid: below minimum
        with pytest.raises(ValidationError) as exc_info:
            ExtractionConfigRequest(evaluation_samples=999)

        assert "greater_than_equal" in str(exc_info.value)

    def test_evaluation_samples_maximum_boundary(self):
        """Test evaluation_samples maximum constraint (100000)."""
        # Valid: exactly at maximum
        config = ExtractionConfigRequest(evaluation_samples=100000)
        assert config.evaluation_samples == 100000

        # Invalid: above maximum
        with pytest.raises(ValidationError) as exc_info:
            ExtractionConfigRequest(evaluation_samples=100001)

        assert "less_than_equal" in str(exc_info.value)

    def test_top_k_examples_minimum_boundary(self):
        """Test top_k_examples minimum constraint (10)."""
        # Valid: exactly at minimum
        config = ExtractionConfigRequest(top_k_examples=10)
        assert config.top_k_examples == 10

        # Invalid: below minimum
        with pytest.raises(ValidationError) as exc_info:
            ExtractionConfigRequest(top_k_examples=9)

        assert "greater_than_equal" in str(exc_info.value)

    def test_top_k_examples_maximum_boundary(self):
        """Test top_k_examples maximum constraint (1000)."""
        # Valid: exactly at maximum
        config = ExtractionConfigRequest(top_k_examples=1000)
        assert config.top_k_examples == 1000

        # Invalid: above maximum
        with pytest.raises(ValidationError) as exc_info:
            ExtractionConfigRequest(top_k_examples=1001)

        assert "less_than_equal" in str(exc_info.value)

    def test_invalid_types(self):
        """Test that invalid types are rejected."""
        with pytest.raises(ValidationError):
            ExtractionConfigRequest(evaluation_samples="not a number")

        with pytest.raises(ValidationError):
            ExtractionConfigRequest(top_k_examples="not a number")

    def test_negative_values(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError):
            ExtractionConfigRequest(evaluation_samples=-1000)

        with pytest.raises(ValidationError):
            ExtractionConfigRequest(top_k_examples=-100)


class TestExtractionStatusResponse:
    """Test ExtractionStatusResponse serialization."""

    def test_minimal_response(self):
        """Test response with minimal required fields."""
        response = ExtractionStatusResponse(
            id="extr_12345",
            training_id="train_67890",
            status="queued",
            config={"evaluation_samples": 10000, "top_k_examples": 100},
            created_at=datetime(2025, 1, 15, 10, 0, 0),
            updated_at=datetime(2025, 1, 15, 10, 0, 0)
        )

        assert response.id == "extr_12345"
        assert response.training_id == "train_67890"
        assert response.status == "queued"
        assert response.progress is None
        assert response.features_extracted is None
        assert response.total_features is None
        assert response.error_message is None
        assert response.statistics is None
        assert response.completed_at is None

    def test_complete_response(self):
        """Test response with all fields populated."""
        response = ExtractionStatusResponse(
            id="extr_12345",
            training_id="train_67890",
            status="completed",
            progress=1.0,
            features_extracted=512,
            total_features=512,
            error_message=None,
            config={"evaluation_samples": 10000, "top_k_examples": 100},
            statistics={
                "total_features": 512,
                "interpretable_count": 384,
                "avg_activation_frequency": 0.023
            },
            created_at=datetime(2025, 1, 15, 10, 0, 0),
            updated_at=datetime(2025, 1, 15, 10, 30, 0),
            completed_at=datetime(2025, 1, 15, 10, 30, 0)
        )

        assert response.id == "extr_12345"
        assert response.status == "completed"
        assert response.progress == 1.0
        assert response.features_extracted == 512
        assert response.total_features == 512
        assert response.statistics["interpretable_count"] == 384
        assert response.completed_at == datetime(2025, 1, 15, 10, 30, 0)

    def test_failed_response(self):
        """Test response for failed extraction."""
        response = ExtractionStatusResponse(
            id="extr_12345",
            training_id="train_67890",
            status="failed",
            progress=0.45,
            features_extracted=230,
            total_features=512,
            error_message="Out of memory during activation extraction",
            config={"evaluation_samples": 10000, "top_k_examples": 100},
            created_at=datetime(2025, 1, 15, 10, 0, 0),
            updated_at=datetime(2025, 1, 15, 10, 15, 0)
        )

        assert response.status == "failed"
        assert response.error_message == "Out of memory during activation extraction"
        assert response.progress == 0.45
        assert response.features_extracted == 230

    def test_model_dump(self):
        """Test serialization to dictionary."""
        response = ExtractionStatusResponse(
            id="extr_12345",
            training_id="train_67890",
            status="extracting",
            progress=0.67,
            config={"evaluation_samples": 10000},
            created_at=datetime(2025, 1, 15, 10, 0, 0),
            updated_at=datetime(2025, 1, 15, 10, 20, 0)
        )

        data = response.model_dump()

        assert data["id"] == "extr_12345"
        assert data["status"] == "extracting"
        assert data["progress"] == 0.67
        assert isinstance(data["created_at"], datetime)

    def test_from_orm_attributes(self):
        """Test creation from ORM object attributes."""
        # Simulate ORM object
        class MockExtractionJob:
            id = "extr_12345"
            training_id = "train_67890"
            status = "completed"
            progress = 1.0
            features_extracted = 512
            total_features = 512
            error_message = None
            config = {"evaluation_samples": 10000}
            statistics = {"total_features": 512}
            created_at = datetime(2025, 1, 15, 10, 0, 0)
            updated_at = datetime(2025, 1, 15, 10, 30, 0)
            completed_at = datetime(2025, 1, 15, 10, 30, 0)

        mock_job = MockExtractionJob()
        response = ExtractionStatusResponse.model_validate(mock_job)

        assert response.id == "extr_12345"
        assert response.status == "completed"
        assert response.progress == 1.0
