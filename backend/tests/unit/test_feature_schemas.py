"""
Unit tests for feature Pydantic schemas.

Tests request validation, field constraints, SQL injection sanitization,
and response serialization for feature discovery endpoints.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from src.schemas.feature import (
    FeatureSearchRequest,
    FeatureResponse,
    FeatureListResponse,
    FeatureDetailResponse,
    FeatureUpdateRequest,
    FeatureActivationExample,
    FeatureStatistics,
    LogitLensResponse,
    CorrelatedFeature,
    CorrelationsResponse,
    AblationResponse,
)


class TestFeatureSearchRequest:
    """Test FeatureSearchRequest validation and SQL injection prevention."""

    def test_default_values(self):
        """Test default search parameters."""
        request = FeatureSearchRequest()

        assert request.search is None
        assert request.sort_by == "activation_freq"
        assert request.sort_order == "desc"
        assert request.is_favorite is None
        assert request.limit == 50
        assert request.offset == 0

    def test_custom_valid_values(self):
        """Test valid custom search parameters."""
        request = FeatureSearchRequest(
            search="question words",
            sort_by="interpretability",
            sort_order="asc",
            is_favorite=True,
            limit=100,
            offset=50
        )

        assert request.search == "question words"
        assert request.sort_by == "interpretability"
        assert request.sort_order == "asc"
        assert request.is_favorite is True
        assert request.limit == 100
        assert request.offset == 50

    def test_search_query_sanitization_sql_injection(self):
        """Test that dangerous SQL characters are removed from search query."""
        dangerous_queries = [
            ("'; DROP TABLE features; --", " DROP TABLE features ")
        ]

        for dangerous, expected in dangerous_queries:
            request = FeatureSearchRequest(search=dangerous)
            # All dangerous chars should be removed
            assert "'" not in request.search
            assert '"' not in request.search
            assert ";" not in request.search
            assert "--" not in request.search

    def test_search_query_sanitization_xss(self):
        """Test that XSS attempts are sanitized."""
        request = FeatureSearchRequest(search="<script>alert('xss')</script>")

        # Script tags should remain but quotes removed
        assert request.search is not None
        assert '"' not in request.search
        assert "'" not in request.search

    def test_search_query_whitespace_handling(self):
        """Test whitespace trimming in search query."""
        # Leading/trailing whitespace should be stripped
        request = FeatureSearchRequest(search="  question words  ")
        assert request.search == "question words"

        # Empty string after stripping should return None
        request = FeatureSearchRequest(search="   ")
        assert request.search is None

    def test_search_query_none(self):
        """Test that None search query is preserved."""
        request = FeatureSearchRequest(search=None)
        assert request.search is None

    def test_search_query_max_length(self):
        """Test search query maximum length constraint."""
        # Valid: exactly at max
        request = FeatureSearchRequest(search="x" * 500)
        assert len(request.search) == 500

        # Invalid: over max
        with pytest.raises(ValidationError) as exc_info:
            FeatureSearchRequest(search="x" * 501)

        assert "String should have at most 500 characters" in str(exc_info.value)

    def test_sort_by_valid_values(self):
        """Test valid sort_by field values."""
        for sort_field in ["activation_freq", "interpretability", "feature_id"]:
            request = FeatureSearchRequest(sort_by=sort_field)
            assert request.sort_by == sort_field

    def test_sort_by_invalid_value(self):
        """Test that invalid sort_by value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FeatureSearchRequest(sort_by="invalid_field")

        assert "Input should be" in str(exc_info.value)

    def test_sort_order_valid_values(self):
        """Test valid sort_order values."""
        request_asc = FeatureSearchRequest(sort_order="asc")
        request_desc = FeatureSearchRequest(sort_order="desc")

        assert request_asc.sort_order == "asc"
        assert request_desc.sort_order == "desc"

    def test_sort_order_invalid_value(self):
        """Test that invalid sort_order value is rejected."""
        with pytest.raises(ValidationError):
            FeatureSearchRequest(sort_order="invalid")

    def test_is_favorite_filter_values(self):
        """Test is_favorite filter with None, True, False."""
        request_all = FeatureSearchRequest(is_favorite=None)
        request_favorites = FeatureSearchRequest(is_favorite=True)
        request_non_favorites = FeatureSearchRequest(is_favorite=False)

        assert request_all.is_favorite is None
        assert request_favorites.is_favorite is True
        assert request_non_favorites.is_favorite is False

    def test_limit_boundary_values(self):
        """Test limit field constraints (1-500)."""
        # Valid: minimum
        request = FeatureSearchRequest(limit=1)
        assert request.limit == 1

        # Valid: maximum
        request = FeatureSearchRequest(limit=500)
        assert request.limit == 500

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            FeatureSearchRequest(limit=0)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            FeatureSearchRequest(limit=501)

    def test_offset_boundary_values(self):
        """Test offset field constraints (>= 0)."""
        # Valid: minimum
        request = FeatureSearchRequest(offset=0)
        assert request.offset == 0

        # Valid: large value
        request = FeatureSearchRequest(offset=10000)
        assert request.offset == 10000

        # Invalid: negative
        with pytest.raises(ValidationError):
            FeatureSearchRequest(offset=-1)


class TestFeatureActivationExample:
    """Test FeatureActivationExample schema."""

    def test_valid_example(self):
        """Test creating a valid activation example."""
        example = FeatureActivationExample(
            tokens=["what", "is", "this"],
            activations=[0.95, 0.32, 0.15],
            max_activation=0.95,
            sample_index=42
        )

        assert example.tokens == ["what", "is", "this"]
        assert example.activations == [0.95, 0.32, 0.15]
        assert example.max_activation == 0.95
        assert example.sample_index == 42

    def test_from_orm_attributes(self):
        """Test creation from ORM object."""
        class MockActivation:
            tokens = ["test"]
            activations = [1.0]
            max_activation = 1.0
            sample_index = 1

        mock = MockActivation()
        example = FeatureActivationExample.model_validate(mock)

        assert example.tokens == ["test"]
        assert example.sample_index == 1


class TestFeatureResponse:
    """Test FeatureResponse schema."""

    def test_minimal_feature_response(self):
        """Test feature response with minimal required fields."""
        response = FeatureResponse(
            id="feat_00042",
            training_id="train_12345",
            extraction_job_id="extr_67890",
            neuron_index=42,
            name="Question Pattern",
            label_source="auto",
            activation_frequency=0.023,
            interpretability_score=0.87,
            max_activation=12.5,
            is_favorite=False,
            created_at=datetime(2025, 1, 15, 10, 0, 0),
            updated_at=datetime(2025, 1, 15, 10, 0, 0)
        )

        assert response.id == "feat_00042"
        assert response.neuron_index == 42
        assert response.name == "Question Pattern"
        assert response.label_source == "auto"
        assert response.description is None
        assert response.notes is None
        assert response.example_context is None

    def test_complete_feature_response(self):
        """Test feature response with all fields populated."""
        example = FeatureActivationExample(
            tokens=["what", "is", "this"],
            activations=[0.95, 0.32, 0.15],
            max_activation=0.95,
            sample_index=42
        )

        response = FeatureResponse(
            id="feat_00042",
            training_id="train_12345",
            extraction_job_id="extr_67890",
            neuron_index=42,
            name="Question Pattern",
            description="Activates on question words (what/how/why/when/where)",
            label_source="user",
            activation_frequency=0.023,
            interpretability_score=0.87,
            max_activation=12.5,
            mean_activation=2.3,
            is_favorite=True,
            notes="High confidence - clear semantic meaning",
            created_at=datetime(2025, 1, 15, 10, 0, 0),
            updated_at=datetime(2025, 1, 15, 11, 30, 0),
            example_context=example
        )

        assert response.description == "Activates on question words (what/how/why/when/where)"
        assert response.label_source == "user"
        assert response.mean_activation == 2.3
        assert response.is_favorite is True
        assert response.notes == "High confidence - clear semantic meaning"
        assert response.example_context is not None
        assert response.example_context.tokens == ["what", "is", "this"]


class TestFeatureStatistics:
    """Test FeatureStatistics schema."""

    def test_valid_statistics(self):
        """Test creating valid feature statistics."""
        stats = FeatureStatistics(
            total_features=512,
            interpretable_percentage=75.2,
            avg_activation_frequency=0.019
        )

        assert stats.total_features == 512
        assert stats.interpretable_percentage == 75.2
        assert stats.avg_activation_frequency == 0.019


class TestFeatureListResponse:
    """Test FeatureListResponse schema."""

    def test_valid_list_response(self):
        """Test creating a valid paginated list response."""
        features = [
            FeatureResponse(
                id="feat_00001",
                training_id="train_12345",
                extraction_job_id="extr_67890",
                neuron_index=1,
                name="Feature 1",
                label_source="auto",
                activation_frequency=0.025,
                interpretability_score=0.9,
                max_activation=15.0,
                is_favorite=False,
                created_at=datetime(2025, 1, 15, 10, 0, 0),
                updated_at=datetime(2025, 1, 15, 10, 0, 0)
            )
        ]

        statistics = FeatureStatistics(
            total_features=512,
            interpretable_percentage=75.2,
            avg_activation_frequency=0.019
        )

        response = FeatureListResponse(
            features=features,
            total=512,
            limit=50,
            offset=0,
            statistics=statistics
        )

        assert len(response.features) == 1
        assert response.total == 512
        assert response.limit == 50
        assert response.offset == 0
        assert response.statistics.total_features == 512


class TestFeatureDetailResponse:
    """Test FeatureDetailResponse schema."""

    def test_feature_detail_with_computed_fields(self):
        """Test detailed feature response with computed active_samples field."""
        response = FeatureDetailResponse(
            id="feat_00042",
            training_id="train_12345",
            extraction_job_id="extr_67890",
            neuron_index=42,
            name="Question Pattern",
            label_source="auto",
            activation_frequency=0.023,
            interpretability_score=0.87,
            max_activation=12.5,
            is_favorite=False,
            created_at=datetime(2025, 1, 15, 10, 0, 0),
            updated_at=datetime(2025, 1, 15, 10, 0, 0),
            active_samples=230  # Computed from activation_frequency * total_samples
        )

        assert response.active_samples == 230
        assert response.activation_frequency == 0.023


class TestFeatureUpdateRequest:
    """Test FeatureUpdateRequest schema."""

    def test_update_all_fields(self):
        """Test updating all mutable fields."""
        update = FeatureUpdateRequest(
            name="Question Words (what/how/why)",
            description="Updated description with more detail",
            notes="Updated notes after analysis"
        )

        assert update.name == "Question Words (what/how/why)"
        assert update.description == "Updated description with more detail"
        assert update.notes == "Updated notes after analysis"

    def test_update_partial_fields(self):
        """Test updating only some fields."""
        update = FeatureUpdateRequest(name="New Name")

        assert update.name == "New Name"
        assert update.description is None
        assert update.notes is None

    def test_name_max_length(self):
        """Test name field max length constraint."""
        # Valid: exactly at max
        update = FeatureUpdateRequest(name="x" * 500)
        assert len(update.name) == 500

        # Invalid: over max
        with pytest.raises(ValidationError):
            FeatureUpdateRequest(name="x" * 501)


class TestLogitLensResponse:
    """Test LogitLensResponse schema."""

    def test_valid_logit_lens_response(self):
        """Test creating a valid logit lens analysis response."""
        response = LogitLensResponse(
            top_tokens=["what", "how", "why", "when", "where"],
            probabilities=[0.32, 0.28, 0.15, 0.14, 0.11],
            interpretation="Feature strongly predicts question words",
            computed_at=datetime(2025, 1, 15, 12, 0, 0)
        )

        assert len(response.top_tokens) == 5
        assert sum(response.probabilities) == pytest.approx(1.0)
        assert response.interpretation == "Feature strongly predicts question words"
        assert response.computed_at == datetime(2025, 1, 15, 12, 0, 0)


class TestCorrelatedFeature:
    """Test CorrelatedFeature schema."""

    def test_valid_correlated_feature(self):
        """Test creating a correlated feature."""
        corr = CorrelatedFeature(
            feature_id="feat_00123",
            feature_name="Interrogative Syntax",
            correlation=0.78
        )

        assert corr.feature_id == "feat_00123"
        assert corr.feature_name == "Interrogative Syntax"
        assert corr.correlation == 0.78


class TestCorrelationsResponse:
    """Test CorrelationsResponse schema."""

    def test_valid_correlations_response(self):
        """Test creating a valid correlations analysis response."""
        correlated_features = [
            CorrelatedFeature(
                feature_id="feat_00123",
                feature_name="Interrogative Syntax",
                correlation=0.78
            ),
            CorrelatedFeature(
                feature_id="feat_00456",
                feature_name="Sentence Start",
                correlation=0.65
            )
        ]

        response = CorrelationsResponse(
            correlated_features=correlated_features,
            computed_at=datetime(2025, 1, 15, 12, 0, 0)
        )

        assert len(response.correlated_features) == 2
        assert response.correlated_features[0].correlation == 0.78
        assert response.computed_at == datetime(2025, 1, 15, 12, 0, 0)


class TestAblationResponse:
    """Test AblationResponse schema."""

    def test_valid_ablation_response(self):
        """Test creating a valid ablation analysis response."""
        response = AblationResponse(
            perplexity_delta=2.5,
            impact_score=0.42,
            baseline_perplexity=15.3,
            ablated_perplexity=17.8,
            computed_at=datetime(2025, 1, 15, 12, 0, 0)
        )

        assert response.perplexity_delta == 2.5
        assert response.impact_score == 0.42
        assert response.baseline_perplexity == 15.3
        assert response.ablated_perplexity == 17.8
        # Verify calculation: ablated - baseline = delta
        assert response.ablated_perplexity - response.baseline_perplexity == pytest.approx(
            response.perplexity_delta
        )
