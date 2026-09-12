"""Unit tests for the aqua-star protected-label guard and provenance (Feature 010)."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.schemas.feature import FeatureUpdateRequest
from src.services.feature_service import FeatureService, ProtectedLabelError


def make_feature(**overrides):
    defaults = dict(
        id="feat_x_1",
        training_id="train_x",
        extraction_job_id="ext_x",
        neuron_index=1,
        name="old name",
        category="old category",
        description="old description",
        notes=None,
        label_source="enhanced_llm",
        star_color="aqua",
        is_favorite=False,
        activation_frequency=0.1,
        interpretability_score=0.5,
        max_activation=1.0,
        mean_activation=0.5,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        labeled_at=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_service(feature):
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = feature
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return FeatureService(db)


class TestAquaGuard:
    @pytest.mark.asyncio
    async def test_name_edit_on_aqua_raises(self):
        service = make_service(make_feature(star_color="aqua"))
        with pytest.raises(ProtectedLabelError):
            await service.update_feature("feat_x_1", FeatureUpdateRequest(name="new name"))

    @pytest.mark.asyncio
    async def test_category_and_description_also_guarded(self):
        service = make_service(make_feature(star_color="aqua"))
        with pytest.raises(ProtectedLabelError):
            await service.update_feature("feat_x_1", FeatureUpdateRequest(category="new cat"))
        with pytest.raises(ProtectedLabelError):
            await service.update_feature("feat_x_1", FeatureUpdateRequest(description="new desc"))

    @pytest.mark.asyncio
    async def test_notes_only_edit_allowed_on_aqua(self):
        feature = make_feature(star_color="aqua")
        service = make_service(feature)
        result = await service.update_feature("feat_x_1", FeatureUpdateRequest(notes="evidence"))
        assert result is not None
        assert feature.notes == "evidence"
        assert feature.name == "old name"

    @pytest.mark.asyncio
    async def test_override_allows_edit(self):
        feature = make_feature(star_color="aqua")
        service = make_service(feature)
        result = await service.update_feature(
            "feat_x_1", FeatureUpdateRequest(name="validated name", override_protected=True)
        )
        assert result is not None
        assert feature.name == "validated name"

    @pytest.mark.asyncio
    async def test_unstarred_feature_not_guarded(self):
        feature = make_feature(star_color=None)
        service = make_service(feature)
        result = await service.update_feature("feat_x_1", FeatureUpdateRequest(name="new name"))
        assert result is not None
        assert feature.name == "new name"

    @pytest.mark.asyncio
    async def test_same_value_edit_not_treated_as_protected_change(self):
        feature = make_feature(star_color="aqua")
        service = make_service(feature)
        # Sending the identical name is a no-op, not a protected edit
        result = await service.update_feature("feat_x_1", FeatureUpdateRequest(name="old name"))
        assert result is not None


class TestProvenance:
    @pytest.mark.asyncio
    async def test_mcp_agent_label_source_persisted(self):
        feature = make_feature(star_color=None)
        service = make_service(feature)
        await service.update_feature(
            "feat_x_1",
            FeatureUpdateRequest(name="agent label", label_source="mcp_agent"),
        )
        assert feature.label_source == "mcp_agent"
        assert feature.labeled_at is not None

    @pytest.mark.asyncio
    async def test_name_change_without_source_defaults_to_user(self):
        feature = make_feature(star_color=None)
        service = make_service(feature)
        await service.update_feature("feat_x_1", FeatureUpdateRequest(name="hand edit"))
        assert feature.label_source == "user"

    def test_schema_rejects_non_whitelisted_source(self):
        with pytest.raises(Exception):
            FeatureUpdateRequest(label_source="openai")
