"""
Unit tests for TrainingTemplateService.

Tests cover:
- Template creation with hyperparameters
- Template retrieval (by ID)
- Template listing with filtering, search, pagination, sorting
- Template updates including hyperparameters conversion
- Template deletion
- Favorite toggling and retrieval
- Template export to JSON
- Template import from JSON with duplicate handling
"""

import pytest
from uuid import uuid4, UUID
from datetime import datetime, UTC
from typing import Optional

from src.services.training_template_service import TrainingTemplateService
from src.schemas.training_template import (
    TrainingTemplateCreate,
    TrainingTemplateUpdate
)
from src.schemas.training import (
    TrainingHyperparameters,
    SAEArchitectureType
)


@pytest.mark.asyncio
class TestTrainingTemplateServiceCreate:
    """Test TrainingTemplateService.create_template()."""

    async def test_create_template_success(self, async_session):
        """Test creating a training template with all fields."""
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000,
            warmup_steps=1000,
            weight_decay=0.01,
            checkpoint_interval=1000,
            log_interval=100
        )

        # Note: model_id/dataset_id omitted as they require actual parent records
        # due to FK constraints. FK relationship tested separately with fixtures.
        template_data = TrainingTemplateCreate(
            name="Test Template",
            description="A test training template",
            encoder_type=SAEArchitectureType.STANDARD,
            hyperparameters=hyperparams,
            is_favorite=True,
            extra_metadata={"author": "test_user"}
        )

        template = await TrainingTemplateService.create_template(async_session, template_data)

        assert template is not None
        assert isinstance(template.id, UUID)
        assert template.name == "Test Template"
        assert template.description == "A test training template"
        assert template.model_id is None  # Not set since no FK reference created
        assert template.dataset_id is None
        assert template.encoder_type == SAEArchitectureType.STANDARD.value
        assert template.is_favorite is True
        assert template.extra_metadata == {"author": "test_user"}
        # Hyperparameters stored as dict in database
        assert template.hyperparameters["hidden_dim"] == 512
        assert template.hyperparameters["latent_dim"] == 16384
        assert template.hyperparameters["learning_rate"] == 0.0001

    async def test_create_template_minimal_fields(self, async_session):
        """Test creating template with only required fields."""
        hyperparams = TrainingHyperparameters(
            hidden_dim=256,
            latent_dim=8192,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=32,
            total_steps=5000
        )

        template_data = TrainingTemplateCreate(
            name="Minimal Template",
            encoder_type=SAEArchitectureType.STANDARD,
            hyperparameters=hyperparams
        )

        template = await TrainingTemplateService.create_template(async_session, template_data)

        assert template is not None
        assert template.name == "Minimal Template"
        assert template.description is None
        assert template.model_id is None
        assert template.dataset_id is None
        assert template.is_favorite is False
        assert template.extra_metadata == {}


@pytest.mark.asyncio
class TestTrainingTemplateServiceGet:
    """Test TrainingTemplateService.get_template()."""

    async def test_get_template_success(self, async_session):
        """Test retrieving an existing template."""
        # Create template
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )
        template_data = TrainingTemplateCreate(
            name="Get Test",
            encoder_type=SAEArchitectureType.STANDARD,
            hyperparameters=hyperparams
        )
        created = await TrainingTemplateService.create_template(async_session, template_data)

        # Retrieve template
        retrieved = await TrainingTemplateService.get_template(async_session, created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "Get Test"

    async def test_get_template_not_found(self, async_session):
        """Test retrieving a non-existent template returns None."""
        non_existent_id = uuid4()
        result = await TrainingTemplateService.get_template(async_session, non_existent_id)

        assert result is None


@pytest.mark.asyncio
class TestTrainingTemplateServiceList:
    """Test TrainingTemplateService.list_templates()."""

    async def test_list_templates_empty(self, async_session):
        """Test listing templates when none exist."""
        templates, total = await TrainingTemplateService.list_templates(async_session)

        assert templates == []
        assert total == 0

    async def test_list_templates_multiple(self, async_session):
        """Test listing multiple templates."""
        # Create 3 templates
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )
        for i in range(3):
            template_data = TrainingTemplateCreate(
                name=f"Template {i}",
                encoder_type=SAEArchitectureType.STANDARD,
                hyperparameters=hyperparams
            )
            await TrainingTemplateService.create_template(async_session, template_data)

        templates, total = await TrainingTemplateService.list_templates(async_session)

        assert len(templates) == 3
        assert total == 3

    async def test_list_templates_pagination(self, async_session):
        """Test pagination with skip and limit."""
        # Create 5 templates
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )
        for i in range(5):
            template_data = TrainingTemplateCreate(
                name=f"Template {i}",
                encoder_type=SAEArchitectureType.STANDARD,
                hyperparameters=hyperparams
            )
            await TrainingTemplateService.create_template(async_session, template_data)

        # Get page 2 (skip 2, limit 2)
        templates, total = await TrainingTemplateService.list_templates(
            async_session,
            skip=2,
            limit=2
        )

        assert len(templates) == 2
        assert total == 5

    async def test_list_templates_search(self, async_session):
        """Test searching templates by name."""
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )

        # Create templates with different names
        await TrainingTemplateService.create_template(
            async_session,
            TrainingTemplateCreate(
                name="Vanilla SAE Config",
                encoder_type=SAEArchitectureType.STANDARD,
                hyperparameters=hyperparams
            )
        )
        await TrainingTemplateService.create_template(
            async_session,
            TrainingTemplateCreate(
                name="TopK SAE Config",
                encoder_type=SAEArchitectureType.SKIP,
                hyperparameters=hyperparams
            )
        )
        await TrainingTemplateService.create_template(
            async_session,
            TrainingTemplateCreate(
                name="Other Template",
                encoder_type=SAEArchitectureType.STANDARD,
                hyperparameters=hyperparams
            )
        )

        # Search for "SAE"
        templates, total = await TrainingTemplateService.list_templates(
            async_session,
            search="SAE"
        )

        assert len(templates) == 2
        assert total == 2
        assert all("SAE" in t.name for t in templates)

    async def test_list_templates_filter_by_favorite(self, async_session):
        """Test filtering templates by favorite status."""
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )

        # Create 2 favorite and 1 non-favorite
        await TrainingTemplateService.create_template(
            async_session,
            TrainingTemplateCreate(
                name="Favorite 1",
                encoder_type=SAEArchitectureType.STANDARD,
                hyperparameters=hyperparams,
                is_favorite=True
            )
        )
        await TrainingTemplateService.create_template(
            async_session,
            TrainingTemplateCreate(
                name="Not Favorite",
                encoder_type=SAEArchitectureType.STANDARD,
                hyperparameters=hyperparams,
                is_favorite=False
            )
        )
        await TrainingTemplateService.create_template(
            async_session,
            TrainingTemplateCreate(
                name="Favorite 2",
                encoder_type=SAEArchitectureType.SKIP,
                hyperparameters=hyperparams,
                is_favorite=True
            )
        )

        # Filter for favorites only
        templates, total = await TrainingTemplateService.list_templates(
            async_session,
            is_favorite=True
        )

        assert len(templates) == 2
        assert total == 2
        assert all(t.is_favorite for t in templates)

    async def test_list_templates_sort_by_name(self, async_session):
        """Test sorting templates by name."""
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )

        # Create templates in random order
        for name in ["Charlie", "Alpha", "Bravo"]:
            await TrainingTemplateService.create_template(
                async_session,
                TrainingTemplateCreate(
                    name=name,
                    encoder_type=SAEArchitectureType.STANDARD,
                    hyperparameters=hyperparams
                )
            )

        # Sort ascending
        templates_asc, _ = await TrainingTemplateService.list_templates(
            async_session,
            sort_by="name",
            order="asc"
        )

        assert templates_asc[0].name == "Alpha"
        assert templates_asc[1].name == "Bravo"
        assert templates_asc[2].name == "Charlie"

        # Sort descending
        templates_desc, _ = await TrainingTemplateService.list_templates(
            async_session,
            sort_by="name",
            order="desc"
        )

        assert templates_desc[0].name == "Charlie"
        assert templates_desc[1].name == "Bravo"
        assert templates_desc[2].name == "Alpha"


@pytest.mark.asyncio
class TestTrainingTemplateServiceUpdate:
    """Test TrainingTemplateService.update_template()."""

    async def test_update_template_name_and_description(self, async_session):
        """Test updating template name and description."""
        # Create template
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )
        template_data = TrainingTemplateCreate(
            name="Original Name",
            description="Original description",
            encoder_type=SAEArchitectureType.STANDARD,
            hyperparameters=hyperparams
        )
        created = await TrainingTemplateService.create_template(async_session, template_data)

        # Update template
        updates = TrainingTemplateUpdate(
            name="Updated Name",
            description="Updated description"
        )
        updated = await TrainingTemplateService.update_template(async_session, created.id, updates)

        assert updated is not None
        assert updated.name == "Updated Name"
        assert updated.description == "Updated description"

    async def test_update_template_hyperparameters(self, async_session):
        """Test updating hyperparameters with conversion."""
        # Create template
        original_hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.001,
            batch_size=64,
            total_steps=10000
        )
        template_data = TrainingTemplateCreate(
            name="Hyperparams Test",
            encoder_type=SAEArchitectureType.STANDARD,
            hyperparameters=original_hyperparams
        )
        created = await TrainingTemplateService.create_template(async_session, template_data)

        # Update hyperparameters
        new_hyperparams = TrainingHyperparameters(
            hidden_dim=1024,
            latent_dim=32768,
            l1_alpha=0.0005,
            learning_rate=0.00005,
            batch_size=128,
            total_steps=20000
        )
        updates = TrainingTemplateUpdate(hyperparameters=new_hyperparams)
        updated = await TrainingTemplateService.update_template(async_session, created.id, updates)

        assert updated is not None
        assert updated.hyperparameters["hidden_dim"] == 1024
        assert updated.hyperparameters["latent_dim"] == 32768
        assert updated.hyperparameters["learning_rate"] == 0.00005
        assert updated.hyperparameters["batch_size"] == 128
        assert updated.hyperparameters["total_steps"] == 20000

    async def test_update_template_encoder_type(self, async_session):
        """Test updating encoder type with enum conversion."""
        # Create template
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )
        template_data = TrainingTemplateCreate(
            name="Encoder Test",
            encoder_type=SAEArchitectureType.STANDARD,
            hyperparameters=hyperparams
        )
        created = await TrainingTemplateService.create_template(async_session, template_data)

        # Update encoder type
        updates = TrainingTemplateUpdate(encoder_type=SAEArchitectureType.TRANSCODER)
        updated = await TrainingTemplateService.update_template(async_session, created.id, updates)

        assert updated is not None
        assert updated.encoder_type == SAEArchitectureType.TRANSCODER.value

    async def test_update_template_not_found(self, async_session):
        """Test updating non-existent template returns None."""
        non_existent_id = uuid4()
        updates = TrainingTemplateUpdate(name="Should Not Work")
        result = await TrainingTemplateService.update_template(async_session, non_existent_id, updates)

        assert result is None


@pytest.mark.asyncio
class TestTrainingTemplateServiceDelete:
    """Test TrainingTemplateService.delete_template()."""

    async def test_delete_template_success(self, async_session):
        """Test deleting an existing template."""
        # Create template
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )
        template_data = TrainingTemplateCreate(
            name="To Delete",
            encoder_type=SAEArchitectureType.STANDARD,
            hyperparameters=hyperparams
        )
        created = await TrainingTemplateService.create_template(async_session, template_data)

        # Delete template
        result = await TrainingTemplateService.delete_template(async_session, created.id)

        assert result is True

        # Verify deletion
        retrieved = await TrainingTemplateService.get_template(async_session, created.id)
        assert retrieved is None

    async def test_delete_template_not_found(self, async_session):
        """Test deleting non-existent template returns False."""
        non_existent_id = uuid4()
        result = await TrainingTemplateService.delete_template(async_session, non_existent_id)

        assert result is False


@pytest.mark.asyncio
class TestTrainingTemplateServiceToggleFavorite:
    """Test TrainingTemplateService.toggle_favorite()."""

    async def test_toggle_favorite_to_true(self, async_session):
        """Test toggling favorite from False to True."""
        # Create non-favorite template
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )
        template_data = TrainingTemplateCreate(
            name="Toggle Test",
            encoder_type=SAEArchitectureType.STANDARD,
            hyperparameters=hyperparams,
            is_favorite=False
        )
        created = await TrainingTemplateService.create_template(async_session, template_data)

        # Toggle to favorite
        toggled = await TrainingTemplateService.toggle_favorite(async_session, created.id)

        assert toggled is not None
        assert toggled.is_favorite is True

    async def test_toggle_favorite_to_false(self, async_session):
        """Test toggling favorite from True to False."""
        # Create favorite template
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )
        template_data = TrainingTemplateCreate(
            name="Toggle Test",
            encoder_type=SAEArchitectureType.STANDARD,
            hyperparameters=hyperparams,
            is_favorite=True
        )
        created = await TrainingTemplateService.create_template(async_session, template_data)

        # Toggle to non-favorite
        toggled = await TrainingTemplateService.toggle_favorite(async_session, created.id)

        assert toggled is not None
        assert toggled.is_favorite is False

    async def test_toggle_favorite_not_found(self, async_session):
        """Test toggling favorite on non-existent template returns None."""
        non_existent_id = uuid4()
        result = await TrainingTemplateService.toggle_favorite(async_session, non_existent_id)

        assert result is None


@pytest.mark.asyncio
class TestTrainingTemplateServiceGetFavorites:
    """Test TrainingTemplateService.get_favorites()."""

    async def test_get_favorites_success(self, async_session):
        """Test retrieving only favorite templates."""
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )

        # Create 2 favorites and 1 non-favorite
        for i in range(2):
            await TrainingTemplateService.create_template(
                async_session,
                TrainingTemplateCreate(
                    name=f"Favorite {i}",
                    encoder_type=SAEArchitectureType.STANDARD,
                    hyperparameters=hyperparams,
                    is_favorite=True
                )
            )
        await TrainingTemplateService.create_template(
            async_session,
            TrainingTemplateCreate(
                name="Not Favorite",
                encoder_type=SAEArchitectureType.STANDARD,
                hyperparameters=hyperparams,
                is_favorite=False
            )
        )

        # Get favorites
        favorites, total = await TrainingTemplateService.get_favorites(async_session)

        assert len(favorites) == 2
        assert total == 2
        assert all(t.is_favorite for t in favorites)


@pytest.mark.asyncio
class TestTrainingTemplateServiceExport:
    """Test TrainingTemplateService.export_templates()."""

    async def test_export_all_templates(self, async_session):
        """Test exporting all templates to JSON format."""
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )

        # Create 2 templates
        for i in range(2):
            await TrainingTemplateService.create_template(
                async_session,
                TrainingTemplateCreate(
                    name=f"Export Template {i}",
                    description=f"Description {i}",
                    encoder_type=SAEArchitectureType.STANDARD,
                    hyperparameters=hyperparams,
                    extra_metadata={"key": f"value{i}"}
                )
            )

        # Export all
        export_data = await TrainingTemplateService.export_templates(async_session)

        assert export_data["version"] == "1.0"
        assert len(export_data["templates"]) == 2
        assert "exported_at" in export_data

        # Check first template structure
        template_data = export_data["templates"][0]
        assert "id" in template_data
        assert "name" in template_data
        assert "encoder_type" in template_data
        assert "hyperparameters" in template_data
        assert "created_at" in template_data
        assert "updated_at" in template_data

    async def test_export_specific_templates(self, async_session):
        """Test exporting specific templates by ID."""
        hyperparams = TrainingHyperparameters(
            hidden_dim=512,
            latent_dim=16384,
            l1_alpha=0.001,
            learning_rate=0.0001,
            batch_size=64,
            total_steps=10000
        )

        # Create 3 templates
        created_templates = []
        for i in range(3):
            template = await TrainingTemplateService.create_template(
                async_session,
                TrainingTemplateCreate(
                    name=f"Template {i}",
                    encoder_type=SAEArchitectureType.STANDARD,
                    hyperparameters=hyperparams
                )
            )
            created_templates.append(template)

        # Export only first 2 templates
        export_data = await TrainingTemplateService.export_templates(
            async_session,
            template_ids=[created_templates[0].id, created_templates[1].id]
        )

        assert len(export_data["templates"]) == 2


@pytest.mark.asyncio
class TestTrainingTemplateServiceImport:
    """Test TrainingTemplateService.import_templates()."""

    async def test_import_templates_new(self, async_session):
        """Test importing new templates."""
        # Note: model_id/dataset_id omitted as they require actual parent records
        import_data = {
            "version": "1.0",
            "templates": [
                {
                    "name": "Imported Template 1",
                    "description": "First import",
                    "encoder_type": "standard",
                    "hyperparameters": {
                        "hidden_dim": 512,
                        "latent_dim": 16384,
                        "l1_alpha": 0.001,
                        "learning_rate": 0.0001,
                        "batch_size": 64,
                        "total_steps": 10000
                    },
                    "is_favorite": False,
                    "extra_metadata": {}
                },
                {
                    "name": "Imported Template 2",
                    "description": "Second import",
                    "encoder_type": "skip",
                    "hyperparameters": {
                        "hidden_dim": 1024,
                        "latent_dim": 32768,
                        "l1_alpha": 0.0005,
                        "learning_rate": 0.00005,
                        "batch_size": 128,
                        "total_steps": 20000
                    },
                    "is_favorite": True,
                    "extra_metadata": {"source": "import"}
                }
            ]
        }

        result = await TrainingTemplateService.import_templates(async_session, import_data)

        assert result["created"] == 2
        assert result["updated"] == 0
        assert result["skipped"] == 0
        assert result["total"] == 2

        # Verify templates were created
        templates, total = await TrainingTemplateService.list_templates(async_session)
        assert total == 2

    async def test_import_templates_duplicate_skip(self, async_session):
        """Test importing templates with duplicate names (skip mode)."""
        # Create existing template
        hyperparams = TrainingHyperparameters(
            hidden_dim=256,
            latent_dim=8192,
            l1_alpha=0.001,
            learning_rate=0.002,
            batch_size=32,
            total_steps=5000
        )
        await TrainingTemplateService.create_template(
            async_session,
            TrainingTemplateCreate(
                name="Existing Template",
                encoder_type=SAEArchitectureType.STANDARD,
                hyperparameters=hyperparams
            )
        )

        # Import with same name
        import_data = {
            "version": "1.0",
            "templates": [
                {
                    "name": "Existing Template",
                    "encoder_type": "standard",
                    "hyperparameters": {
                        "hidden_dim": 512,
                        "latent_dim": 16384,
                        "l1_alpha": 0.001,
                        "learning_rate": 0.0001,
                        "batch_size": 64,
                        "total_steps": 10000
                    }
                }
            ]
        }

        result = await TrainingTemplateService.import_templates(
            async_session,
            import_data,
            overwrite_duplicates=False
        )

        assert result["created"] == 0
        assert result["updated"] == 0
        assert result["skipped"] == 1

        # Verify original template unchanged
        templates, _ = await TrainingTemplateService.list_templates(async_session)
        assert templates[0].hyperparameters["hidden_dim"] == 256  # Original value

    async def test_import_templates_duplicate_overwrite(self, async_session):
        """Test importing templates with duplicate names (overwrite mode)."""
        # Create existing template
        hyperparams = TrainingHyperparameters(
            hidden_dim=256,
            latent_dim=8192,
            l1_alpha=0.001,
            learning_rate=0.002,
            batch_size=32,
            total_steps=5000
        )
        await TrainingTemplateService.create_template(
            async_session,
            TrainingTemplateCreate(
                name="Existing Template",
                encoder_type=SAEArchitectureType.STANDARD,
                hyperparameters=hyperparams
            )
        )

        # Import with same name
        import_data = {
            "version": "1.0",
            "templates": [
                {
                    "name": "Existing Template",
                    "encoder_type": "transcoder",
                    "hyperparameters": {
                        "hidden_dim": 512,
                        "learning_rate": 0.001,
                        "batch_size": 64
                    }
                }
            ]
        }

        result = await TrainingTemplateService.import_templates(
            async_session,
            import_data,
            overwrite_duplicates=True
        )

        assert result["created"] == 0
        assert result["updated"] == 1
        assert result["skipped"] == 0

        # Verify template was updated
        templates, _ = await TrainingTemplateService.list_templates(async_session)
        assert templates[0].hyperparameters["hidden_dim"] == 512  # Updated value
        assert templates[0].encoder_type == "transcoder"  # Updated type

    async def test_import_templates_invalid_version(self, async_session):
        """Test importing with unsupported version raises error."""
        import_data = {
            "version": "2.0",
            "templates": []
        }

        with pytest.raises(ValueError, match="Unsupported import version"):
            await TrainingTemplateService.import_templates(async_session, import_data)

    async def test_import_templates_no_templates(self, async_session):
        """Test importing with no templates raises error."""
        import_data = {
            "version": "1.0",
            "templates": []
        }

        with pytest.raises(ValueError, match="No templates found"):
            await TrainingTemplateService.import_templates(async_session, import_data)
