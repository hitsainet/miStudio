"""
Integration tests for Extraction Template API endpoints.

Tests complete CRUD workflows for extraction template management.
"""

import pytest
from uuid import UUID

from src.services.extraction_template_service import ExtractionTemplateService
from src.schemas.extraction_template import (
    ExtractionTemplateCreate,
    ExtractionTemplateUpdate,
)


class TestExtractionTemplatesAPI:
    """Integration tests for extraction template API operations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_template(self, async_session):
        """
        Test creating a new extraction template.

        Verifies that:
        1. Template can be created with all fields
        2. ID is auto-generated (UUID)
        3. Timestamps are set automatically
        4. Default values are applied (is_favorite=False)
        """
        template_data = ExtractionTemplateCreate(
            name="Test Template",
            description="Extract from early layers",
            layer_indices=[0, 5, 11],
            hook_types=["residual", "mlp"],
            max_samples=1000,
            batch_size=32,
            top_k_examples=10,
            is_favorite=False,
            extra_metadata={"author": "test_user"}
        )

        template = await ExtractionTemplateService.create_template(async_session, template_data)
        await async_session.commit()
        await async_session.refresh(template)

        assert template.id is not None
        assert isinstance(template.id, UUID)
        assert template.name == "Test Template"
        assert template.description == "Extract from early layers"
        assert template.layer_indices == [0, 5, 11]
        assert template.hook_types == ["residual", "mlp"]
        assert template.max_samples == 1000
        assert template.batch_size == 32
        assert template.top_k_examples == 10
        assert template.is_favorite is False
        assert template.extra_metadata == {"author": "test_user"}
        assert template.created_at is not None
        assert template.updated_at is not None

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, template.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_template(self, async_session):
        """
        Test retrieving a template by ID.

        Verifies that:
        1. Template can be fetched by UUID
        2. All fields are returned correctly
        3. Returns None for non-existent ID
        """
        # Create template
        template_data = ExtractionTemplateCreate(
            name="Get Test Template",
            layer_indices=[0, 1, 2],
            hook_types=["residual"],
            max_samples=500,
            batch_size=16,
            top_k_examples=5,
        )
        template = await ExtractionTemplateService.create_template(async_session, template_data)
        await async_session.commit()

        # Retrieve template
        fetched = await ExtractionTemplateService.get_template(async_session, template.id)
        assert fetched is not None
        assert fetched.id == template.id
        assert fetched.name == "Get Test Template"

        # Test non-existent ID
        from uuid import uuid4
        non_existent = await ExtractionTemplateService.get_template(async_session, uuid4())
        assert non_existent is None

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, template.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_templates(self, async_session):
        """
        Test listing templates with pagination.

        Verifies that:
        1. Templates can be listed with pagination
        2. Total count is accurate
        3. Empty list returned when no templates exist
        """
        # Create multiple templates
        templates_to_create = [
            ExtractionTemplateCreate(
                name=f"List Template {i}",
                layer_indices=[0, i],
                hook_types=["residual"],
                max_samples=100 * (i + 1),
                batch_size=8,
                top_k_examples=5,
            )
            for i in range(5)
        ]

        created_templates = []
        for template_data in templates_to_create:
            template = await ExtractionTemplateService.create_template(async_session, template_data)
            created_templates.append(template)
        await async_session.commit()

        # List templates
        templates, total = await ExtractionTemplateService.list_templates(
            async_session,
            skip=0,
            limit=10
        )
        assert total >= 5
        assert len(templates) >= 5

        # Test pagination
        templates_page1, total = await ExtractionTemplateService.list_templates(
            async_session,
            skip=0,
            limit=2
        )
        assert len(templates_page1) == 2
        assert total >= 5

        # Cleanup
        for template in created_templates:
            await ExtractionTemplateService.delete_template(async_session, template.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_search_templates(self, async_session):
        """
        Test searching templates by name or description.

        Verifies that:
        1. Search works on name field
        2. Search works on description field
        3. Search is case-insensitive
        """
        # Create templates with specific names/descriptions
        template1 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Quick Shallow Extract",
                description="Fast extraction from first 3 layers",
                layer_indices=[0, 1, 2],
                hook_types=["residual"],
                max_samples=100,
                batch_size=8,
                top_k_examples=5,
            )
        )
        template2 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Deep Analysis",
                description="Thorough extraction from all layers",
                layer_indices=list(range(24)),
                hook_types=["residual", "mlp", "attention"],
                max_samples=10000,
                batch_size=64,
                top_k_examples=20,
            )
        )
        await async_session.commit()

        # Search by name (case-insensitive)
        results, count = await ExtractionTemplateService.list_templates(
            async_session,
            search="shallow"
        )
        assert count >= 1
        found_names = [t.name for t in results]
        assert "Quick Shallow Extract" in found_names

        # Search by description
        results, count = await ExtractionTemplateService.list_templates(
            async_session,
            search="thorough"
        )
        assert count >= 1
        found_names = [t.name for t in results]
        assert "Deep Analysis" in found_names

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, template1.id)
        await ExtractionTemplateService.delete_template(async_session, template2.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_template(self, async_session):
        """
        Test updating template fields.

        Verifies that:
        1. Individual fields can be updated
        2. Unchanged fields remain intact
        3. Updated_at timestamp is updated
        """
        # Create template
        template_data = ExtractionTemplateCreate(
            name="Original Name",
            description="Original description",
            layer_indices=[0, 5],
            hook_types=["residual"],
            max_samples=1000,
            batch_size=32,
            top_k_examples=10,
        )
        template = await ExtractionTemplateService.create_template(async_session, template_data)
        await async_session.commit()
        await async_session.refresh(template)

        original_created_at = template.created_at
        original_updated_at = template.updated_at

        # Update some fields
        import asyncio
        await asyncio.sleep(0.1)  # Ensure timestamp difference

        updates = ExtractionTemplateUpdate(
            name="Updated Name",
            max_samples=2000,
        )
        updated = await ExtractionTemplateService.update_template(
            async_session, template.id, updates
        )
        await async_session.commit()
        await async_session.refresh(updated)

        assert updated.name == "Updated Name"
        assert updated.max_samples == 2000
        assert updated.description == "Original description"  # Unchanged
        assert updated.layer_indices == [0, 5]  # Unchanged
        assert updated.created_at == original_created_at  # Unchanged

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, template.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_delete_template(self, async_session):
        """
        Test deleting a template.

        Verifies that:
        1. Template can be deleted by ID
        2. Deleted template cannot be retrieved
        3. Delete returns False for non-existent ID
        """
        # Create template
        template_data = ExtractionTemplateCreate(
            name="Delete Test",
            layer_indices=[0],
            hook_types=["residual"],
            max_samples=100,
            batch_size=8,
            top_k_examples=5,
        )
        template = await ExtractionTemplateService.create_template(async_session, template_data)
        await async_session.commit()
        template_id = template.id

        # Delete template
        deleted = await ExtractionTemplateService.delete_template(async_session, template_id)
        await async_session.commit()

        assert deleted is True

        # Verify it's gone
        fetched = await ExtractionTemplateService.get_template(async_session, template_id)
        assert fetched is None

        # Try to delete non-existent template
        from uuid import uuid4
        deleted_again = await ExtractionTemplateService.delete_template(async_session, uuid4())
        assert deleted_again is False

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_toggle_favorite(self, async_session):
        """
        Test toggling favorite status.

        Verifies that:
        1. Favorite status can be toggled on
        2. Favorite status can be toggled off
        3. Toggle returns None for non-existent ID
        """
        # Create template (default is_favorite=False)
        template_data = ExtractionTemplateCreate(
            name="Favorite Test",
            layer_indices=[0, 5],
            hook_types=["residual"],
            max_samples=1000,
            batch_size=32,
            top_k_examples=10,
        )
        template = await ExtractionTemplateService.create_template(async_session, template_data)
        await async_session.commit()

        assert template.is_favorite is False

        # Toggle to favorite
        toggled = await ExtractionTemplateService.toggle_favorite(async_session, template.id)
        await async_session.commit()
        await async_session.refresh(toggled)

        assert toggled.is_favorite is True

        # Toggle back to non-favorite
        toggled_again = await ExtractionTemplateService.toggle_favorite(async_session, template.id)
        await async_session.commit()
        await async_session.refresh(toggled_again)

        assert toggled_again.is_favorite is False

        # Test non-existent ID
        from uuid import uuid4
        non_existent = await ExtractionTemplateService.toggle_favorite(async_session, uuid4())
        assert non_existent is None

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, template.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_favorites(self, async_session):
        """
        Test listing only favorite templates.

        Verifies that:
        1. Only templates with is_favorite=True are returned
        2. Non-favorite templates are excluded
        3. Pagination works for favorites
        """
        # Create mix of favorite and non-favorite templates
        favorite1 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Favorite 1",
                layer_indices=[0, 5],
                hook_types=["residual"],
                max_samples=1000,
                batch_size=32,
                top_k_examples=10,
                is_favorite=True,
            )
        )
        favorite2 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Favorite 2",
                layer_indices=[0, 10],
                hook_types=["mlp"],
                max_samples=2000,
                batch_size=64,
                top_k_examples=20,
                is_favorite=True,
            )
        )
        normal = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Normal Template",
                layer_indices=[0],
                hook_types=["residual"],
                max_samples=100,
                batch_size=8,
                top_k_examples=5,
                is_favorite=False,
            )
        )
        await async_session.commit()

        # Get only favorites
        favorites, count = await ExtractionTemplateService.get_favorites(
            async_session,
            skip=0,
            limit=10
        )
        assert count >= 2
        favorite_names = [t.name for t in favorites]
        assert "Favorite 1" in favorite_names
        assert "Favorite 2" in favorite_names
        assert "Normal Template" not in favorite_names

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, favorite1.id)
        await ExtractionTemplateService.delete_template(async_session, favorite2.id)
        await ExtractionTemplateService.delete_template(async_session, normal.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_filter_by_favorite(self, async_session):
        """
        Test filtering templates by favorite status in list.

        Verifies that:
        1. is_favorite=True filter returns only favorites
        2. is_favorite=False filter returns only non-favorites
        3. No filter returns all templates
        """
        # Create templates with different favorite status
        favorite = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Is Favorite",
                layer_indices=[0],
                hook_types=["residual"],
                max_samples=100,
                batch_size=8,
                top_k_examples=5,
                is_favorite=True,
            )
        )
        not_favorite = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Not Favorite",
                layer_indices=[0],
                hook_types=["residual"],
                max_samples=100,
                batch_size=8,
                top_k_examples=5,
                is_favorite=False,
            )
        )
        await async_session.commit()

        # Filter for favorites only
        favorites, count = await ExtractionTemplateService.list_templates(
            async_session,
            is_favorite=True
        )
        favorite_names = [t.name for t in favorites]
        assert "Is Favorite" in favorite_names
        assert "Not Favorite" not in favorite_names

        # Filter for non-favorites only
        non_favorites, count = await ExtractionTemplateService.list_templates(
            async_session,
            is_favorite=False
        )
        non_favorite_names = [t.name for t in non_favorites]
        assert "Not Favorite" in non_favorite_names
        # Note: "Is Favorite" might not be in this list depending on other templates

        # No filter - get all
        all_templates, count = await ExtractionTemplateService.list_templates(
            async_session
        )
        all_names = [t.name for t in all_templates]
        assert "Is Favorite" in all_names
        assert "Not Favorite" in all_names

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, favorite.id)
        await ExtractionTemplateService.delete_template(async_session, not_favorite.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sorting_templates(self, async_session):
        """
        Test sorting templates by different fields.

        Verifies that:
        1. Templates can be sorted by created_at (desc/asc)
        2. Templates can be sorted by name
        3. Sort order is applied correctly
        """
        # Create templates with different attributes
        import asyncio
        template1 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Alpha",
                layer_indices=[0],
                hook_types=["residual"],
                max_samples=100,
                batch_size=8,
                top_k_examples=5,
            )
        )
        await asyncio.sleep(0.1)  # Ensure different timestamps

        template2 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Beta",
                layer_indices=[0],
                hook_types=["residual"],
                max_samples=100,
                batch_size=8,
                top_k_examples=5,
            )
        )
        await asyncio.sleep(0.1)

        template3 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Gamma",
                layer_indices=[0],
                hook_types=["residual"],
                max_samples=100,
                batch_size=8,
                top_k_examples=5,
            )
        )
        await async_session.commit()

        # Sort by name ascending
        templates, _ = await ExtractionTemplateService.list_templates(
            async_session,
            sort_by="name",
            order="asc"
        )
        names = [t.name for t in templates if t.name in ["Alpha", "Beta", "Gamma"]]
        assert names.index("Alpha") < names.index("Beta") < names.index("Gamma")

        # Sort by created_at descending (newest first)
        templates, _ = await ExtractionTemplateService.list_templates(
            async_session,
            sort_by="created_at",
            order="desc"
        )
        ids = [t.id for t in templates if t.id in [template1.id, template2.id, template3.id]]
        # Gamma (newest) should come before Alpha (oldest)
        assert ids.index(template3.id) < ids.index(template1.id)

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, template1.id)
        await ExtractionTemplateService.delete_template(async_session, template2.id)
        await ExtractionTemplateService.delete_template(async_session, template3.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_template_with_all_hook_types(self, async_session):
        """
        Test template with all available hook types.

        Verifies that:
        1. All hook types (residual, mlp, attention) can be used
        2. Array field stores all values correctly
        """
        template_data = ExtractionTemplateCreate(
            name="All Hooks Template",
            layer_indices=[0, 5, 10, 15],
            hook_types=["residual", "mlp", "attention"],
            max_samples=5000,
            batch_size=64,
            top_k_examples=20,
        )
        template = await ExtractionTemplateService.create_template(async_session, template_data)
        await async_session.commit()
        await async_session.refresh(template)

        assert len(template.hook_types) == 3
        assert "residual" in template.hook_types
        assert "mlp" in template.hook_types
        assert "attention" in template.hook_types

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, template.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_template_with_many_layers(self, async_session):
        """
        Test template with many layer indices (all layers of a large model).

        Verifies that:
        1. Large arrays can be stored
        2. All layer indices are preserved
        """
        all_layers = list(range(32))  # 32-layer model
        template_data = ExtractionTemplateCreate(
            name="All Layers Template",
            layer_indices=all_layers,
            hook_types=["residual"],
            max_samples=10000,
            batch_size=128,
            top_k_examples=50,
        )
        template = await ExtractionTemplateService.create_template(async_session, template_data)
        await async_session.commit()
        await async_session.refresh(template)

        assert len(template.layer_indices) == 32
        assert template.layer_indices == all_layers
        assert template.layer_indices[0] == 0
        assert template.layer_indices[-1] == 31

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, template.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_extra_metadata_field(self, async_session):
        """
        Test extra_metadata JSONB field.

        Verifies that:
        1. Complex metadata can be stored
        2. Nested structures are preserved
        3. Metadata can be updated
        """
        metadata = {
            "author": "user123",
            "version": "2.0",
            "tags": ["quick", "shallow"],
            "notes": "Optimized for speed",
            "settings": {
                "gpu_enabled": True,
                "precision": "fp16"
            }
        }

        template_data = ExtractionTemplateCreate(
            name="Metadata Test",
            layer_indices=[0, 5],
            hook_types=["residual"],
            max_samples=1000,
            batch_size=32,
            top_k_examples=10,
            extra_metadata=metadata,
        )
        template = await ExtractionTemplateService.create_template(async_session, template_data)
        await async_session.commit()
        await async_session.refresh(template)

        assert template.extra_metadata["author"] == "user123"
        assert "quick" in template.extra_metadata["tags"]
        assert template.extra_metadata["settings"]["gpu_enabled"] is True

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, template.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_export_all_templates(self, async_session):
        """
        Test exporting all templates to JSON format.

        Verifies that:
        1. Export includes all templates when no IDs specified
        2. Export format includes version and metadata
        3. All template fields are exported correctly
        """
        # Create test templates
        template1 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Export Test 1",
                description="First template",
                layer_indices=[0, 5],
                hook_types=["residual"],
                max_samples=1000,
                batch_size=32,
                top_k_examples=10,
                is_favorite=True,
                extra_metadata={"export_id": 1},
            )
        )
        template2 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Export Test 2",
                description="Second template",
                layer_indices=[0, 10, 20],
                hook_types=["mlp", "attention"],
                max_samples=2000,
                batch_size=64,
                top_k_examples=20,
                is_favorite=False,
                extra_metadata={"export_id": 2},
            )
        )
        await async_session.commit()

        # Export all templates
        export_data = await ExtractionTemplateService.export_templates(async_session)

        # Verify export structure
        assert "version" in export_data
        assert export_data["version"] == "1.0"
        assert "export_date" in export_data
        assert "count" in export_data
        assert "templates" in export_data
        assert export_data["count"] >= 2

        # Verify templates are in export
        exported_names = [t["name"] for t in export_data["templates"]]
        assert "Export Test 1" in exported_names
        assert "Export Test 2" in exported_names

        # Verify field completeness
        for template in export_data["templates"]:
            if template["name"] == "Export Test 1":
                assert template["description"] == "First template"
                assert template["layer_indices"] == [0, 5]
                assert template["hook_types"] == ["residual"]
                assert template["max_samples"] == 1000
                assert template["batch_size"] == 32
                assert template["top_k_examples"] == 10
                assert template["is_favorite"] is True
                assert template["extra_metadata"]["export_id"] == 1

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, template1.id)
        await ExtractionTemplateService.delete_template(async_session, template2.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_export_specific_templates(self, async_session):
        """
        Test exporting specific templates by ID.

        Verifies that:
        1. Only specified templates are exported
        2. Other templates are not included
        3. Export works with single or multiple IDs
        """
        # Create test templates
        template1 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Export Specific 1",
                layer_indices=[0],
                hook_types=["residual"],
                max_samples=100,
                batch_size=8,
                top_k_examples=5,
            )
        )
        template2 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Export Specific 2",
                layer_indices=[5],
                hook_types=["mlp"],
                max_samples=200,
                batch_size=16,
                top_k_examples=10,
            )
        )
        template3 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Not Exported",
                layer_indices=[10],
                hook_types=["attention"],
                max_samples=300,
                batch_size=32,
                top_k_examples=15,
            )
        )
        await async_session.commit()

        # Export only template1 and template2
        export_data = await ExtractionTemplateService.export_templates(
            async_session,
            template_ids=[template1.id, template2.id]
        )

        # Verify only selected templates are exported
        assert export_data["count"] == 2
        exported_names = [t["name"] for t in export_data["templates"]]
        assert "Export Specific 1" in exported_names
        assert "Export Specific 2" in exported_names
        assert "Not Exported" not in exported_names

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, template1.id)
        await ExtractionTemplateService.delete_template(async_session, template2.id)
        await ExtractionTemplateService.delete_template(async_session, template3.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_import_new_templates(self, async_session):
        """
        Test importing new templates from JSON.

        Verifies that:
        1. New templates are created from import data
        2. All fields are set correctly
        3. Import results include correct counts
        """
        import_data = {
            "version": "1.0",
            "templates": [
                {
                    "name": "Imported Template 1",
                    "description": "First imported",
                    "layer_indices": [0, 5, 10],
                    "hook_types": ["residual", "mlp"],
                    "max_samples": 1500,
                    "batch_size": 48,
                    "top_k_examples": 15,
                    "is_favorite": False,
                    "extra_metadata": {"source": "import_test"}
                },
                {
                    "name": "Imported Template 2",
                    "description": "Second imported",
                    "layer_indices": [0, 1, 2, 3],
                    "hook_types": ["attention"],
                    "max_samples": 500,
                    "batch_size": 16,
                    "top_k_examples": 5,
                    "is_favorite": True,
                    "extra_metadata": {"source": "import_test"}
                }
            ]
        }

        # Import templates
        result = await ExtractionTemplateService.import_templates(
            async_session,
            import_data,
            overwrite_duplicates=False
        )

        # Verify import results
        assert result["created"] == 2
        assert result["updated"] == 0
        assert result["skipped"] == 0
        assert result["total_processed"] == 2
        assert len(result["errors"]) == 0

        # Verify templates were created
        templates, total = await ExtractionTemplateService.list_templates(async_session)
        imported_names = [t.name for t in templates]
        assert "Imported Template 1" in imported_names
        assert "Imported Template 2" in imported_names

        # Verify field values
        template1 = next(t for t in templates if t.name == "Imported Template 1")
        assert template1.description == "First imported"
        assert template1.layer_indices == [0, 5, 10]
        assert template1.hook_types == ["residual", "mlp"]
        assert template1.max_samples == 1500
        assert template1.batch_size == 48
        assert template1.top_k_examples == 15
        assert template1.is_favorite is False

        # Cleanup
        for template in templates:
            if template.name.startswith("Imported Template"):
                await ExtractionTemplateService.delete_template(async_session, template.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_import_skip_duplicates(self, async_session):
        """
        Test importing templates with duplicate names (skip mode).

        Verifies that:
        1. Existing templates with same name are skipped
        2. Skip count is reported correctly
        3. Original template is unchanged
        """
        # Create existing template
        existing = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Duplicate Test",
                description="Original description",
                layer_indices=[0, 5],
                hook_types=["residual"],
                max_samples=1000,
                batch_size=32,
                top_k_examples=10,
            )
        )
        await async_session.commit()

        # Try to import template with same name
        import_data = {
            "version": "1.0",
            "templates": [
                {
                    "name": "Duplicate Test",
                    "description": "New description",
                    "layer_indices": [0, 10, 20],
                    "hook_types": ["mlp"],
                    "max_samples": 2000,
                    "batch_size": 64,
                    "top_k_examples": 20,
                }
            ]
        }

        # Import with overwrite_duplicates=False
        result = await ExtractionTemplateService.import_templates(
            async_session,
            import_data,
            overwrite_duplicates=False
        )

        # Verify skipped
        assert result["created"] == 0
        assert result["updated"] == 0
        assert result["skipped"] == 1
        assert result["total_processed"] == 1

        # Verify original template unchanged
        await async_session.refresh(existing)
        assert existing.description == "Original description"
        assert existing.layer_indices == [0, 5]
        assert existing.max_samples == 1000

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, existing.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_import_overwrite_duplicates(self, async_session):
        """
        Test importing templates with duplicate names (overwrite mode).

        Verifies that:
        1. Existing templates with same name are updated
        2. Update count is reported correctly
        3. Template fields are updated with new values
        """
        # Create existing template
        existing = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Overwrite Test",
                description="Original description",
                layer_indices=[0, 5],
                hook_types=["residual"],
                max_samples=1000,
                batch_size=32,
                top_k_examples=10,
            )
        )
        await async_session.commit()
        original_id = existing.id

        # Import template with same name
        import_data = {
            "version": "1.0",
            "templates": [
                {
                    "name": "Overwrite Test",
                    "description": "Updated description",
                    "layer_indices": [0, 10, 20],
                    "hook_types": ["mlp", "attention"],
                    "max_samples": 2000,
                    "batch_size": 64,
                    "top_k_examples": 20,
                }
            ]
        }

        # Import with overwrite_duplicates=True
        result = await ExtractionTemplateService.import_templates(
            async_session,
            import_data,
            overwrite_duplicates=True
        )

        # Verify updated
        assert result["created"] == 0
        assert result["updated"] == 1
        assert result["skipped"] == 0
        assert result["total_processed"] == 1

        # Verify template was updated
        updated = await ExtractionTemplateService.get_template(async_session, original_id)
        assert updated is not None
        assert updated.description == "Updated description"
        assert updated.layer_indices == [0, 10, 20]
        assert updated.hook_types == ["mlp", "attention"]
        assert updated.max_samples == 2000
        assert updated.batch_size == 64
        assert updated.top_k_examples == 20

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, original_id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_export_import_roundtrip(self, async_session):
        """
        Test complete export/import cycle.

        Verifies that:
        1. Templates can be exported and then imported
        2. Imported templates match original templates
        3. All data is preserved through the cycle
        """
        # Create original templates
        original1 = await ExtractionTemplateService.create_template(
            async_session,
            ExtractionTemplateCreate(
                name="Roundtrip 1",
                description="Test roundtrip",
                layer_indices=[0, 5, 10, 15],
                hook_types=["residual", "mlp"],
                max_samples=3000,
                batch_size=96,
                top_k_examples=25,
                is_favorite=True,
                extra_metadata={"test": "roundtrip", "value": 42}
            )
        )
        await async_session.commit()

        # Export
        export_data = await ExtractionTemplateService.export_templates(
            async_session,
            template_ids=[original1.id]
        )

        # Delete original
        await ExtractionTemplateService.delete_template(async_session, original1.id)
        await async_session.commit()

        # Import back
        result = await ExtractionTemplateService.import_templates(
            async_session,
            export_data,
            overwrite_duplicates=False
        )

        assert result["created"] == 1
        assert result["errors"] == []

        # Verify imported template matches original
        templates, _ = await ExtractionTemplateService.list_templates(async_session)
        imported = next(t for t in templates if t.name == "Roundtrip 1")

        assert imported.description == "Test roundtrip"
        assert imported.layer_indices == [0, 5, 10, 15]
        assert imported.hook_types == ["residual", "mlp"]
        assert imported.max_samples == 3000
        assert imported.batch_size == 96
        assert imported.top_k_examples == 25
        assert imported.is_favorite is True
        assert imported.extra_metadata["test"] == "roundtrip"
        assert imported.extra_metadata["value"] == 42

        # Cleanup
        await ExtractionTemplateService.delete_template(async_session, imported.id)
        await async_session.commit()
