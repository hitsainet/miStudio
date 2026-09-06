"""
Unit tests for multi-SAE import functionality.

Tests the ability to import multiple SAEs from a single training that
produces SAEs for multiple layers and/or hook types.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import json
import tempfile
import os

from src.schemas.sae import (
    SAEImportFromTrainingRequest,
    SAEImportFromTrainingResponse,
    AvailableSAEInfo,
    TrainingAvailableSAEsResponse,
)
from src.models.training import Training, TrainingStatus
from src.models.external_sae import ExternalSAE, SAESource, SAEStatus, SAEFormat


class TestAvailableSAEInfoSchema:
    """Tests for AvailableSAEInfo schema."""

    def test_available_sae_info_creation(self):
        """Test creating AvailableSAEInfo with valid data."""
        info = AvailableSAEInfo(
            layer=0,
            hook_type="hook_resid_pre",
            path="/data/trainings/123/community_format/layer_0_hook_resid_pre",
            size_bytes=1024 * 1024 * 50  # 50MB
        )
        assert info.layer == 0
        assert info.hook_type == "hook_resid_pre"
        assert info.size_bytes == 50 * 1024 * 1024

    def test_available_sae_info_optional_size(self):
        """Test that size_bytes is optional."""
        info = AvailableSAEInfo(
            layer=1,
            hook_type="hook_mlp_out",
            path="/some/path"
        )
        assert info.size_bytes is None


class TestTrainingAvailableSAEsResponse:
    """Tests for TrainingAvailableSAEsResponse schema."""

    def test_response_with_multiple_saes(self):
        """Test response listing multiple available SAEs."""
        response = TrainingAvailableSAEsResponse(
            training_id="training-123",
            available_saes=[
                AvailableSAEInfo(layer=0, hook_type="hook_resid_pre", path="/path/0_pre"),
                AvailableSAEInfo(layer=0, hook_type="hook_resid_post", path="/path/0_post"),
                AvailableSAEInfo(layer=1, hook_type="hook_resid_pre", path="/path/1_pre"),
                AvailableSAEInfo(layer=1, hook_type="hook_resid_post", path="/path/1_post"),
            ],
            total_count=4
        )
        assert response.training_id == "training-123"
        assert len(response.available_saes) == 4
        assert response.total_count == 4

    def test_response_empty_saes(self):
        """Test response with no available SAEs."""
        response = TrainingAvailableSAEsResponse(
            training_id="training-456",
            available_saes=[],
            total_count=0
        )
        assert len(response.available_saes) == 0
        assert response.total_count == 0


class TestSAEImportFromTrainingRequest:
    """Tests for SAEImportFromTrainingRequest schema."""

    def test_import_all_default(self):
        """Test that import_all defaults to True."""
        request = SAEImportFromTrainingRequest(training_id="training-123")
        assert request.import_all is True
        assert request.layers is None
        assert request.hook_types is None

    def test_import_specific_layers(self):
        """Test importing specific layers only."""
        request = SAEImportFromTrainingRequest(
            training_id="training-123",
            import_all=False,
            layers=[0, 2, 4]
        )
        assert request.import_all is False
        assert request.layers == [0, 2, 4]
        assert request.hook_types is None

    def test_import_specific_hook_types(self):
        """Test importing specific hook types only."""
        request = SAEImportFromTrainingRequest(
            training_id="training-123",
            import_all=False,
            hook_types=["hook_resid_pre", "hook_mlp_out"]
        )
        assert request.import_all is False
        assert request.layers is None
        assert request.hook_types == ["hook_resid_pre", "hook_mlp_out"]

    def test_import_with_name_and_description(self):
        """Test request with optional name and description."""
        request = SAEImportFromTrainingRequest(
            training_id="training-123",
            name="My Custom SAE",
            description="SAE trained on specific data"
        )
        assert request.name == "My Custom SAE"
        assert request.description == "SAE trained on specific data"


class TestSAEImportFromTrainingResponse:
    """Tests for SAEImportFromTrainingResponse schema."""

    def test_response_multiple_saes_imported(self):
        """Test response when multiple SAEs are imported."""
        # Create mock SAE response objects
        from datetime import datetime

        mock_sae_data = {
            "id": "sae-001",
            "name": "SAE L0 resid_pre",
            "description": "Test SAE",
            "source": SAESource.TRAINED,
            "status": SAEStatus.READY,
            "format": SAEFormat.COMMUNITY_STANDARD,
            "layer": 0,
            "hook_type": "hook_resid_pre",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        response = SAEImportFromTrainingResponse(
            imported_count=4,
            sae_ids=["sae-001", "sae-002", "sae-003", "sae-004"],
            saes=[],  # Would normally contain full SAE objects
            training_id="training-123",
            message="Imported 4 SAE(s) from training training-123"
        )

        assert response.imported_count == 4
        assert len(response.sae_ids) == 4
        assert response.training_id == "training-123"
        assert "4" in response.message

    def test_response_single_sae_imported(self):
        """Test response when a single SAE is imported."""
        response = SAEImportFromTrainingResponse(
            imported_count=1,
            sae_ids=["sae-001"],
            saes=[],
            training_id="training-456",
            message="Imported 1 SAE(s) from training training-456"
        )

        assert response.imported_count == 1
        assert len(response.sae_ids) == 1


class TestDirectoryStructureParsing:
    """Tests for parsing training checkpoint directory structure."""

    def test_parse_layer_hook_from_directory_name(self):
        """Test extracting layer and hook_type from directory names."""
        # The service should parse directory names like:
        # layer_0_hook_resid_pre -> (layer=0, hook_type="hook_resid_pre")
        # layer_12_hook_mlp_out -> (layer=12, hook_type="hook_mlp_out")

        test_cases = [
            ("layer_0_hook_resid_pre", 0, "hook_resid_pre"),
            ("layer_1_hook_resid_post", 1, "hook_resid_post"),
            ("layer_12_hook_mlp_out", 12, "hook_mlp_out"),
            ("layer_5_hook_attn_out", 5, "hook_attn_out"),
        ]

        for dir_name, expected_layer, expected_hook in test_cases:
            # Parse the directory name
            parts = dir_name.split("_", 2)  # Split into: "layer", "N", "hook_type"
            layer = int(parts[1])
            hook_type = parts[2]

            assert layer == expected_layer, f"Failed for {dir_name}"
            assert hook_type == expected_hook, f"Failed for {dir_name}"

    def test_create_temp_community_format_structure(self):
        """Test creating and scanning a mock community_format directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock community_format structure
            community_format = Path(tmpdir) / "community_format"
            community_format.mkdir()

            # Create multiple layer/hook directories
            test_saes = [
                ("layer_0_hook_resid_pre", {"d_in": 768, "d_sae": 6144}),
                ("layer_0_hook_resid_post", {"d_in": 768, "d_sae": 6144}),
                ("layer_1_hook_resid_pre", {"d_in": 768, "d_sae": 6144}),
            ]

            for dir_name, cfg_data in test_saes:
                sae_dir = community_format / dir_name
                sae_dir.mkdir()

                # Create cfg.json
                with open(sae_dir / "cfg.json", "w") as f:
                    json.dump(cfg_data, f)

                # Create dummy weights file
                with open(sae_dir / "sae_weights.safetensors", "wb") as f:
                    f.write(b"dummy weights")

            # Verify structure
            found_dirs = list(community_format.iterdir())
            assert len(found_dirs) == 3

            # Parse each directory
            for sae_dir in found_dirs:
                dir_name = sae_dir.name
                assert dir_name.startswith("layer_")

                # Verify cfg.json exists
                cfg_path = sae_dir / "cfg.json"
                assert cfg_path.exists()

                # Verify weights file exists
                weights_path = sae_dir / "sae_weights.safetensors"
                assert weights_path.exists()


class TestFilteringLogic:
    """Tests for filtering available SAEs by layers and hook_types."""

    def test_filter_by_layers(self):
        """Test filtering SAEs by specific layers."""
        available_saes = [
            AvailableSAEInfo(layer=0, hook_type="hook_resid_pre", path="/path/0_pre"),
            AvailableSAEInfo(layer=0, hook_type="hook_resid_post", path="/path/0_post"),
            AvailableSAEInfo(layer=1, hook_type="hook_resid_pre", path="/path/1_pre"),
            AvailableSAEInfo(layer=1, hook_type="hook_resid_post", path="/path/1_post"),
            AvailableSAEInfo(layer=2, hook_type="hook_resid_pre", path="/path/2_pre"),
        ]

        # Filter to only layer 0
        layers = [0]
        filtered = [s for s in available_saes if s.layer in layers]
        assert len(filtered) == 2
        assert all(s.layer == 0 for s in filtered)

        # Filter to layers 0 and 2
        layers = [0, 2]
        filtered = [s for s in available_saes if s.layer in layers]
        assert len(filtered) == 3

    def test_filter_by_hook_types(self):
        """Test filtering SAEs by specific hook types."""
        available_saes = [
            AvailableSAEInfo(layer=0, hook_type="hook_resid_pre", path="/path/0_pre"),
            AvailableSAEInfo(layer=0, hook_type="hook_resid_post", path="/path/0_post"),
            AvailableSAEInfo(layer=1, hook_type="hook_resid_pre", path="/path/1_pre"),
            AvailableSAEInfo(layer=1, hook_type="hook_mlp_out", path="/path/1_mlp"),
        ]

        # Filter to only hook_resid_pre
        hook_types = ["hook_resid_pre"]
        filtered = [s for s in available_saes if s.hook_type in hook_types]
        assert len(filtered) == 2
        assert all(s.hook_type == "hook_resid_pre" for s in filtered)

    def test_filter_by_both_layers_and_hook_types(self):
        """Test filtering SAEs by both layers and hook types."""
        available_saes = [
            AvailableSAEInfo(layer=0, hook_type="hook_resid_pre", path="/path/0_pre"),
            AvailableSAEInfo(layer=0, hook_type="hook_resid_post", path="/path/0_post"),
            AvailableSAEInfo(layer=1, hook_type="hook_resid_pre", path="/path/1_pre"),
            AvailableSAEInfo(layer=1, hook_type="hook_resid_post", path="/path/1_post"),
        ]

        layers = [0]
        hook_types = ["hook_resid_pre"]

        filtered = [
            s for s in available_saes
            if s.layer in layers and s.hook_type in hook_types
        ]

        assert len(filtered) == 1
        assert filtered[0].layer == 0
        assert filtered[0].hook_type == "hook_resid_pre"

    def test_no_filter_returns_all(self):
        """Test that None filters return all SAEs."""
        available_saes = [
            AvailableSAEInfo(layer=0, hook_type="hook_resid_pre", path="/path/0_pre"),
            AvailableSAEInfo(layer=1, hook_type="hook_resid_post", path="/path/1_post"),
        ]

        layers = None
        hook_types = None

        # Filter logic: if filter is None, include all
        filtered = [
            s for s in available_saes
            if (layers is None or s.layer in layers)
            and (hook_types is None or s.hook_type in hook_types)
        ]

        assert len(filtered) == 2


class TestNameGeneration:
    """Tests for generating SAE names with layer/hook suffixes."""

    def test_generate_name_with_layer_and_hook(self):
        """Test generating SAE name with layer and hook type suffix."""
        base_name = "SAE from GPT-2"
        layer = 5
        hook_type = "hook_resid_pre"

        # Expected format: "SAE from GPT-2 (L5-resid_pre)"
        generated_name = f"{base_name} (L{layer}-{hook_type.replace('hook_', '')})"

        assert generated_name == "SAE from GPT-2 (L5-resid_pre)"

    def test_generate_name_without_base_name(self):
        """Test generating SAE name when no base name provided."""
        training_id = "abc123"
        layer = 0
        hook_type = "hook_mlp_out"

        # When no base name, use training ID
        generated_name = f"SAE {training_id[:8]} (L{layer}-{hook_type.replace('hook_', '')})"

        assert generated_name == "SAE abc123 (L0-mlp_out)"

    def test_hook_type_formatting(self):
        """Test formatting hook types for display."""
        hook_types = [
            ("hook_resid_pre", "resid_pre"),
            ("hook_resid_post", "resid_post"),
            ("hook_mlp_out", "mlp_out"),
            ("hook_attn_out", "attn_out"),
        ]

        for full_hook, expected_display in hook_types:
            display = full_hook.replace("hook_", "")
            assert display == expected_display


class TestExternalSAEHookType:
    """Tests for hook_type field in ExternalSAE model."""

    def test_external_sae_has_hook_type_field(self):
        """Verify ExternalSAE model has hook_type column."""
        # Check the model has the hook_type attribute
        assert hasattr(ExternalSAE, 'hook_type')

    def test_hook_type_is_optional(self):
        """Verify hook_type can be None for backward compatibility."""
        # The column should be nullable for existing SAEs
        column = ExternalSAE.__table__.columns.get('hook_type')
        assert column is not None
        assert column.nullable is True
