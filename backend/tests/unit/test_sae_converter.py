"""
Unit tests for SAE Converter Service.

Tests format detection, conversion between SAELens/Community Standard and miStudio formats,
and model inference from dimensions.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
from safetensors.torch import save_file, load_file

from src.services.sae_converter import SAEConverterService, MODEL_DIMENSION_MAP


class TestSAEConverterService:
    """Tests for SAEConverterService."""

    def test_infer_model_from_dimensions_known(self):
        """Test model inference for known dimensions."""
        assert SAEConverterService.infer_model_from_dimensions(768) == "gpt2"
        assert SAEConverterService.infer_model_from_dimensions(2048) == "google/gemma-2b"
        assert SAEConverterService.infer_model_from_dimensions(4096) == "meta-llama/Llama-2-7b"

    def test_infer_model_from_dimensions_unknown(self):
        """Test model inference returns None for unknown dimensions."""
        assert SAEConverterService.infer_model_from_dimensions(999) is None
        assert SAEConverterService.infer_model_from_dimensions(0) is None

    def test_detect_format_community_standard(self):
        """Test format detection for Community Standard format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create Community Standard format files
            config = {
                "model_name": "gpt2",
                "hook_point": "blocks.6.hook_resid_post",
                "hook_point_layer": 6,
                "d_in": 768,
                "d_sae": 16384,
            }
            with open(path / "cfg.json", "w") as f:
                json.dump(config, f)

            # Create dummy weights
            weights = {
                "W_enc": torch.randn(768, 16384),
                "b_enc": torch.randn(16384),
                "W_dec": torch.randn(16384, 768),
                "b_dec": torch.randn(768),
            }
            save_file(weights, str(path / "sae_weights.safetensors"))

            assert SAEConverterService.detect_format(str(path)) == "community_standard"

    def test_detect_format_mistudio(self):
        """Test format detection for miStudio format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create miStudio format checkpoint with model.* prefix
            weights = {
                "model.encoder.weight": torch.randn(16384, 768),
                "model.encoder.bias": torch.randn(16384),
                "model.decoder.weight": torch.randn(768, 16384),
                "model.decoder_bias": torch.randn(768),
            }
            save_file(weights, str(path / "checkpoint.safetensors"))

            assert SAEConverterService.detect_format(str(path)) == "mistudio"

    def test_detect_format_unknown(self):
        """Test format detection returns unknown for invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            # Empty directory
            assert SAEConverterService.detect_format(str(path)) == "unknown"

    def test_get_sae_info_community_standard(self):
        """Test getting SAE info from Community Standard format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create Community Standard format
            config = {
                "model_name": "gpt2",
                "hook_point": "blocks.6.hook_resid_post",
                "hook_point_layer": 6,
                "d_in": 768,
                "d_sae": 16384,
                "architecture": "standard",
            }
            with open(path / "cfg.json", "w") as f:
                json.dump(config, f)

            weights = {
                "W_enc": torch.randn(768, 16384),
                "b_enc": torch.randn(16384),
                "W_dec": torch.randn(16384, 768),
                "b_dec": torch.randn(768),
            }
            save_file(weights, str(path / "sae_weights.safetensors"))

            info = SAEConverterService.get_sae_info(str(path))

            assert info["format"] == "community_standard"
            assert info["model_name"] == "gpt2"
            assert info["layer"] == 6
            assert info["d_in"] == 768
            assert info["d_sae"] == 16384
            assert info["architecture"] == "standard"

    def test_saelens_to_mistudio_conversion(self):
        """Test conversion from Community Standard to miStudio format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            target_path = Path(tmpdir) / "target"
            source_path.mkdir()

            # Create Community Standard format
            d_in, d_sae = 768, 16384
            config = {
                "model_name": "gpt2",
                "hook_point": "blocks.6.hook_resid_post",
                "hook_point_layer": 6,
                "d_in": d_in,
                "d_sae": d_sae,
                "architecture": "standard",
            }
            with open(source_path / "cfg.json", "w") as f:
                json.dump(config, f)

            # Community Standard weights: W_enc [d_in, d_sae], W_dec [d_sae, d_in]
            original_w_enc = torch.randn(d_in, d_sae)
            original_w_dec = torch.randn(d_sae, d_in)
            weights = {
                "W_enc": original_w_enc,
                "b_enc": torch.randn(d_sae),
                "W_dec": original_w_dec,
                "b_dec": torch.randn(d_in),
            }
            save_file(weights, str(source_path / "sae_weights.safetensors"))

            # Convert
            checkpoint_path, metadata = SAEConverterService.saelens_to_mistudio(
                str(source_path), str(target_path)
            )

            # Verify output
            assert Path(checkpoint_path).exists()
            assert metadata["format"] == "mistudio"
            assert metadata["model_name"] == "gpt2"
            assert metadata["layer"] == 6

            # Verify weights are properly transposed for miStudio format
            converted_weights = load_file(checkpoint_path)

            # miStudio encoder.weight should be [d_sae, d_in] (transposed from W_enc)
            assert "model.encoder.weight" in converted_weights
            assert converted_weights["model.encoder.weight"].shape == (d_sae, d_in)

    def test_mistudio_to_saelens_conversion(self):
        """Test conversion from miStudio to Community Standard format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            target_path = Path(tmpdir) / "target"
            source_path.mkdir()

            # Create miStudio format checkpoint
            d_in, d_sae = 768, 16384
            # miStudio format: encoder.weight [d_sae, d_in], decoder.weight [d_in, d_sae]
            weights = {
                "model.encoder.weight": torch.randn(d_sae, d_in),
                "model.encoder.bias": torch.randn(d_sae),
                "model.decoder.weight": torch.randn(d_in, d_sae),
                "model.decoder_bias": torch.randn(d_in),
            }
            save_file(weights, str(source_path / "checkpoint.safetensors"))

            # Convert
            output_dir = SAEConverterService.mistudio_to_saelens(
                str(source_path),
                str(target_path),
                model_name="gpt2",
                layer=6,
            )

            # Verify output
            assert Path(output_dir).exists()
            assert (Path(output_dir) / "cfg.json").exists()
            assert (Path(output_dir) / "sae_weights.safetensors").exists()

            # Verify config
            with open(Path(output_dir) / "cfg.json") as f:
                config = json.load(f)
            # Service normalizes model names (gpt2 -> gpt2-small)
            assert config["model_name"] == "gpt2-small"
            assert config["hook_point_layer"] == 6
            assert config["d_in"] == d_in
            assert config["d_sae"] == d_sae

            # Verify weights are properly transposed for Community Standard
            converted_weights = load_file(str(Path(output_dir) / "sae_weights.safetensors"))

            # Community Standard W_enc should be [d_in, d_sae]
            assert "W_enc" in converted_weights
            assert converted_weights["W_enc"].shape == (d_in, d_sae)


class TestRateLimiter:
    """Tests for the steering endpoint rate limiter."""

    def test_rate_limiter_allows_requests_under_limit(self):
        """Test that requests under the limit are allowed."""
        from src.api.v1.endpoints.steering import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=60)
        client_id = "test_client"

        # First 5 requests should be allowed
        for i in range(5):
            assert limiter.is_allowed(client_id) is True

        # 6th request should be denied
        assert limiter.is_allowed(client_id) is False

    def test_rate_limiter_different_clients(self):
        """Test that different clients have separate limits."""
        from src.api.v1.endpoints.steering import RateLimiter

        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # Client A uses both requests
        assert limiter.is_allowed("client_a") is True
        assert limiter.is_allowed("client_a") is True
        assert limiter.is_allowed("client_a") is False

        # Client B still has their full quota
        assert limiter.is_allowed("client_b") is True
        assert limiter.is_allowed("client_b") is True
        assert limiter.is_allowed("client_b") is False

    def test_rate_limiter_time_until_allowed(self):
        """Test time until allowed calculation."""
        from src.api.v1.endpoints.steering import RateLimiter

        limiter = RateLimiter(max_requests=1, window_seconds=60)
        client_id = "test_client"

        # Before any requests, should return 0
        assert limiter.time_until_allowed(client_id) == 0

        # Use up the quota
        limiter.is_allowed(client_id)

        # Should return a positive value
        time_until = limiter.time_until_allowed(client_id)
        assert 0 < time_until <= 60
