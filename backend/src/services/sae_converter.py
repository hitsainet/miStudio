"""
SAE Converter Service.

Provides conversion between SAELens/Community Standard format and miStudio format.
Wraps the lower-level community_format.py utilities with a service-oriented interface.

Formats supported:
- Community Standard (SAELens, Neuronpedia, HuggingFace)
- miStudio native checkpoint format
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from sqlalchemy.ext.asyncio import AsyncSession

from ..ml.community_format import (
    CommunityStandardConfig,
    load_sae_community_format,
    save_sae_community_format,
    load_sae_auto_detect,
    is_community_format,
    is_mistudio_format,
    migrate_mistudio_to_community,
    convert_community_to_mistudio_weights,
)

logger = logging.getLogger(__name__)


# Model dimension to name mapping for inference
MODEL_DIMENSION_MAP: Dict[int, str] = {
    768: "gpt2",
    1024: "gpt2-medium",
    1280: "gpt2-large",
    1600: "gpt2-xl",
    2048: "google/gemma-2b",
    2304: "google/gemma-2-2b",
    3072: "google/gemma-7b",
    3584: "google/gemma-2-9b",
    4096: "meta-llama/Llama-2-7b",
    5120: "meta-llama/Llama-2-13b",
    8192: "meta-llama/Llama-2-70b",
    4544: "EleutherAI/pythia-6.9b",
    6144: "EleutherAI/pythia-12b",
}


class SAEConverterService:
    """
    Service for converting SAEs between different formats.

    Handles:
    - SAELens/Community Standard → miStudio
    - miStudio → SAELens/Community Standard
    - Format auto-detection
    - Model inference from dimensions
    """

    @staticmethod
    def saelens_to_mistudio(
        source_dir: str,
        target_dir: str,
        device: str = "cpu",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Convert SAELens/Community Standard format to miStudio format.

        Args:
            source_dir: Directory containing cfg.json and sae_weights.safetensors
            target_dir: Directory to save miStudio checkpoint
            device: Device to load tensors onto

        Returns:
            Tuple of (checkpoint_path, metadata)
        """
        source_path = Path(source_dir)
        target_path = Path(target_dir)

        # Validate source
        if not is_community_format(source_path):
            raise ValueError(
                f"Source directory {source_dir} is not in Community Standard format. "
                "Expected cfg.json and sae_weights.safetensors"
            )

        # Load Community Standard format
        state_dict, config, sparsity = load_sae_community_format(source_path, device)

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        # Save in miStudio checkpoint format
        from safetensors.torch import save_file

        # Add "model." prefix for miStudio format
        mistudio_weights = {f"model.{k}": v for k, v in state_dict.items()}

        checkpoint_path = target_path / "checkpoint.safetensors"
        save_file(mistudio_weights, str(checkpoint_path))

        # Create metadata
        metadata = {
            "format": "mistudio",
            "original_format": "community_standard",
            "model_name": config.model_name if config else None,
            "layer": config.hook_point_layer if config else None,
            "d_in": config.d_in if config else None,
            "d_sae": config.d_sae if config else None,
            "architecture": config.architecture if config else "standard",
            "activation_fn": config.activation_fn_str if config else "relu",
        }

        # Save metadata
        import json
        metadata_path = target_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Converted SAELens to miStudio format: {checkpoint_path}")
        return str(checkpoint_path), metadata

    @staticmethod
    def mistudio_to_saelens(
        source_path: str,
        target_dir: str,
        model_name: str,
        layer: int,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Convert miStudio checkpoint to SAELens/Community Standard format.

        Args:
            source_path: Path to miStudio checkpoint (directory or file)
            target_dir: Directory to save Community Standard format
            model_name: Name of the target model
            layer: Target layer index
            hyperparams: Optional training hyperparameters

        Returns:
            Path to output directory
        """
        return migrate_mistudio_to_community(
            source_path=Path(source_path),
            output_dir=Path(target_dir),
            model_name=model_name,
            layer=layer,
            hyperparams=hyperparams,
        )

    @staticmethod
    def infer_model_from_dimensions(hidden_dim: int) -> Optional[str]:
        """
        Infer model name from hidden dimension.

        Args:
            hidden_dim: Model hidden dimension (d_model)

        Returns:
            Model name/ID or None if not recognized
        """
        return MODEL_DIMENSION_MAP.get(hidden_dim)

    @staticmethod
    def detect_format(path: str) -> str:
        """
        Detect the format of an SAE at the given path.

        Args:
            path: Path to SAE directory or file

        Returns:
            Format string: "community_standard", "mistudio", or "unknown"
        """
        p = Path(path)

        if is_community_format(p):
            return "community_standard"
        elif is_mistudio_format(p):
            return "mistudio"
        else:
            return "unknown"

    @staticmethod
    def load_auto(
        path: str,
        device: str = "cpu",
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]], str]:
        """
        Auto-detect and load SAE from any supported format.

        Args:
            path: Path to SAE directory or file
            device: Device to load tensors onto

        Returns:
            Tuple of (state_dict in miStudio format, metadata dict, format_type)
        """
        state_dict, config, format_type = load_sae_auto_detect(Path(path), device)

        # Convert config to metadata dict
        metadata = None
        if config:
            metadata = {
                "model_name": config.model_name,
                "hook_point": config.hook_point,
                "layer": config.hook_point_layer,
                "d_in": config.d_in,
                "d_sae": config.d_sae,
                "architecture": config.architecture,
                "activation_fn": config.activation_fn_str,
                "l1_coefficient": config.l1_coefficient,
            }

        return state_dict, metadata, format_type

    @staticmethod
    def get_sae_info(path: str) -> Dict[str, Any]:
        """
        Get information about an SAE without loading weights.

        Args:
            path: Path to SAE directory

        Returns:
            Dictionary with SAE information
        """
        p = Path(path)
        format_type = SAEConverterService.detect_format(path)

        info = {
            "path": str(p),
            "format": format_type,
        }

        if format_type == "community_standard":
            # Load config only
            import json
            config_path = p / "cfg.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                info.update({
                    "model_name": config.get("model_name"),
                    "hook_point": config.get("hook_point"),
                    "layer": config.get("hook_point_layer"),
                    "d_in": config.get("d_in"),
                    "d_sae": config.get("d_sae"),
                    "architecture": config.get("architecture"),
                })

        elif format_type == "mistudio":
            # Check for metadata.json
            metadata_path = p / "metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                info.update(metadata)
            else:
                # Try to infer from weights
                from safetensors.torch import safe_open
                checkpoint_path = p / "checkpoint.safetensors"
                if not checkpoint_path.exists():
                    layer_dirs = list(p.glob("layer_*"))
                    if layer_dirs:
                        checkpoint_path = layer_dirs[0] / "checkpoint.safetensors"

                if checkpoint_path.exists():
                    with safe_open(str(checkpoint_path), framework="pt") as f:
                        for key in f.keys():
                            if "encoder.weight" in key:
                                shape = f.get_slice(key).get_shape()
                                info["d_sae"] = shape[0]
                                info["d_in"] = shape[1]
                                info["inferred_model"] = SAEConverterService.infer_model_from_dimensions(shape[1])
                                break

        return info


# Singleton-style function to get service
_converter_service: Optional[SAEConverterService] = None


def get_sae_converter() -> SAEConverterService:
    """Get SAE converter service instance."""
    global _converter_service
    if _converter_service is None:
        _converter_service = SAEConverterService()
    return _converter_service
