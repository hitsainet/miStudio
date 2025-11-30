"""
Checkpoint service layer for SAE checkpoint management.

This module contains the CheckpointService class which handles
checkpoint saving, loading, and management operations.

Supports two checkpoint formats:
1. miStudio format: Internal format with optimizer states for training resumption
2. Community Standard format: Standard format for interoperability with major SAE tools
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import uuid4
from pathlib import Path
import os

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from safetensors.torch import save_file, load_file
import torch

from ..models.checkpoint import Checkpoint
from ..ml.sparse_autoencoder import SparseAutoencoder, SkipAutoencoder, Transcoder
from ..ml.community_format import (
    CommunityStandardConfig,
    save_sae_community_format,
    load_sae_community_format,
    is_community_format,
    is_mistudio_format,
    convert_mistudio_to_community_weights,
)


logger = logging.getLogger(__name__)


class CheckpointService:
    """Service class for checkpoint operations."""

    @staticmethod
    async def create_checkpoint(
        db: AsyncSession,
        training_id: str,
        step: int,
        loss: float,
        storage_path: str,
        l0_sparsity: Optional[float] = None,
        is_best: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """
        Create a new checkpoint record.

        Args:
            db: Database session
            training_id: Training job ID
            step: Training step
            loss: Loss at checkpoint
            storage_path: Path to .safetensors file
            l0_sparsity: L0 sparsity at checkpoint
            is_best: Whether this is the best checkpoint
            extra_metadata: Additional metadata

        Returns:
            Created checkpoint object
        """
        checkpoint_id = f"ckpt_{uuid4().hex[:8]}"

        # Get file size if file exists
        file_size_bytes = None
        if os.path.exists(storage_path):
            file_size_bytes = os.path.getsize(storage_path)

        db_checkpoint = Checkpoint(
            id=checkpoint_id,
            training_id=training_id,
            step=step,
            loss=loss,
            l0_sparsity=l0_sparsity,
            storage_path=storage_path,
            file_size_bytes=file_size_bytes,
            is_best=is_best,
            extra_metadata=extra_metadata or {},
        )

        db.add(db_checkpoint)
        await db.commit()
        await db.refresh(db_checkpoint)

        return db_checkpoint

    @staticmethod
    async def get_checkpoint(
        db: AsyncSession,
        checkpoint_id: str
    ) -> Optional[Checkpoint]:
        """
        Get a checkpoint by ID.

        Args:
            db: Database session
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint object or None if not found
        """
        result = await db.execute(
            select(Checkpoint).where(Checkpoint.id == checkpoint_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_checkpoints(
        db: AsyncSession,
        training_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[List[Checkpoint], int]:
        """
        List checkpoints for a training job.

        Args:
            db: Database session
            training_id: Training job ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            Tuple of (list of checkpoints, total count)
        """
        # Get total count
        from sqlalchemy import func
        count_query = select(func.count()).select_from(Checkpoint).where(
            Checkpoint.training_id == training_id
        )
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Get paginated results (ordered by step descending)
        query = (
            select(Checkpoint)
            .where(Checkpoint.training_id == training_id)
            .order_by(Checkpoint.step.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        checkpoints = list(result.scalars().all())

        return checkpoints, total

    @staticmethod
    async def get_best_checkpoint(
        db: AsyncSession,
        training_id: str
    ) -> Optional[Checkpoint]:
        """
        Get the best checkpoint for a training job.

        Args:
            db: Database session
            training_id: Training job ID

        Returns:
            Best checkpoint or None if no checkpoints exist
        """
        query = (
            select(Checkpoint)
            .where(
                and_(
                    Checkpoint.training_id == training_id,
                    Checkpoint.is_best == True
                )
            )
            .limit(1)
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_latest_checkpoint(
        db: AsyncSession,
        training_id: str
    ) -> Optional[Checkpoint]:
        """
        Get the most recent checkpoint for a training job.

        Args:
            db: Database session
            training_id: Training job ID

        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        query = (
            select(Checkpoint)
            .where(Checkpoint.training_id == training_id)
            .order_by(Checkpoint.step.desc())
            .limit(1)
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def update_best_checkpoint(
        db: AsyncSession,
        training_id: str,
        checkpoint_id: str
    ) -> Optional[Checkpoint]:
        """
        Mark a checkpoint as the best, unmarking all others.

        Args:
            db: Database session
            training_id: Training job ID
            checkpoint_id: Checkpoint ID to mark as best

        Returns:
            Updated checkpoint or None if not found
        """
        # Unmark all checkpoints as best for this training
        all_checkpoints_query = select(Checkpoint).where(
            Checkpoint.training_id == training_id
        )
        result = await db.execute(all_checkpoints_query)
        all_checkpoints = result.scalars().all()

        for ckpt in all_checkpoints:
            ckpt.is_best = False

        # Mark the specified checkpoint as best
        target_checkpoint = None
        for ckpt in all_checkpoints:
            if ckpt.id == checkpoint_id:
                ckpt.is_best = True
                target_checkpoint = ckpt
                break

        if target_checkpoint:
            await db.commit()
            await db.refresh(target_checkpoint)

        return target_checkpoint

    @staticmethod
    async def delete_checkpoint(
        db: AsyncSession,
        checkpoint_id: str,
        delete_file: bool = True
    ) -> bool:
        """
        Delete a checkpoint record and optionally its file and parent directories.

        For multi-layer checkpoints, this will:
        1. Delete the checkpoint file (e.g., checkpoint.safetensors)
        2. Delete the layer directory if empty (e.g., layer_7/)
        3. Delete the checkpoint step directory if empty (e.g., checkpoint_498000/)

        Args:
            db: Database session
            checkpoint_id: Checkpoint ID
            delete_file: Whether to delete the checkpoint file and empty directories

        Returns:
            True if deleted, False if not found

        Example structure:
            checkpoints/
            └── checkpoint_498000/         # Checkpoint step directory
                ├── layer_7/               # Layer directory
                │   └── checkpoint.safetensors  # Checkpoint file
                ├── layer_14/
                │   └── checkpoint.safetensors
                └── layer_18/
                    └── checkpoint.safetensors
        """
        db_checkpoint = await CheckpointService.get_checkpoint(db, checkpoint_id)
        if not db_checkpoint:
            return False

        # Delete file and parent directories if requested
        if delete_file and os.path.exists(db_checkpoint.storage_path):
            storage_path = Path(db_checkpoint.storage_path)

            try:
                # Step 1: Delete the checkpoint file
                storage_path.unlink()
                logger.info(f"Deleted checkpoint file: {storage_path}")

                # Step 2: Delete layer directory if empty (e.g., layer_7/)
                layer_dir = storage_path.parent
                if layer_dir.exists() and not any(layer_dir.iterdir()):
                    layer_dir.rmdir()
                    logger.info(f"Deleted empty layer directory: {layer_dir}")

                    # Step 3: Delete checkpoint step directory if empty (e.g., checkpoint_498000/)
                    checkpoint_dir = layer_dir.parent
                    if checkpoint_dir.exists() and not any(checkpoint_dir.iterdir()):
                        checkpoint_dir.rmdir()
                        logger.info(f"Deleted empty checkpoint directory: {checkpoint_dir}")

            except OSError as e:
                logger.warning(f"Error deleting checkpoint files/directories: {e}")
                # Continue with database deletion even if file deletion fails

        # Delete database record
        await db.delete(db_checkpoint)
        await db.commit()

        return True

    @staticmethod
    def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        storage_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save model and optimizer state to disk using safetensors.

        Args:
            model: PyTorch model to save
            optimizer: Optimizer to save
            step: Current training step
            storage_path: Path to save checkpoint
            extra_metadata: Additional metadata to save
        """
        # Ensure directory exists
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

        # Prepare state dict
        state_dict = {
            # Model weights
            **{f"model.{k}": v for k, v in model.state_dict().items()},
            # Optimizer state
            **{f"optimizer.{k}": v for k, v in optimizer.state_dict().items() if torch.is_tensor(v)},
        }

        # Save metadata as tensors (safetensors requirement)
        metadata = {
            "step": str(step),
            "architecture": model.__class__.__name__,
        }
        if extra_metadata:
            for k, v in extra_metadata.items():
                if isinstance(v, (int, float, str)):
                    metadata[k] = str(v)

        # Save to file
        save_file(state_dict, storage_path, metadata=metadata)

    @staticmethod
    def load_checkpoint(
        storage_path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load checkpoint from disk.

        Args:
            storage_path: Path to checkpoint file
            model: Model to load weights into (optional)
            optimizer: Optimizer to load state into (optional)
            device: Device to load tensors onto

        Returns:
            Dictionary with metadata and state
        """
        if not os.path.exists(storage_path):
            raise FileNotFoundError(f"Checkpoint not found: {storage_path}")

        # Load tensors
        state_dict = load_file(storage_path, device=device)

        # Separate model and optimizer states
        model_state = {}
        optimizer_state = {}

        for key, value in state_dict.items():
            if key.startswith("model."):
                model_state[key.replace("model.", "")] = value
            elif key.startswith("optimizer."):
                optimizer_state[key.replace("optimizer.", "")] = value

        # Load into model if provided
        if model is not None and model_state:
            model.load_state_dict(model_state)

        # Load into optimizer if provided
        if optimizer is not None and optimizer_state:
            # Note: Full optimizer state restoration is more complex
            # This loads only tensor states; full restoration needs param_groups
            pass

        return {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "storage_path": storage_path,
        }

    @staticmethod
    def save_multilayer_checkpoint(
        models: Dict[int, torch.nn.Module],
        optimizers: Dict[int, torch.optim.Optimizer],
        step: int,
        base_storage_path: str,
        training_layers: List[int],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """
        Save checkpoints for multi-layer training.

        Creates directory structure: checkpoint_{step}/layer_{idx}/checkpoint.safetensors

        Args:
            models: Dictionary of models per layer {layer_idx: model}
            optimizers: Dictionary of optimizers per layer {layer_idx: optimizer}
            step: Current training step
            base_storage_path: Base directory for checkpoints
            training_layers: List of layer indices being trained
            extra_metadata: Additional metadata to save

        Returns:
            Dictionary mapping layer_idx to checkpoint path
        """
        checkpoint_dir = Path(base_storage_path) / f"checkpoint_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_paths = {}

        for layer_idx in training_layers:
            layer_dir = checkpoint_dir / f"layer_{layer_idx}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            storage_path = str(layer_dir / "checkpoint.safetensors")

            # Add layer-specific metadata
            layer_metadata = {
                "layer_idx": layer_idx,
                **(extra_metadata or {})
            }

            CheckpointService.save_checkpoint(
                model=models[layer_idx],
                optimizer=optimizers[layer_idx],
                step=step,
                storage_path=storage_path,
                extra_metadata=layer_metadata,
            )

            checkpoint_paths[layer_idx] = storage_path

        return checkpoint_paths

    @staticmethod
    def load_multilayer_checkpoint(
        checkpoint_dir: str,
        models: Dict[int, torch.nn.Module],
        optimizers: Dict[int, torch.optim.Optimizer],
        training_layers: List[int],
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load checkpoints for multi-layer training.

        Args:
            checkpoint_dir: Directory containing layer subdirectories
            models: Dictionary of models to load into {layer_idx: model}
            optimizers: Dictionary of optimizers to load into {layer_idx: optimizer}
            training_layers: List of layer indices to restore
            device: Device to load tensors onto

        Returns:
            Dictionary with metadata and layer information
        """
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        loaded_layers = {}

        for layer_idx in training_layers:
            layer_checkpoint_path = checkpoint_path / f"layer_{layer_idx}" / "checkpoint.safetensors"

            if not layer_checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint for layer {layer_idx} not found: {layer_checkpoint_path}"
                )

            # Load checkpoint for this layer
            checkpoint_data = CheckpointService.load_checkpoint(
                storage_path=str(layer_checkpoint_path),
                model=models.get(layer_idx),
                optimizer=optimizers.get(layer_idx),
                device=device,
            )

            loaded_layers[layer_idx] = checkpoint_data

        return {
            "loaded_layers": loaded_layers,
            "checkpoint_dir": str(checkpoint_path),
            "training_layers": training_layers,
        }

    @staticmethod
    def save_community_checkpoint(
        model: torch.nn.Module,
        output_dir: str,
        model_name: str,
        layer: int,
        hyperparams: Dict[str, Any],
        training_id: Optional[str] = None,
        checkpoint_step: Optional[int] = None,
        sparsity: Optional[torch.Tensor] = None,
        tied_weights: bool = False,
    ) -> str:
        """
        Save model checkpoint in Community Standard format.

        Creates:
            {output_dir}/
            ├── cfg.json
            ├── sae_weights.safetensors
            └── sparsity.safetensors (optional)

        Args:
            model: PyTorch SAE model to save
            output_dir: Directory to save files
            model_name: Name of the target model (e.g., "gpt2-small")
            layer: Target layer index
            hyperparams: Training hyperparameters dict
            training_id: Optional training job ID for provenance
            checkpoint_step: Optional checkpoint step number
            sparsity: Optional feature sparsity tensor [d_sae]
            tied_weights: Whether model uses tied weights

        Returns:
            Path to the output directory

        Raises:
            ValueError: If model state_dict is empty or conversion fails
            RuntimeError: If saved file is too small
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Validate model before saving
        state_dict = model.state_dict()
        if not state_dict:
            raise ValueError(
                f"Model has empty state_dict! Model class: {model.__class__.__name__}"
            )
        logger.info(
            f"Saving Community Standard checkpoint: model={model.__class__.__name__}, "
            f"layer={layer}, keys={list(state_dict.keys())}"
        )

        # Create Community Standard config from training hyperparams
        config = CommunityStandardConfig.from_training_hyperparams(
            hyperparams=hyperparams,
            model_name=model_name,
            layer=layer,
            training_id=training_id,
            checkpoint_step=checkpoint_step,
        )

        # Save in Community Standard format
        save_sae_community_format(
            model=model,
            config=config,
            output_dir=output_path,
            sparsity=sparsity,
            tied_weights=tied_weights,
        )

        logger.info(f"Saved Community Standard checkpoint to {output_path}")

        return str(output_path)

    @staticmethod
    def save_multilayer_community_checkpoint(
        models: Dict[int, torch.nn.Module],
        base_output_dir: str,
        model_name: str,
        training_layers: List[int],
        hyperparams: Dict[str, Any],
        training_id: Optional[str] = None,
        checkpoint_step: Optional[int] = None,
        sparsity_per_layer: Optional[Dict[int, torch.Tensor]] = None,
        tied_weights: bool = False,
    ) -> Dict[int, str]:
        """
        Save multi-layer SAE checkpoints in Community Standard format.

        Creates directory structure:
            {base_output_dir}/
            └── layer_{idx}/
                ├── cfg.json
                ├── sae_weights.safetensors
                └── sparsity.safetensors (optional)

        Args:
            models: Dictionary of models per layer {layer_idx: model}
            base_output_dir: Base directory for checkpoints
            model_name: Name of the target model
            training_layers: List of layer indices being trained
            hyperparams: Training hyperparameters dict
            training_id: Optional training job ID
            checkpoint_step: Optional checkpoint step number
            sparsity_per_layer: Optional dict of sparsity tensors per layer
            tied_weights: Whether models use tied weights

        Returns:
            Dictionary mapping layer_idx to output directory path

        Raises:
            ValueError: If any model has empty state_dict or conversion fails
            RuntimeError: If any saved file is too small
        """
        base_path = Path(base_output_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        # Log summary of models being saved
        logger.info(
            f"Saving {len(training_layers)} SAE(s) to Community Standard format: "
            f"layers={training_layers}, output={base_path}, "
            f"hyperparams.architecture_type={hyperparams.get('architecture_type', 'unknown')}"
        )

        output_paths = {}

        for layer_idx in training_layers:
            model = models[layer_idx]
            layer_dir = base_path / f"layer_{layer_idx}"

            # Get sparsity for this layer if provided
            sparsity = None
            if sparsity_per_layer and layer_idx in sparsity_per_layer:
                sparsity = sparsity_per_layer[layer_idx]

            output_path = CheckpointService.save_community_checkpoint(
                model=model,
                output_dir=str(layer_dir),
                model_name=model_name,
                layer=layer_idx,
                hyperparams=hyperparams,
                training_id=training_id,
                checkpoint_step=checkpoint_step,
                sparsity=sparsity,
                tied_weights=tied_weights,
            )

            output_paths[layer_idx] = output_path

        logger.info(
            f"Saved Community Standard checkpoints for {len(training_layers)} layers "
            f"to {base_path}"
        )

        return output_paths
