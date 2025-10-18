"""
Checkpoint service layer for SAE checkpoint management.

This module contains the CheckpointService class which handles
checkpoint saving, loading, and management operations.
"""

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
        Delete a checkpoint record and optionally its file.

        Args:
            db: Database session
            checkpoint_id: Checkpoint ID
            delete_file: Whether to delete the checkpoint file

        Returns:
            True if deleted, False if not found
        """
        db_checkpoint = await CheckpointService.get_checkpoint(db, checkpoint_id)
        if not db_checkpoint:
            return False

        # Delete file if requested and exists
        if delete_file and os.path.exists(db_checkpoint.storage_path):
            os.remove(db_checkpoint.storage_path)

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
