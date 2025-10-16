"""
Model service layer.

This module contains business logic for model management operations,
including downloading models from HuggingFace, managing quantization,
and tracking model metadata.
"""

import logging
from typing import Optional, Tuple, List
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..models.model import Model, ModelStatus, QuantizationFormat
from ..schemas.model import ModelCreate, ModelUpdate, ModelDownloadRequest

logger = logging.getLogger(__name__)


class ModelService:
    """Service class for model operations."""

    @staticmethod
    def generate_model_id() -> str:
        """
        Generate a unique model ID.

        Returns:
            Model ID in format m_{uuid_hex[:8]}
        """
        return f"m_{uuid4().hex[:8]}"

    @staticmethod
    async def initiate_model_download(
        db: AsyncSession,
        download_request: ModelDownloadRequest
    ) -> Model:
        """
        Initiate a model download from HuggingFace.

        Creates a model database record in DOWNLOADING state.
        The actual download should be handled by a background task.

        Args:
            db: Database session
            download_request: Download request with repo_id and quantization

        Returns:
            Created model in DOWNLOADING state
        """
        model_id = ModelService.generate_model_id()

        # Extract model name from repo_id (e.g., "meta-llama/Llama-2-7b-hf" -> "Llama-2-7b-hf")
        model_name = download_request.repo_id.split("/")[-1]

        # Create model record
        db_model = Model(
            id=model_id,
            name=model_name,
            repo_id=download_request.repo_id,  # Store HuggingFace repo ID
            architecture="",  # Will be filled by background task after config load
            params_count=0,  # Will be filled by background task
            quantization=download_request.quantization,
            status=ModelStatus.DOWNLOADING,
            progress=0.0,
            file_path=str(settings.models_dir / "raw" / model_id),
        )

        db.add(db_model)
        await db.commit()
        await db.refresh(db_model)

        logger.info(f"Initiated download for model {model_id} from {download_request.repo_id}")

        return db_model

    @staticmethod
    async def get_model(db: AsyncSession, model_id: str) -> Optional[Model]:
        """
        Get a model by ID.

        Args:
            db: Database session
            model_id: Model ID (string format)

        Returns:
            Model if found, None otherwise
        """
        result = await db.execute(
            select(Model).where(Model.id == model_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_model_by_name(db: AsyncSession, name: str) -> Optional[Model]:
        """
        Get a model by name.

        Args:
            db: Database session
            name: Model name

        Returns:
            Model if found, None otherwise
        """
        result = await db.execute(
            select(Model).where(Model.name == name)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_models(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        architecture: Optional[str] = None,
        quantization: Optional[QuantizationFormat] = None,
        status: Optional[ModelStatus] = None,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> Tuple[List[Model], int]:
        """
        List models with filtering, pagination, and sorting.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            search: Search query for name or repo_id
            architecture: Filter by architecture type
            quantization: Filter by quantization format
            status: Filter by status
            sort_by: Column to sort by
            order: Sort order (asc or desc)

        Returns:
            Tuple of (list of models, total count)
        """
        # Build base query
        query = select(Model)

        # Apply filters
        if search:
            search_filter = or_(
                Model.name.ilike(f"%{search}%"),
                Model.repo_id.ilike(f"%{search}%")
            )
            query = query.where(search_filter)

        if architecture:
            query = query.where(Model.architecture == architecture)

        if quantization:
            query = query.where(Model.quantization == quantization)

        if status:
            query = query.where(Model.status == status)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply sorting
        sort_column = getattr(Model, sort_by, Model.created_at)
        if order == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Apply pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await db.execute(query)
        models = result.scalars().all()

        return list(models), total

    @staticmethod
    async def update_model(
        db: AsyncSession,
        model_id: str,
        updates: ModelUpdate
    ) -> Optional[Model]:
        """
        Update a model.

        Args:
            db: Database session
            model_id: Model ID
            updates: Update data

        Returns:
            Updated model if found, None otherwise
        """
        db_model = await ModelService.get_model(db, model_id)
        if not db_model:
            return None

        # Update fields
        update_data = updates.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            setattr(db_model, field, value)

        await db.commit()
        await db.refresh(db_model)

        return db_model

    @staticmethod
    async def update_model_progress(
        db: AsyncSession,
        model_id: str,
        progress: float,
        status: Optional[ModelStatus] = None
    ) -> Optional[Model]:
        """
        Update model download/processing progress.

        Args:
            db: Database session
            model_id: Model ID
            progress: Progress percentage (0-100)
            status: Optional status update

        Returns:
            Updated model if found, None otherwise
        """
        db_model = await ModelService.get_model(db, model_id)
        if not db_model:
            return None

        db_model.progress = progress
        if status:
            db_model.status = status

        await db.commit()
        await db.refresh(db_model)

        return db_model

    @staticmethod
    async def mark_model_ready(
        db: AsyncSession,
        model_id: str,
        architecture: str,
        params_count: int,
        architecture_config: dict,
        memory_required_bytes: int,
        disk_size_bytes: int,
        file_path: str,
        quantized_path: Optional[str] = None
    ) -> Optional[Model]:
        """
        Mark a model as ready after successful download and quantization.

        Args:
            db: Database session
            model_id: Model ID
            architecture: Model architecture (e.g., "llama", "gpt2")
            params_count: Number of parameters
            architecture_config: Architecture configuration dict
            memory_required_bytes: Estimated memory requirement
            disk_size_bytes: Disk size in bytes
            file_path: Path to raw model files
            quantized_path: Path to quantized model files (if applicable)

        Returns:
            Updated model if found, None otherwise
        """
        db_model = await ModelService.get_model(db, model_id)
        if not db_model:
            return None

        db_model.status = ModelStatus.READY
        db_model.progress = 100.0
        db_model.architecture = architecture
        db_model.params_count = params_count
        db_model.architecture_config = architecture_config
        db_model.memory_required_bytes = memory_required_bytes
        db_model.disk_size_bytes = disk_size_bytes
        db_model.file_path = file_path
        db_model.quantized_path = quantized_path

        await db.commit()
        await db.refresh(db_model)

        logger.info(f"Model {model_id} marked as READY")

        return db_model

    @staticmethod
    async def mark_model_error(
        db: AsyncSession,
        model_id: str,
        error_message: str
    ) -> Optional[Model]:
        """
        Mark a model as failed with an error message.

        Args:
            db: Database session
            model_id: Model ID
            error_message: Error description

        Returns:
            Updated model if found, None otherwise
        """
        db_model = await ModelService.get_model(db, model_id)
        if not db_model:
            return None

        db_model.status = ModelStatus.ERROR
        db_model.error_message = error_message

        await db.commit()
        await db.refresh(db_model)

        logger.error(f"Model {model_id} marked as ERROR: {error_message}")

        return db_model

    @staticmethod
    async def delete_model(db: AsyncSession, model_id: str) -> Optional[dict]:
        """
        Delete a model and return file paths for cleanup.

        This will cascade delete all related activation extractions.

        Args:
            db: Database session
            model_id: Model ID

        Returns:
            Dict with deletion status and file paths if deleted, None if not found
            Format: {
                "deleted": True,
                "model_id": str,
                "file_path": Optional[str],
                "quantized_path": Optional[str],
                "deleted_extractions": int
            }
        """
        db_model = await ModelService.get_model(db, model_id)
        if not db_model:
            return None

        # Capture file paths before deletion
        file_path = db_model.file_path
        quantized_path = db_model.quantized_path

        # Delete related activation extractions first (cascade delete)
        from ..models.activation_extraction import ActivationExtraction
        deletion_result = await db.execute(
            select(ActivationExtraction).where(ActivationExtraction.model_id == model_id)
        )
        related_extractions = deletion_result.scalars().all()
        extraction_count = len(related_extractions)

        for extraction in related_extractions:
            await db.delete(extraction)

        if extraction_count > 0:
            logger.info(f"Deleted {extraction_count} activation extraction(s) for model {model_id}")

        # Now delete the model
        await db.delete(db_model)
        await db.commit()

        logger.info(f"Model {model_id} deleted from database")

        return {
            "deleted": True,
            "model_id": model_id,
            "file_path": file_path,
            "quantized_path": quantized_path,
            "deleted_extractions": extraction_count,
        }

    @staticmethod
    async def get_model_architecture_info(
        db: AsyncSession,
        model_id: str
    ) -> Optional[dict]:
        """
        Get detailed architecture information for a model.

        Args:
            db: Database session
            model_id: Model ID

        Returns:
            Architecture info dict if found, None otherwise
        """
        db_model = await ModelService.get_model(db, model_id)
        if not db_model:
            return None

        return {
            "model_id": db_model.id,
            "name": db_model.name,
            "architecture": db_model.architecture,
            "params_count": db_model.params_count,
            "quantization": db_model.quantization.value,
            "architecture_config": db_model.architecture_config,
            "memory_required_bytes": db_model.memory_required_bytes,
            "disk_size_bytes": db_model.disk_size_bytes,
        }
