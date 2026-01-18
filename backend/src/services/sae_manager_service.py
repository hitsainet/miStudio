"""
SAE Manager service layer.

This module contains business logic for SAE management operations,
including listing, creating, importing from training, and deleting SAEs.
"""

import logging
import shutil
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..models.external_sae import ExternalSAE, SAESource, SAEStatus, SAEFormat
from ..models.training import Training, TrainingStatus
from ..schemas.sae import (
    SAEDownloadRequest,
    SAEImportFromTrainingRequest,
    SAEImportFromFileRequest,
    SAEResponse,
    AvailableSAEInfo,
    TrainingAvailableSAEsResponse,
    SAEImportFromTrainingResponse,
)
from .huggingface_sae_service import HuggingFaceSAEService

logger = logging.getLogger(__name__)


class SAEManagerService:
    """Service class for SAE management operations."""

    @staticmethod
    def generate_sae_id() -> str:
        """
        Generate a unique SAE ID.

        Returns:
            SAE ID in format sae_{uuid_hex[:12]}
        """
        return f"sae_{uuid4().hex[:12]}"

    @staticmethod
    async def get_sae(db: AsyncSession, sae_id: str) -> Optional[ExternalSAE]:
        """
        Get an SAE by ID.

        Args:
            db: Database session
            sae_id: SAE ID

        Returns:
            ExternalSAE if found, None otherwise
        """
        result = await db.execute(
            select(ExternalSAE).where(ExternalSAE.id == sae_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_saes(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        source: Optional[SAESource] = None,
        status: Optional[SAEStatus] = None,
        model_name: Optional[str] = None,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> Tuple[List[ExternalSAE], int]:
        """
        List SAEs with filtering, pagination, and sorting.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            search: Search query for name or description
            source: Filter by source type
            status: Filter by status
            model_name: Filter by model name
            sort_by: Column to sort by
            order: Sort order (asc or desc)

        Returns:
            Tuple of (list of SAEs, total count)
        """
        # Build base query
        query = select(ExternalSAE).where(ExternalSAE.status != SAEStatus.DELETED.value)

        # Apply filters
        if search:
            search_filter = or_(
                ExternalSAE.name.ilike(f"%{search}%"),
                ExternalSAE.description.ilike(f"%{search}%"),
                ExternalSAE.hf_repo_id.ilike(f"%{search}%")
            )
            query = query.where(search_filter)

        if source:
            query = query.where(ExternalSAE.source == source.value)

        if status:
            query = query.where(ExternalSAE.status == status.value)

        if model_name:
            query = query.where(ExternalSAE.model_name.ilike(f"%{model_name}%"))

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar_one()

        # Apply sorting
        sort_column = getattr(ExternalSAE, sort_by, ExternalSAE.created_at)
        if order.lower() == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Apply pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await db.execute(query)
        saes = list(result.scalars().all())

        return saes, total

    @staticmethod
    async def initiate_download(
        db: AsyncSession,
        request: SAEDownloadRequest
    ) -> ExternalSAE:
        """
        Initiate an SAE download from HuggingFace.

        Creates an SAE database record in PENDING state.
        The actual download should be handled by a Celery task.

        Args:
            db: Database session
            request: Download request

        Returns:
            Created ExternalSAE in PENDING state
        """
        sae_id = SAEManagerService.generate_sae_id()

        # Generate name if not provided
        name = request.name
        if not name:
            name = f"{request.repo_id.split('/')[-1]}/{request.filepath}"

        # Create local storage path
        local_path = HuggingFaceSAEService.get_sae_storage_path(sae_id)

        # Create SAE record
        db_sae = ExternalSAE(
            id=sae_id,
            name=name,
            description=request.description,
            source=SAESource.HUGGINGFACE.value,
            status=SAEStatus.PENDING.value,
            hf_repo_id=request.repo_id,
            hf_filepath=request.filepath,
            hf_revision=request.revision,
            model_name=request.model_name,
            model_id=request.model_id,  # Link to local model for steering
            format=SAEFormat.COMMUNITY_STANDARD.value,
            local_path=str(local_path),
            progress=0.0,
            sae_metadata={}
        )

        db.add(db_sae)
        await db.commit()
        await db.refresh(db_sae)

        logger.info(f"Initiated SAE download {sae_id} from {request.repo_id}/{request.filepath}")

        return db_sae

    @staticmethod
    async def get_available_saes_from_training(
        db: AsyncSession,
        training_id: str
    ) -> TrainingAvailableSAEsResponse:
        """
        Get list of available SAEs from a completed training.

        Scans the community_format directory for layer subdirectories
        and extracts layer/hook_type information. Filters out SAEs that
        have already been imported to the SAE repository.

        Directory naming conventions:
        - Single hook: layer_{idx}/
        - Multi-hook: layer_{idx}_{hook_type}/

        Args:
            db: Database session
            training_id: Training job ID

        Returns:
            TrainingAvailableSAEsResponse with list of available SAEs (excluding already imported)
        """
        # Get the training job
        training_result = await db.execute(
            select(Training).where(Training.id == training_id)
        )
        training = training_result.scalar_one_or_none()

        if not training:
            raise ValueError(f"Training job not found: {training_id}")

        if training.status != TrainingStatus.COMPLETED.value:
            raise ValueError(f"Training job is not completed: {training.status}")

        # Get already-imported SAEs for this training
        # These should be filtered out from the available list
        imported_result = await db.execute(
            select(ExternalSAE.layer, ExternalSAE.hook_type)
            .where(ExternalSAE.training_id == training_id)
            .where(ExternalSAE.status != SAEStatus.DELETED.value)
        )
        imported_saes = set()
        for row in imported_result:
            # Create a key for already-imported SAEs
            imported_saes.add((row.layer, row.hook_type))

        # Check for Community Standard format
        training_base_dir = settings.data_dir / "trainings" / training_id
        community_format_dir = training_base_dir / "community_format"

        available_saes: List[AvailableSAEInfo] = []

        if community_format_dir.exists():
            # Scan for layer directories
            for item in community_format_dir.iterdir():
                if not item.is_dir():
                    continue

                dir_name = item.name

                # Parse directory name: layer_{idx} or layer_{idx}_{hook_type}
                if not dir_name.startswith("layer_"):
                    continue

                parts = dir_name.split("_", 2)  # ["layer", "{idx}", "{hook_type}"] or ["layer", "{idx}"]

                if len(parts) < 2:
                    continue

                try:
                    layer_idx = int(parts[1])
                except ValueError:
                    continue

                # Determine hook_type from directory name or default
                # For single-hook trainings (layer_13), use "resid_pre" as default
                hook_type = parts[2] if len(parts) > 2 else "resid_pre"

                # Skip if this SAE has already been imported
                if (layer_idx, hook_type) in imported_saes:
                    logger.debug(f"Skipping already-imported SAE: layer {layer_idx}, hook_type {hook_type}")
                    continue

                # Calculate size
                total_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())

                available_saes.append(AvailableSAEInfo(
                    layer=layer_idx,
                    hook_type=hook_type,
                    path=str(item.relative_to(training_base_dir)),
                    size_bytes=total_size
                ))

        # Sort by layer, then hook_type
        available_saes.sort(key=lambda x: (x.layer, x.hook_type))

        return TrainingAvailableSAEsResponse(
            training_id=training_id,
            available_saes=available_saes,
            total_count=len(available_saes)
        )

    @staticmethod
    async def _import_single_sae(
        db: AsyncSession,
        training: Training,
        source_dir: Path,
        layer: int,
        hook_type: str,
        name_prefix: Optional[str],
        description: Optional[str],
    ) -> ExternalSAE:
        """
        Import a single SAE from a source directory.

        Args:
            db: Database session
            training: Training record
            source_dir: Directory containing SAE files
            layer: Layer index
            hook_type: Hook type string
            name_prefix: Optional name prefix
            description: Optional description

        Returns:
            Created ExternalSAE record
        """
        hyperparams = training.hyperparameters or {}

        sae_id = SAEManagerService.generate_sae_id()

        # Generate name with layer/hook suffix
        if name_prefix:
            name = f"{name_prefix} (L{layer}-{hook_type})"
        else:
            name = f"SAE from {training.id} (L{layer}-{hook_type})"

        # Copy checkpoint to SAE storage
        local_path = HuggingFaceSAEService.get_sae_storage_path(sae_id)
        local_path.mkdir(parents=True, exist_ok=True)

        # Copy the files
        for item in source_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, local_path / item.name)
            elif item.is_dir():
                shutil.copytree(item, local_path / item.name)

        # Calculate total size
        total_size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())

        # Create SAE record
        db_sae = ExternalSAE(
            id=sae_id,
            name=name,
            description=description,
            source=SAESource.TRAINED.value,
            status=SAEStatus.READY.value,
            training_id=training.id,
            model_id=training.model_id,
            model_name=None,
            layer=layer,
            hook_type=hook_type,
            n_features=hyperparams.get("latent_dim"),
            d_model=hyperparams.get("hidden_dim"),
            architecture=hyperparams.get("architecture_type", "standard"),
            format=SAEFormat.COMMUNITY_STANDARD.value,
            local_path=str(local_path),
            file_size_bytes=total_size,
            progress=100.0,
            sae_metadata={
                "training_hyperparameters": hyperparams,
                "training_status": training.status,
                "final_loss": training.current_loss,
                "final_l0_sparsity": training.current_l0_sparsity,
                "format_source": "community_format",
            },
            downloaded_at=datetime.utcnow()
        )

        db.add(db_sae)

        logger.info(f"Imported SAE {sae_id} from training {training.id} (L{layer}-{hook_type})")

        return db_sae

    @staticmethod
    async def import_from_training(
        db: AsyncSession,
        request: SAEImportFromTrainingRequest
    ) -> SAEImportFromTrainingResponse:
        """
        Import SAE(s) from a completed training job.

        Supports importing multiple SAEs from multi-layer/multi-hook trainings.
        Uses Community Standard format if available.

        Args:
            db: Database session
            request: Import request with training_id and optional filters

        Returns:
            SAEImportFromTrainingResponse with list of created SAEs
        """
        # Get the training job
        training_result = await db.execute(
            select(Training).where(Training.id == request.training_id)
        )
        training = training_result.scalar_one_or_none()

        if not training:
            raise ValueError(f"Training job not found: {request.training_id}")

        if training.status != TrainingStatus.COMPLETED.value:
            raise ValueError(f"Training job is not completed: {training.status}")

        # Get available SAEs
        available_response = await SAEManagerService.get_available_saes_from_training(
            db, request.training_id
        )
        available_saes = available_response.available_saes

        if not available_saes:
            raise ValueError("No SAEs found in training checkpoint")

        # Filter if not importing all
        if not request.import_all:
            filtered = []
            for sae_info in available_saes:
                # Check layer filter
                if request.layers and sae_info.layer not in request.layers:
                    continue
                # Check hook_type filter
                if request.hook_types and sae_info.hook_type not in request.hook_types:
                    continue
                filtered.append(sae_info)
            available_saes = filtered

        if not available_saes:
            raise ValueError("No SAEs match the specified filters")

        # Import each SAE
        training_base_dir = settings.data_dir / "trainings" / request.training_id
        created_saes: List[ExternalSAE] = []

        for sae_info in available_saes:
            source_dir = training_base_dir / sae_info.path

            db_sae = await SAEManagerService._import_single_sae(
                db=db,
                training=training,
                source_dir=source_dir,
                layer=sae_info.layer,
                hook_type=sae_info.hook_type,
                name_prefix=request.name,
                description=request.description,
            )
            created_saes.append(db_sae)

        # Commit all at once
        await db.commit()

        # Refresh all SAEs
        for sae in created_saes:
            await db.refresh(sae)

        # Build response
        sae_responses = [SAEResponse.model_validate(sae) for sae in created_saes]

        logger.info(
            f"Imported {len(created_saes)} SAE(s) from training {request.training_id}"
        )

        return SAEImportFromTrainingResponse(
            imported_count=len(created_saes),
            sae_ids=[sae.id for sae in created_saes],
            saes=sae_responses,
            training_id=request.training_id,
            message=f"Successfully imported {len(created_saes)} SAE(s)"
        )

    @staticmethod
    async def import_from_file(
        db: AsyncSession,
        request: SAEImportFromFileRequest
    ) -> ExternalSAE:
        """
        Import an SAE from a local file.

        Args:
            db: Database session
            request: Import request with file path

        Returns:
            Created ExternalSAE
        """
        source_path = settings.resolve_data_path(request.file_path)

        if not source_path.exists():
            raise ValueError(f"File not found: {request.file_path}")

        sae_id = SAEManagerService.generate_sae_id()

        # Copy to SAE storage
        local_path = HuggingFaceSAEService.get_sae_storage_path(sae_id)
        local_path.mkdir(parents=True, exist_ok=True)

        if source_path.is_file():
            shutil.copy2(source_path, local_path / source_path.name)
            total_size = source_path.stat().st_size
        else:
            shutil.copytree(source_path, local_path, dirs_exist_ok=True)
            total_size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())

        # Create SAE record
        db_sae = ExternalSAE(
            id=sae_id,
            name=request.name,
            description=request.description,
            source=SAESource.LOCAL.value,
            status=SAEStatus.READY.value,
            model_name=request.model_name,
            layer=request.layer,
            format=request.format.value,
            local_path=str(local_path),
            file_size_bytes=total_size,
            progress=100.0,
            sae_metadata={
                "original_path": str(source_path)
            },
            downloaded_at=datetime.utcnow()
        )

        db.add(db_sae)
        await db.commit()
        await db.refresh(db_sae)

        logger.info(f"Imported SAE {sae_id} from {request.file_path}")

        return db_sae

    @staticmethod
    async def update_download_progress(
        db: AsyncSession,
        sae_id: str,
        progress: float,
        status: Optional[SAEStatus] = None,
        error_message: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> Optional[ExternalSAE]:
        """
        Update SAE download progress.

        Args:
            db: Database session
            sae_id: SAE ID
            progress: Download progress (0-100)
            status: New status (optional)
            error_message: Error message if failed
            metadata_updates: Additional metadata to merge

        Returns:
            Updated ExternalSAE if found
        """
        sae = await SAEManagerService.get_sae(db, sae_id)
        if not sae:
            return None

        sae.progress = progress

        if status:
            sae.status = status.value

        if error_message:
            sae.error_message = error_message

        if metadata_updates:
            current_metadata = sae.sae_metadata or {}
            current_metadata.update(metadata_updates)
            sae.sae_metadata = current_metadata

        if status == SAEStatus.READY:
            sae.downloaded_at = datetime.utcnow()

        await db.commit()
        await db.refresh(sae)

        return sae

    @staticmethod
    async def update_sae_info(
        db: AsyncSession,
        sae_id: str,
        layer: Optional[int] = None,
        n_features: Optional[int] = None,
        d_model: Optional[int] = None,
        architecture: Optional[str] = None,
        model_name: Optional[str] = None,
        file_size_bytes: Optional[int] = None
    ) -> Optional[ExternalSAE]:
        """
        Update SAE architecture info after download/conversion.

        Args:
            db: Database session
            sae_id: SAE ID
            layer: Target layer
            n_features: Number of features
            d_model: Model dimension
            architecture: SAE architecture type
            model_name: Target model name
            file_size_bytes: File size

        Returns:
            Updated ExternalSAE if found
        """
        sae = await SAEManagerService.get_sae(db, sae_id)
        if not sae:
            return None

        if layer is not None:
            sae.layer = layer
        if n_features is not None:
            sae.n_features = n_features
        if d_model is not None:
            sae.d_model = d_model
        if architecture is not None:
            sae.architecture = architecture
        if model_name is not None:
            sae.model_name = model_name
        if file_size_bytes is not None:
            sae.file_size_bytes = file_size_bytes

        await db.commit()
        await db.refresh(sae)

        return sae

    @staticmethod
    async def delete_sae(
        db: AsyncSession,
        sae_id: str,
        delete_files: bool = True
    ) -> bool:
        """
        Delete an SAE.

        Args:
            db: Database session
            sae_id: SAE ID
            delete_files: Whether to delete local files

        Returns:
            True if deleted, False if not found
        """
        sae = await SAEManagerService.get_sae(db, sae_id)
        if not sae:
            return False

        # Delete local files if requested
        if delete_files and sae.local_path:
            local_path = settings.resolve_data_path(sae.local_path)
            if local_path.exists():
                try:
                    if local_path.is_dir():
                        shutil.rmtree(local_path)
                    else:
                        local_path.unlink()
                    logger.info(f"Deleted SAE files at {local_path}")
                except Exception as e:
                    logger.warning(f"Error deleting SAE files: {e}")

        # Soft delete by setting status
        sae.status = SAEStatus.DELETED.value
        await db.commit()

        logger.info(f"Deleted SAE {sae_id}")

        return True

    @staticmethod
    async def delete_saes_batch(
        db: AsyncSession,
        sae_ids: List[str],
        delete_files: bool = True
    ) -> Dict[str, Any]:
        """
        Delete multiple SAEs.

        Args:
            db: Database session
            sae_ids: List of SAE IDs to delete
            delete_files: Whether to delete local files

        Returns:
            Dict with deleted_count, failed_count, deleted_ids, failed_ids, errors
        """
        deleted_ids = []
        failed_ids = []
        errors = {}

        for sae_id in sae_ids:
            try:
                success = await SAEManagerService.delete_sae(db, sae_id, delete_files)
                if success:
                    deleted_ids.append(sae_id)
                else:
                    failed_ids.append(sae_id)
                    errors[sae_id] = "SAE not found"
            except Exception as e:
                failed_ids.append(sae_id)
                errors[sae_id] = str(e)

        return {
            "deleted_count": len(deleted_ids),
            "failed_count": len(failed_ids),
            "deleted_ids": deleted_ids,
            "failed_ids": failed_ids,
            "errors": errors
        }

    @staticmethod
    async def get_ready_saes_for_steering(
        db: AsyncSession,
        model_name: Optional[str] = None
    ) -> List[ExternalSAE]:
        """
        Get SAEs that are ready for use in steering.

        Args:
            db: Database session
            model_name: Optional filter by model name

        Returns:
            List of ready SAEs
        """
        query = select(ExternalSAE).where(
            ExternalSAE.status == SAEStatus.READY.value
        )

        if model_name:
            query = query.where(ExternalSAE.model_name.ilike(f"%{model_name}%"))

        query = query.order_by(ExternalSAE.created_at.desc())

        result = await db.execute(query)
        return list(result.scalars().all())
