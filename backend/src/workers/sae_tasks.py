"""
Celery tasks for SAE management operations.

This module contains background tasks for downloading SAEs from HuggingFace,
uploading SAEs to HuggingFace, and format conversion operations.
"""

import logging
import traceback
from pathlib import Path
from typing import Optional

from celery import Task

from ..core.celery_app import celery_app
from ..core.config import settings
from ..core.database import get_sync_db
from ..models.external_sae import ExternalSAE, SAEStatus
from .websocket_emitter import emit_sae_download_progress, emit_sae_upload_progress

logger = logging.getLogger(__name__)


# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF = 60  # seconds


class SAETask(Task):
    """Base task class for SAE operations with retry logic."""

    autoretry_for = (Exception,)
    retry_backoff = True
    retry_backoff_max = 300  # max 5 minutes between retries
    retry_jitter = True
    max_retries = MAX_RETRIES


def update_sae_status(
    sae_id: str,
    status: SAEStatus,
    progress: float = 0.0,
    error_message: Optional[str] = None,
    **kwargs,
):
    """Update SAE status in database."""
    with get_sync_db() as db:
        sae = db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
        if sae:
            sae.status = status.value
            sae.progress = progress
            if error_message:
                sae.error_message = error_message
            # Update any additional fields
            for key, value in kwargs.items():
                if hasattr(sae, key):
                    setattr(sae, key, value)
            db.commit()


@celery_app.task(
    bind=True,
    base=SAETask,
    name="sae.download",
    queue="sae",
    time_limit=1800,  # 30 minutes max
    soft_time_limit=1500,  # 25 minutes soft limit
)
def download_sae_task(
    self: Task,
    sae_id: str,
    repo_id: str,
    filepath: str,
    access_token: Optional[str] = None,
    revision: Optional[str] = None,
) -> dict:
    """
    Download SAE from HuggingFace and convert to miStudio format.

    This task:
    1. Downloads SAE files from HuggingFace (0-50% progress)
    2. Converts to miStudio format if needed (50-90%)
    3. Updates database with metadata (90-100%)

    Args:
        sae_id: SAE database ID
        repo_id: HuggingFace repository ID
        filepath: Path within repository
        access_token: Optional HuggingFace access token
        revision: Optional Git revision/branch

    Returns:
        Dict with status and local_path
    """
    logger.info(f"Starting SAE download: {sae_id} from {repo_id}/{filepath}")

    try:
        # Update status to downloading
        update_sae_status(sae_id, SAEStatus.DOWNLOADING, progress=0.0)
        emit_sae_download_progress(
            sae_id=sae_id,
            progress=0.0,
            status="downloading",
            message="Starting download...",
            stage="download",
        )

        # Import service (avoid circular imports)
        from ..services.huggingface_sae_service import HuggingFaceSAEService

        # Get local storage path
        local_path = HuggingFaceSAEService.get_sae_storage_path(sae_id)
        local_path.mkdir(parents=True, exist_ok=True)

        # Download from HuggingFace
        def progress_callback(downloaded: float, total: float, message: str):
            """Callback for download progress."""
            if total > 0:
                progress = min(50.0, (downloaded / total) * 50.0)
            else:
                progress = 25.0  # Unknown total, show partial progress
            emit_sae_download_progress(
                sae_id=sae_id,
                progress=progress,
                status="downloading",
                message=message,
                stage="download",
            )
            update_sae_status(sae_id, SAEStatus.DOWNLOADING, progress=progress)

        # Perform download
        download_result = HuggingFaceSAEService.download_sae(
            repo_id=repo_id,
            filepath=filepath,
            local_dir=local_path,
            token=access_token,
            revision=revision,
            progress_callback=progress_callback,
        )

        # Update progress - download complete
        emit_sae_download_progress(
            sae_id=sae_id,
            progress=50.0,
            status="converting",
            message="Download complete. Converting format...",
            stage="convert",
        )
        update_sae_status(sae_id, SAEStatus.CONVERTING, progress=50.0)

        # Get the actual downloaded path from the result
        # For Gemma Scope, this is the directory containing params.npz
        downloaded_path = Path(download_result.get("local_path", str(local_path)))
        # If it's a file (like params.npz), use its parent directory
        if downloaded_path.is_file():
            sae_path = downloaded_path.parent
        else:
            sae_path = downloaded_path

        logger.info(f"Downloaded SAE path: {sae_path}")

        # Check format and convert if needed
        from ..services.sae_converter import SAEConverterService

        format_type = SAEConverterService.detect_format(str(sae_path))

        metadata = {}
        if format_type == "community_standard":
            # Already in standard format - just extract metadata
            info = SAEConverterService.get_sae_info(str(sae_path))
            metadata = {
                "model_name": info.get("model_name"),
                "layer": info.get("layer"),
                "d_in": info.get("d_in"),
                "d_sae": info.get("d_sae"),
                "architecture": info.get("architecture"),
            }
        elif format_type == "gemma_scope":
            # Gemma Scope format (params.npz) - extract metadata
            info = SAEConverterService.get_sae_info(str(sae_path))
            metadata = {
                "model_name": info.get("model_name"),
                "layer": info.get("layer"),
                "d_in": info.get("d_in"),
                "d_sae": info.get("d_sae"),
                "architecture": info.get("architecture", "jumprelu"),
            }
        elif format_type == "unknown":
            # Try to detect and convert
            logger.warning(f"Unknown format for {sae_id}, attempting auto-detection")

        emit_sae_download_progress(
            sae_id=sae_id,
            progress=90.0,
            status="converting",
            message="Updating database...",
            stage="convert",
        )

        # Update database with final info - use sae_path which points to actual SAE files
        with get_sync_db() as db:
            sae = db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
            if sae:
                sae.status = SAEStatus.READY.value
                sae.progress = 100.0
                sae.local_path = str(sae_path)  # Use the actual SAE path, not root
                if metadata:
                    sae.model_name = metadata.get("model_name") or sae.model_name
                    sae.layer = metadata.get("layer")
                    sae.d_model = metadata.get("d_in")
                    sae.n_features = metadata.get("d_sae")
                    sae.architecture = metadata.get("architecture")
                db.commit()

        # Emit completion
        emit_sae_download_progress(
            sae_id=sae_id,
            progress=100.0,
            status="ready",
            message="SAE downloaded and ready",
            stage="complete",
        )

        logger.info(f"SAE download complete: {sae_id}")
        return {
            "status": "success",
            "sae_id": sae_id,
            "local_path": str(local_path),
        }

    except Exception as e:
        error_msg = f"SAE download failed: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

        # Update database
        update_sae_status(sae_id, SAEStatus.FAILED, progress=0.0, error_message=error_msg)

        # Emit failure
        emit_sae_download_progress(
            sae_id=sae_id,
            progress=0.0,
            status="failed",
            message=error_msg,
        )

        # Re-raise for Celery retry
        raise


@celery_app.task(
    bind=True,
    base=SAETask,
    name="sae.upload",
    queue="sae",
    time_limit=1800,  # 30 minutes max
    soft_time_limit=1500,  # 25 minutes soft limit
)
def upload_sae_task(
    self: Task,
    sae_id: str,
    repo_id: str,
    access_token: str,
    private: bool = False,
    commit_message: Optional[str] = None,
) -> dict:
    """
    Upload SAE to HuggingFace in Community Standard format.

    This task:
    1. Converts SAE to Community Standard format if needed (0-50% progress)
    2. Uploads to HuggingFace (50-100%)

    Args:
        sae_id: SAE database ID
        repo_id: Target HuggingFace repository ID
        access_token: HuggingFace access token
        private: Whether to create private repository
        commit_message: Optional commit message

    Returns:
        Dict with status and repo_url
    """
    logger.info(f"Starting SAE upload: {sae_id} to {repo_id}")

    try:
        # Get SAE from database
        with get_sync_db() as db:
            sae = db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
            if not sae:
                raise ValueError(f"SAE not found: {sae_id}")
            if not sae.local_path:
                raise ValueError(f"SAE has no local path: {sae_id}")

            local_path = Path(sae.local_path)
            model_name = sae.model_name
            layer = sae.layer

        if not local_path.exists():
            raise ValueError(f"SAE local path does not exist: {local_path}")

        # Emit initial progress
        emit_sae_upload_progress(
            sae_id=sae_id,
            progress=0.0,
            status="converting",
            message="Preparing SAE for upload...",
            stage="convert",
        )

        # Import services
        from ..services.sae_converter import SAEConverterService
        from ..services.huggingface_sae_service import HuggingFaceSAEService

        # Check format and convert if needed
        format_type = SAEConverterService.detect_format(str(local_path))

        upload_dir = local_path
        if format_type == "mistudio":
            # Need to convert to community standard
            emit_sae_upload_progress(
                sae_id=sae_id,
                progress=10.0,
                status="converting",
                message="Converting to Community Standard format...",
                stage="convert",
            )

            # Create temp directory for conversion
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                converted_path = SAEConverterService.mistudio_to_saelens(
                    source_path=str(local_path),
                    target_dir=temp_dir,
                    model_name=model_name or "unknown",
                    layer=layer or 0,
                )
                upload_dir = Path(converted_path)

        emit_sae_upload_progress(
            sae_id=sae_id,
            progress=50.0,
            status="uploading",
            message="Uploading to HuggingFace...",
            stage="upload",
        )

        # Upload to HuggingFace
        def progress_callback(uploaded: float, total: float, message: str):
            """Callback for upload progress."""
            if total > 0:
                progress = 50.0 + min(50.0, (uploaded / total) * 50.0)
            else:
                progress = 75.0
            emit_sae_upload_progress(
                sae_id=sae_id,
                progress=progress,
                status="uploading",
                message=message,
                stage="upload",
            )

        repo_url = HuggingFaceSAEService.upload_sae(
            local_dir=upload_dir,
            repo_id=repo_id,
            token=access_token,
            private=private,
            commit_message=commit_message or f"Upload SAE {sae_id}",
            progress_callback=progress_callback,
        )

        # Emit completion
        emit_sae_upload_progress(
            sae_id=sae_id,
            progress=100.0,
            status="completed",
            message="Upload complete!",
            stage="complete",
            repo_url=repo_url,
        )

        logger.info(f"SAE upload complete: {sae_id} -> {repo_url}")
        return {
            "status": "success",
            "sae_id": sae_id,
            "repo_url": repo_url,
        }

    except Exception as e:
        error_msg = f"SAE upload failed: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

        # Emit failure
        emit_sae_upload_progress(
            sae_id=sae_id,
            progress=0.0,
            status="failed",
            message=error_msg,
        )

        # Re-raise for Celery retry
        raise
