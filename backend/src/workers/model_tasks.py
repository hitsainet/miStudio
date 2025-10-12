"""
Celery tasks for model management operations.

This module contains background tasks for downloading, loading, and quantizing
language models from HuggingFace.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..core.config import settings
from ..core.websocket import ws_manager
from ..ml.model_loader import (
    load_model_from_hf,
    ModelLoadError,
    OutOfMemoryError,
)
from ..models.model import Model, ModelStatus, QuantizationFormat
from ..services.model_service import ModelService

logger = logging.getLogger(__name__)

# Create synchronous database session for Celery tasks
sync_engine = create_engine(str(settings.database_url_sync))
SyncSessionLocal = sessionmaker(bind=sync_engine, autoflush=False, autocommit=False)


def send_progress_update(model_id: str, progress: float, status: str, message: str):
    """
    Send progress update via WebSocket using HTTP request.

    Args:
        model_id: Model ID
        progress: Progress percentage (0-100)
        status: Current status
        message: Status message
    """
    try:
        import requests

        # Send WebSocket update via internal API endpoint
        requests.post(
            "http://localhost:8000/api/internal/ws/emit",
            json={
                "channel": f"models/{model_id}/progress",
                "event": "progress",
                "data": {
                    "type": "model_progress",
                    "model_id": model_id,
                    "progress": progress,
                    "status": status,
                    "message": message,
                }
            },
            timeout=1.0
        )
    except Exception as e:
        logger.warning(f"Failed to send WebSocket update for model {model_id}: {e}")


@shared_task(
    name="workers.model_tasks.download_and_load_model",
    bind=True,
    max_retries=3,
    default_retry_delay=300,  # 5 minutes
)
def download_and_load_model(
    self,
    model_id: str,
    repo_id: str,
    quantization: str,
    access_token: Optional[str] = None
):
    """
    Download and load a model from HuggingFace with specified quantization.

    This task:
    1. Downloads the model from HuggingFace
    2. Loads it into memory with specified quantization
    3. Extracts architecture configuration
    4. Calculates resource requirements
    5. Saves metadata to database
    6. Sends progress updates via WebSocket

    Args:
        model_id: Model database ID
        repo_id: HuggingFace repository ID
        quantization: Quantization format string
        access_token: Optional HuggingFace access token

    Returns:
        dict with model metadata
    """
    db = SyncSessionLocal()

    try:
        logger.info(f"Starting model download: {model_id} from {repo_id}")

        # Convert quantization string to enum
        quant_format = QuantizationFormat(quantization)

        # Update status to DOWNLOADING
        model = db.query(Model).filter_by(id=model_id).first()

        if not model:
            raise ModelLoadError(f"Model {model_id} not found in database")

        model.status = ModelStatus.DOWNLOADING
        model.progress = 0.0
        db.commit()

        # Send initial progress
        send_progress_update(
            model_id=model_id,
            progress=0.0,
            status="downloading",
            message=f"Starting download from {repo_id}"
        )

        # Determine cache directory
        cache_dir = settings.models_dir / "raw" / model_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Update progress: downloading
        send_progress_update(
            model_id=model_id,
            progress=10.0,
            status="downloading",
            message="Downloading model files from HuggingFace"
        )

        # Load model from HuggingFace (this handles download + quantization)
        try:
            logger.info(f"Loading model {repo_id} with {quant_format.value} quantization")

            model_obj, tokenizer, config, metadata = load_model_from_hf(
                repo_id=repo_id,
                quant_format=quant_format,
                cache_dir=cache_dir,
                device_map="auto",
                trust_remote_code=False,
                hf_token=access_token,
                auto_fallback=True,
            )

            # Model loaded successfully
            send_progress_update(
                model_id=model_id,
                progress=70.0,
                status="loading",
                message=f"Model loaded with {metadata['quantization']} quantization"
            )

        except OutOfMemoryError as e:
            logger.error(f"Out of memory loading model {model_id}: {e}")
            model.status = ModelStatus.ERROR
            model.error_message = str(e)
            db.commit()

            send_progress_update(
                model_id=model_id,
                progress=0.0,
                status="error",
                message=f"Out of memory: {str(e)}"
            )
            raise

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            model.status = ModelStatus.ERROR
            model.error_message = f"Failed to load model: {str(e)}"
            db.commit()

            send_progress_update(
                model_id=model_id,
                progress=0.0,
                status="error",
                message=f"Download failed: {str(e)}"
            )
            raise

        # Calculate disk size
        disk_size = sum(
            f.stat().st_size
            for f in cache_dir.rglob("*")
            if f.is_file()
        )

        # Prepare quantized path if quantization was applied
        quantized_path = None
        if quant_format != QuantizationFormat.FP32:
            quantized_path = str(settings.models_dir / "quantized" / f"{model_id}_{quant_format.value}")

        # Update model in database with all metadata
        model.architecture = metadata["architecture"]
        model.params_count = metadata["params_count"]
        model.architecture_config = metadata["architecture_config"]
        model.memory_required_bytes = metadata["memory_required_bytes"]
        model.disk_size_bytes = disk_size
        model.file_path = str(cache_dir)
        model.quantized_path = quantized_path
        model.status = ModelStatus.READY
        model.progress = 100.0
        model.error_message = None
        db.commit()

        logger.info(f"Model {model_id} successfully loaded and ready")

        # Send final progress
        send_progress_update(
            model_id=model_id,
            progress=100.0,
            status="ready",
            message=f"Model ready with {metadata['params_count']:,} parameters"
        )

        return {
            "model_id": model_id,
            "repo_id": repo_id,
            "architecture": metadata["architecture"],
            "params_count": metadata["params_count"],
            "quantization": metadata["quantization"],
            "status": "ready",
        }

    except Exception as exc:
        logger.exception(f"Task failed for model {model_id}: {exc}")

        # Update database with error
        try:
            model = db.query(Model).filter_by(id=model_id).first()
            if model:
                model.status = ModelStatus.ERROR
                model.error_message = str(exc)
                db.commit()
        except Exception as db_exc:
            logger.error(f"Failed to update error state in database: {db_exc}")

        # Retry if not at max retries
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying task for model {model_id} (attempt {self.request.retries + 1})")
            raise self.retry(exc=exc)

        raise

    finally:
        db.close()


@shared_task(name="workers.model_tasks.delete_model_files")
def delete_model_files(model_id: str, file_path: Optional[str] = None, quantized_path: Optional[str] = None):
    """
    Delete model files from disk after database deletion.

    Args:
        model_id: Model ID
        file_path: Path to raw model files
        quantized_path: Path to quantized model files

    Returns:
        dict with deletion status
    """
    deleted_files = []
    errors = []

    try:
        # Delete raw model files
        if file_path and os.path.exists(file_path):
            import shutil
            shutil.rmtree(file_path)
            deleted_files.append(file_path)
            logger.info(f"Deleted raw model files: {file_path}")

        # Delete quantized model files
        if quantized_path and os.path.exists(quantized_path):
            import shutil
            shutil.rmtree(quantized_path)
            deleted_files.append(quantized_path)
            logger.info(f"Deleted quantized model files: {quantized_path}")

        return {
            "model_id": model_id,
            "deleted_files": deleted_files,
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"Failed to delete files for model {model_id}: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)

        return {
            "model_id": model_id,
            "deleted_files": deleted_files,
            "errors": errors,
        }


@shared_task(name="workers.model_tasks.update_model_progress")
def update_model_progress(model_id: str, progress: float, status: Optional[str] = None):
    """
    Update model download/loading progress in database.

    This is a lightweight task that can be called frequently during downloads.

    Args:
        model_id: Model ID
        progress: Progress percentage (0-100)
        status: Optional status update

    Returns:
        dict with update status
    """
    db = SyncSessionLocal()

    try:
        model = db.query(Model).filter_by(id=model_id).first()

        if not model:
            logger.warning(f"Model {model_id} not found for progress update")
            return {"error": "Model not found"}

        model.progress = progress
        if status:
            model.status = ModelStatus(status)

        db.commit()

        # Send WebSocket update
        send_progress_update(
            model_id=model_id,
            progress=progress,
            status=status or model.status.value,
            message=f"Progress: {progress:.1f}%"
        )

        return {"model_id": model_id, "progress": progress, "status": status}

    except Exception as e:
        logger.error(f"Failed to update progress for model {model_id}: {e}")
        return {"error": str(e)}

    finally:
        db.close()
