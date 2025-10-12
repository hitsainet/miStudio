"""
Celery tasks for model management operations.

This module contains background tasks for downloading, loading, and quantizing
language models from HuggingFace, as well as extracting activations from models.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional, List

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
from ..services.activation_service import ActivationService, ActivationExtractionError
from .websocket_emitter import emit_model_progress, emit_extraction_progress

logger = logging.getLogger(__name__)

# Create synchronous database session for Celery tasks
sync_engine = create_engine(str(settings.database_url_sync))
SyncSessionLocal = sessionmaker(bind=sync_engine, autoflush=False, autocommit=False)


def get_directory_size(path: Path) -> int:
    """
    Calculate total size of all files in a directory recursively.

    Args:
        path: Directory path

    Returns:
        Total size in bytes
    """
    total_size = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
    except Exception as e:
        logger.warning(f"Error calculating directory size for {path}: {e}")
    return total_size


class DownloadProgressMonitor:
    """
    Monitor download progress by watching cache directory size growth.

    This provides approximate progress updates during HuggingFace model downloads
    by periodically checking the size of downloaded files.
    """

    def __init__(self, cache_dir: Path, model_id: str, estimated_size_gb: float = 5.0):
        """
        Initialize progress monitor.

        Args:
            cache_dir: Directory where model files are being downloaded
            model_id: Model ID for progress updates
            estimated_size_gb: Estimated total size in GB (used for progress calculation)
        """
        self.cache_dir = cache_dir
        self.model_id = model_id
        self.estimated_size_bytes = int(estimated_size_gb * 1024 * 1024 * 1024)
        self.initial_size = 0
        self.running = False
        self.thread = None

    def start(self):
        """Start monitoring in background thread."""
        self.initial_size = get_directory_size(self.cache_dir)
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"[ProgressMonitor] Started for {self.model_id}, initial size: {self.initial_size / (1024**2):.2f} MB")

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info(f"[ProgressMonitor] Stopped for {self.model_id}")

    def _monitor_loop(self):
        """Monitor loop that runs in background thread."""
        last_progress = 0
        check_interval = 3.0  # Check every 3 seconds

        while self.running:
            try:
                current_size = get_directory_size(self.cache_dir)
                downloaded_bytes = current_size - self.initial_size

                # Calculate progress (capped at 90% since we don't know exact size)
                progress = min(90, (downloaded_bytes / self.estimated_size_bytes) * 100)

                # Only send update if progress increased by at least 1%
                if progress >= last_progress + 1:
                    downloaded_mb = downloaded_bytes / (1024 * 1024)
                    estimated_mb = self.estimated_size_bytes / (1024 * 1024)

                    # Update database
                    try:
                        db = SyncSessionLocal()
                        model = db.query(Model).filter_by(id=self.model_id).first()
                        if model:
                            model.progress = progress
                            db.commit()
                        db.close()
                    except Exception as db_e:
                        logger.warning(f"[ProgressMonitor] Failed to update database: {db_e}")

                    # Send WebSocket update
                    send_progress_update(
                        model_id=self.model_id,
                        progress=progress,
                        status="downloading",
                        message=f"Downloaded {downloaded_mb:.0f} MB / ~{estimated_mb:.0f} MB"
                    )

                    last_progress = progress
                    logger.info(
                        f"[ProgressMonitor] {self.model_id}: {progress:.1f}% "
                        f"({downloaded_mb:.1f} MB downloaded)"
                    )

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"[ProgressMonitor] Error: {e}")
                time.sleep(check_interval)


def send_progress_update(model_id: str, progress: float, status: str, message: str):
    """
    Send model progress update via WebSocket.

    Wrapper function that uses the shared websocket emitter utility.

    Args:
        model_id: Model ID
        progress: Progress percentage (0-100)
        status: Current status
        message: Status message
    """
    emit_model_progress(
        model_id=model_id,
        event="progress",
        data={
            "type": "model_progress",
            "model_id": model_id,
            "progress": progress,
            "status": status,
            "message": message,
        }
    )


@shared_task(
    name="workers.model_tasks.download_and_load_model",
    bind=True,
    max_retries=3,
    default_retry_delay=300,  # 5 minutes
    queue="processing",
)
def download_and_load_model(
    self,
    model_id: str,
    repo_id: str,
    quantization: str,
    access_token: Optional[str] = None,
    trust_remote_code: bool = False
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

        # Estimate model size based on repo name (rough heuristics)
        # This helps show more accurate progress during download
        estimated_size_gb = 5.0  # Default estimate
        repo_lower = repo_id.lower()
        if "70b" in repo_lower or "72b" in repo_lower:
            estimated_size_gb = 40.0
        elif "13b" in repo_lower or "12b" in repo_lower:
            estimated_size_gb = 15.0
        elif "7b" in repo_lower or "8b" in repo_lower:
            estimated_size_gb = 10.0
        elif "3b" in repo_lower:
            estimated_size_gb = 4.0
        elif "1b" in repo_lower or "1.1b" in repo_lower:
            estimated_size_gb = 2.0
        elif "nemo" in repo_lower:
            estimated_size_gb = 12.0  # Mistral-Nemo is ~12B params

        logger.info(f"Estimated model size: {estimated_size_gb} GB")

        # Start progress monitor to track download
        progress_monitor = DownloadProgressMonitor(
            cache_dir=cache_dir,
            model_id=model_id,
            estimated_size_gb=estimated_size_gb
        )
        progress_monitor.start()

        # Load model from HuggingFace (this handles download + quantization)
        try:
            logger.info(f"Loading model {repo_id} with {quant_format.value} quantization")

            model_obj, tokenizer, config, metadata = load_model_from_hf(
                repo_id=repo_id,
                quant_format=quant_format,
                cache_dir=cache_dir,
                device_map="auto",
                trust_remote_code=trust_remote_code,
                hf_token=access_token,
                auto_fallback=True,
            )

            # Stop progress monitor
            progress_monitor.stop()

            # Model loaded successfully
            send_progress_update(
                model_id=model_id,
                progress=95.0,
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


@shared_task(
    name="workers.model_tasks.extract_activations",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    queue="extraction",
)
def extract_activations(
    self,
    model_id: str,
    dataset_id: str,
    layer_indices: List[int],
    hook_types: List[str],
    max_samples: int,
    batch_size: int = 8,
    extraction_id: Optional[str] = None,
):
    """
    Extract activations from a model using a tokenized dataset.

    This task:
    1. Loads the model and tokenized dataset from disk
    2. Registers forward hooks on specified layers
    3. Runs batched inference to capture activations
    4. Saves activations as .npy files with metadata
    5. Calculates statistics (mean, max, std, sparsity)
    6. Sends progress updates via WebSocket

    Args:
        model_id: Model database ID
        dataset_id: Dataset database ID
        layer_indices: List of layer indices to extract from (e.g., [0, 5, 10])
        hook_types: List of hook types ('residual', 'mlp', 'attention')
        max_samples: Maximum number of samples to process
        batch_size: Batch size for processing (default: 8)
        extraction_id: Optional extraction ID (generated if not provided)

    Returns:
        dict with extraction metadata

    Raises:
        ActivationExtractionError: If extraction fails
        OutOfMemoryError: If GPU runs out of memory
    """
    db = SyncSessionLocal()

    try:
        logger.info(
            f"Starting activation extraction for model {model_id}, "
            f"dataset {dataset_id}, layers {layer_indices}, hooks {hook_types}"
        )

        # Get model from database
        model = db.query(Model).filter_by(id=model_id).first()
        if not model:
            raise ActivationExtractionError(f"Model {model_id} not found in database")

        if model.status != ModelStatus.READY:
            raise ActivationExtractionError(
                f"Model {model_id} is not ready (status: {model.status.value})"
            )

        # Get dataset path (assuming we have a Dataset model similar to Model)
        # For now, we'll construct the path from dataset_id
        dataset_path = str(settings.data_dir / "datasets" / dataset_id / "tokenized")

        if not Path(dataset_path).exists():
            raise ActivationExtractionError(f"Dataset path not found: {dataset_path}")

        # Generate extraction ID if not provided
        if extraction_id is None:
            from datetime import datetime
            extraction_id = f"ext_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Send initial progress
        send_extraction_progress(
            model_id=model_id,
            extraction_id=extraction_id,
            progress=0.0,
            status="starting",
            message=f"Starting extraction with {len(layer_indices)} layers, {len(hook_types)} hook types"
        )

        # Create activation service
        activation_service = ActivationService()

        # Start extraction with progress callbacks
        original_batch_size = batch_size

        try:
            # Update progress: loading model
            send_extraction_progress(
                model_id=model_id,
                extraction_id=extraction_id,
                progress=10.0,
                status="loading",
                message="Loading model and dataset"
            )

            # Run extraction
            result = activation_service.extract_activations(
                model_id=model_id,
                model_path=model.file_path,
                architecture=model.architecture,
                quantization=model.quantization,
                dataset_path=dataset_path,
                layer_indices=layer_indices,
                hook_types=hook_types,
                max_samples=max_samples,
                batch_size=batch_size,
                extraction_id=extraction_id,
            )

            # Update progress: extraction complete
            send_extraction_progress(
                model_id=model_id,
                extraction_id=extraction_id,
                progress=90.0,
                status="saving",
                message=f"Saved {len(result['saved_files'])} activation files"
            )

            logger.info(
                f"Extraction {extraction_id} complete: "
                f"{result['num_samples']} samples, {len(result['saved_files'])} files"
            )

            # Send final progress
            send_extraction_progress(
                model_id=model_id,
                extraction_id=extraction_id,
                progress=100.0,
                status="complete",
                message=f"Extraction complete: {result['num_samples']} samples processed"
            )

            return result

        except OutOfMemoryError as e:
            logger.warning(f"OOM during extraction {extraction_id}, attempting retry with smaller batch")

            # Try with reduced batch size if this is first retry
            if self.request.retries == 0 and batch_size > 1:
                new_batch_size = max(1, batch_size // 2)
                logger.info(f"Retrying with batch_size={new_batch_size} (was {batch_size})")

                send_extraction_progress(
                    model_id=model_id,
                    extraction_id=extraction_id,
                    progress=0.0,
                    status="retrying",
                    message=f"OOM detected, retrying with batch_size={new_batch_size}"
                )

                # Retry with smaller batch size
                raise self.retry(
                    exc=e,
                    kwargs={
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "layer_indices": layer_indices,
                        "hook_types": hook_types,
                        "max_samples": max_samples,
                        "batch_size": new_batch_size,
                        "extraction_id": extraction_id,
                    }
                )

            # If already retried or batch_size is 1, fail
            send_extraction_progress(
                model_id=model_id,
                extraction_id=extraction_id,
                progress=0.0,
                status="error",
                message=f"Out of memory: {str(e)}"
            )
            raise

        except ActivationExtractionError as e:
            logger.error(f"Extraction failed for {extraction_id}: {e}")

            send_extraction_progress(
                model_id=model_id,
                extraction_id=extraction_id,
                progress=0.0,
                status="error",
                message=f"Extraction failed: {str(e)}"
            )
            raise

    except Exception as exc:
        logger.exception(f"Task failed for extraction {extraction_id}: {exc}")

        # Send error update
        if extraction_id:
            send_extraction_progress(
                model_id=model_id,
                extraction_id=extraction_id or "unknown",
                progress=0.0,
                status="error",
                message=f"Error: {str(exc)}"
            )

        # Retry if not at max retries
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying extraction (attempt {self.request.retries + 1})")
            raise self.retry(exc=exc)

        raise

    finally:
        db.close()


@shared_task(name="workers.model_tasks.cancel_download")
def cancel_download(model_id: str, task_id: Optional[str] = None):
    """
    Cancel an in-progress model download.

    This task:
    1. Revokes the download Celery task
    2. Updates model status to ERROR with "Cancelled by user"
    3. Cleans up partial download files
    4. Sends WebSocket notification

    Args:
        model_id: Model database ID
        task_id: Optional Celery task ID to revoke

    Returns:
        dict with cancellation status
    """
    db = SyncSessionLocal()

    try:
        logger.info(f"Cancelling download for model {model_id}")

        # Get model from database
        model = db.query(Model).filter_by(id=model_id).first()

        if not model:
            return {"error": f"Model {model_id} not found"}

        # Check if model is in a cancellable state
        if model.status not in [ModelStatus.DOWNLOADING, ModelStatus.LOADING, ModelStatus.QUANTIZING]:
            return {
                "error": f"Model {model_id} is not in a cancellable state (status: {model.status.value})"
            }

        # Revoke the Celery task if task_id provided
        if task_id:
            from celery import current_app
            current_app.control.revoke(task_id, terminate=True)
            logger.info(f"Revoked Celery task {task_id} for model {model_id}")

        # Clean up partial download files
        cache_dir = settings.models_dir / "raw" / model_id
        if cache_dir.exists():
            import shutil
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"Cleaned up cache directory: {cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up cache directory {cache_dir}: {e}")

        # Update model status
        model.status = ModelStatus.ERROR
        model.error_message = "Cancelled by user"
        model.progress = 0.0
        db.commit()

        # Send WebSocket notification
        send_progress_update(
            model_id=model_id,
            progress=0.0,
            status="error",
            message="Download cancelled by user"
        )

        logger.info(f"Successfully cancelled download for model {model_id}")

        return {
            "model_id": model_id,
            "status": "cancelled",
            "message": "Download cancelled successfully"
        }

    except Exception as e:
        error_msg = f"Failed to cancel download for model {model_id}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

    finally:
        db.close()
