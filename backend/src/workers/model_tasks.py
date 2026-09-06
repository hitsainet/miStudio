"""
Celery tasks for model management operations.

This module contains background tasks for downloading, loading, and quantizing
language models from HuggingFace, as well as extracting activations from models.
"""

import logging
from datetime import datetime, timezone
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional, List

from ..core.cancellation import (
    OperatorCancelled,
    cancel_checker,
    clear_cancel_request,
    cooperative_cancel,
    record_progress,
    request_cancel,
)
from ..core.celery_app import celery_app
from ..core.config import settings
from ..core.database import get_sync_db
from ..ml.model_loader import (
    load_model_from_hf,
    ModelLoadError,
    OutOfMemoryError,
)
from ..models.model import Model, ModelStatus, QuantizationFormat
from ..services.model_service import ModelService
from ..services.activation_service import ActivationService, ActivationExtractionError
from ..services.extraction_db_service import ExtractionDatabaseService
from ..models.activation_extraction import ExtractionStatus
from .base_task import DatabaseTask
from .websocket_emitter import emit_model_progress, emit_extraction_progress, emit_extraction_failed

logger = logging.getLogger(__name__)


def classify_extraction_error(error: Exception, batch_size: int) -> tuple[str, dict]:
    """
    Classify extraction error and suggest retry parameters.

    Args:
        error: The exception that occurred
        batch_size: Current batch size

    Returns:
        Tuple of (error_type, suggested_retry_params)
    """
    error_str = str(error).lower()
    error_type = "UNKNOWN"
    suggested_params = {}

    # Check for OOM errors
    if isinstance(error, OutOfMemoryError) or "out of memory" in error_str or "cuda oom" in error_str:
        error_type = "OOM"
        # Suggest half the batch size, minimum 1
        suggested_batch_size = max(1, batch_size // 2)
        suggested_params = {"batch_size": suggested_batch_size}

    # Check for validation errors
    elif isinstance(error, ActivationExtractionError):
        if "not found" in error_str or "not ready" in error_str:
            error_type = "VALIDATION"
        else:
            error_type = "EXTRACTION"

    # Check for timeout errors
    elif "timeout" in error_str or "timed out" in error_str:
        error_type = "TIMEOUT"
        # Suggest smaller batch size for timeout
        suggested_batch_size = max(1, batch_size // 2)
        suggested_params = {"batch_size": suggested_batch_size}

    return error_type, suggested_params


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
        self.cancel_seen = False
        self._cancel = None
        if model_id:
            from ..core.cancellation import cancel_checker

            self._cancel = cancel_checker("model_download", model_id)
        self._stop_event = threading.Event()
        self._stop_event.set()  # Not running until start()
        self.thread = None

    @property
    def running(self) -> bool:
        return not self._stop_event.is_set()

    @running.setter
    def running(self, value: bool):
        if value:
            self._stop_event.clear()
        else:
            self._stop_event.set()

    def start(self):
        """Start monitoring in background thread."""
        self.initial_size = get_directory_size(self.cache_dir)
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"[ProgressMonitor] Started for {self.model_id}, initial size: {self.initial_size / (1024**2):.2f} MB")

    def stop(self):
        """Stop monitoring."""
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info(f"[ProgressMonitor] Stopped for {self.model_id}")

    def _monitor_loop(self):
        """Monitor loop that runs in background thread."""
        last_progress = 0
        check_interval = 3.0  # Check every 3 seconds

        while not self._stop_event.is_set():
            try:
                current_size = get_directory_size(self.cache_dir)
                downloaded_bytes = current_size - self.initial_size

                # Calculate progress (capped at 90% since we don't know exact size)
                progress = min(90, (downloaded_bytes / self.estimated_size_bytes) * 100)

                # Re-check stop AFTER the (potentially slow) directory walk so a
                # stale "downloading" update can't land after the task marks READY
                if self._stop_event.is_set():
                    break

                # THE CANCEL OBSERVER. This thread is the only thing running
                # while `snapshot_download` blocks the task, so it is where the
                # request is first SEEN — but seeing is all it can do: raising
                # here would die in a worker thread, and HuggingFace exposes no
                # abort hook. It records the fact and stops narrating; the task
                # acts on it at its next real boundary. See the note on
                # `download_and_load_model` for what that costs.
                if self._cancel is not None and self._cancel():
                    logger.info(
                        "[ProgressMonitor] %s: cancellation requested; the "
                        "download will stop at the next phase boundary",
                        self.model_id,
                    )
                    self.cancel_seen = True
                    self._stop_event.set()
                    break

                # Only send update if progress increased by at least 1%
                if progress >= last_progress + 1:
                    downloaded_mb = downloaded_bytes / (1024 * 1024)
                    estimated_mb = self.estimated_size_bytes / (1024 * 1024)

                    # Update database
                    try:
                        with get_sync_db() as db:
                            model = db.query(Model).filter_by(id=self.model_id).first()
                            if model:
                                model.progress = progress
                                db.commit()
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

                # Interruptible sleep: wakes immediately when stop() is called
                self._stop_event.wait(check_interval)

            except Exception as e:
                logger.error(f"[ProgressMonitor] Error: {e}")
                self._stop_event.wait(check_interval)


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


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="workers.model_tasks.download_and_load_model",
    max_retries=0,  # No auto-retry - user must manually retry
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
    try:
        # Apply transformers compatibility patches for newer models (Phi-4, etc.)
        # This must be done in the task because Celery uses forked child processes
        from ..ml.transformers_compat import patch_transformers_compatibility
        patch_transformers_compatibility()

        logger.info(f"Starting model download: {model_id} from {repo_id}")

        # Convert quantization string to enum
        quant_format = QuantizationFormat(quantization)

        # Update status to DOWNLOADING
        with self.get_db() as db:
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
        # BEFORE THE DOWNLOAD. A model cancelled while queued must not pull
        # 15 GB first.
        clear_cancel_request("model_download", model_id)
        _model_cancel = cancel_checker("model_download", model_id)
        _model_cancel.raise_if_cancelled("stopped before the download began")

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
            # AFTER THE DOWNLOAD, BEFORE ANYTHING ELSE. If the monitor saw the
            # request mid-transfer this is the first point the task can act on
            # it, and it is also the boundary before quantization and the
            # architecture pass — the expensive work that follows.
            if progress_monitor.cancel_seen:
                _model_cancel.poll_now()
            _model_cancel.raise_if_cancelled(
                "stopped after the download, before loading")

            # Model loaded successfully
            send_progress_update(
                model_id=model_id,
                progress=95.0,
                status="loading",
                message=f"Model loaded with {metadata['quantization']} quantization"
            )

        except OutOfMemoryError as e:
            logger.error(f"Out of memory loading model {model_id}: {e}")

            with self.get_db() as db:
                model = db.query(Model).filter_by(id=model_id).first()
                if model:
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

            with self.get_db() as db:
                model = db.query(Model).filter_by(id=model_id).first()
                if model:
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
        with self.get_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if model:
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

        # Close out any task_queue rows from a prior failed attempt (retry success)
        from .base_task import mark_task_queue_entries_completed
        with self.get_db() as db:
            mark_task_queue_entries_completed(db, model_id, "model", "download")

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

    except OperatorCancelled as cancelled:
        # NO HANDLER EXISTED. The raise escaped the task entirely: `except
        # Exception` below cannot catch a BaseException, and celery re-raises
        # rather than recording FAILURE — so the acks_late message was not
        # acked and would have been redelivered only after the full 12-hour
        # visibility timeout. That is the precise outcome this whole design
        # exists to avoid, reintroduced by adding a checkpoint without a
        # handler to receive it.
        logger.info("Model download %s cancelled: %s", model_id, cancelled.detail)

        # AND THE TASK OWNS ITS PARTIAL OUTPUT. `cancel_download` deliberately
        # stops deleting the cache directory once the job has started, on the
        # promise that the task removes it here. Without this the promise was
        # false and a cancelled 40 GB download was orphaned forever — invisible
        # to `delete_model_files`, which resolves `model.file_path`, a column
        # that is not written until after the checkpoint.
        # BOUND BEFORE THE try, and resolved against the deletable roots.
        #
        # Computing it inside the try meant a failure on that very line would
        # leave `cache_dir` unbound for the logger in the except — an error
        # raised inside an error handler, which nothing catches, reproducing
        # the unacked-acks_late strand this handler exists to prevent.
        #
        # `resolve_deletable_path` is the MIS-E2E-071 guard every other
        # deletion in this change already goes through: it refuses the trusted
        # roots and their top-level directories, so an empty model_id cannot
        # collapse the target onto `models_dir/raw` — every downloaded model.
        cache_dir = settings.models_dir / "raw" / model_id
        try:
            target = settings.resolve_deletable_path(str(cache_dir))
        except ValueError as guard_exc:
            logger.error("Refusing to delete %s: %s", cache_dir, guard_exc)
            target = None
        if target is not None and target.exists():
            try:
                shutil.rmtree(target)
                logger.info("Removed the cancelled download's partial output: %s", target)
            except Exception as cleanup_exc:  # noqa: BLE001 - must not mask the cancel
                logger.warning("Could not remove %s: %s", target, cleanup_exc)

        # Close the task_queue row as CANCELLED, not completed.
        #
        # `mark_task_queue_entries_completed` writes status="completed",
        # progress=100.0 — so R2 traded "shows as still running" for "shows as
        # finished successfully", the same durable-right/visible-wrong shape it
        # had just fixed two hunks earlier. TaskQueue documents a `cancelled`
        # value; this uses it.
        try:
            from ..models.task_queue import TaskQueue

            with get_sync_db() as _queue_db:
                # The same (entity_type, task_type) pair the success path uses
                # at the end of this task — a different one would silently
                # match nothing and leave the ghost entry in place.
                for _entry in (
                    _queue_db.query(TaskQueue)
                    .filter_by(entity_id=model_id, entity_type="model",
                               task_type="download")
                    .filter(TaskQueue.status.in_(("queued", "running")))
                    .all()
                ):
                    _entry.status = "cancelled"
                    _entry.completed_at = datetime.now(timezone.utc)
                _queue_db.commit()
        except Exception:  # noqa: BLE001 - bookkeeping must not mask the cancel
            logger.debug("Could not close the task_queue row for %s", model_id)

        record_progress(
            "model_download", model_id,
            status=ModelStatus.ERROR.value,
            error_message="Cancelled by user",
        )
        return {
            "status": "cancelled",
            "model_id": model_id,
            "detail": cancelled.detail,
        }

    except Exception as exc:
        logger.exception(f"Task failed for model {model_id}: {exc}")

        # Update database with error
        try:
            with self.get_db() as db:
                model = db.query(Model).filter_by(id=model_id).first()
                if model:
                    model.status = ModelStatus.ERROR
                    model.error_message = str(exc)
                    db.commit()
        except Exception as db_exc:
            logger.error(f"Failed to update error state in database: {db_exc}")

        # Save failure state to task_queue for manual retry
        try:
            from ..models.task_queue import TaskQueue
            import uuid

            with self.get_db() as db:
                # Check if there's an existing queued task_queue entry for this entity
                # (which would indicate this is a retry)
                existing_entry = db.query(TaskQueue).filter_by(
                    entity_id=model_id,
                    entity_type="model",
                    task_type="download"
                ).filter(
                    TaskQueue.status.in_(["queued", "running"])
                ).first()

                if existing_entry:
                    # This is a retry that failed - update the existing entry
                    existing_entry.status = "failed"
                    existing_entry.error_message = str(exc)
                    existing_entry.task_id = self.request.id
                    db.commit()
                    logger.info(f"Updated failed retry in task_queue: {existing_entry.id} (retry #{existing_entry.retry_count})")
                else:
                    # This is an initial failure - create new entry
                    task_queue_entry = TaskQueue(
                        id=f"tq_{uuid.uuid4().hex[:12]}",
                        task_id=self.request.id,
                        task_type="download",
                        entity_id=model_id,
                        entity_type="model",
                        status="failed",
                        progress=0.0,
                        error_message=str(exc),
                        retry_params={
                            "repo_id": repo_id,
                            "quantization": quantization,
                            "trust_remote_code": trust_remote_code,
                        },
                        retry_count=0,
                    )
                    db.add(task_queue_entry)
                    db.commit()
                    logger.info(f"Saved failed task to task_queue: {task_queue_entry.id}")
        except Exception as queue_exc:
            logger.error(f"Failed to save task to queue: {queue_exc}")

        # Send WebSocket notification of failure
        send_progress_update(
            model_id=model_id,
            progress=0.0,
            status="error",
            message=f"Download failed: {str(exc)}"
        )

        raise

    finally:
        # GIVE THE WEIGHTS BACK. This task loads the model for one reason — to
        # read `architecture`, `params_count` and `architecture_config` off it
        # for the database — and `device_map="auto"` puts them on the GPU to do
        # it. Nothing downstream wants them resident: training and extraction
        # each load what they need themselves. Without this the weights sit on
        # the card until the worker is restarted, which is days.
        #
        # Measured: downloading LFM2.5-2.6B (2,697,198,592 params at FP32) left
        # 10,696 MiB held on a 24 GB card for an hour, against a ~3 GB CUDA
        # context floor. The next model to want the card gets what is left.
        #
        # The pre-extraction path already calls `empty_cache()` with the comment
        # "in case previous task didn't complete cleanup" — that is this leak,
        # worked around at the far end. A workaround there cannot help anything
        # that is not an extraction, which is what the serving process is.
        #
        # EVERY REFERENCE, then gc, then `empty_cache`. Freeing the local alone
        # returns the blocks to torch's allocator and not to the driver, so the
        # memory stays unavailable to every other process on the card, which is
        # the whole complaint. `tokenizer` and `config` are nulled with it: they
        # are small, but they are the kind of thing that keeps a module alive.
        try:
            import gc

            import torch

            model_obj = None  # noqa: F841 - the assignment IS the release
            tokenizer = None  # noqa: F841
            config = None  # noqa: F841
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(
                    "Released the download-inspection model from GPU memory "
                    f"(reserved now {torch.cuda.memory_reserved() / 1024**3:.2f} GB)"
                )
        except Exception as cleanup_exc:  # noqa: BLE001
            # A failure here must not replace the real outcome — including a
            # propagating exception, which a raise from `finally` would discard.
            logger.warning(f"GPU cleanup after download failed: {cleanup_exc}")


@celery_app.task(name="workers.model_tasks.delete_model_files")
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
    from ..core.config import settings

    deleted_files = []
    errors = []

    try:
        # Resolve Docker-style /data/ paths for native mode compatibility
        # MIS-E2E-071 — file_path/quantized_path are API-writable via
        # ModelUpdate and land straight in rmtree.
        resolved_file_path = None
        if file_path:
            try:
                resolved_file_path = str(settings.resolve_deletable_path(file_path))
            except ValueError as e:
                errors.append(f"Refusing to delete file_path {file_path!r}: {e}")
                logger.error(errors[-1])
        resolved_quantized_path = None
        if quantized_path:
            try:
                resolved_quantized_path = str(
                    settings.resolve_deletable_path(quantized_path)
                )
            except ValueError as e:
                errors.append(f"Refusing to delete quantized_path {quantized_path!r}: {e}")
                logger.error(errors[-1])

        # Delete raw model files
        if resolved_file_path and os.path.exists(resolved_file_path):
            import shutil
            shutil.rmtree(resolved_file_path)
            deleted_files.append(resolved_file_path)
            logger.info(f"Deleted raw model files: {resolved_file_path}")

        # Delete quantized model files
        if resolved_quantized_path and os.path.exists(resolved_quantized_path):
            import shutil
            shutil.rmtree(resolved_quantized_path)
            deleted_files.append(resolved_quantized_path)
            logger.info(f"Deleted quantized model files: {resolved_quantized_path}")

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


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="workers.model_tasks.update_model_progress"
)
def update_model_progress(self, model_id: str, progress: float, status: Optional[str] = None):
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
    try:
        with self.get_db() as db:
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
            status=status or model.status.value if model else "unknown",
            message=f"Progress: {progress:.1f}%"
        )

        return {"model_id": model_id, "progress": progress, "status": status}

    except Exception as e:
        logger.error(f"Failed to update progress for model {model_id}: {e}")
        return {"error": str(e)}


class PermanentExtractionError(Exception):
    """An extraction failure that retrying cannot fix.

    Celery's retry exists for transient faults — a busy GPU, a flaky mount. A
    missing model row or a missing extraction row is not transient: the second
    attempt fails identically, and on 2026-08-24 each attempt re-resolved the
    model and started downloading it again. `autoretry_for` must never include
    this.
    """


def build_extraction_progress_callback(task, model_id: str, extraction_id: str, cancelled):
    """The extraction's progress callback — and its ONLY cancellation checkpoint.

    MODULE LEVEL SO A TEST CAN DRIVE THE REAL ONE. This was a closure inside
    `extract_activations`, unreachable from any test, so the Shape-A test drove
    a hand-written reconstruction instead. Deleting the `raise_if_cancelled`
    from the production copy then turned exactly ONE test red — a source scrape
    — and this repo's record is that a source-scraping guard fails open. The
    capability was, by the reachability rule, not shipped.

    `task` is the bound Celery task, for its `get_db()`.
    """
    def on_extraction_progress(samples_processed: int, total_samples: int):
        """Update database and emit WebSocket progress during extraction."""
        cancelled.raise_if_cancelled(
            f"stopped after {samples_processed} of {total_samples} samples"
        )

        # Calculate progress (10% for loading, 10-90% for extraction, 90-100% for saving)
        extraction_progress = 10.0 + (samples_processed / total_samples) * 80.0

        # Update database
        try:
            with task.get_db() as db:
                ExtractionDatabaseService.update_progress(
                    db=db,
                    extraction_id=extraction_id,
                    progress=extraction_progress,
                    status=ExtractionStatus.EXTRACTING,
                    samples_processed=samples_processed,
                )
        except Exception as db_e:
            logger.warning(f"Failed to update extraction progress in database: {db_e}")

        # Emit WebSocket update
        emit_extraction_progress(
            model_id=model_id,
            extraction_id=extraction_id,
            progress=extraction_progress,
            status="extracting",
            message=f"Processing samples: {samples_processed}/{total_samples}"
        )

    return on_extraction_progress


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="workers.model_tasks.extract_activations",
    max_retries=2,
    default_retry_delay=60,
    queue="extraction",
)
@cooperative_cancel("activation_extraction")
def extract_activations(
    self,
    model_id: str,
    dataset_id: str,
    layer_indices: List[int],
    hook_types: List[str],
    max_samples: int,
    batch_size: int = 8,
    micro_batch_size: Optional[int] = None,
    extraction_id: Optional[str] = None,
    gpu_id: int = 0,
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
        micro_batch_size: GPU micro-batch size for memory efficiency (defaults to batch_size)
        extraction_id: Optional extraction ID (generated if not provided)
        gpu_id: GPU device ID to use for extraction (default: 0)

    Returns:
        dict with extraction metadata

    Raises:
        ActivationExtractionError: If extraction fails
        OutOfMemoryError: If GPU runs out of memory
    """
    try:
        logger.info(
            f"Starting activation extraction for model {model_id}, "
            f"dataset {dataset_id}, layers {layer_indices}, hooks {hook_types}, gpu={gpu_id}"
        )

        # REFUSE TO RUN WITHOUT SOMEWHERE TO REPORT.
        #
        # 2026-08-24: this task ran for 3.5 hours against an `extraction_id`
        # whose row was never created. Every progress write logged
        # "not found for progress update" and continued — roughly 300 times —
        # so the UI sat on "Starting extraction..." while the GPU was pinned at
        # 100%, the failure at the end could not be recorded either ("not found
        # to mark failed"), and the retry that followed re-resolved the model
        # and kicked off a spurious 15 GB download.
        #
        # Every one of those symptoms is downstream of the same fact, known in
        # the first millisecond: there is nowhere to write the result. A job
        # whose outcome cannot be recorded has no reason to consume a GPU.
        if extraction_id:
            with self.get_db() as _db:
                _row = ExtractionDatabaseService.get_extraction(_db, extraction_id)
            if _row is None:
                raise PermanentExtractionError(
                    f"Extraction {extraction_id} has no database row, so this run "
                    f"could never be recorded. Refusing to start. (The row is "
                    f"created by the API before dispatch; if it is missing the "
                    f"request did not commit.)"
                )

        # Pre-task GPU memory check - ensure clean state before loading model
        import torch
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)  # GB
            reserved_before = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)    # GB
            logger.info(
                f"[Pre-extraction {self.request.id}] GPU {gpu_id} memory before cleanup: "
                f"Allocated={allocated_before:.2f} GB, Reserved={reserved_before:.2f} GB"
            )

            # Force cleanup in case previous task didn't complete cleanup
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
            import gc
            gc.collect()

            allocated_after = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
            reserved_after = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
            logger.info(
                f"[Pre-extraction {self.request.id}] GPU {gpu_id} memory after cleanup: "
                f"Allocated={allocated_after:.2f} GB, Reserved={reserved_after:.2f} GB"
            )

        # Get model and dataset from database
        from ..models.dataset import Dataset as DatasetModel

        with self.get_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if not model:
                raise ActivationExtractionError(f"Model {model_id} not found in database")

            if model.status != ModelStatus.READY:
                raise ActivationExtractionError(
                    f"Model {model_id} is not ready (status: {model.status.value})"
                )

            # Get model file paths before closing session - resolve to absolute using settings helper
            model_file_path = str(settings.resolve_data_path(model.file_path))
            model_architecture = model.architecture
            model_quantization = model.quantization

            # Get dataset and tokenized path (with eager loading of tokenizations)
            from sqlalchemy.orm import joinedload
            dataset = db.query(DatasetModel).options(joinedload(DatasetModel.tokenizations)).filter_by(id=dataset_id).first()
            if not dataset:
                raise ActivationExtractionError(f"Dataset {dataset_id} not found in database")

            # Import DatasetStatus enum
            from ..models.dataset import DatasetStatus

            if dataset.status != DatasetStatus.READY:
                raise ActivationExtractionError(
                    f"Dataset {dataset_id} is not ready (status: {dataset.status.value})"
                )

            # Check if dataset has tokenizations
            if not dataset.tokenizations or len(dataset.tokenizations) == 0:
                raise ActivationExtractionError(f"Dataset {dataset_id} has no tokenizations")

            # Get first tokenization's path (most common case is one tokenization per dataset)
            tokenization = dataset.tokenizations[0]
            if not tokenization.tokenized_path:
                raise ActivationExtractionError(f"Dataset {dataset_id} tokenization has no path")

            # Get tokenized path - resolve to absolute using settings helper
            dataset_path = str(settings.resolve_data_path(tokenization.tokenized_path))

        # Validate model path exists
        if not Path(model_file_path).exists():
            raise ActivationExtractionError(f"Model path not found: {model_file_path}")

        if not Path(dataset_path).exists():
            raise ActivationExtractionError(f"Dataset path not found: {dataset_path}")

        # Generate extraction ID if not provided (first attempt only)
        if extraction_id is None:
            from datetime import datetime
            extraction_id = f"ext_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Generated new extraction_id: {extraction_id}")
        else:
            logger.info(f"Reusing existing extraction_id: {extraction_id} (retry attempt {self.request.retries})")

        # Create or update database record for extraction tracking
        try:
            with self.get_db() as db:
                # Check if extraction already exists (from previous retry)
                from ..models.activation_extraction import ActivationExtraction
                existing = db.query(ActivationExtraction).filter_by(id=extraction_id).first()

                if existing:
                    # Update existing record with retry attempt
                    logger.info(f"Found existing extraction record {extraction_id}, updating for retry")
                    existing.status = ExtractionStatus.QUEUED
                    existing.progress = 0.0
                    existing.samples_processed = 0
                    existing.retry_count = self.request.retries
                    existing.celery_task_id = self.request.id
                    # Clear the previous attempt's failure. Without this the UI
                    # keeps rendering its "Error Details" panel (it keys off a
                    # non-empty error_message) for the whole successful retry and
                    # after it completes — e.g. a transient CUDA OOM shown against
                    # a healthy run sitting at 58%.
                    existing.error_message = None
                    existing.error_type = None
                    db.commit()
                else:
                    # Create new record (first attempt)
                    ExtractionDatabaseService.create_extraction(
                        db=db,
                        extraction_id=extraction_id,
                        model_id=model_id,
                        dataset_id=dataset_id,
                        layer_indices=layer_indices,
                        hook_types=hook_types,
                        max_samples=max_samples,
                        batch_size=batch_size,
                        micro_batch_size=micro_batch_size,
                        celery_task_id=self.request.id,
                        gpu_id=gpu_id,
                    )
                    logger.info(f"Created database record for extraction {extraction_id} (GPU {gpu_id})")
        except Exception as db_e:
            # Don't fail extraction if database tracking fails
            logger.warning(f"Failed to create/update extraction database record: {db_e}")

        # Send initial progress
        emit_extraction_progress(
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
            try:
                with self.get_db() as db:
                    ExtractionDatabaseService.update_progress(
                        db=db,
                        extraction_id=extraction_id,
                        progress=10.0,
                        status=ExtractionStatus.LOADING,
                        samples_processed=0,
                    )
            except Exception as db_e:
                logger.warning(f"Failed to update extraction progress in database: {db_e}")

            emit_extraction_progress(
                model_id=model_id,
                extraction_id=extraction_id,
                progress=10.0,
                status="loading",
                message="Loading model and dataset"
            )

            # THE CANCELLATION CHECKPOINT.
            #
            # This callback already fires every ~10 samples, which is the finest
            # boundary at which this task can cleanly abandon work — a partially
            # written batch is not something to resume from. The checker's own
            # 2-second throttle decides whether the call reaches the database,
            # so calling it here costs nothing at any sample rate.
            #
            # The raise crosses `activation_service`'s
            # `except Exception: logger.warning("Progress callback failed")`,
            # which sits directly around this call. That is precisely why
            # `OperatorCancelled` derives from BaseException: an
            # Exception-derived cancel raised here would be logged at WARNING
            # and the extraction would carry on for hours.
            cancelled = cancel_checker("activation_extraction", extraction_id)
            on_extraction_progress = build_extraction_progress_callback(
                self, model_id, extraction_id, cancelled
            )

            # POLL BEFORE THE INDIVISIBLE STEP. Everything past this line is one
            # `extract_activations` call, and the first checkpoint inside it is
            # not reached until the model is loaded onto the GPU — minutes for a
            # large model. `poll_now` ignores the throttle because being two
            # seconds stale is the wrong trade immediately before that.
            if cancelled.poll_now():
                raise OperatorCancelled(
                    "activation_extraction", extraction_id, cancelled.reason or "cancelled",
                    "stopped before the extraction pass began",
                )

            # Run extraction with progress callback
            result = activation_service.extract_activations(
                model_id=model_id,
                model_path=model_file_path,
                architecture=model_architecture,
                quantization=model_quantization,
                dataset_path=dataset_path,
                layer_indices=layer_indices,
                hook_types=hook_types,
                max_samples=max_samples,
                batch_size=batch_size,
                micro_batch_size=micro_batch_size,
                extraction_id=extraction_id,
                progress_callback=on_extraction_progress,
                gpu_id=gpu_id,
            )

            # Update progress: saving
            try:
                with self.get_db() as db:
                    ExtractionDatabaseService.update_progress(
                        db=db,
                        extraction_id=extraction_id,
                        progress=90.0,
                        status=ExtractionStatus.SAVING,
                        samples_processed=result['num_samples'],
                    )
            except Exception as db_e:
                logger.warning(f"Failed to update extraction progress in database: {db_e}")

            emit_extraction_progress(
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

            # Mark extraction as completed in database (critical step - wrap in robust try-catch)
            completion_success = False
            try:
                with self.get_db() as db:
                    ExtractionDatabaseService.mark_completed(
                        db=db,
                        extraction_id=extraction_id,
                        statistics=result['statistics'],
                        saved_files=result['saved_files'],
                    )
                    completion_success = True
                    logger.info(f"Successfully marked extraction {extraction_id} as COMPLETED in database")
            except Exception as db_e:
                logger.error(
                    f"CRITICAL: Failed to mark extraction {extraction_id} as completed in database: {db_e}",
                    exc_info=True
                )
                # Try to emit failure event if completion fails
                emit_extraction_failed(
                    model_id=model_id,
                    extraction_id=extraction_id,
                    error_message=f"Failed to save completion status: {str(db_e)}",
                    error_type="DATABASE",
                    suggested_retry_params={}
                )
                # Re-raise to trigger retry - this is a critical failure
                raise ActivationExtractionError(f"Failed to save extraction completion to database: {db_e}")

            # Send final progress only if completion succeeded
            if completion_success:
                emit_extraction_progress(
                    model_id=model_id,
                    extraction_id=extraction_id,
                    progress=100.0,
                    status="complete",
                    message=f"Extraction complete: {result['num_samples']} samples processed"
                )
                logger.info(f"Extraction {extraction_id} fully completed and saved")

            return result

        except OutOfMemoryError as e:
            logger.warning(f"OOM during extraction {extraction_id}, attempting retry with smaller batch")

            # Try with reduced batch size if this is first retry
            if self.request.retries == 0 and batch_size > 1:
                new_batch_size = max(1, batch_size // 2)
                logger.info(f"Retrying with batch_size={new_batch_size} (was {batch_size})")

                emit_extraction_progress(
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
                        "micro_batch_size": micro_batch_size,
                        "extraction_id": extraction_id,
                    }
                )

            # If already retried or batch_size is 1, fail
            try:
                with self.get_db() as db:
                    ExtractionDatabaseService.mark_failed(
                        db=db,
                        extraction_id=extraction_id,
                        error_message=f"Out of memory: {str(e)}"
                    )
            except Exception as db_e:
                logger.warning(f"Failed to mark extraction as failed in database: {db_e}")

            # Classify error and emit dedicated failure event
            error_type, suggested_params = classify_extraction_error(e, batch_size)
            emit_extraction_failed(
                model_id=model_id,
                extraction_id=extraction_id,
                error_message=f"Out of memory: {str(e)}",
                error_type=error_type,
                suggested_retry_params=suggested_params
            )
            raise

        except ActivationExtractionError as e:
            logger.error(f"Extraction failed for {extraction_id}: {e}")

            try:
                with self.get_db() as db:
                    ExtractionDatabaseService.mark_failed(
                        db=db,
                        extraction_id=extraction_id,
                        error_message=str(e)
                    )
            except Exception as db_e:
                logger.warning(f"Failed to mark extraction as failed in database: {db_e}")

            # Classify error and emit dedicated failure event
            error_type, suggested_params = classify_extraction_error(e, batch_size)
            emit_extraction_failed(
                model_id=model_id,
                extraction_id=extraction_id,
                error_message=f"Extraction failed: {str(e)}",
                error_type=error_type,
                suggested_retry_params=suggested_params
            )
            raise

    except Exception as exc:
        logger.exception(f"Task failed for extraction {extraction_id}: {exc}")

        # Mark extraction as failed in database
        if extraction_id:
            try:
                with self.get_db() as db:
                    ExtractionDatabaseService.mark_failed(
                        db=db,
                        extraction_id=extraction_id,
                        error_message=str(exc)
                    )
            except Exception as db_e:
                logger.warning(f"Failed to mark extraction as failed in database: {db_e}")

            # Classify error and emit dedicated failure event
            error_type, suggested_params = classify_extraction_error(exc, batch_size)
            emit_extraction_failed(
                model_id=model_id,
                extraction_id=extraction_id or "unknown",
                error_message=f"Error: {str(exc)}",
                error_type=error_type,
                suggested_retry_params=suggested_params
            )

        # DO NOT RETRY WHAT CANNOT SUCCEED.
        #
        # A missing model row or a missing extraction row fails identically on
        # every attempt. On 2026-08-24 each retry re-resolved the model and
        # began downloading it again — three attempts, one spurious 15 GB
        # fetch, and a UI that showed nothing throughout. Retry is for
        # transient faults; this is a permanent one.
        if isinstance(exc, PermanentExtractionError) or "not found in database" in str(exc):
            logger.error(
                "Not retrying extraction %s: %s. This cannot succeed on a "
                "second attempt, and retrying re-triggers model resolution.",
                extraction_id, exc,
            )
            raise

        # Retry if not at max retries
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying extraction (attempt {self.request.retries + 1}) with extraction_id={extraction_id}")
            # Pass extraction_id in kwargs to preserve it across retries
            raise self.retry(
                exc=exc,
                kwargs={
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "layer_indices": layer_indices,
                    "hook_types": hook_types,
                    "max_samples": max_samples,
                    "batch_size": batch_size,
                    "micro_batch_size": micro_batch_size,
                    "extraction_id": extraction_id,  # Preserve extraction_id
                }
            )

        raise

    finally:
        # CRITICAL: Ensure GPU cache is cleared even if service cleanup didn't work
        # This is a safety net for sequential extraction jobs
        import torch
        if torch.cuda.is_available():
            import gc
            gc.collect()
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()

            allocated_final = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
            reserved_final = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
            logger.info(
                f"[Post-extraction {self.request.id}] GPU {gpu_id} memory after task cleanup: "
                f"Allocated={allocated_final:.2f} GB, Reserved={reserved_final:.2f} GB"
            )


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="workers.model_tasks.cancel_download"
)
def cancel_download(self, model_id: str, task_id: Optional[str] = None):
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
    try:
        logger.info(f"Cancelling download for model {model_id}")

        # Get model from database
        with self.get_db() as db:
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

            # WRITE THE REQUEST FIRST, so a running download can see it. The
            # revoke above is inert on a --pool=solo worker; the flag is the
            # channel.
            request_cancel(
                "model_download", model_id, reason="Cancelled by user",
            )

            # DO NOT DELETE WHAT IS STILL BEING WRITTEN. This rmtree'd the
            # cache directory `snapshot_download` was actively filling, and the
            # task then recreated parts of it — a half-tree nothing could read
            # and nothing would clean up. Only a job that had NOT started is
            # cleaned up here; a started one removes its own partial output at
            # its next phase boundary.
            job_had_started = (model.progress or 0) > 0
            cache_dir = settings.models_dir / "raw" / model_id
            if job_had_started:
                logger.info(
                    "Not deleting %s: the download is live and removes its own "
                    "partial output when it stops", cache_dir,
                )
            elif cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    logger.info(f"Cleaned up cache directory: {cache_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up cache directory {cache_dir}: {e}")

            # Update model status. The enum has no CANCELLED member (native PG
            # type), so ERROR remains — but `cancel_requested_at` is now set,
            # which is what distinguishes this from a crash.
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
