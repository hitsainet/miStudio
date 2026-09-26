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
from .gpu_job import gpu_job
from ..core.database import get_sync_db
# At import time, not inside the task: tests stand a stub in for `torch` while
# the task runs, and a first import of this module under that stub would bind
# the stub into it for the rest of the process.
from ..ml.model_devices import detach_dispatch_hooks
from ..ml.model_loader import (
    load_model_from_hf,
    ModelLoadError,
    OutOfMemoryError,
)
from ..models.model import Model, ModelStatus, QuantizationFormat
from ..services.model_service import ModelService
from ..services.activation_service import (
    ActivationExtractionError,
    ActivationService,
    ModelLoadOutOfMemory,
    load_format_for,
    resolve_model_snapshot,
)
from ..services.extraction_db_service import ExtractionDatabaseService
from ..services.gpu_placement import AUTO, GpuPlacementError, place_job
from ..models.activation_extraction import EXTRACTION_PHASES, ExtractionPhase, ExtractionStatus
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

    # A MODEL LOAD that ran out of memory: no batch ran, so no smaller batch helps.
    # Checked before the message match, which its "out of memory" text also meets.
    if isinstance(error, ModelLoadOutOfMemory):
        error_type = "MODEL_LOAD_OOM"

    # Check for OOM errors
    elif isinstance(error, OutOfMemoryError) or "out of memory" in error_str or "cuda oom" in error_str:
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


def record_download_card(
    db, model_id: str, gpu_uuid: Optional[str], gpu_uuids: Optional[List[str]] = None
) -> int:
    """Stamp the card(s) on this download's live task_queue rows. Returns how many.

    A download has a task_queue row only once it has failed (then the row is its
    retry handle), so the row that is live here is a retry's. A first attempt has
    none, and its card reaches the row the failure handler creates. ``gpu_uuids``
    lists every card of an inspection load split across several; None for one card.
    """
    from ..models.task_queue import TaskQueue

    rows = (
        db.query(TaskQueue)
        .filter_by(entity_id=model_id, entity_type="model", task_type="download")
        .filter(TaskQueue.status.in_(("queued", "running")))
        .all()
    )
    for row in rows:
        row.gpu_uuid = gpu_uuid
        row.gpu_uuids = gpu_uuids
    db.commit()
    return len(rows)


def required_mb_for_load(config_source: str, quantization: str, **config_kwargs) -> Optional[float]:
    """The GPU memory a model load needs, in MiB, read from its config alone.

    WHY A JOB IS SIZED BEFORE IT IS PLACED. Placement without a size takes the
    card with the most free memory and never splits — `place_job` has nothing to
    split against. So a model that fits no single card (Qwen2.5-14B-Instruct is
    ~29.5 GB at bf16; the node's cards are 12 and 24 GB) went to the 3090 and was
    refused by the loader's preflight, and could not even be registered.

    The SAME arithmetic as `resource_config.preflight_gpu_capacity` — bytes per
    parameter at this quantization, plus its fixed headroom — so a card the
    placement accepts is a card the preflight accepts, and a split the placement
    sizes is one the preflight sums to the same total.

    None when the config cannot be read or sized; the job is then placed as
    before (most free card), and the loader's own checks still apply.
    """
    from ..ml.model_loader import estimate_parameter_count
    from ..services.resource_config import _ACTIVATION_HEADROOM_GB, _BYTES_PER_PARAM

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(config_source, **config_kwargs)
    except Exception as exc:  # noqa: BLE001 - sizing is advisory; the load reports its own failure
        logger.warning(
            "Could not read the config of %s to size its GPU placement (%s). Placing "
            "it without a size: it goes to the card with the most free memory and "
            "cannot be split.", config_source, exc,
        )
        return None

    params = estimate_parameter_count(config)
    if not params:
        logger.warning(
            "The config of %s does not give a parameter count, so its GPU placement "
            "is unsized and cannot be split.", config_source,
        )
        return None
    per_param = _BYTES_PER_PARAM.get(str(quantization).upper(), 2.0)
    return params * per_param / (1024 ** 2) + _ACTIVATION_HEADROOM_GB * 1024


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="workers.model_tasks.download_and_load_model",
    max_retries=0,  # No auto-retry - user must manually retry
    # high_priority, to AGREE with task_routes, which wins. This read
    # "processing" while celery resolved the task to high_priority, so the
    # decorator described a queue the task never used (OSD-10). Both are kept
    # deliberately: if a task_routes entry is ever lost, an explicit queue here
    # is what stops the task falling back to the default queue silently.
    queue="high_priority",
)
# In place: the GPU step is the inspection load AFTER the download.
@gpu_job("model_download", handoff=False)
def download_and_load_model(
    self,
    model_id: str,
    repo_id: str,
    quantization: str,
    access_token: Optional[str] = None,
    trust_remote_code: bool = False,
    gpu_request: Optional[str] = None,
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
        gpu_request: "auto" or a GPU UUID, resolved at submit. None (a message
            or retry row from before this existed) means auto.

    Returns:
        dict with model metadata
    """
    # Bound before the try: the `finally` reads it, and a failure before the
    # placement must still reach the release there.
    placement = None
    # Likewise the model: the `finally` detaches a split model's hooks, and a
    # load that raised never bound it.
    model_obj = None
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

        # WHICH CARD THE INSPECTION LOAD USES. This was device_map="auto",
        # which spread the weights over every visible GPU — including a card
        # miLLM was serving from. Placed against the memory free now; a card
        # that cannot take it raises GpuPlacementError, which the handler below
        # records as this download's failure. No other card is tried.
        #
        # SIZED FIRST, AND SPLIT ONLY WHEN NO ONE CARD CAN HOLD IT. Unsized, the
        # placement took the most-free card, and a model larger than every
        # card (Qwen2.5-14B-Instruct, ~29.5 GB at bf16, against 12 + 24 GB) was
        # refused by the loader's preflight — it could not even be registered,
        # so nothing downstream could split it either. A model that fits one
        # card is placed on one card exactly as before.
        #
        # The inspection LOAD is kept rather than replaced by a config-only
        # read. It is what fetches exactly the weight files `from_pretrained`
        # uses (a whole-repo snapshot would also pull duplicate .bin/original
        # checkpoints), what proves the model loads at the requested
        # quantization (never retried at another), and what counts the
        # parameters recorded on the row — bitsandbytes' packed tensors
        # included. Replacing it changes all three for every model to serve
        # the rare one that needs a split.
        from ..services.gpu_placement import AUTO, place_job

        required_mb = required_mb_for_load(
            repo_id,
            quant_format.value,
            cache_dir=str(cache_dir),
            trust_remote_code=trust_remote_code,
            token=access_token,
        )
        placement = place_job(gpu_request, required_mb=required_mb, allow_shard=True)
        logger.info(
            "Model download %s: inspection load on %s (requested %s, ~%s MB)",
            model_id, placement.describe(), gpu_request or AUTO,
            "unknown" if required_mb is None else f"{required_mb:,.0f}",
        )
        # Recorded BEFORE the load, on the retry row when one is live.
        with self.get_db() as db:
            record_download_card(db, model_id, **placement.gpu_columns())

        progress_monitor.start()

        # Load model from HuggingFace (this handles download + quantization)
        try:
            logger.info(f"Loading model {repo_id} with {quant_format.value} quantization")

            model_obj, tokenizer, config, metadata = load_model_from_hf(
                repo_id=repo_id,
                quant_format=quant_format,
                cache_dir=cache_dir,
                # The placed card, which the loader's preflight also measures —
                # or the split's own device map over its cards, within their
                # GPU-only budget. Both passed through as the placement says.
                device_map=placement.device_map,
                max_memory=placement.max_memory,
                trust_remote_code=trust_remote_code,
                hf_token=access_token,
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
        from ..core.cancellation import GPU_LEASE_LOST

        if cancelled.reason == GPU_LEASE_LOST:
            # NOT THE OPERATOR'S CANCEL (multi-GPU Phase 3, review round 2). The
            # download is placed BEFORE the transfer, so its lease can lapse at any
            # point of a multi-hour download, and the check after the download raises
            # this. The operator's branch below deletes the partial output and says
            # "Cancelled by user": here that was every downloaded weight. The job
            # stops as a FAILURE instead, keeps its files, and is offered for retry.
            message = (
                f"Stopped: {cancelled.detail or 'its GPU lease was lost'}. The files downloaded "
                "so far are kept; retry the download."
            )
            logger.error("Model download %s stopped: %s", model_id, cancelled.detail)
            _record_failed_download(
                self, model_id=model_id, message=message, placement=placement, repo_id=repo_id,
                quantization=quantization, trust_remote_code=trust_remote_code, gpu_request=gpu_request,
            )
            return {
                "status": "failed",
                "model_id": model_id,
                "reason": GPU_LEASE_LOST,
                "detail": cancelled.detail,
            }

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
        _record_failed_download(
            self, model_id=model_id, message=str(exc), placement=placement, repo_id=repo_id,
            quantization=quantization, trust_remote_code=trust_remote_code, gpu_request=gpu_request,
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

            # A SPLIT inspection model carries accelerate's dispatch hooks, which
            # hold references to its modules on every card. Removed before the
            # model is dropped; a model on one card has none. Its own try: a
            # failure here must not skip the release below.
            if model_obj is not None:
                try:
                    detach_dispatch_hooks(model_obj)
                except Exception as hook_exc:  # noqa: BLE001
                    logger.warning(f"Could not remove dispatch hooks from the inspection model: {hook_exc}")
            model_obj = None  # noqa: F841 - the assignment IS the release
            tokenizer = None  # noqa: F841
            config = None  # noqa: F841
            gc.collect()
            if torch.cuda.is_available():
                on_card = placement is not None and placement.device.type == "cuda"
                if on_card:
                    # EVERY card the load used. The caching allocator is per
                    # device, and `empty_cache()` with no device context acts on
                    # the current one — the first card of a split — which left
                    # the other card's share of the weights reserved until the
                    # worker restarted.
                    for placed in placement.all_devices:
                        with torch.cuda.device(placed):
                            torch.cuda.empty_cache()
                    reserved = "; ".join(
                        f"{torch.cuda.memory_reserved(placed) / 1024**3:.2f} GB on {placed}"
                        for placed in placement.all_devices
                    )
                else:
                    torch.cuda.empty_cache()
                    reserved = "n/a, never placed on a GPU"
                logger.info(
                    "Released the download-inspection model from GPU memory "
                    f"(reserved now {reserved})"
                )
        except Exception as cleanup_exc:  # noqa: BLE001
            # A failure here must not replace the real outcome — including a
            # propagating exception, which a raise from `finally` would discard.
            logger.warning(f"GPU cleanup after download failed: {cleanup_exc}")


def _record_failed_download(
    task,
    *,
    model_id: str,
    message: str,
    placement,
    repo_id: str,
    quantization: str,
    trust_remote_code: bool,
    gpu_request: Optional[str],
) -> None:
    """Record a download that did not finish: the model row, a retryable task_queue row, the UI.

    ONE writer for the task's failure and for a stop that is not the operator's (a lost
    GPU lease), so both offer the same retry with the same card. Never raises.
    """
    # Update database with error
    try:
        with task.get_db() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if model:
                model.status = ModelStatus.ERROR
                model.error_message = message
                db.commit()
    except Exception as db_exc:
        logger.error(f"Failed to update error state in database: {db_exc}")

    # Save failure state to task_queue for manual retry
    try:
        from ..models.task_queue import TaskQueue
        import uuid

        with task.get_db() as db:
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
                existing_entry.error_message = message
                existing_entry.task_id = task.request.id
                existing_entry.gpu_uuid = placement.uuid if placement else None
                existing_entry.gpu_uuids = (
                    placement.gpu_columns()["gpu_uuids"] if placement else None
                )
                db.commit()
                logger.info(f"Updated failed retry in task_queue: {existing_entry.id} (retry #{existing_entry.retry_count})")
            else:
                # This is an initial failure - create new entry
                task_queue_entry = TaskQueue(
                    id=f"tq_{uuid.uuid4().hex[:12]}",
                    task_id=task.request.id,
                    task_type="download",
                    entity_id=model_id,
                    entity_type="model",
                    status="failed",
                    progress=0.0,
                    error_message=message,
                    # The card it ran on, if it got that far — and every
                    # card, when the inspection load was split.
                    gpu_uuid=placement.uuid if placement else None,
                    gpu_uuids=placement.gpu_columns()["gpu_uuids"] if placement else None,
                    retry_params={
                        "repo_id": repo_id,
                        "quantization": quantization,
                        "trust_remote_code": trust_remote_code,
                        # A retry asks for the same card, not whatever
                        # the retry endpoint would default to.
                        "gpu_request": gpu_request,
                    },
                    retry_count=0,
                )
                db.add(task_queue_entry)
                db.commit()
                logger.info(f"Saved failed task to task_queue: {task_queue_entry.id}")
    except Exception as queue_exc:
        logger.error(f"Failed to save task to queue: {queue_exc}")

    # Send WebSocket notification of failure
    try:
        send_progress_update(
            model_id=model_id,
            progress=0.0,
            status="error",
            message=f"Download failed: {message}"
        )
    except Exception as emit_exc:  # noqa: BLE001 - the recorded failure is what matters
        logger.warning(f"Could not send the download failure for {model_id}: {emit_exc}")


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


def select_tokenization_for_model(tokenizations, model_id: str, dataset_id: str):
    """Pick the tokenization built with THIS model's tokenizer.

    WHY THIS IS NOT `tokenizations[0]`. It used to be, justified in a comment as
    "the most common case is one tokenization per dataset". A dataset tokenized
    for several models has several, the relationship is unordered, and whichever
    row Postgres returned first won — so an extraction could feed a model
    another tokenizer's ids. `activation_service` then CLAMPS out-of-range ids
    to `vocab_size - 1`, collapsing them onto a single token, with nothing but a
    per-sample warning to show for it.

    Observed in production, not hypothetical, and far worse than clamping.
    Measured 2026-09-11 over the 22 extractions on disk: **20 of them** ran
    against a tokenization built with a different model's tokenizer. Comparing
    the vocabularies id-by-id, agreement is **0.00%** for every pair involved —
    including LiquidAI/LFM2.5-1.2B-Instruct vs LiquidAI/LFM2.5-2.6B, which share
    a family name and share nothing else (2 ids agree out of 64,402 for
    granite-4.1-8b vs LFM2.5-1.2B; 1 of 100,352 for gemma-4-12B-it vs granite).

    So clamping is the *visible* symptom, not the damage. Decoding Bloomberg row
    0 from the LFM2.5-2.6B tokenization: the text written was

        'Ivory Coast Keeps Cocoa Export Tax Below 22%, Document Shows'

    and what LFM2.5-1.2B-Instruct actually read was

        '<|file_end|>\x1b129 express展<|reserved_330|>ショ<|reserved_63|> nood
         impression actividades<|reserved_219|> theseeten regression her iss'

    The clamped 3.7% is the only part that warned. The other 96.3% was in range,
    raised nothing, and decoded to different text. And when the tokenization's
    vocabulary is SMALLER than the model's (granite ids into gemma, LFM2.5-1.2B
    ids into granite) there is no clamping and therefore no warning at all — the
    silent case is the common one.

    Refuses rather than falling back, because the fallback IS the bug.
    """
    matching = [
        t for t in (tokenizations or [])
        if t.model_id == model_id and t.tokenized_path
    ]
    if not matching:
        available = sorted({t.model_id for t in (tokenizations or [])})
        raise ActivationExtractionError(
            f"Dataset {dataset_id} has no usable tokenization for model {model_id}. "
            f"Tokenized for: {available}. Refusing to extract against another "
            f"model's token ids — they would be clamped into range and the "
            f"activations would be silently wrong."
        )
    # Deterministic when one model has several max_lengths: take the longest,
    # which is the most context the extraction can use. Tie-broken on id so the
    # choice never depends on row order.
    chosen = max(matching, key=lambda t: (t.max_length or 0, t.id))
    if len(matching) > 1:
        logger.info(
            "Dataset %s has %d tokenizations for model %s; using max_length=%s",
            dataset_id, len(matching), model_id, chosen.max_length,
        )
    return chosen


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
                    phase=ExtractionPhase.EXTRACTING.value,
                )
        except Exception as db_e:
            logger.warning(f"Failed to update extraction progress in database: {db_e}")

        # Emit WebSocket update
        emit_extraction_progress(
            model_id=model_id,
            extraction_id=extraction_id,
            progress=extraction_progress,
            status="extracting",
            message=f"Processing samples: {samples_processed}/{total_samples}",
            phase=ExtractionPhase.EXTRACTING.value,
        )

    return on_extraction_progress


#: How often a post-GPU phase (merging, statistics) touches the extraction row.
#: The activation janitor treats a started row as abandoned once `updated_at` is
#: 600 s old and fails it once the row is an hour stale, so anything well under
#: 600 s keeps a live phase visibly alive.
STATISTICS_HEARTBEAT_SECONDS = 60
MERGE_HEARTBEAT_SECONDS = 60


def write_extraction_phase(extraction_id: str, phase: str) -> bool:
    """Record which step a running extraction is in, and move its clock.

    THROUGH `record_progress`, NOT A DIRECT WRITE. A phase is progress, and the
    guard refuses any progress write onto a CANCELLED, FAILED or COMPLETED row.
    A heartbeat already in flight when the operator presses Stop therefore
    cannot make a cancelled row read "merging" again.

    `updated_at` is written explicitly. SQLAlchemy emits no UPDATE for an
    attribute set to the value it already holds, so a heartbeat repeating
    "merging" would otherwise leave the janitor's clock where it was.

    Returns whether the row was written.
    """
    if phase not in EXTRACTION_PHASES:
        raise ValueError(
            f"unknown extraction phase {phase!r}; expected one of {sorted(EXTRACTION_PHASES)}"
        )
    return record_progress(
        "activation_extraction",
        extraction_id,
        phase=phase,
        updated_at=datetime.now(timezone.utc),
    )


def _build_phase_heartbeat(
    model_id: str,
    extraction_id: str,
    cancelled,
    *,
    phase: str,
    describe,
    stopped,
    clock,
    interval_s: float,
):
    """One post-GPU phase's heartbeat: `(layer_index, n_layers, done, total)`.

    Every call polls cancellation (the checker throttles its own reads). The
    FIRST call writes the phase at once, because it is the transition the UI
    must show; later calls write it and emit a WebSocket message at most once
    per `interval_s`, on TIME, since one unit of work can take microseconds or
    minutes depending on the layer.
    """
    last = {"at": None}

    def on_phase_progress(layer_index: int, n_layers: int, done: int, total: int):
        cancelled.raise_if_cancelled(stopped(layer_index, n_layers))

        now = clock()
        if last["at"] is not None and now - last["at"] < interval_s:
            return
        last["at"] = now

        fraction = (layer_index + done / max(total, 1)) / max(n_layers, 1)
        write_extraction_phase(extraction_id, phase)
        try:
            emit_extraction_progress(
                model_id=model_id,
                extraction_id=extraction_id,
                progress=90.0,
                status="extracting",
                message=describe(layer_index, n_layers, fraction),
                phase=phase,
            )
        except Exception as emit_e:
            logger.warning(f"Failed to emit {phase} progress: {emit_e}")

    return on_phase_progress


def build_merge_heartbeat(
    model_id: str,
    extraction_id: str,
    cancelled,
    *,
    clock=time.monotonic,
    interval_s: float = MERGE_HEARTBEAT_SECONDS,
):
    """Liveness, cancellation and the "merging" phase for the shard merge.

    After the GPU pass reaches N/N, every layer's per-batch temporary files are
    copied into one memory-mapped file. That is disk-bound and, on a 10,000 x
    2,048-token extraction, long. The per-sample callback is silent by then, so
    without this the row is untouched for the whole merge: the UI reloads to
    "Extracting N/N", and the janitor sees a row going stale.
    """
    return _build_phase_heartbeat(
        model_id,
        extraction_id,
        cancelled,
        phase=ExtractionPhase.MERGING.value,
        describe=lambda layer, n_layers, fraction: (
            f"Merging batch files: layer {layer + 1} of {n_layers} "
            f"({fraction * 100:.0f}% of the merge)"
        ),
        stopped=lambda layer, n_layers: (
            f"stopped merging batch files at layer {layer + 1} of {n_layers}"
        ),
        clock=clock,
        interval_s=interval_s,
    )


def build_statistics_heartbeat(
    model_id: str,
    extraction_id: str,
    cancelled,
    *,
    clock=time.monotonic,
    interval_s: float = STATISTICS_HEARTBEAT_SECONDS,
):
    """Liveness, cancellation and the "statistics" phase for the statistics.

    FOUND ON A LIVE JOB, 2026-09-12. The per-sample progress callback is the
    only thing that writes the extraction row, and it stops when the GPU pass
    ends. What follows — statistics over 10,000 x 2,048 x 2,048 float16 values
    per layer — ran 85 minutes with the row untouched. Celery reports PENDING
    for these tasks, so the janitor falls back to the row's age: at the first
    sweep an hour after the last sample it would have failed a healthy job and
    told the UI so, inviting a delete of 390 GB of valid output mid-run. It was
    kept alive by hand.

    `cancelled.raise_if_cancelled` runs on every call — the checker throttles
    its own database reads — so Stop works during this phase too. The row write
    and the WebSocket message are throttled here, on TIME: a chunk can take
    seconds on a large layer or microseconds on a small one.
    """
    return _build_phase_heartbeat(
        model_id,
        extraction_id,
        cancelled,
        phase=ExtractionPhase.STATISTICS.value,
        describe=lambda layer, n_layers, fraction: (
            f"Computing statistics: layer {layer + 1} of {n_layers} "
            f"({fraction * 100:.0f}% of the statistics phase)"
        ),
        stopped=lambda layer, n_layers: (
            f"stopped computing statistics at layer {layer + 1} of {n_layers}"
        ),
        clock=clock,
        interval_s=interval_s,
    )


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="workers.model_tasks.extract_activations",
    max_retries=2,
    default_retry_delay=60,
    queue="extraction",
)
@gpu_job("activation_extraction")
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
    # Both were threaded into the service and set by nothing, so the token
    # budget was permanently its default and could not be tuned, and truncation
    # could not be turned on at all.
    max_seq_length: Optional[int] = None,
    micro_batch_token_budget: Optional[int] = None,
    extraction_id: Optional[str] = None,
    # "auto" or a GPU UUID, as the endpoint resolved it and stored on the row.
    # None — a message queued before this argument existed — means auto.
    gpu_request: Optional[str] = None,
    # LEGACY, IGNORED. Messages queued before 2026-09-13 carry an NVML index
    # here. Accepted so they do not fail on an unknown keyword, and not used:
    # index 0 named the 3090 when they were queued and names the 3080 Ti now.
    gpu_id: Optional[int] = None,
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
        gpu_request: "auto" or a GPU UUID. The job is placed on a card when it
            STARTS, against live free memory; a card that is missing or cannot
            take the job fails it with the reason, and it is not retried.
        gpu_id: Legacy NVML index from messages queued before 2026-09-13; ignored.

    Returns:
        dict with extraction metadata

    Raises:
        ActivationExtractionError: If extraction fails
        OutOfMemoryError: If GPU runs out of memory
    """
    # Bound before the try: the finally releases memory on the placed card, and
    # a placement that failed placed nothing.
    placement = None
    try:
        logger.info(
            f"Starting activation extraction for model {model_id}, "
            f"dataset {dataset_id}, layers {layer_indices}, hooks {hook_types}, "
            f"gpu_request={gpu_request or AUTO}"
        )
        if gpu_id is not None:
            logger.warning(
                "Ignoring legacy gpu_id=%s on extraction %s: an NVML index names a "
                "different card since the second GPU was added. Placing on %r.",
                gpu_id, extraction_id, gpu_request or AUTO,
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

            tokenization = select_tokenization_for_model(
                dataset.tokenizations, model_id, dataset_id
            )

            # Get tokenized path - resolve to absolute using settings helper
            dataset_path = str(settings.resolve_data_path(tokenization.tokenized_path))

        # Validate model path exists
        if not Path(model_file_path).exists():
            raise ActivationExtractionError(f"Model path not found: {model_file_path}")

        if not Path(dataset_path).exists():
            raise ActivationExtractionError(f"Dataset path not found: {dataset_path}")

        # PLACE THE JOB BEFORE ANYTHING TOUCHES A GPU.
        #
        # This took `gpu_id: int = 0` and used it as a CUDA index, so every
        # extraction ran on index 0 — the 3080 Ti since 2026-09-13, whatever card
        # was chosen. `place_job` picks against live free memory (Auto) or
        # honours the named card, and makes it current so code that asks CUDA
        # for "the current device" (bitsandbytes) lands on it too. A card that
        # cannot be used raises GpuPlacementError: the job fails with that
        # message and is neither retried nor moved to another card.
        #
        # SIZED FROM THE MODEL'S OWN CONFIG, AND SPLIT WHEN NO ONE CARD HOLDS IT
        # (Multi-GPU Phase 2). Placed unsized, Auto took the most-free card and
        # could never split, so a model larger than every card was loaded onto
        # the 3090 and died out of memory. The size is read first, which is why
        # the model and dataset rows are looked up above; neither touches a GPU.
        #
        # `allow_shard=True` because this path runs split: the loader spreads
        # the model over the placement's cards within a GPU-only budget and
        # refuses any CPU or disk offload, inputs go to the embedding's card,
        # each hook copies its layer's output to the CPU from whichever card
        # the layer ran on, and memory is measured and released on every card.
        # A model that fits one card still gets one card; "all" splits across
        # every card, as asked.
        required_mb = required_mb_for_load(
            resolve_model_snapshot(model_file_path),
            load_format_for(model_quantization).value,
            local_files_only=True,
        )
        placement = place_job(gpu_request or AUTO, required_mb=required_mb, allow_shard=True)
        logger.info(
            f"[Extraction {extraction_id}] placed on {placement.describe()} as "
            f"{' + '.join(str(placed) for placed in placement.all_devices)} "
            f"(requested {gpu_request or AUTO!r}, "
            f"~{'unknown' if required_mb is None else f'{required_mb:,.0f}'} MB)"
        )

        # Recorded and committed BEFORE the model loads, so a job that dies
        # loading still says which card(s) it died on. A job that mints its own
        # id records them when it creates its row, below.
        if extraction_id:
            with self.get_db() as _db:
                ExtractionDatabaseService.record_gpu_uuid(
                    _db, extraction_id, **placement.gpu_columns()
                )

        # Pre-task GPU memory check - ensure clean state before loading model,
        # on EVERY card the job will use: a split loads onto each of them.
        import torch
        placed_cards = [placed for placed in placement.all_devices if placed.type == "cuda"]
        for placed in placed_cards:
            allocated_before = torch.cuda.memory_allocated(placed) / (1024 ** 3)  # GB
            reserved_before = torch.cuda.memory_reserved(placed) / (1024 ** 3)    # GB
            logger.info(
                f"[Pre-extraction {self.request.id}] {placed} memory before cleanup: "
                f"Allocated={allocated_before:.2f} GB, Reserved={reserved_before:.2f} GB"
            )

        if placed_cards:
            # Force cleanup in case previous task didn't complete cleanup
            for placed in placed_cards:
                with torch.cuda.device(placed):
                    torch.cuda.empty_cache()
            import gc
            gc.collect()

            for placed in placed_cards:
                allocated_after = torch.cuda.memory_allocated(placed) / (1024 ** 3)
                reserved_after = torch.cuda.memory_reserved(placed) / (1024 ** 3)
                logger.info(
                    f"[Pre-extraction {self.request.id}] {placed} memory after cleanup: "
                    f"Allocated={allocated_after:.2f} GB, Reserved={reserved_after:.2f} GB"
                )

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
                    # Nor is the previous attempt's step where this one is.
                    existing.phase = None
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
                        # Not max_seq_length / micro_batch_token_budget: the row
                        # has no such columns and create_extraction takes no
                        # such arguments. Passing them raised a TypeError that
                        # the handler below swallowed as a warning, so this
                        # branch created no row at all.
                        celery_task_id=self.request.id,
                        gpu_request=gpu_request or AUTO,
                        gpu_uuid=placement.uuid,
                        gpu_uuids=placement.gpu_columns()["gpu_uuids"],
                    )
                    logger.info(
                        f"Created database record for extraction {extraction_id} "
                        f"({placement.describe()})"
                    )
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
            # The per-sample callback goes quiet when the GPU pass ends; these
            # carry liveness, Stop and the persisted phase through the shard
            # merge and the statistics after it.
            on_merge_progress = build_merge_heartbeat(
                model_id, extraction_id, cancelled
            )
            on_statistics_progress = build_statistics_heartbeat(
                model_id, extraction_id, cancelled
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
                max_seq_length=max_seq_length,
                micro_batch_token_budget=micro_batch_token_budget,
                extraction_id=extraction_id,
                progress_callback=on_extraction_progress,
                # The whole placement: a split's loader needs its device_map
                # and max_memory, and its cleanup needs every card.
                placement=placement,
                merge_progress_callback=on_merge_progress,
                statistics_progress_callback=on_statistics_progress,
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
                        # These three were missing, so a retried attempt lost
                        # its sequence length, its token budget and the card it
                        # was asked to run on (the task's own defaults applied).
                        "max_seq_length": max_seq_length,
                        "micro_batch_token_budget": micro_batch_token_budget,
                        "gpu_request": gpu_request,
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
        # A RETRY IS NOT A FAILURE, AND IT HAS ALREADY BEEN SENT.
        #
        # `self.retry()` queues the new attempt and THEN raises celery's
        # `Retry`, which is an `Exception`. The OOM back-off above raised it
        # straight into this handler, which marked the extraction FAILED,
        # emitted a failure, and called `self.retry()` a SECOND time with the
        # original batch size: two queued attempts for one extraction id, one
        # of them not halved.
        from celery.exceptions import Retry

        if isinstance(exc, Retry):
            raise
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
        # A card that is missing or cannot take the job is not transient either:
        # a named card is honoured or refused, never retried into a different
        # answer, and the message already names the cards and their free memory.
        if (
            isinstance(exc, PermanentExtractionError)
            or isinstance(exc, GpuPlacementError)
            or "not found in database" in str(exc)
        ):
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
                    # Missing until 2026-09-13: a retried attempt ran with the
                    # task's defaults for these, not what was asked for.
                    "max_seq_length": max_seq_length,
                    "micro_batch_token_budget": micro_batch_token_budget,
                    "gpu_request": gpu_request,
                    "extraction_id": extraction_id,  # Preserve extraction_id
                }
            )

        raise

    finally:
        # CRITICAL: Ensure GPU cache is cleared even if service cleanup didn't work
        # This is a safety net for sequential extraction jobs. It acts on the
        # card the job was placed on; a job that never placed touched no card.
        import torch
        if placement is not None and placement.device.type == "cuda":
            import gc
            gc.collect()
            # EVERY card the job was placed on: a split held weights on each,
            # and the allocator's cache is per device.
            for placed in placement.all_devices:
                with torch.cuda.device(placed):
                    torch.cuda.empty_cache()

                allocated_final = torch.cuda.memory_allocated(placed) / (1024 ** 3)
                reserved_final = torch.cuda.memory_reserved(placed) / (1024 ** 3)
                logger.info(
                    f"[Post-extraction {self.request.id}] {placed} memory after task cleanup: "
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
