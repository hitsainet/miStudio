"""
Celery tasks for dataset operations.

This module contains background tasks for downloading, processing,
and tokenizing datasets with real-time progress updates via WebSocket.
"""

from datetime import datetime, UTC
from pathlib import Path
from typing import Optional
from uuid import UUID
import signal
import os
import psutil
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# NOTE: Do NOT import load_dataset at module level!
# It must be imported inside the task function AFTER patching tqdm
# Otherwise, datasets will import tqdm before we can patch it
from sqlalchemy.orm import Session
import redis

from ..core.celery_app import celery_app
from ..core.config import settings
from ..models.dataset import DatasetStatus, Dataset
from ..models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from ..models.model import Model
from ..schemas.dataset import DatasetUpdate
from ..services.dataset_service import DatasetService
from ..services.tokenization_service import TokenizationService
from .base_task import DatabaseTask
from .websocket_emitter import emit_dataset_progress, emit_tokenization_progress
from .tqdm_websocket_bridge import create_tqdm_websocket_callback
import time


def cleanup_child_processes():
    """
    Terminate all child processes of the current process.

    This is crucial for cleaning up multiprocessing workers spawned by
    HuggingFace datasets when a tokenization task is cancelled.
    """
    try:
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)

        if children:
            print(f"[CLEANUP] Terminating {len(children)} child processes")
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            # Wait for processes to terminate gracefully
            gone, alive = psutil.wait_procs(children, timeout=3)

            # Force kill any remaining processes
            if alive:
                print(f"[CLEANUP] Force killing {len(alive)} remaining processes")
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
    except Exception as e:
        print(f"[CLEANUP] Error cleaning up child processes: {e}")


def release_redis_lock(dataset_id: str, redis_client: redis.Redis = None) -> None:
    """
    Release Redis distributed lock for a dataset operation.

    Args:
        dataset_id: Dataset UUID
        redis_client: Redis client (will create if None)
    """
    try:
        if redis_client is None:
            redis_url = str(settings.redis_url)
            redis_client = redis.from_url(redis_url, decode_responses=True)

        lock_key = f"tokenization_lock:{dataset_id}"
        deleted = redis_client.delete(lock_key)
        if deleted:
            print(f"[LOCK] Released tokenization lock for dataset {dataset_id}")
        else:
            print(f"[LOCK] No lock found for dataset {dataset_id} (may have expired)")
    except Exception as e:
        print(f"[LOCK] Warning: Failed to release lock for dataset {dataset_id}: {e}")


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.dataset_tasks.download_dataset_task",
    max_retries=0,  # No auto-retry - user must manually retry
)
def download_dataset_task(
    self,
    dataset_id: str,
    repo_id: str,
    access_token: Optional[str] = None,
    split: Optional[str] = None,
    config: Optional[str] = None,
):
    """
    Download dataset from HuggingFace Hub.

    Args:
        dataset_id: Dataset UUID
        repo_id: HuggingFace repository ID (e.g., 'roneneldan/TinyStories')
        access_token: HuggingFace access token for gated datasets
        split: Dataset split to download (e.g., 'train', 'validation', 'test')
        config: Dataset configuration name (e.g., 'en', 'zh') for datasets with multiple configs

    Returns:
        dict: Download result with dataset path and statistics
    """
    import logging
    logger = logging.getLogger(__name__)

    # Debug: Log token presence in celery task
    token_provided = bool(access_token and access_token.strip())
    logger.info(f"[Dataset Task] Starting download: repo={repo_id}, token_provided={token_provided}, "
                f"token_length={len(access_token) if access_token else 0}")
    print(f"[Dataset Task] Starting download: repo={repo_id}, token_provided={token_provided}, "
          f"token_length={len(access_token) if access_token else 0}")

    try:
        dataset_uuid = UUID(dataset_id)

        # Update status to downloading
        with self.get_db() as db:
            self.update_progress(
                db=db,
                model_class=Dataset,
                record_id=dataset_id,
                progress=0.0,
                status=DatasetStatus.DOWNLOADING.value,
            )

        emit_dataset_progress(
            dataset_id,
            "progress",
            {
                "dataset_id": dataset_id,
                "progress": 0.0,
                "status": "downloading",
                "message": f"Starting download of {repo_id}...",
            },
        )

        # Prepare download directory using settings
        data_dir = settings.datasets_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        raw_path = data_dir / repo_id.replace("/", "_")

        # Download dataset from HuggingFace
        emit_dataset_progress(
            dataset_id,
            "progress",
            {
                "dataset_id": dataset_id,
                "progress": 10.0,
                "status": "downloading",
                "message": "Downloading from HuggingFace Hub...",
            },
        )

        # Monkey-patch tqdm to emit WebSocket progress during download
        # Maps HuggingFace's tqdm (0-100%) to our progress range (10-70%)
        TqdmWebSocket = create_tqdm_websocket_callback(
            dataset_id=dataset_id,
            base_progress=10.0,
            progress_range=60.0,  # 10% → 70%
            throttle_seconds=0.5  # Emit at most every 0.5 seconds
        )

        # Patch tqdm at multiple locations where datasets might use it
        import sys
        from tqdm import tqdm as original_tqdm

        # Save original tqdm
        sys.modules['tqdm'].tqdm = TqdmWebSocket
        sys.modules['tqdm.auto'].tqdm = TqdmWebSocket

        # Configure HuggingFace Hub timeouts BEFORE importing load_dataset
        # Default is 10 seconds which is too short for large datasets
        import huggingface_hub.constants
        huggingface_hub.constants.DEFAULT_DOWNLOAD_TIMEOUT = 300  # 5 minutes
        huggingface_hub.constants.DEFAULT_REQUEST_TIMEOUT = 300  # 5 minutes
        huggingface_hub.constants.DEFAULT_ETAG_TIMEOUT = 300  # 5 minutes

        # NOW import load_dataset - it will use our patched tqdm and extended timeouts
        from datasets import load_dataset

        try:
            dataset = load_dataset(
                repo_id,
                name=config,
                split=split,
                cache_dir=str(data_dir),
                token=access_token,
                trust_remote_code=True,
            )
        finally:
            # Restore original tqdm regardless of success/failure
            sys.modules['tqdm'].tqdm = original_tqdm
            sys.modules['tqdm.auto'].tqdm = original_tqdm

        # Update progress: saving to disk
        emit_dataset_progress(
            dataset_id,
            "progress",
            {
                "dataset_id": dataset_id,
                "progress": 70.0,
                "status": "downloading",
                "message": "Saving dataset to disk...",
            },
        )

        # Save dataset to our organized location
        dataset.save_to_disk(str(raw_path))

        # Optional: Clean up temporary download cache
        if settings.auto_cleanup_after_download:
            import shutil
            import logging
            logger = logging.getLogger(__name__)

            try:
                # Clean up downloads cache (temporary download chunks)
                downloads_dir = data_dir / "downloads"
                if downloads_dir.exists():
                    downloads_size = sum(f.stat().st_size for f in downloads_dir.rglob('*') if f.is_file())
                    shutil.rmtree(downloads_dir)
                    logger.info(f"Cleaned up downloads cache: {downloads_dir} ({downloads_size / 1024**3:.2f} GB)")

                # Clean up HF cache format (vietgpt___openwebtext_en vs vietgpt_openwebtext_en)
                hf_cache_dir = data_dir / repo_id.replace("/", "___")
                if hf_cache_dir.exists() and hf_cache_dir != raw_path:
                    hf_cache_size = sum(f.stat().st_size for f in hf_cache_dir.rglob('*') if f.is_file())
                    shutil.rmtree(hf_cache_dir)
                    logger.info(f"Cleaned up HF cache: {hf_cache_dir} ({hf_cache_size / 1024**3:.2f} GB)")

            except Exception as e:
                logger.warning(f"Cleanup failed (non-critical): {e}")

        # Update progress
        emit_dataset_progress(
            dataset_id,
            "progress",
            {
                "dataset_id": dataset_id,
                "progress": 90.0,
                "status": "downloading",
                "message": "Download complete, processing metadata...",
            },
        )

        # Calculate dataset statistics
        num_samples = len(dataset) if hasattr(dataset, "__len__") else None
        size_bytes = dataset.size_in_bytes if hasattr(dataset, "size_in_bytes") else None

        # Update dataset record with download results
        with self.get_db() as db:
            try:
                dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
                if not dataset_obj:
                    raise ValueError(f"Dataset {dataset_id} not found")

                dataset_obj.status = DatasetStatus.READY
                dataset_obj.progress = 100.0
                dataset_obj.raw_path = str(raw_path)
                dataset_obj.num_samples = num_samples
                dataset_obj.size_bytes = size_bytes

                db.commit()
                db.refresh(dataset_obj)

                # Only emit "completed" event after successful database commit
                emit_dataset_progress(
                    dataset_id,
                    "completed",
                    {
                        "dataset_id": dataset_id,
                        "progress": 100.0,
                        "status": "ready",
                        "message": "Dataset downloaded successfully",
                        "num_samples": num_samples,
                        "size_bytes": size_bytes,
                    },
                )

            except Exception as commit_error:
                # Rollback transaction on failure
                db.rollback()
                error_msg = f"Failed to save download results: {str(commit_error)}"
                print(f"Database commit error: {error_msg}")

                # Emit error event instead of success
                emit_dataset_progress(
                    dataset_id,
                    "error",
                    {
                        "dataset_id": dataset_id,
                        "status": "error",
                        "message": error_msg,
                    },
                )

                # Re-raise to trigger outer error handler
                raise

        return {
            "dataset_id": dataset_id,
            "status": "ready",
            "raw_path": str(raw_path),
            "num_samples": num_samples,
            "size_bytes": size_bytes,
        }

    except Exception as e:
        # Handle errors with better messages for common issues
        error_str = str(e)

        # Detect gated repository errors (401 Unauthorized)
        if "401" in error_str or "Unauthorized" in error_str or "gated" in error_str.lower():
            error_message = (
                f"Access denied for {repo_id}. This dataset requires you to accept its license agreement "
                f"on HuggingFace's website first. Visit https://huggingface.co/datasets/{repo_id} and click "
                f"'Agree and access repository', then retry with your token."
            )
        # Detect invalid token
        elif "Invalid token" in error_str or "Invalid credentials" in error_str:
            error_message = (
                f"Invalid HuggingFace token. Please check your token at https://huggingface.co/settings/tokens "
                f"and ensure it has 'read' permissions."
            )
        # Detect repository not found
        elif "404" in error_str or "not found" in error_str.lower():
            error_message = f"Dataset '{repo_id}' not found. Please check the repository ID is correct."
        else:
            error_message = f"Download failed: {error_str}"

        print(f"Dataset download error: {error_message}")

        with self.get_db() as db:
            self.update_progress(
                db=db,
                model_class=Dataset,
                record_id=dataset_id,
                progress=None,
                status=DatasetStatus.ERROR.value,
                error_message=error_message,
            )

        emit_dataset_progress(
            dataset_id,
            "error",
            {
                "dataset_id": dataset_id,
                "status": "error",
                "message": error_message,
            },
        )

        # Save failure state to task_queue for manual retry
        try:
            from ..models.task_queue import TaskQueue
            import uuid

            with self.get_db() as db:
                # Check if there's an existing queued task_queue entry for this entity
                existing_entry = db.query(TaskQueue).filter_by(
                    entity_id=dataset_id,
                    entity_type="dataset",
                    task_type="download"
                ).filter(
                    TaskQueue.status.in_(["queued", "running"])
                ).first()

                if existing_entry:
                    # This is a retry that failed - update the existing entry
                    existing_entry.status = "failed"
                    existing_entry.error_message = error_message
                    existing_entry.task_id = self.request.id
                    db.commit()
                    print(f"Updated failed retry in task_queue: {existing_entry.id} (retry #{existing_entry.retry_count})")
                else:
                    # This is an initial failure - create new entry
                    task_queue_entry = TaskQueue(
                        id=f"tq_{uuid.uuid4().hex[:12]}",
                        task_id=self.request.id,
                        task_type="download",
                        entity_id=dataset_id,
                        entity_type="dataset",
                        status="failed",
                        progress=0.0,
                        error_message=error_message,
                        retry_params={
                            "repo_id": repo_id,
                            "access_token": access_token,
                            "split": split,
                            "config": config,
                        },
                        retry_count=0,
                    )
                    db.add(task_queue_entry)
                    db.commit()
                    print(f"Saved failed dataset download to task_queue: {task_queue_entry.id}")
        except Exception as queue_exc:
            print(f"Failed to save task to queue: {queue_exc}")

        raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.dataset_tasks.tokenize_dataset_task",
    max_retries=0,  # No auto-retry - user must manually retry
)
def tokenize_dataset_task(
    self,
    dataset_id: str,
    model_id: str,
    max_length: int = 512,
    stride: int = 0,
    padding: str = "max_length",
    truncation: str = "longest_first",
    add_special_tokens: bool = True,
    return_attention_mask: bool = True,
    enable_cleaning: bool = True,
):
    """
    Tokenize dataset using model's tokenizer.

    Args:
        dataset_id: Dataset UUID
        model_id: Model ID whose tokenizer will be used
        max_length: Maximum sequence length
        stride: Sliding window stride for long sequences
        padding: Padding strategy ('max_length', 'longest', or 'do_not_pad')
        truncation: Truncation strategy ('longest_first', 'only_first', 'only_second', or 'do_not_truncate')
        add_special_tokens: Add special tokens (BOS, EOS, PAD, etc.)
        return_attention_mask: Return attention mask
        enable_cleaning: Enable text cleaning (removes HTML tags, control characters, excessive punctuation, normalizes Unicode)

    Returns:
        dict: Tokenization result with statistics
    """
    # Graceful shutdown state - allows completion if we're past the critical tokenization point
    graceful_shutdown_requested = False
    tokenization_complete = False
    saved_tokenized_dataset = None
    saved_tokenized_path = None

    # Register signal handlers for proper cleanup of child processes
    def signal_handler(signum, frame):
        nonlocal graceful_shutdown_requested
        print(f"[SIGNAL] Received signal {signum}")
        graceful_shutdown_requested = True

        # If tokenization is already complete, let it finish saving
        if tokenization_complete:
            print(f"[SIGNAL] Tokenization complete, allowing graceful save...")
            return  # Don't raise, let the task finish saving

        # Otherwise, clean up and terminate
        print(f"[SIGNAL] Tokenization in progress, cleaning up child processes...")
        cleanup_child_processes()
        raise SystemExit(f"Task terminated by signal {signum}")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize variables that might be referenced in exception handler
    dataset_uuid = None
    tokenization_id = None

    try:
        dataset_uuid = UUID(dataset_id)

        # Create tokenization ID: tok_{dataset_id}_{model_id}_{max_length}
        # Include max_length in ID to allow multiple tokenizations per dataset+model
        tokenization_id = f"tok_{str(dataset_uuid).replace('-', '')}_{model_id}_{max_length}"

        # Track start time for progress estimates
        start_time = time.time()
        started_at = datetime.now(UTC).isoformat()

        # Get dataset, model, and create/get tokenization record
        with self.get_db() as db:
            dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
            if not dataset_obj:
                raise ValueError(f"Dataset {dataset_id} not found")

            if not dataset_obj.raw_path:
                raise ValueError(f"Dataset {dataset_id} has no raw_path - download dataset first")

            model_obj = db.query(Model).filter_by(id=model_id).first()
            if not model_obj:
                raise ValueError(f"Model {model_id} not found")

            tokenizer_name = model_obj.repo_id
            # Get model's local cache directory for tokenizer files
            # Resolve relative path to absolute using settings.data_dir
            if model_obj.file_path:
                resolved_path = settings.resolve_data_path(model_obj.file_path)
                # Check if file_path is a HF cache directory (contains snapshots/ subdirectory)
                # or starts with "models--". In that case, use its parent as cache_dir.
                # HuggingFace expects cache_dir to be the PARENT of "models--{org}--{model}"
                if (resolved_path / "snapshots").exists() or resolved_path.name.startswith("models--"):
                    model_cache_dir = str(resolved_path.parent)
                    print(f"[TOKENIZER] Detected HF cache format, using parent as cache_dir")
                else:
                    model_cache_dir = str(resolved_path)
            else:
                model_cache_dir = None
            print(f"Using tokenizer from model {model_id}: {tokenizer_name}")
            print(f"Model cache directory: {model_cache_dir}")

            # Check if tokenization already exists
            tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()

            if tokenization_obj:
                # Task-level deduplication: Check if already successfully tokenized
                if (tokenization_obj.status == TokenizationStatus.READY and
                    tokenization_obj.tokenized_path and
                    settings.resolve_data_path(tokenization_obj.tokenized_path).exists()):
                    print(f"[DEDUP] Tokenization {tokenization_id} already complete, skipping duplicate task")
                    return {
                        'tokenization_id': tokenization_id,
                        'dataset_id': dataset_id,
                        'model_id': model_id,
                        'status': 'skipped',
                        'reason': 'already_tokenized',
                        'tokenized_path': tokenization_obj.tokenized_path
                    }
            else:
                # Create new tokenization record
                tokenization_obj = DatasetTokenization(
                    id=tokenization_id,
                    dataset_id=dataset_uuid,
                    model_id=model_id,
                    max_length=max_length,
                    tokenizer_repo_id=tokenizer_name,
                    status=TokenizationStatus.QUEUED,
                    progress=0.0,
                    celery_task_id=self.request.id,
                )
                db.add(tokenization_obj)
                db.commit()
                print(f"Created new tokenization record: {tokenization_id} (max_length={max_length})")

        # Update status to processing (both tokenization and parent dataset)
        with self.get_db() as db:
            tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
            if tokenization_obj:
                tokenization_obj.status = TokenizationStatus.PROCESSING
                tokenization_obj.progress = 0.0

            # Also update parent dataset status to PROCESSING
            dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
            if dataset_obj:
                dataset_obj.status = DatasetStatus.PROCESSING
                dataset_obj.progress = 0.0
                dataset_obj.error_message = None  # Clear any previous error on new attempt

            db.commit()

        # Update Celery task state: Starting
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'percent': 0.0,
                'status': 'Starting tokenization...'
            }
        )

        # Get dataset from database to retrieve raw_path and filter config
        with self.get_db() as db:
            dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
            if not dataset_obj:
                raise ValueError(f"Dataset {dataset_id} not found")
            if not dataset_obj.raw_path:
                raise ValueError(f"Dataset {dataset_id} has no raw_path")
            raw_path = str(settings.resolve_data_path(dataset_obj.raw_path))
            # Load filter configuration from tokenization object (per-job config)
            tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
            if tokenization_obj:
                filter_enabled = dataset_obj.tokenization_filter_enabled  # Still from dataset for backward compat
                filter_mode = dataset_obj.tokenization_filter_mode
                filter_threshold = dataset_obj.tokenization_junk_ratio_threshold
                # Load new per-tokenization filter settings
                remove_all_punctuation = tokenization_obj.remove_all_punctuation
                custom_filter_chars = tokenization_obj.custom_filter_chars
            else:
                # Fallback to dataset-level settings
                filter_enabled = dataset_obj.tokenization_filter_enabled
                filter_mode = dataset_obj.tokenization_filter_mode
                filter_threshold = dataset_obj.tokenization_junk_ratio_threshold
                remove_all_punctuation = False
                custom_filter_chars = None

            # DEBUG: Log the loaded filter configuration values
            logger.info(
                f"[FILTER_CONFIG] Loaded from DB - "
                f"filter_enabled={filter_enabled} (type={type(filter_enabled).__name__}), "
                f"filter_mode={filter_mode}, "
                f"filter_threshold={filter_threshold}, "
                f"remove_all_punctuation={remove_all_punctuation}, "
                f"custom_filter_chars={custom_filter_chars}"
            )

        # Update Celery task state: Loading tokenizer
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 10,
                'total': 100,
                'percent': 10.0,
                'status': 'Loading tokenizer...'
            }
        )

        # Load tokenizer from model's local cache (downloaded with model)
        # This avoids needing HuggingFace auth for gated models
        tokenizer = TokenizationService.load_tokenizer(
            tokenizer_name,
            cache_dir=model_cache_dir
        )

        # Update Celery task state: Loading dataset
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 20,
                'total': 100,
                'percent': 20.0,
                'status': 'Loading dataset...'
            }
        )

        # Update tokenization progress
        with self.get_db() as db:
            tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
            if tokenization_obj:
                tokenization_obj.progress = 20.0
                db.commit()

        # Load dataset from disk
        dataset = TokenizationService.load_dataset_from_disk(raw_path)

        # Emit detailed progress: loading stage complete
        elapsed = time.time() - start_time
        emit_tokenization_progress(
            dataset_id=dataset_id,
            tokenization_id=tokenization_id,
            progress=20.0,
            stage="loading",
            samples_processed=0,
            total_samples=len(dataset),
            started_at=started_at,
            elapsed_seconds=elapsed,
        )

        # Handle DatasetDict (multi-split datasets) - use 'train' split by default
        from datasets import DatasetDict
        if isinstance(dataset, DatasetDict):
            # Try to get train split, or first available split
            if 'train' in dataset:
                dataset = dataset['train']
            else:
                # Get first available split
                first_split = next(iter(dataset.keys()))
                dataset = dataset[first_split]

        # Analyze dataset schema to determine best text column
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 30,
                'total': 100,
                'percent': 30.0,
                'status': 'Analyzing dataset schema...'
            }
        )

        with self.get_db() as db:
            tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
            if tokenization_obj:
                tokenization_obj.progress = 30.0
                db.commit()

        schema_info = TokenizationService.analyze_dataset_schema(dataset)

        # Log schema information (detailed for debugging)
        print(f"Dataset schema analysis:")
        print(f"  Available text columns: {schema_info['text_columns']}")
        print(f"  List/sequence columns: {schema_info.get('list_columns', [])}")
        print(f"  Conversation columns detected: {len(schema_info.get('conversation_columns', []))}")
        print(f"  Requires preprocessing: {schema_info.get('requires_preprocessing', False)}")
        print(f"  All columns: {schema_info['all_columns']}")
        print(f"  Column types: {schema_info['column_info']}")

        # Log any warnings from schema analysis
        for warning in schema_info.get('warnings', []):
            print(f"  [WARNING] {warning}")
            logger.warning(f"Schema analysis warning: {warning}")

        # Log suggestions
        for suggestion in schema_info.get('suggestions', []):
            print(f"  [SUGGESTION] {suggestion}")

        # Validate that we have a text column to tokenize
        if not schema_info['recommended_column']:
            raise ValueError(
                f"No suitable text column found in dataset. "
                f"Available columns: {schema_info['all_columns']}. "
                f"Column types: {schema_info['column_info']}. "
                f"Please ensure the dataset has at least one string-type column "
                f"or a conversation column with recognized format (OpenAI, ShareGPT, etc.)."
            )

        # Handle conversation datasets that need preprocessing
        if schema_info.get('requires_preprocessing') and schema_info.get('preprocessing_config'):
            preprocessing_config = schema_info['preprocessing_config']

            print(f"Dataset requires conversation preprocessing:")
            print(f"  Source column: {preprocessing_config['source_column']}")
            print(f"  Format: {preprocessing_config['format']}")
            print(f"  Text key: {preprocessing_config['text_key']}")
            print(f"  Role key: {preprocessing_config['role_key']}")

            # Update task state
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': 35,
                    'total': 100,
                    'percent': 35.0,
                    'status': f"Preprocessing conversations from '{preprocessing_config['source_column']}'..."
                }
            )

            with self.get_db() as db:
                tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
                if tokenization_obj:
                    tokenization_obj.progress = 35.0
                    db.commit()

            # Create progress callback for preprocessing
            def preprocessing_progress(pct, msg):
                print(f"[PREPROCESS] {msg} ({pct:.1f}%)")
                emit_tokenization_progress(
                    dataset_id=dataset_id,
                    tokenization_id=tokenization_id,
                    progress=30.0 + (pct / 100.0 * 10.0),  # Map 0-100% to 30-40%
                    stage="preprocessing",
                    samples_processed=0,
                    total_samples=len(dataset),
                    started_at=started_at,
                    elapsed_seconds=time.time() - start_time,
                )

            # Preprocess the dataset to extract text from conversations
            dataset = TokenizationService.preprocess_conversation_dataset(
                dataset=dataset,
                preprocessing_config=preprocessing_config,
                progress_callback=preprocessing_progress,
            )

            # After preprocessing, use the output column for tokenization
            text_column = preprocessing_config.get('output_column', 'text')
            print(f"Preprocessing complete. Using '{text_column}' column for tokenization.")

        else:
            # No preprocessing needed - use recommended column directly
            text_column = schema_info['recommended_column']

        print(f"Final text column for tokenization: {text_column}")
        print(f"  Is multi-column: {schema_info['is_multi_column']}")

        # Update Celery task state: Starting tokenization
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 40,
                'total': 100,
                'percent': 40.0,
                'status': f'Tokenizing {len(dataset):,} samples using column \'{text_column}\'...'
            }
        )

        with self.get_db() as db:
            tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
            if tokenization_obj:
                tokenization_obj.progress = 40.0
                db.commit()

        # Emit detailed progress: starting tokenization stage
        total_samples = len(dataset)
        elapsed = time.time() - start_time
        emit_tokenization_progress(
            dataset_id=dataset_id,
            tokenization_id=tokenization_id,
            progress=40.0,
            stage="tokenizing",
            samples_processed=0,
            total_samples=total_samples,
            started_at=started_at,
            elapsed_seconds=elapsed,
        )

        # Map truncation strategy to tokenizer parameter
        truncation_config = {
            "longest_first": True,  # Default HuggingFace behavior
            "only_first": "only_first",
            "only_second": "only_second",
            "do_not_truncate": False,
        }
        truncation_param = truncation_config.get(truncation, True)

        # Monkey-patch tqdm to emit WebSocket progress during tokenization
        # Maps HuggingFace's tqdm (0-100%) to our progress range (40-80%)
        TqdmTokenization = create_tqdm_websocket_callback(
            dataset_id=dataset_id,
            tokenization_id=tokenization_id,  # Pass tokenization_id for correct WebSocket channel
            base_progress=40.0,
            progress_range=40.0,  # 40% → 80%
            throttle_seconds=0.5,  # Emit at most every 0.5 seconds OR every 1%
            stage="tokenizing",  # Current stage
            started_at=started_at  # Pass start timestamp
        )

        # Patch tqdm at multiple locations where datasets might use it
        import sys
        from tqdm import tqdm as original_tqdm

        # Save original tqdm
        sys.modules['tqdm'].tqdm = TqdmTokenization
        sys.modules['tqdm.auto'].tqdm = TqdmTokenization

        # Tokenize dataset using multiprocessing (avoids OOM on large datasets)
        # Progress is now tracked via tqdm WebSocket bridge

        # DEBUG: Log values being passed to tokenizer
        logger.info(
            f"[FILTER_CONFIG] Passing to tokenizer - "
            f"enable_filtering={filter_enabled} (type={type(filter_enabled).__name__}), "
            f"filter_mode={filter_mode}, "
            f"junk_ratio_threshold={filter_threshold}"
        )

        try:
            tokenized_dataset = TokenizationService.tokenize_dataset(
                dataset=dataset,
                tokenizer=tokenizer,
                text_column=text_column,
                max_length=max_length,
                stride=stride,
                truncation=truncation_param,  # Use dynamic truncation strategy from request
                padding=padding,  # Use dynamic padding strategy from request
                add_special_tokens=add_special_tokens,
                return_attention_mask=return_attention_mask,
                enable_cleaning=enable_cleaning,
                batch_size=1000,
                progress_callback=None,  # Disabled for multiprocessing (using tqdm bridge instead)
                num_proc=None,  # Auto-detect CPU cores for parallel processing (prevents OOM)
                # Use filter config from tokenization (per-job) instead of global settings
                enable_filtering=filter_enabled,
                filter_mode=filter_mode,
                junk_ratio_threshold=filter_threshold,
                remove_all_punctuation=remove_all_punctuation,
                custom_filter_chars=custom_filter_chars,
                # Use model's local cache for tokenizer (avoids HF auth for gated models)
                cache_dir=model_cache_dir,
            )
        finally:
            # Always restore original tqdm regardless of success/failure
            sys.modules['tqdm'].tqdm = original_tqdm
            sys.modules['tqdm.auto'].tqdm = original_tqdm

        # Mark tokenization as complete - from here on, graceful shutdown is allowed
        tokenization_complete = True
        print(f"[TOKENIZATION] Tokenization complete, marking for graceful shutdown protection")

        # Force cleanup of input dataset to prevent multiprocessing cleanup issues
        try:
            del dataset
        except:
            pass

        # Force garbage collection to clean up multiprocessing resources
        import gc
        gc.collect()

        # CRITICAL: Save tokenized dataset IMMEDIATELY after tokenization
        # This ensures data is saved even if SIGTERM is received during statistics
        print(f"[TOKENIZATION] Saving tokenized dataset immediately to prevent data loss...")
        tokenized_path = Path(raw_path).parent / f"{Path(raw_path).name}_tokenized_{model_id}"
        TokenizationService.save_tokenized_dataset(
            tokenized_dataset,
            tokenized_path,
        )
        saved_tokenized_path = str(tokenized_path)
        print(f"[TOKENIZATION] Dataset saved to {saved_tokenized_path}")

        # Check if graceful shutdown was requested during tokenization
        if graceful_shutdown_requested:
            print(f"[SIGNAL] Graceful shutdown requested, skipping statistics calculation")
            # Still update database with basic success status
            with self.get_db() as db:
                tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
                if tokenization_obj:
                    tokenization_obj.status = TokenizationStatus.READY
                    tokenization_obj.progress = 100.0
                    tokenization_obj.completed_at = datetime.now(UTC)
                    tokenization_obj.tokenized_path = saved_tokenized_path

                dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
                if dataset_obj:
                    dataset_obj.status = DatasetStatus.READY
                    dataset_obj.progress = 100.0

                db.commit()

            # Release Redis lock and return
            release_redis_lock(dataset_id)
            return {
                "tokenization_id": tokenization_id,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "status": "ready",
                "tokenized_path": saved_tokenized_path,
                "statistics": None,  # Statistics were skipped due to shutdown
                "graceful_shutdown": True,
            }

        # Update Celery task state: Calculating statistics
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 80,
                'total': 100,
                'percent': 80.0,
                'status': 'Calculating statistics...'
            }
        )

        with self.get_db() as db:
            tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
            if tokenization_obj:
                tokenization_obj.progress = 80.0
                db.commit()

        # Define callback for statistics calculation progress (80-90% range)
        def stats_progress_callback(pct: float):
            """Update database with statistics calculation progress (maps 0-100% to 80-90%)"""
            try:
                mapped_progress = 80.0 + (pct / 100.0 * 10.0)  # Map 0-100% to 80-90%
                print(f"[STATS CALLBACK] Called with pct={pct:.1f}%, mapped to {mapped_progress:.1f}%")
                with self.get_db() as db:
                    tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
                    if tokenization_obj:
                        old_progress = tokenization_obj.progress
                        tokenization_obj.progress = mapped_progress
                        db.commit()
                        print(f"[STATS CALLBACK] Updated progress from {old_progress} to {mapped_progress}")

                        # Emit WebSocket progress update
                        elapsed = time.time() - start_time
                        samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
                        emit_tokenization_progress(
                            dataset_id=dataset_id,
                            tokenization_id=tokenization_id,
                            progress=mapped_progress,
                            stage="saving",  # Statistics calculation is part of saving stage
                            samples_processed=total_samples,  # All samples are processed at this point
                            total_samples=total_samples,
                            started_at=started_at,
                            elapsed_seconds=elapsed,
                            samples_per_second=samples_per_sec,
                        )
                    else:
                        print(f"[STATS CALLBACK] ERROR: Tokenization {tokenization_id} not found")
            except Exception as e:
                print(f"[STATS CALLBACK] EXCEPTION: {e}")

        # Calculate statistics with progress callback
        stats = TokenizationService.calculate_statistics(
            tokenized_dataset,
            progress_callback=stats_progress_callback
        )

        # IMPORTANT: Override vocab_size with tokenizer's actual vocab_size
        # calculate_statistics returns unique tokens USED in dataset, but we need
        # the tokenizer's full vocab_size for validation during extraction
        # (to ensure dataset was tokenized with a compatible tokenizer)
        stats["unique_tokens_used"] = stats["vocab_size"]  # Keep for informational purposes
        stats["vocab_size"] = tokenizer.vocab_size
        print(f"[TOKENIZATION] Tokenizer vocab_size: {tokenizer.vocab_size}, unique tokens in dataset: {stats['unique_tokens_used']}")

        # Emit detailed progress: saving stage
        elapsed = time.time() - start_time
        emit_tokenization_progress(
            dataset_id=dataset_id,
            tokenization_id=tokenization_id,
            progress=90.0,
            stage="saving",
            samples_processed=total_samples,
            total_samples=total_samples,
            started_at=started_at,
            elapsed_seconds=elapsed,
        )

        # Update Celery task state: Finalizing (dataset already saved above)
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 95,
                'total': 100,
                'percent': 95.0,
                'status': 'Finalizing...'
            }
        )

        with self.get_db() as db:
            tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
            if tokenization_obj:
                tokenization_obj.progress = 95.0
                db.commit()


        # Update tokenization record with results
        with self.get_db() as db:
            try:
                tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
                if not tokenization_obj:
                    raise ValueError(f"Tokenization {tokenization_id} not found")

                # Update tokenization record with results
                tokenization_obj.status = TokenizationStatus.READY
                tokenization_obj.progress = 100.0
                tokenization_obj.completed_at = datetime.now(UTC)
                tokenization_obj.tokenized_path = saved_tokenized_path
                tokenization_obj.vocab_size = stats["vocab_size"]
                tokenization_obj.num_tokens = stats["num_tokens"]
                tokenization_obj.avg_seq_length = stats["avg_seq_length"]

                # Update parent dataset status to READY
                dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
                if dataset_obj:
                    dataset_obj.status = DatasetStatus.READY
                    dataset_obj.progress = 100.0
                    dataset_obj.error_message = None  # Clear error on successful completion

                db.commit()
                db.refresh(tokenization_obj)

                # Emit detailed progress: complete
                elapsed = time.time() - start_time
                emit_tokenization_progress(
                    dataset_id=dataset_id,
                    tokenization_id=tokenization_id,
                    progress=100.0,
                    stage="complete",
                    samples_processed=total_samples,
                    total_samples=total_samples,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                )

                # Update Celery task state: Complete
                self.update_state(
                    state='SUCCESS',
                    meta={
                        'current': 100,
                        'total': 100,
                        'percent': 100.0,
                        'status': 'Tokenization complete',
                        'statistics': stats
                    }
                )

                # Only emit "completed" event after successful database commit
                emit_dataset_progress(
                    dataset_id,
                    "completed",
                    {
                        "dataset_id": dataset_id,
                        "progress": 100.0,
                        "status": "ready",
                        "message": "Tokenization complete",
                        "statistics": stats,
                    },
                )

                # Release Redis distributed lock on success
                release_redis_lock(dataset_id)

            except Exception as commit_error:
                # Rollback transaction on failure
                db.rollback()
                error_msg = f"Failed to save tokenization results: {str(commit_error)}"
                print(f"Database commit error: {error_msg}")

                # Emit error event instead of success
                emit_dataset_progress(
                    dataset_id,
                    "error",
                    {
                        "dataset_id": dataset_id,
                        "status": "error",
                        "message": error_msg,
                    },
                )

                # Re-raise to trigger outer error handler
                raise

        return {
            "tokenization_id": tokenization_id,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "status": "ready",
            "tokenized_path": saved_tokenized_path,
            "statistics": stats,
        }

    except BaseException as e:
        # IMPORTANT: Catch BaseException to handle SystemExit from signal handler
        # SystemExit is NOT a subclass of Exception, so "except Exception" won't catch it
        error_message = f"Tokenization failed: {str(e)}"
        print(f"Tokenization error: {error_message}")

        # Clean up any child processes
        cleanup_child_processes()

        # Check if data was already saved before the error
        if saved_tokenized_path and Path(saved_tokenized_path).exists():
            print(f"[RECOVERY] Tokenized data was saved before error at {saved_tokenized_path}")
            # Mark as READY since data was saved successfully
            with self.get_db() as db:
                tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
                if tokenization_obj:
                    tokenization_obj.status = TokenizationStatus.READY
                    tokenization_obj.progress = 100.0
                    tokenization_obj.completed_at = datetime.now(UTC)
                    tokenization_obj.tokenized_path = saved_tokenized_path
                    # Note: statistics may be missing, but data is saved

                dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
                if dataset_obj:
                    dataset_obj.status = DatasetStatus.READY
                    dataset_obj.progress = 100.0

                db.commit()

            release_redis_lock(dataset_id)
            return {
                "tokenization_id": tokenization_id,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "status": "ready",
                "tokenized_path": saved_tokenized_path,
                "statistics": None,
                "recovered_after_error": True,
            }

        # Update tokenization AND dataset status to ERROR (data was not saved)
        # Guard against early failure where tokenization_id/dataset_uuid might not be set
        if tokenization_id and dataset_uuid:
            with self.get_db() as db:
                tokenization_obj = db.query(DatasetTokenization).filter_by(id=tokenization_id).first()
                if tokenization_obj:
                    tokenization_obj.status = TokenizationStatus.ERROR
                    tokenization_obj.error_message = error_message
                    tokenization_obj.progress = None

                # Also update parent dataset status to READY (allow retry)
                dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
                if dataset_obj:
                    dataset_obj.status = DatasetStatus.READY  # Allow retry
                    dataset_obj.progress = 0.0
                    dataset_obj.error_message = error_message

                db.commit()

        emit_dataset_progress(
            dataset_id,
            "error",
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "tokenization_id": tokenization_id,
                "status": "error",
                "message": error_message,
            },
        )

        # Release Redis distributed lock on error
        release_redis_lock(dataset_id)

        # Save failure state to task_queue for manual retry
        try:
            from ..models.task_queue import TaskQueue
            import uuid

            with self.get_db() as db:
                # Check if there's an existing queued task_queue entry for this entity
                existing_entry = db.query(TaskQueue).filter_by(
                    entity_id=dataset_id,
                    entity_type="dataset",
                    task_type="tokenization"
                ).filter(
                    TaskQueue.status.in_(["queued", "running"])
                ).first()

                if existing_entry:
                    # This is a retry that failed - update the existing entry
                    existing_entry.status = "failed"
                    existing_entry.error_message = error_message
                    existing_entry.task_id = self.request.id
                    db.commit()
                    print(f"Updated failed retry in task_queue: {existing_entry.id} (retry #{existing_entry.retry_count})")
                else:
                    # This is an initial failure - create new entry
                    task_queue_entry = TaskQueue(
                        id=f"tq_{uuid.uuid4().hex[:12]}",
                        task_id=self.request.id,
                        task_type="tokenization",
                        entity_id=dataset_id,
                        entity_type="dataset",
                        status="failed",
                        progress=0.0,
                        error_message=error_message,
                        retry_params={
                            "tokenizer_name": tokenizer_name,
                            "max_length": max_length,
                            "stride": stride,
                            "padding": padding,
                            "truncation": truncation,
                            "add_special_tokens": add_special_tokens,
                            "text_column": text_column,
                        },
                        retry_count=0,
                    )
                    db.add(task_queue_entry)
                    db.commit()
                    print(f"Saved failed tokenization to task_queue: {task_queue_entry.id}")
        except Exception as queue_exc:
            print(f"Failed to save task to queue: {queue_exc}")

        raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.dataset_tasks.cancel_dataset_download"
)
def cancel_dataset_download(self, dataset_id: str, task_id: Optional[str] = None):
    """
    Cancel an in-progress dataset download or tokenization.

    This task:
    1. Revokes the Celery task if task_id provided
    2. Updates dataset status to ERROR with "Cancelled by user"
    3. Cleans up partial download/tokenization files
    4. Sends WebSocket notification

    Args:
        dataset_id: Dataset UUID
        task_id: Optional Celery task ID to revoke

    Returns:
        dict with cancellation status
    """
    import logging
    import shutil

    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Cancelling download/processing for dataset {dataset_id}")

        dataset_uuid = UUID(dataset_id)

        # Get dataset from database
        with self.get_db() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_uuid).first()

            if not dataset:
                return {"error": f"Dataset {dataset_id} not found"}

            # Check if dataset is in a cancellable state
            if dataset.status not in [DatasetStatus.DOWNLOADING, DatasetStatus.PROCESSING]:
                return {
                    "error": f"Dataset {dataset_id} is not in a cancellable state (status: {dataset.status.value})"
                }

            # Revoke the Celery task if task_id provided
            if task_id:
                from celery import current_app
                current_app.control.revoke(task_id, terminate=True)
                logger.info(f"Revoked Celery task {task_id} for dataset {dataset_id}")

            # Clean up partial download files ONLY if dataset was downloading
            # If dataset was processing (tokenizing), do NOT delete raw files - they're needed!
            if dataset.status == DatasetStatus.DOWNLOADING and dataset.raw_path:
                raw_path = settings.resolve_data_path(dataset.raw_path)
                if raw_path.exists():
                    try:
                        shutil.rmtree(raw_path)
                        logger.info(f"Cleaned up partial raw dataset files: {raw_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up raw files {raw_path}: {e}")

            # Clean up partial tokenization files from tokenizations relationship
            if dataset.tokenizations:
                for tokenization in dataset.tokenizations:
                    if tokenization.tokenized_path:
                        tokenized_path = settings.resolve_data_path(tokenization.tokenized_path)
                        if tokenized_path.exists():
                            try:
                                shutil.rmtree(tokenized_path)
                                logger.info(f"Cleaned up tokenized files: {tokenized_path}")
                            except Exception as e:
                                logger.warning(f"Failed to clean up tokenized files {tokenized_path}: {e}")

            # Update dataset status
            dataset.status = DatasetStatus.ERROR
            dataset.error_message = "Cancelled by user"
            dataset.progress = 0.0
            db.commit()

        # Send WebSocket notification
        emit_dataset_progress(
            dataset_id,
            "error",
            {
                "dataset_id": dataset_id,
                "progress": 0.0,
                "status": "error",
                "message": "Download/processing cancelled by user"
            }
        )

        logger.info(f"Successfully cancelled download/processing for dataset {dataset_id}")

        return {
            "dataset_id": dataset_id,
            "status": "cancelled",
            "message": "Download/processing cancelled successfully"
        }

    except Exception as e:
        error_msg = f"Failed to cancel download/processing for dataset {dataset_id}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@celery_app.task(name="src.workers.dataset_tasks.delete_dataset_files")
def delete_dataset_files(dataset_id: str, raw_path: Optional[str] = None, tokenized_path: Optional[str] = None):
    """
    Delete dataset files from disk after database deletion.

    This task runs in the background to clean up dataset files without
    blocking the API response.

    Args:
        dataset_id: Dataset UUID
        raw_path: Path to raw dataset files (may be Docker-style /data/ path)
        tokenized_path: Path to tokenized dataset files (may be Docker-style /data/ path)

    Returns:
        dict with deletion status
    """
    import logging
    import os
    import shutil

    logger = logging.getLogger(__name__)
    deleted_files = []
    errors = []

    try:
        # Resolve Docker-style /data/ paths for native mode compatibility
        resolved_raw_path = str(settings.resolve_data_path(raw_path)) if raw_path else None
        resolved_tokenized_path = str(settings.resolve_data_path(tokenized_path)) if tokenized_path else None

        # Delete raw dataset files
        if resolved_raw_path and os.path.exists(resolved_raw_path):
            try:
                shutil.rmtree(resolved_raw_path)
                deleted_files.append(resolved_raw_path)
                logger.info(f"Deleted raw dataset files: {resolved_raw_path}")
            except Exception as e:
                error_msg = f"Failed to delete raw dataset files {resolved_raw_path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Delete tokenized dataset files
        if resolved_tokenized_path and os.path.exists(resolved_tokenized_path):
            try:
                shutil.rmtree(resolved_tokenized_path)
                deleted_files.append(resolved_tokenized_path)
                logger.info(f"Deleted tokenized dataset files: {resolved_tokenized_path}")
            except Exception as e:
                error_msg = f"Failed to delete tokenized dataset files {resolved_tokenized_path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Clean up HuggingFace cache directories and lock files
        # These are created during download but not tracked in the database
        from pathlib import Path
        import glob

        if resolved_raw_path:
            try:
                data_dir = Path(resolved_raw_path).parent
                dataset_name = Path(resolved_raw_path).name

                # Pattern 1: HuggingFace cache format (triple underscore)
                # e.g., "vietgpt_openwebtext_en" -> "vietgpt___openwebtext_en"
                hf_cache_name = dataset_name.replace('_', '___', 1)  # Replace first underscore with triple
                hf_cache_path = data_dir / hf_cache_name

                if hf_cache_path.exists() and hf_cache_path != Path(resolved_raw_path):
                    try:
                        shutil.rmtree(hf_cache_path)
                        deleted_files.append(str(hf_cache_path))
                        logger.info(f"Deleted HuggingFace cache: {hf_cache_path}")
                    except Exception as e:
                        error_msg = f"Failed to delete HuggingFace cache {hf_cache_path}: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)

                # Pattern 2: Lock files
                # e.g., "data_datasets_vietgpt___openwebtext_en_default_*.lock"
                lock_pattern = str(data_dir / f"data_datasets_{hf_cache_name}_*.lock")
                lock_files = glob.glob(lock_pattern)
                for lock_file in lock_files:
                    try:
                        os.remove(lock_file)
                        deleted_files.append(lock_file)
                        logger.info(f"Deleted lock file: {lock_file}")
                    except Exception as e:
                        error_msg = f"Failed to delete lock file {lock_file}: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)

            except Exception as e:
                error_msg = f"Failed during HF cache cleanup: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        return {
            "dataset_id": dataset_id,
            "deleted_files": deleted_files,
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"Failed to delete files for dataset {dataset_id}: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)

        return {
            "dataset_id": dataset_id,
            "deleted_files": deleted_files,
            "errors": errors,
        }


# Export tasks
__all__ = [
    "download_dataset_task",
    "tokenize_dataset_task",
    "cancel_dataset_download",
    "delete_dataset_files",
]
