"""
Celery tasks for dataset operations.

This module contains background tasks for downloading, processing,
and tokenizing datasets with real-time progress updates via WebSocket.
"""

from datetime import datetime, UTC
from pathlib import Path
from typing import Optional
from uuid import UUID

# NOTE: Do NOT import load_dataset at module level!
# It must be imported inside the task function AFTER patching tqdm
# Otherwise, datasets will import tqdm before we can patch it
from sqlalchemy.orm import Session

from ..core.celery_app import celery_app
from ..core.config import settings
from ..models.dataset import DatasetStatus, Dataset
from ..schemas.dataset import DatasetUpdate
from ..services.dataset_service import DatasetService
from ..services.tokenization_service import TokenizationService
from .base_task import DatabaseTask
from .websocket_emitter import emit_dataset_progress
from .tqdm_websocket_bridge import create_tqdm_websocket_callback


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

        # Prepare download directory (use relative path for development)
        data_dir = Path("./data/datasets")
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

        # NOW import load_dataset - it will use our patched tqdm
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
        # Handle errors
        error_message = f"Download failed: {str(e)}"
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
    tokenizer_name: str,
    max_length: int = 512,
    stride: int = 0,
    padding: str = "max_length",
    truncation: str = "longest_first",
    add_special_tokens: bool = True,
    return_attention_mask: bool = True,
):
    """
    Tokenize dataset using specified tokenizer.

    Args:
        dataset_id: Dataset UUID
        tokenizer_name: HuggingFace tokenizer name (e.g., 'gpt2')
        max_length: Maximum sequence length
        stride: Sliding window stride for long sequences
        padding: Padding strategy ('max_length', 'longest', or 'do_not_pad')
        truncation: Truncation strategy ('longest_first', 'only_first', 'only_second', or 'do_not_truncate')
        add_special_tokens: Add special tokens (BOS, EOS, PAD, etc.)
        return_attention_mask: Return attention mask

    Returns:
        dict: Tokenization result with statistics
    """
    try:
        dataset_uuid = UUID(dataset_id)

        # Update status to processing
        with self.get_db() as db:
            self.update_progress(
                db=db,
                model_class=Dataset,
                record_id=dataset_id,
                progress=0.0,
                status=DatasetStatus.PROCESSING.value,
            )

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

        # Get dataset from database to retrieve raw_path
        with self.get_db() as db:
            dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
            if not dataset_obj:
                raise ValueError(f"Dataset {dataset_id} not found")
            if not dataset_obj.raw_path:
                raise ValueError(f"Dataset {dataset_id} has no raw_path")
            raw_path = dataset_obj.raw_path

            # Update database progress
            dataset_obj.progress = 0.0
            dataset_obj.status = DatasetStatus.PROCESSING
            db.commit()

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

        # Load tokenizer
        tokenizer = TokenizationService.load_tokenizer(tokenizer_name)

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

        # Update database progress
        with self.get_db() as db:
            dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
            if dataset_obj:
                dataset_obj.progress = 20.0
                db.commit()

        # Load dataset from disk
        dataset = TokenizationService.load_dataset_from_disk(raw_path)

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
            dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
            if dataset_obj:
                dataset_obj.progress = 30.0
                db.commit()

        schema_info = TokenizationService.analyze_dataset_schema(dataset)

        # Validate that we have a text column to tokenize
        if not schema_info['recommended_column']:
            raise ValueError(
                f"No suitable text column found in dataset. "
                f"Available columns: {schema_info['all_columns']}. "
                f"Column types: {schema_info['column_info']}. "
                f"Please ensure the dataset has at least one string-type column."
            )

        text_column = schema_info['recommended_column']

        # Log schema information
        print(f"Dataset schema analysis:")
        print(f"  Available text columns: {schema_info['text_columns']}")
        print(f"  Selected column: {text_column}")
        print(f"  All columns: {schema_info['all_columns']}")
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
            dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
            if dataset_obj:
                dataset_obj.progress = 40.0
                db.commit()

        # Map truncation strategy to tokenizer parameter
        truncation_config = {
            "longest_first": True,  # Default HuggingFace behavior
            "only_first": "only_first",
            "only_second": "only_second",
            "do_not_truncate": False,
        }
        truncation_param = truncation_config.get(truncation, True)

        # Tokenize dataset using multiprocessing (avoids OOM on large datasets)
        # Note: progress_callback removed because it doesn't work with multiprocessing
        # Progress is tracked via Celery task state updates at milestones
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
                batch_size=1000,
                progress_callback=None,  # Disabled for multiprocessing
                num_proc=None,  # Auto-detect CPU cores for parallel processing (prevents OOM)
            )

            # Force cleanup of input dataset to prevent multiprocessing cleanup issues
            del dataset

            # Force garbage collection to clean up multiprocessing resources
            import gc
            gc.collect()

        except SystemExit as e:
            # HuggingFace datasets sometimes raises SystemExit during multiprocessing cleanup
            # This is a known issue - if tokenization completed, we can safely continue
            print(f"Warning: SystemExit during tokenization cleanup (exit code: {e.code})")
            print("Tokenization likely completed successfully, continuing...")
            # Re-raise if this happened during actual tokenization (exit code != -241)
            if e.code != -241:
                raise

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
            dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
            if dataset_obj:
                dataset_obj.progress = 80.0
                db.commit()

        # Define callback for statistics calculation progress (80-90% range)
        def stats_progress_callback(pct: float):
            """Update database with statistics calculation progress (maps 0-100% to 80-90%)"""
            try:
                mapped_progress = 80.0 + (pct / 100.0 * 10.0)  # Map 0-100% to 80-90%
                print(f"[STATS CALLBACK] Called with pct={pct:.1f}%, mapped to {mapped_progress:.1f}%")
                with self.get_db() as db:
                    dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
                    if dataset_obj:
                        old_progress = dataset_obj.progress
                        dataset_obj.progress = mapped_progress
                        db.commit()
                        print(f"[STATS CALLBACK] Updated progress from {old_progress} to {mapped_progress}")
                    else:
                        print(f"[STATS CALLBACK] ERROR: Dataset {dataset_uuid} not found")
            except Exception as e:
                print(f"[STATS CALLBACK] EXCEPTION: {e}")

        # Calculate statistics with progress callback
        stats = TokenizationService.calculate_statistics(
            tokenized_dataset,
            progress_callback=stats_progress_callback
        )

        # Update Celery task state: Saving dataset
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 90,
                'total': 100,
                'percent': 90.0,
                'status': 'Saving tokenized dataset...'
            }
        )

        # Save tokenized dataset
        tokenized_path = Path(raw_path).parent / f"{Path(raw_path).name}_tokenized"
        TokenizationService.save_tokenized_dataset(
            tokenized_dataset,
            tokenized_path,
        )

        # Update Celery task state: Finalizing
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
            dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
            if dataset_obj:
                dataset_obj.progress = 95.0
                db.commit()


        # Update dataset with tokenization results
        with self.get_db() as db:
            try:
                dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
                if not dataset_obj:
                    raise ValueError(f"Dataset {dataset_id} not found")

                # Update dataset metadata with tokenization stats and schema info
                dataset_obj.status = DatasetStatus.READY
                dataset_obj.progress = 100.0
                dataset_obj.tokenized_path = str(tokenized_path)
                # Update top-level statistics columns for easy querying
                dataset_obj.num_tokens = stats["num_tokens"]
                dataset_obj.avg_seq_length = stats["avg_seq_length"]
                dataset_obj.vocab_size = stats["vocab_size"]
                # Merge tokenization metadata with existing metadata (don't overwrite!)
                existing_metadata = dataset_obj.extra_metadata or {}
                dataset_obj.extra_metadata = {
                    **existing_metadata,  # Preserve existing metadata (split, config, etc.)
                    "schema": {
                        "text_columns": schema_info["text_columns"],
                        "column_info": schema_info["column_info"],
                        "all_columns": schema_info["all_columns"],
                        "is_multi_column": schema_info["is_multi_column"],
                    },
                    "tokenization": {
                        "tokenizer_name": tokenizer_name,
                        "text_column_used": text_column,
                        "max_length": max_length,
                        "stride": stride,
                        "padding": padding,
                        "truncation": truncation,
                        "add_special_tokens": add_special_tokens,
                        "return_attention_mask": return_attention_mask,
                        "num_tokens": stats["num_tokens"],
                        "avg_seq_length": stats["avg_seq_length"],
                        "min_seq_length": stats["min_seq_length"],
                        "max_seq_length": stats["max_seq_length"],
                        "median_seq_length": stats["median_seq_length"],
                        "vocab_size": stats["vocab_size"],
                        "length_distribution": stats["length_distribution"],
                        "split_distribution": stats.get("split_distribution"),
                    }
                }

                db.commit()
                db.refresh(dataset_obj)

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
            "dataset_id": dataset_id,
            "status": "ready",
            "tokenized_path": str(tokenized_path),
            "statistics": stats,
        }

    except Exception as e:
        error_message = f"Tokenization failed: {str(e)}"
        print(f"Dataset tokenization error: {error_message}")

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

            # Clean up partial download files
            if dataset.raw_path:
                raw_path = Path(dataset.raw_path)
                if raw_path.exists():
                    try:
                        shutil.rmtree(raw_path)
                        logger.info(f"Cleaned up raw dataset files: {raw_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up raw files {raw_path}: {e}")

            # Clean up partial tokenization files
            if dataset.tokenized_path:
                tokenized_path = Path(dataset.tokenized_path)
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
        raw_path: Path to raw dataset files
        tokenized_path: Path to tokenized dataset files

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
        # Delete raw dataset files
        if raw_path and os.path.exists(raw_path):
            try:
                shutil.rmtree(raw_path)
                deleted_files.append(raw_path)
                logger.info(f"Deleted raw dataset files: {raw_path}")
            except Exception as e:
                error_msg = f"Failed to delete raw dataset files {raw_path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Delete tokenized dataset files
        if tokenized_path and os.path.exists(tokenized_path):
            try:
                shutil.rmtree(tokenized_path)
                deleted_files.append(tokenized_path)
                logger.info(f"Deleted tokenized dataset files: {tokenized_path}")
            except Exception as e:
                error_msg = f"Failed to delete tokenized dataset files {tokenized_path}: {str(e)}"
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
