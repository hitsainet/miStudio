"""
Celery tasks for dataset operations.

This module contains background tasks for downloading, processing,
and tokenizing datasets with real-time progress updates via WebSocket.
"""

import asyncio
import httpx
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional
from uuid import UUID

from celery import Task
from datasets import load_dataset
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.celery_app import celery_app
from ..core.config import settings
from ..core.database import AsyncSessionLocal
from ..models.dataset import DatasetStatus
from ..schemas.dataset import DatasetUpdate
from ..services.dataset_service import DatasetService
from ..services.tokenization_service import TokenizationService


class DatasetTask(Task):
    """Base class for dataset-related Celery tasks with progress tracking."""

    def __init__(self):
        super().__init__()
        self._session: Optional[AsyncSession] = None

    async def get_session(self) -> AsyncSession:
        """Get or create async database session."""
        if self._session is None:
            self._session = AsyncSessionLocal()
        return self._session

    async def close_session(self):
        """Close database session."""
        if self._session:
            await self._session.close()
            self._session = None

    def emit_progress(
        self,
        dataset_id: str,
        event: str,
        data: dict,
    ):
        """
        Emit progress update via WebSocket through HTTP callback.

        Args:
            dataset_id: Dataset UUID
            event: Event type (progress, completed, error)
            data: Event data payload
        """
        channel = f"datasets/{dataset_id}/progress"

        try:
            # Use httpx synchronous client to call the FastAPI internal endpoint
            # Use configured WebSocket emit URL (supports both local and Docker deployments)
            api_url = settings.websocket_emit_url

            with httpx.Client() as client:
                response = client.post(
                    api_url,
                    json={
                        "channel": channel,
                        "event": event,
                        "data": data,
                    },
                    timeout=1.0,
                )
                print(f"WebSocket emit: {event} to {channel} - Status: {response.status_code}")
        except Exception as e:
            print(f"Failed to emit WebSocket event: {e}")

    async def update_dataset_status(
        self,
        dataset_id: UUID,
        status: DatasetStatus,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
    ):
        """
        Update dataset status in database.

        Args:
            dataset_id: Dataset UUID
            status: New status
            progress: Progress percentage (0-100)
            error_message: Error message if status is ERROR
        """
        session = await self.get_session()

        update_data = {"status": status.value}
        if progress is not None:
            update_data["progress"] = progress
        if error_message is not None:
            update_data["error_message"] = error_message

        updates = DatasetUpdate(**update_data)
        await DatasetService.update_dataset(session, dataset_id, updates)
        await session.commit()


@celery_app.task(
    bind=True,
    base=DatasetTask,
    name="src.workers.dataset_tasks.download_dataset_task",
    max_retries=3,
    default_retry_delay=60,
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
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        dataset_uuid = UUID(dataset_id)

        # Update status to downloading
        loop.run_until_complete(
            self.update_dataset_status(dataset_uuid, DatasetStatus.DOWNLOADING, progress=0.0)
        )
        self.emit_progress(
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
        # Note: load_dataset is synchronous but has its own progress
        self.emit_progress(
            dataset_id,
            "progress",
            {
                "dataset_id": dataset_id,
                "progress": 10.0,
                "status": "downloading",
                "message": "Downloading from HuggingFace Hub...",
            },
        )

        dataset = load_dataset(
            repo_id,
            name=config,
            split=split,
            cache_dir=str(data_dir),
            token=access_token,
            trust_remote_code=True,
        )

        # Update progress: saving to disk
        self.emit_progress(
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
        self.emit_progress(
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
        async def finalize_download():
            session = await self.get_session()

            try:
                updates = DatasetUpdate(
                    status=DatasetStatus.READY.value,
                    progress=100.0,
                    raw_path=str(raw_path),
                    num_samples=num_samples,
                    size_bytes=size_bytes,
                )
                await DatasetService.update_dataset(session, dataset_uuid, updates)
                await session.commit()

                # Only emit "completed" event after successful database commit
                self.emit_progress(
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
                await session.rollback()
                error_msg = f"Failed to save download results: {str(commit_error)}"
                print(f"Database commit error: {error_msg}")

                # Emit error event instead of success
                self.emit_progress(
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

        loop.run_until_complete(finalize_download())

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

        async def handle_error():
            await self.update_dataset_status(
                UUID(dataset_id),
                DatasetStatus.ERROR,
                error_message=error_message,
            )
            self.emit_progress(
                dataset_id,
                "error",
                {
                    "dataset_id": dataset_id,
                    "status": "error",
                    "message": error_message,
                },
            )

        loop.run_until_complete(handle_error())

        # Retry with exponential backoff if not max retries
        # Formula: base_delay * (2 ** retry_count)
        # Attempt 1: 60s, Attempt 2: 120s, Attempt 3: 240s
        if self.request.retries < self.max_retries:
            base_delay = 60
            retry_delay = base_delay * (2 ** self.request.retries)
            print(f"Retrying download (attempt {self.request.retries + 1}/{self.max_retries}) in {retry_delay}s...")
            raise self.retry(exc=e, countdown=retry_delay)

        raise

    finally:
        loop.run_until_complete(self.close_session())
        loop.close()


@celery_app.task(
    bind=True,
    base=DatasetTask,
    name="src.workers.dataset_tasks.tokenize_dataset_task",
    max_retries=2,
    default_retry_delay=120,
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
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        dataset_uuid = UUID(dataset_id)

        # Update status to processing
        loop.run_until_complete(
            self.update_dataset_status(dataset_uuid, DatasetStatus.PROCESSING, progress=0.0)
        )
        self.emit_progress(
                dataset_id,
                "progress",
                {
                    "dataset_id": dataset_id,
                    "progress": 0.0,
                    "status": "processing",
                    "message": "Starting tokenization...",
                },
        )

        # Get dataset from database to retrieve raw_path
        async def get_dataset_info():
            session = await self.get_session()
            dataset_obj = await DatasetService.get_dataset(session, dataset_uuid)
            if not dataset_obj:
                raise ValueError(f"Dataset {dataset_id} not found")
            if not dataset_obj.raw_path:
                raise ValueError(f"Dataset {dataset_id} has no raw_path")
            return dataset_obj.raw_path

        raw_path = loop.run_until_complete(get_dataset_info())

        # Emit progress: 10%
        self.emit_progress(
                dataset_id,
                "progress",
                {
                    "dataset_id": dataset_id,
                    "progress": 10.0,
                    "status": "processing",
                    "message": "Loading tokenizer...",
                },
        )

        # Load tokenizer
        tokenizer = TokenizationService.load_tokenizer(tokenizer_name)

        # Emit progress: 20%
        self.emit_progress(
                dataset_id,
                "progress",
                {
                    "dataset_id": dataset_id,
                    "progress": 20.0,
                    "status": "processing",
                    "message": "Loading dataset...",
                },
        )

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
        self.emit_progress(
                dataset_id,
                "progress",
                {
                    "dataset_id": dataset_id,
                    "progress": 30.0,
                    "status": "processing",
                    "message": "Analyzing dataset schema...",
                },
        )

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

        # Emit progress: 40%
        self.emit_progress(
                dataset_id,
                "progress",
                {
                    "dataset_id": dataset_id,
                    "progress": 40.0,
                    "status": "processing",
                    "message": f"Tokenizing {len(dataset)} samples using column '{text_column}'...",
                },
        )

        # Define progress callback for tokenization
        def tokenization_progress(progress_pct: float, message: str):
            """Report tokenization progress."""
            self.emit_progress(
                dataset_id,
                "progress",
                {
                    "dataset_id": dataset_id,
                    "progress": progress_pct,
                    "status": "processing",
                    "message": message,
                },
            )

        # Map truncation strategy to tokenizer parameter
        truncation_config = {
            "longest_first": True,  # Default HuggingFace behavior
            "only_first": "only_first",
            "only_second": "only_second",
            "do_not_truncate": False,
        }
        truncation_param = truncation_config.get(truncation, True)

        # Tokenize dataset with progress tracking
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
            progress_callback=tokenization_progress,
        )

        # Emit progress: 80%
        self.emit_progress(
                dataset_id,
                "progress",
                {
                    "dataset_id": dataset_id,
                    "progress": 80.0,
                    "status": "processing",
                    "message": "Calculating statistics...",
                },
        )

        # Calculate statistics
        stats = TokenizationService.calculate_statistics(tokenized_dataset)

        # Save tokenized dataset
        tokenized_path = Path(raw_path).parent / f"{Path(raw_path).name}_tokenized"
        TokenizationService.save_tokenized_dataset(
            tokenized_dataset,
            tokenized_path,
        )

        # Emit progress: 95%
        self.emit_progress(
                dataset_id,
                "progress",
                {
                    "dataset_id": dataset_id,
                    "progress": 95.0,
                    "status": "processing",
                    "message": "Saving results...",
                },
        )

        # Update dataset with tokenization results
        async def finalize_tokenization():
            session = await self.get_session()

            try:
                # Update dataset metadata with tokenization stats and schema info
                updates = DatasetUpdate(
                    status=DatasetStatus.READY.value,
                    progress=100.0,
                    tokenized_path=str(tokenized_path),
                    metadata={
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
                    },
                )
                await DatasetService.update_dataset(session, dataset_uuid, updates)
                await session.commit()

                # Only emit "completed" event after successful database commit
                self.emit_progress(
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
                await session.rollback()
                error_msg = f"Failed to save tokenization results: {str(commit_error)}"
                print(f"Database commit error: {error_msg}")

                # Emit error event instead of success
                self.emit_progress(
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

        loop.run_until_complete(finalize_tokenization())

        return {
            "dataset_id": dataset_id,
            "status": "ready",
            "tokenized_path": str(tokenized_path),
            "statistics": stats,
        }

    except Exception as e:
        error_message = f"Tokenization failed: {str(e)}"
        print(f"Dataset tokenization error: {error_message}")

        async def handle_error():
            await self.update_dataset_status(
                UUID(dataset_id),
                DatasetStatus.ERROR,
                error_message=error_message,
            )
            self.emit_progress(
                dataset_id,
                "error",
                {
                    "dataset_id": dataset_id,
                    "status": "error",
                    "message": error_message,
                },
            )

        loop.run_until_complete(handle_error())

        # Retry with exponential backoff if not max retries
        # Formula: base_delay * (2 ** retry_count)
        # Attempt 1: 120s, Attempt 2: 240s
        if self.request.retries < self.max_retries:
            base_delay = 120
            retry_delay = base_delay * (2 ** self.request.retries)
            print(f"Retrying tokenization (attempt {self.request.retries + 1}/{self.max_retries}) in {retry_delay}s...")
            raise self.retry(exc=e, countdown=retry_delay)

        raise

    finally:
        loop.run_until_complete(self.close_session())
        loop.close()


# Export tasks
__all__ = [
    "download_dataset_task",
    "tokenize_dataset_task",
]
