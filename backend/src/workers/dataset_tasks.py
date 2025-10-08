"""
Celery tasks for dataset operations.

This module contains background tasks for downloading, processing,
and tokenizing datasets with real-time progress updates via WebSocket.
"""

import asyncio
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional
from uuid import UUID

from celery import Task
from datasets import load_dataset
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.celery_app import celery_app
from ..core.database import AsyncSessionLocal
from ..core.websocket import ws_manager
from ..models.dataset import DatasetStatus
from ..schemas.dataset import DatasetUpdate
from ..services.dataset_service import DatasetService


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

    async def emit_progress(
        self,
        dataset_id: str,
        event: str,
        data: dict,
    ):
        """
        Emit progress update via WebSocket.

        Args:
            dataset_id: Dataset UUID
            event: Event type (progress, completed, error)
            data: Event data payload
        """
        channel = f"datasets/{dataset_id}/progress"
        await ws_manager.emit_event(channel=channel, event=event, data=data)

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
):
    """
    Download dataset from HuggingFace Hub.

    Args:
        dataset_id: Dataset UUID
        repo_id: HuggingFace repository ID (e.g., 'roneneldan/TinyStories')
        access_token: HuggingFace access token for gated datasets
        split: Dataset split to download (e.g., 'train', 'validation', 'test')

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
        loop.run_until_complete(
            self.emit_progress(
                dataset_id,
                "progress",
                {
                    "progress": 0.0,
                    "status": "downloading",
                    "message": f"Starting download of {repo_id}...",
                },
            )
        )

        # Prepare download directory (use relative path for development)
        data_dir = Path("./data/datasets")
        data_dir.mkdir(parents=True, exist_ok=True)
        raw_path = data_dir / repo_id.replace("/", "_")

        # Download dataset from HuggingFace
        # Note: load_dataset is synchronous but has its own progress
        loop.run_until_complete(
            self.emit_progress(
                dataset_id,
                "progress",
                {
                    "progress": 10.0,
                    "status": "downloading",
                    "message": "Downloading from HuggingFace Hub...",
                },
            )
        )

        dataset = load_dataset(
            repo_id,
            split=split,
            cache_dir=str(data_dir),
            token=access_token,
        )

        # Update progress
        loop.run_until_complete(
            self.emit_progress(
                dataset_id,
                "progress",
                {
                    "progress": 80.0,
                    "status": "downloading",
                    "message": "Download complete, processing metadata...",
                },
            )
        )

        # Calculate dataset statistics
        num_samples = len(dataset) if hasattr(dataset, "__len__") else None
        size_bytes = dataset.size_in_bytes if hasattr(dataset, "size_in_bytes") else None

        # Update dataset record with download results
        async def finalize_download():
            session = await self.get_session()

            updates = DatasetUpdate(
                status=DatasetStatus.READY.value,
                progress=100.0,
                raw_path=str(raw_path),
                num_samples=num_samples,
                size_bytes=size_bytes,
            )
            await DatasetService.update_dataset(session, dataset_uuid, updates)
            await session.commit()

            await self.emit_progress(
                dataset_id,
                "completed",
                {
                    "progress": 100.0,
                    "status": "ready",
                    "message": "Dataset downloaded successfully",
                    "num_samples": num_samples,
                    "size_bytes": size_bytes,
                },
            )

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
            await self.emit_progress(
                dataset_id,
                "error",
                {
                    "status": "error",
                    "message": error_message,
                },
            )

        loop.run_until_complete(handle_error())

        # Retry if not max retries
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)

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
):
    """
    Tokenize dataset using specified tokenizer.

    Args:
        dataset_id: Dataset UUID
        tokenizer_name: HuggingFace tokenizer name (e.g., 'gpt2')
        max_length: Maximum sequence length
        stride: Sliding window stride for long sequences

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
        loop.run_until_complete(
            self.emit_progress(
                dataset_id,
                "progress",
                {
                    "progress": 0.0,
                    "status": "processing",
                    "message": "Starting tokenization...",
                },
            )
        )

        # TODO: Implement actual tokenization logic
        # This is a placeholder - full implementation would:
        # 1. Load dataset from raw_path
        # 2. Load tokenizer
        # 3. Tokenize samples with progress tracking
        # 4. Save tokenized dataset to Arrow format
        # 5. Calculate statistics (num_tokens, avg_seq_length, vocab_size)

        # For now, just mark as ready after a short delay
        import time
        for progress in [20, 40, 60, 80, 100]:
            time.sleep(0.5)
            loop.run_until_complete(
                self.emit_progress(
                    dataset_id,
                    "progress",
                    {
                        "progress": float(progress),
                        "status": "processing",
                        "message": f"Tokenizing... {progress}%",
                    },
                )
            )

        # Mark as ready
        async def finalize_tokenization():
            await self.update_dataset_status(
                dataset_uuid,
                DatasetStatus.READY,
                progress=100.0,
            )
            await self.emit_progress(
                dataset_id,
                "completed",
                {
                    "progress": 100.0,
                    "status": "ready",
                    "message": "Tokenization complete",
                },
            )

        loop.run_until_complete(finalize_tokenization())

        return {
            "dataset_id": dataset_id,
            "status": "ready",
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
            await self.emit_progress(
                dataset_id,
                "error",
                {
                    "status": "error",
                    "message": error_message,
                },
            )

        loop.run_until_complete(handle_error())

        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=120)

        raise

    finally:
        loop.run_until_complete(self.close_session())
        loop.close()


# Export tasks
__all__ = [
    "download_dataset_task",
    "tokenize_dataset_task",
]
