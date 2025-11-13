"""
TqdmWebSocket Bridge

This module provides a custom tqdm class that bridges HuggingFace dataset
download progress into the application's WebSocket progress system.

Usage:
    from datasets import load_dataset
    from .tqdm_websocket_bridge import create_tqdm_websocket_callback

    # Create callback with dataset context
    tqdm_class = create_tqdm_websocket_callback(
        dataset_id="abc-123",
        base_progress=10.0,
        progress_range=60.0  # Maps tqdm 0-100% to 10-70% in our system
    )

    # Monkey-patch tqdm for this download
    import datasets.utils.file_utils
    original_tqdm = datasets.utils.file_utils.tqdm
    datasets.utils.file_utils.tqdm = tqdm_class

    # Download with progress tracking
    dataset = load_dataset(...)

    # Restore original tqdm
    datasets.utils.file_utils.tqdm = original_tqdm
"""

import logging
from typing import Optional, Callable
from tqdm import tqdm as tqdm_original

from .websocket_emitter import emit_dataset_progress, emit_tokenization_progress

logger = logging.getLogger(__name__)


class TqdmWebSocketCallback(tqdm_original):
    """
    Custom tqdm progress bar that emits WebSocket updates.

    This class intercepts tqdm progress updates from HuggingFace libraries
    and translates them into WebSocket emissions for the frontend.

    Supports both dataset download progress and tokenization progress.
    """

    def __init__(
        self,
        *args,
        dataset_id: Optional[str] = None,
        tokenization_id: Optional[str] = None,
        base_progress: float = 0.0,
        progress_range: float = 100.0,
        throttle_seconds: float = 0.5,
        stage: str = "processing",
        started_at: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize TqdmWebSocketCallback.

        Args:
            dataset_id: Dataset ID for WebSocket channel
            tokenization_id: Tokenization ID (if tracking tokenization progress)
            base_progress: Starting progress percentage (e.g., 10.0)
            progress_range: Range to map tqdm 0-100% into (e.g., 60.0 maps to 10-70%)
            throttle_seconds: Minimum seconds between WebSocket emissions (default: 0.5s)
            stage: Current processing stage (for tokenization: "tokenizing", "saving", etc.)
            started_at: ISO timestamp when processing started
            *args, **kwargs: Passed to tqdm parent class
        """
        super().__init__(*args, **kwargs)
        self.dataset_id = dataset_id
        self.tokenization_id = tokenization_id
        self.base_progress = base_progress
        self.progress_range = progress_range
        self.throttle_seconds = throttle_seconds
        self.stage = stage
        self.started_at = started_at
        self.last_emit_time = 0.0
        self.last_emitted_progress = -1.0
        self.start_time = None  # Track when update() was first called

    def update(self, n=1):
        """
        Override tqdm's update method to emit WebSocket progress.

        Args:
            n: Number of items to increment progress by
        """
        # Call parent update to maintain tqdm functionality
        result = super().update(n)

        # Only emit if we have a dataset_id (not all tqdm instances are for datasets)
        if self.dataset_id and self.total:
            # Calculate current progress percentage (0-100%)
            tqdm_progress = (self.n / self.total) * 100.0

            # Map to our progress range
            mapped_progress = self.base_progress + (tqdm_progress * self.progress_range / 100.0)

            # Throttle emissions: only emit if progress changed by >= 10% or enough time passed
            import time
            current_time = time.time()
            progress_delta = abs(mapped_progress - self.last_emitted_progress)
            time_delta = current_time - self.last_emit_time

            should_emit = (
                progress_delta >= 1.0  # Progress changed by at least 1% (reduced from 10%)
                or time_delta >= self.throttle_seconds  # Or throttle period elapsed
                or tqdm_progress >= 99.9  # Or nearly complete
            )

            if should_emit:
                # Initialize start time on first emission
                if self.start_time is None:
                    self.start_time = current_time

                # Extract description if available
                desc = self.desc or ("Tokenizing" if self.tokenization_id else "Downloading")

                # Format progress message
                if self.total:
                    message = f"{desc}: {self.n:,}/{self.total:,} examples ({tqdm_progress:.1f}%)"
                else:
                    message = f"{desc}: {self.n:,} examples"

                # Emit via WebSocket - different channels for dataset download vs tokenization
                try:
                    if self.tokenization_id:
                        # Emit tokenization progress
                        elapsed = current_time - self.start_time if self.start_time else 0
                        samples_per_sec = self.n / elapsed if elapsed > 0 else 0

                        emit_tokenization_progress(
                            dataset_id=self.dataset_id,
                            tokenization_id=self.tokenization_id,
                            progress=mapped_progress,
                            stage=self.stage,
                            samples_processed=self.n,
                            total_samples=self.total or 0,
                            started_at=self.started_at,
                            elapsed_seconds=elapsed,
                            samples_per_second=samples_per_sec,
                        )
                    else:
                        # Emit dataset download progress
                        emit_dataset_progress(
                            self.dataset_id,
                            "progress",
                            {
                                "dataset_id": self.dataset_id,
                                "progress": mapped_progress,
                                "status": "downloading",
                                "message": message,
                            },
                        )

                    self.last_emitted_progress = mapped_progress
                    self.last_emit_time = current_time
                except Exception as e:
                    # Don't let WebSocket errors break the operation
                    logger.warning(f"Failed to emit progress via WebSocket: {e}")

                # Update database progress (throttled same as WebSocket)
                try:
                    from uuid import UUID
                    from ..db.session import get_db
                    from ..models.dataset import Dataset

                    # Create a new database session for this update
                    db_gen = get_db()
                    db = next(db_gen)
                    try:
                        dataset_uuid = UUID(self.dataset_id)
                        dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
                        if dataset_obj:
                            dataset_obj.progress = mapped_progress / 100.0  # Store as 0.0-1.0 fraction
                            db.commit()
                    finally:
                        # Always close the database session
                        try:
                            next(db_gen)
                        except StopIteration:
                            pass
                except Exception as e:
                    # Don't let database errors break the download
                    logger.warning(f"Failed to update database progress: {e}")

        return result

    def close(self):
        """Override close to emit final progress update."""
        if self.dataset_id and self.total:
            # Emit 100% for this tqdm instance (mapped to our range)
            final_progress = self.base_progress + self.progress_range
            try:
                if self.tokenization_id:
                    # Emit final tokenization progress
                    import time
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    samples_per_sec = self.n / elapsed if elapsed > 0 else 0

                    emit_tokenization_progress(
                        dataset_id=self.dataset_id,
                        tokenization_id=self.tokenization_id,
                        progress=final_progress,
                        stage=self.stage,
                        samples_processed=self.n,
                        total_samples=self.total or 0,
                        started_at=self.started_at,
                        elapsed_seconds=elapsed,
                        samples_per_second=samples_per_sec,
                    )
                else:
                    # Emit final dataset download progress
                    emit_dataset_progress(
                        self.dataset_id,
                        "progress",
                        {
                            "dataset_id": self.dataset_id,
                            "progress": final_progress,
                            "status": "downloading",
                            "message": f"Completed: {self.n:,} examples processed",
                        },
                    )
            except Exception as e:
                logger.warning(f"Failed to emit final progress via WebSocket: {e}")

        super().close()


def create_tqdm_websocket_callback(
    dataset_id: str,
    tokenization_id: Optional[str] = None,
    base_progress: float = 10.0,
    progress_range: float = 60.0,
    throttle_seconds: float = 0.5,
    stage: str = "processing",
    started_at: Optional[str] = None,
) -> type:
    """
    Factory function to create a tqdm class with WebSocket callback configured.

    This returns a CLASS (not instance) that can be used to replace tqdm globally
    or passed to libraries that accept custom tqdm classes.

    Args:
        dataset_id: Dataset ID for WebSocket channel
        tokenization_id: Tokenization ID (if tracking tokenization progress)
        base_progress: Starting progress percentage (e.g., 10.0 means start at 10%)
        progress_range: Range to map tqdm 0-100% into (e.g., 60.0 means 10%-70%)
        throttle_seconds: Minimum seconds between WebSocket emissions
        stage: Current processing stage (for tokenization)
        started_at: ISO timestamp when processing started

    Returns:
        A tqdm class with the callback parameters baked in

    Example:
        # Create a custom tqdm class for dataset download
        TqdmClass = create_tqdm_websocket_callback(
            dataset_id="abc-123",
            base_progress=10.0,
            progress_range=60.0
        )

        # Or for tokenization progress
        TqdmClass = create_tqdm_websocket_callback(
            dataset_id="abc-123",
            tokenization_id="tok_xyz",
            base_progress=40.0,
            progress_range=40.0,
            stage="tokenizing",
            started_at="2025-11-11T21:00:00Z"
        )

        # Monkey-patch HuggingFace datasets to use our tqdm
        import datasets.utils.file_utils
        original_tqdm = datasets.utils.file_utils.tqdm
        datasets.utils.file_utils.tqdm = TqdmClass

        # Download/tokenize with progress tracking
        dataset = load_dataset(...)

        # Restore original tqdm
        datasets.utils.file_utils.tqdm = original_tqdm
    """
    class ConfiguredTqdmWebSocket(TqdmWebSocketCallback):
        def __init__(self, *args, **kwargs):
            # Inject our callback parameters
            kwargs.setdefault('dataset_id', dataset_id)
            kwargs.setdefault('tokenization_id', tokenization_id)
            kwargs.setdefault('base_progress', base_progress)
            kwargs.setdefault('progress_range', progress_range)
            kwargs.setdefault('throttle_seconds', throttle_seconds)
            kwargs.setdefault('stage', stage)
            kwargs.setdefault('started_at', started_at)
            super().__init__(*args, **kwargs)

    return ConfiguredTqdmWebSocket
