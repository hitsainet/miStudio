"""
TqdmWebSocket Bridge

This module provides a custom tqdm class that bridges HuggingFace dataset
download and tokenization progress into the application's WebSocket progress system.

IMPORTANT: For multiprocessing support (num_proc > 1), you MUST patch BOTH:
1. `datasets.utils.tqdm.tqdm` - The module-level tqdm class
2. `datasets.arrow_dataset.hf_tqdm` - The cached reference imported at module load time

HuggingFace's datasets library imports tqdm at module load time via:
    from .utils import tqdm as hf_tqdm
This caches a reference to the original class in arrow_dataset.py. Simply patching
datasets.utils.tqdm.tqdm is NOT enough - you must also patch the cached reference.

Usage:
    from datasets import load_dataset
    from .tqdm_websocket_bridge import create_tqdm_websocket_callback

    # Create callback with dataset context
    tqdm_class = create_tqdm_websocket_callback(
        dataset_id="abc-123",
        base_progress=10.0,
        progress_range=60.0  # Maps tqdm 0-100% to 10-70% in our system
    )

    # Monkey-patch tqdm at ALL locations for multiprocessing support
    import sys
    import importlib
    from tqdm import tqdm as original_tqdm
    # Use importlib to get the MODULE (datasets.utils re-exports the class, not module)
    hf_tqdm_module = importlib.import_module('datasets.utils.tqdm')
    arrow_dataset_module = importlib.import_module('datasets.arrow_dataset')

    # Save originals
    original_hf_tqdm = hf_tqdm_module.tqdm
    original_arrow_tqdm = arrow_dataset_module.hf_tqdm

    # Apply patches
    sys.modules['tqdm'].tqdm = tqdm_class
    sys.modules['tqdm.auto'].tqdm = tqdm_class
    hf_tqdm_module.tqdm = tqdm_class  # Patch module-level class
    arrow_dataset_module.hf_tqdm = tqdm_class  # CRITICAL: Patch cached reference!

    # Download/tokenize with progress tracking
    dataset = load_dataset(...)
    # or tokenized = dataset.map(..., num_proc=4)

    # Restore original tqdm
    sys.modules['tqdm'].tqdm = original_tqdm
    sys.modules['tqdm.auto'].tqdm = original_tqdm
    hf_tqdm_module.tqdm = original_hf_tqdm
    arrow_dataset_module.hf_tqdm = original_arrow_tqdm
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

    #: Consecutive write failures before escalating WARNING -> ERROR.
    DB_FAILURE_ALARM = 5

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
        cancel_scope: Optional[str] = None,
        cancel_target: Optional[str] = None,
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
        # Consecutive failed database progress writes. A single dropped tick is
        # survivable; an unbroken run of them means the row is frozen and the
        # job cannot report its own result (see the handler below).
        self._db_write_failures = 0
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

        # THE ONLY OWNER-PROCESS CHECKPOINT DURING A FORKED MAP.
        #
        # `Dataset.map(num_proc=N)` forks a worker pool, and the mapper runs in
        # the children — a child cannot cleanly stop its siblings, and one that
        # dies turns into "One of the subprocesses has abruptly died", a cancel
        # indistinguishable from a crash. But `update()` runs in the PARENT:
        # datasets funnels every batch's progress back through a manager queue
        # and calls `pbar.update(...)` there. So this is where a cancellation
        # can be both observed and acted on.
        #
        # The same applies to a HuggingFace download, where tqdm is the only
        # in-process callback `snapshot_download` offers at all.
        self._cancel = None
        if cancel_scope and cancel_target:
            from ..core.cancellation import cancel_checker

            self._cancel = cancel_checker(cancel_scope, cancel_target)

    def update(self, n=1):
        """
        Override tqdm's update method to emit WebSocket progress.

        Args:
            n: Number of items to increment progress by
        """
        # POLL FIRST, before the parent bookkeeping and before any emission.
        # `raise_if_cancelled` throttles itself, so this costs one monotonic
        # comparison on the overwhelming majority of ticks.
        #
        # The raise crosses this module's own `except Exception` handlers below
        # AND `datasets`' internal ones. `OperatorCancelled` derives from
        # BaseException for exactly that reason; an Exception here would be
        # logged as a dropped progress tick and the job would run to completion.
        if self._cancel is not None:
            self._cancel.raise_if_cancelled(
                f"stopped at {self.n} of {self.total or '?'}"
            )

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
                # getattr, not attribute access: tqdm's __init__ RETURNS EARLY
                # when the bar is constructed with disable=True and never sets
                # `desc`. HuggingFace disables bars whenever its progress-bar
                # env flag is set — and a Celery worker has no tty. The
                # AttributeError was then caught by the handler below and
                # logged as a dropped progress tick, so the row silently
                # stopped moving while the job ran on.
                desc = getattr(self, "desc", None) or (
                    "Tokenizing" if self.tokenization_id else "Downloading"
                )

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
                    from ..core.database import get_sync_db
                    from ..models.dataset import Dataset

                    with get_sync_db() as db:
                        if self.tokenization_id:
                            # Update tokenization progress
                            #
                            # `DatasetTokenization` lives in
                            # `models/dataset_tokenization.py`, NOT `models/dataset.py`.
                            # The line above correctly imports `Dataset` from
                            # `..models.dataset` and this one copied that path,
                            # so it raised ImportError on EVERY progress tick.
                            # The caller catches it as "Failed to update database
                            # progress" and continues, so tokenization ran to
                            # completion while its row stayed frozen at whatever
                            # value it last held — reported 2026-08-24 as a job
                            # "stuck at 40%" that had in fact finished 789,850
                            # samples in 6m30s.
                            from ..models.dataset_tokenization import DatasetTokenization
                            tokenization_obj = db.query(DatasetTokenization).filter_by(id=self.tokenization_id).first()
                            if tokenization_obj:
                                tokenization_obj.progress = mapped_progress  # Store as 0-100 percentage
                                db.commit()
                        else:
                            # Update dataset progress (for downloads)
                            dataset_uuid = UUID(self.dataset_id)
                            dataset_obj = db.query(Dataset).filter_by(id=dataset_uuid).first()
                            if dataset_obj:
                                dataset_obj.progress = mapped_progress / 100.0  # Store as 0.0-1.0 fraction
                                db.commit()
                except Exception as e:
                    # A dropped progress tick is survivable. A dropped tick
                    # EVERY time is not — the row freezes and the job becomes
                    # unusable however well it computes.
                    #
                    # 2026-08-24: a bad import here raised on every tick for
                    # seven months. Tokenization ran to completion — 789,850
                    # samples in 6m30s — while the row sat at 40%. The operator
                    # saw a stuck job and deleted finished work. "Don't let
                    # database errors break the operation" is right for one
                    # tick and catastrophic as a standing policy, because
                    # nothing ever escalates.
                    #
                    # So: warn once, then escalate. A progress writer that has
                    # never once succeeded is broken, not unlucky, and the job
                    # depending on it cannot report its own result.
                    self._db_write_failures += 1
                    if self._db_write_failures == 1:
                        logger.warning(f"Failed to update database progress: {e}")
                    elif self._db_write_failures == self.DB_FAILURE_ALARM:
                        logger.error(
                            "Progress writes have failed %d consecutive times "
                            "(%s). The row is frozen and this job cannot report "
                            "its own completion — treat any 'stuck' progress as "
                            "this, not as a hung job.",
                            self._db_write_failures, e, exc_info=True,
                        )
                else:
                    self._db_write_failures = 0

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
    cancel_scope: Optional[str] = None,
    cancel_target: Optional[str] = None,
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
        cancel_scope: A `core.cancellation` scope name. When given with
            `cancel_target`, every `update()` becomes a cancellation
            checkpoint — the only one that exists inside a forked
            `Dataset.map` or a HuggingFace `snapshot_download`.
        cancel_target: The row id to poll within that scope.

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
            kwargs.setdefault('cancel_scope', cancel_scope)
            kwargs.setdefault('cancel_target', cancel_target)
            super().__init__(*args, **kwargs)

    return ConfiguredTqdmWebSocket
