"""
Shared WebSocket progress emission utility for Celery workers.

This module provides a standardized way for Celery workers to emit progress
updates via WebSocket through an internal HTTP callback endpoint. This approach
allows workers to communicate with connected clients without direct WebSocket
connections.

Usage:
    from ..workers.websocket_emitter import emit_progress

    # Emit progress update
    emit_progress(
        channel="datasets/abc-123/progress",
        event="progress",
        data={
            "progress": 50.0,
            "status": "processing",
            "message": "Processing dataset..."
        }
    )

    # Or use the convenience functions
    emit_dataset_progress(dataset_id="abc-123", progress=50.0, status="processing", message="...")
    emit_model_progress(model_id="m_xyz-789", progress=50.0, status="downloading", message="...")
"""

import logging
from typing import Any, Dict, Optional

import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)


def emit_progress(
    channel: str,
    event: str,
    data: Dict[str, Any],
    timeout: float = 1.0,
) -> bool:
    """
    Emit progress update via WebSocket through HTTP callback.

    This function sends a WebSocket event through the internal API endpoint,
    which then broadcasts it to all connected WebSocket clients listening
    on the specified channel.

    Args:
        channel: WebSocket channel to emit to (e.g., "datasets/{id}/progress")
        event: Event type (e.g., "progress", "completed", "error")
        data: Event data payload (must be JSON-serializable)
        timeout: Request timeout in seconds (default: 1.0)

    Returns:
        True if emission succeeded, False otherwise

    Examples:
        >>> emit_progress(
        ...     channel="datasets/abc-123/progress",
        ...     event="progress",
        ...     data={"progress": 50.0, "status": "processing"}
        ... )
        True

        >>> emit_progress(
        ...     channel="models/m_xyz-789/progress",
        ...     event="completed",
        ...     data={"status": "ready", "message": "Model loaded successfully"}
        ... )
        True
    """
    try:
        # Use configured WebSocket emit URL (supports both local and Docker deployments)
        api_url = settings.websocket_emit_url

        # Prepare payload
        payload = {
            "channel": channel,
            "event": event,
            "data": data,
        }

        # Send HTTP POST request to internal emission endpoint
        with httpx.Client() as client:
            response = client.post(
                api_url,
                json=payload,
                timeout=timeout,
            )

            # Log result
            if response.status_code == 200:
                logger.debug(f"WebSocket emit: {event} to {channel} - Success")
                return True
            else:
                logger.warning(
                    f"WebSocket emit: {event} to {channel} - "
                    f"Failed with status {response.status_code}"
                )
                return False

    except httpx.TimeoutException:
        logger.warning(f"WebSocket emit timeout: {event} to {channel}")
        return False

    except Exception as e:
        logger.error(f"Failed to emit WebSocket event: {e}", exc_info=True)
        return False


def emit_dataset_progress(
    dataset_id: str,
    event: str,
    data: Dict[str, Any],
) -> bool:
    """
    Emit progress update for a dataset operation.

    Convenience function that automatically constructs the channel name
    for dataset progress updates.

    Args:
        dataset_id: Dataset UUID or ID
        event: Event type (e.g., "progress", "completed", "error")
        data: Event data payload

    Returns:
        True if emission succeeded, False otherwise

    Examples:
        >>> emit_dataset_progress(
        ...     dataset_id="abc-123",
        ...     event="progress",
        ...     data={
        ...         "dataset_id": "abc-123",
        ...         "progress": 50.0,
        ...         "status": "processing",
        ...         "message": "Tokenizing dataset..."
        ...     }
        ... )
        True
    """
    channel = f"datasets/{dataset_id}/progress"
    return emit_progress(channel, event, data)


def emit_model_progress(
    model_id: str,
    event: str,
    data: Dict[str, Any],
) -> bool:
    """
    Emit progress update for a model operation.

    Convenience function that automatically constructs the channel name
    for model progress updates.

    Args:
        model_id: Model ID
        event: Event type (e.g., "progress", "completed", "error")
        data: Event data payload

    Returns:
        True if emission succeeded, False otherwise

    Examples:
        >>> emit_model_progress(
        ...     model_id="m_xyz-789",
        ...     event="progress",
        ...     data={
        ...         "model_id": "m_xyz-789",
        ...         "progress": 75.0,
        ...         "status": "downloading",
        ...         "message": "Downloaded 3GB / 4GB"
        ...     }
        ... )
        True
    """
    channel = f"models/{model_id}/progress"
    return emit_progress(channel, event, data)


def emit_extraction_progress(
    model_id: str,
    extraction_id: str,
    progress: float,
    status: str,
    message: str,
) -> bool:
    """
    Emit progress update for activation extraction.

    Convenience function for activation extraction progress with
    standardized payload structure.

    Args:
        model_id: Model ID being extracted from
        extraction_id: Unique extraction operation ID
        progress: Progress percentage (0-100)
        status: Current status (extracting, processing, saving, complete, error)
        message: Human-readable status message

    Returns:
        True if emission succeeded, False otherwise

    Examples:
        >>> emit_extraction_progress(
        ...     model_id="m_xyz-789",
        ...     extraction_id="ext_20250112_153045",
        ...     progress=60.0,
        ...     status="extracting",
        ...     message="Processing batch 6/10"
        ... )
        True
    """
    channel = f"models/{model_id}/extraction"
    data = {
        "type": "extraction_progress",
        "model_id": model_id,
        "extraction_id": extraction_id,
        "progress": progress,
        "status": status,
        "message": message,
    }
    return emit_progress(channel, "progress", data)


def emit_extraction_failed(
    model_id: str,
    extraction_id: str,
    error_message: str,
    error_type: str = "UNKNOWN",
    suggested_retry_params: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Emit dedicated failure event for activation extraction.

    This function emits a specialized failure event with error classification
    and suggested retry parameters, allowing the frontend to provide actionable
    recovery options to the user.

    Args:
        model_id: Model ID being extracted from
        extraction_id: Unique extraction operation ID
        error_message: Human-readable error message
        error_type: Error classification (OOM, VALIDATION, TIMEOUT, UNKNOWN)
        suggested_retry_params: Optional dict with suggested retry parameters
                               (e.g., {"batch_size": 4} for OOM errors)

    Returns:
        True if emission succeeded, False otherwise

    Examples:
        >>> emit_extraction_failed(
        ...     model_id="m_xyz-789",
        ...     extraction_id="ext_20250115_031430",
        ...     error_message="CUDA out of memory. Tried to allocate 2.00 GiB",
        ...     error_type="OOM",
        ...     suggested_retry_params={"batch_size": 4}
        ... )
        True
    """
    channel = f"models/{model_id}/extraction"
    data = {
        "type": "extraction_failed",
        "model_id": model_id,
        "extraction_id": extraction_id,
        "error_type": error_type,
        "error_message": error_message,
        "suggested_retry_params": suggested_retry_params or {},
        "retry_available": True,
        "cancel_available": True,
    }
    return emit_progress(channel, "failed", data)


# Export public API
__all__ = [
    "emit_progress",
    "emit_dataset_progress",
    "emit_model_progress",
    "emit_extraction_progress",
    "emit_extraction_failed",
]
