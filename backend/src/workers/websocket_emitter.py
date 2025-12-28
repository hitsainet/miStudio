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

        # DEBUG: Print to see if function is called
        print(f"[EMIT DEBUG] Attempting to emit {event} to {channel}")

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
                print(f"[EMIT DEBUG] Success: {event} to {channel}")
                logger.debug(f"WebSocket emit: {event} to {channel} - Success")
                return True
            else:
                print(f"[EMIT DEBUG] Failed with status {response.status_code}: {event} to {channel}")
                logger.warning(
                    f"WebSocket emit: {event} to {channel} - "
                    f"Failed with status {response.status_code}"
                )
                return False

    except httpx.TimeoutException:
        print(f"[EMIT DEBUG] Timeout: {event} to {channel}")
        logger.warning(f"WebSocket emit timeout: {event} to {channel}")
        return False

    except Exception as e:
        print(f"[EMIT DEBUG] Exception: {e}")
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
    for dataset progress updates and adds namespace prefix to event names.

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
    # Add namespace prefix to event name for proper WebSocket routing
    namespaced_event = f"dataset:{event}"
    return emit_progress(channel, namespaced_event, data)


def emit_tokenization_progress(
    dataset_id: str,
    tokenization_id: str,
    progress: float,
    stage: str,
    samples_processed: int,
    total_samples: int,
    started_at: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
    estimated_seconds_remaining: Optional[float] = None,
    samples_per_second: Optional[float] = None,
    filter_stats: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Emit detailed tokenization progress update.

    Sends comprehensive progress information including stage, sample counts,
    time estimates, processing rate, and filtering statistics.

    Args:
        dataset_id: Dataset UUID
        tokenization_id: Tokenization job ID
        progress: Overall progress percentage (0-100)
        stage: Current processing stage ('loading', 'tokenizing', 'filtering', 'saving', 'complete')
        samples_processed: Number of samples processed so far
        total_samples: Total number of samples to process
        started_at: ISO timestamp when tokenization started
        elapsed_seconds: Time elapsed since start
        estimated_seconds_remaining: Estimated time to completion
        samples_per_second: Current processing rate
        filter_stats: Optional filtering statistics dict with keys:
            - samples_filtered: Number of samples filtered out
            - junk_tokens: Number of tokens classified as junk
            - total_tokens: Total tokens processed
            - filter_rate: Percentage of samples filtered

    Returns:
        True if emission succeeded, False otherwise

    Examples:
        >>> emit_tokenization_progress(
        ...     dataset_id="abc-123",
        ...     tokenization_id="tok_abc123",
        ...     progress=45.5,
        ...     stage="tokenizing",
        ...     samples_processed=4550,
        ...     total_samples=10000,
        ...     elapsed_seconds=120.5,
        ...     estimated_seconds_remaining=145.0,
        ...     samples_per_second=37.8,
        ...     filter_stats={
        ...         "samples_filtered": 250,
        ...         "junk_tokens": 15000,
        ...         "total_tokens": 500000,
        ...         "filter_rate": 2.5
        ...     }
        ... )
        True
    """
    data = {
        "tokenization_id": tokenization_id,
        "dataset_id": dataset_id,
        "progress": progress,
        "stage": stage,
        "samples_processed": samples_processed,
        "total_samples": total_samples,
    }

    # Add optional fields if provided
    if started_at is not None:
        data["started_at"] = started_at
    if elapsed_seconds is not None:
        data["elapsed_seconds"] = elapsed_seconds
    if estimated_seconds_remaining is not None:
        data["estimated_seconds_remaining"] = estimated_seconds_remaining
    if samples_per_second is not None:
        data["samples_per_second"] = samples_per_second
    if filter_stats is not None:
        data["filter_stats"] = filter_stats

    channel = f"datasets/{dataset_id}/tokenization/{tokenization_id}"
    namespaced_event = "tokenization:progress"
    return emit_progress(channel, namespaced_event, data)


def emit_model_progress(
    model_id: str,
    event: str,
    data: Dict[str, Any],
) -> bool:
    """
    Emit progress update for a model operation.

    Convenience function that automatically constructs the channel name
    for model progress updates and adds namespace prefix to event names.

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
    # Add namespace prefix to event name for proper WebSocket routing
    namespaced_event = f"model:{event}"
    return emit_progress(channel, namespaced_event, data)


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
    # Add namespace prefix to event name for proper WebSocket routing
    return emit_progress(channel, "extraction:progress", data)


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
    # Add namespace prefix to event name for proper WebSocket routing
    return emit_progress(channel, "extraction:failed", data)


def emit_extraction_deleted(
    extraction_id: str,
    feature_count: int = 0,
) -> bool:
    """
    Emit notification that an extraction job has been deleted.

    This function emits a deletion complete event, allowing the frontend
    to remove the extraction from its state after async background deletion.

    Args:
        extraction_id: ID of the deleted extraction
        feature_count: Number of features that were deleted

    Returns:
        True if emission succeeded, False otherwise

    Examples:
        >>> emit_extraction_deleted(
        ...     extraction_id="extr_20250115_031430_train_abc",
        ...     feature_count=8192
        ... )
        True
    """
    channel = f"extraction/{extraction_id}"
    data = {
        "type": "extraction_deleted",
        "extraction_id": extraction_id,
        "feature_count": feature_count,
        "status": "deleted",
    }
    # Add namespace prefix to event name for proper WebSocket routing
    return emit_progress(channel, "extraction:deleted", data)


def emit_extraction_deletion_progress(
    extraction_id: str,
    features_deleted: int,
    total_features: int,
    progress: float,
    status: str = "deleting",
    message: Optional[str] = None,
) -> bool:
    """
    Emit progress update during extraction deletion.

    Large extractions with many features take significant time to delete
    due to CASCADE deleting feature_activations. This provides real-time
    progress feedback during batch deletion.

    Args:
        extraction_id: ID of the extraction being deleted
        features_deleted: Number of features deleted so far
        total_features: Total number of features to delete
        progress: Progress percentage (0.0 to 1.0)
        status: Current status ('deleting', 'completed', 'failed')
        message: Optional human-readable status message

    Returns:
        True if emission succeeded, False otherwise

    Examples:
        >>> emit_extraction_deletion_progress(
        ...     extraction_id="extr_20250128_120000_train_abc",
        ...     features_deleted=4096,
        ...     total_features=8192,
        ...     progress=0.5,
        ...     status="deleting",
        ...     message="Deleting features batch 2/4..."
        ... )
        True
    """
    channel = f"extraction/{extraction_id}"
    data = {
        "extraction_id": extraction_id,
        "features_deleted": features_deleted,
        "total_features": total_features,
        "progress": progress,
        "status": status,
        "message": message or f"Deleting features ({features_deleted}/{total_features})...",
    }
    # Add namespace prefix to event name for proper WebSocket routing
    return emit_progress(channel, "extraction:deletion_progress", data)


def emit_training_progress(
    training_id: str,
    event: str,
    data: Dict[str, Any],
) -> bool:
    """
    Emit progress update for a training operation.

    Convenience function that automatically constructs the channel name
    for training progress updates.

    Args:
        training_id: Training job ID
        event: Event type (e.g., "created", "progress", "status_changed", "completed", "failed")
        data: Event data payload

    Returns:
        True if emission succeeded, False otherwise

    Examples:
        >>> emit_training_progress(
        ...     training_id="train_abc123",
        ...     event="progress",
        ...     data={
        ...         "training_id": "train_abc123",
        ...         "current_step": 1000,
        ...         "total_steps": 100000,
        ...         "progress": 1.0,
        ...         "loss": 0.0234,
        ...         "l0_sparsity": 0.05,
        ...         "dead_neurons": 42,
        ...         "learning_rate": 0.0003
        ...     }
        ... )
        True
    """
    channel = f"trainings/{training_id}/progress"
    return emit_progress(channel, event, data)


def emit_checkpoint_created(
    training_id: str,
    checkpoint_id: str,
    step: int,
    loss: float,
    is_best: bool,
    storage_path: str,
) -> bool:
    """
    Emit checkpoint creation event.

    Args:
        training_id: Training job ID
        checkpoint_id: Checkpoint ID
        step: Training step at checkpoint
        loss: Loss at checkpoint
        is_best: Whether this is the best checkpoint
        storage_path: Path to checkpoint file

    Returns:
        True if emission succeeded, False otherwise

    Examples:
        >>> emit_checkpoint_created(
        ...     training_id="train_abc123",
        ...     checkpoint_id="ckpt_def456",
        ...     step=5000,
        ...     loss=0.0198,
        ...     is_best=True,
        ...     storage_path="/data/trainings/train_abc123/checkpoints/step_5000.safetensors"
        ... )
        True
    """
    channel = f"trainings/{training_id}/checkpoints"
    data = {
        "training_id": training_id,
        "checkpoint_id": checkpoint_id,
        "step": step,
        "loss": loss,
        "is_best": is_best,
        "storage_path": storage_path,
    }
    return emit_progress(channel, "checkpoint_created", data)


# ============================================================================
# System Monitoring Emission Functions
# ============================================================================


def emit_gpu_metrics(
    gpu_id: int,
    metrics: Dict[str, Any],
) -> bool:
    """
    Emit metrics update for a specific GPU.

    Convenience function for GPU monitoring that automatically constructs
    the channel name for GPU-specific metrics.

    Args:
        gpu_id: GPU device ID (0, 1, 2, etc.)
        metrics: GPU metrics payload including utilization, memory, temperature, etc.

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        system/gpu/{gpu_id}

    Examples:
        >>> emit_gpu_metrics(
        ...     gpu_id=0,
        ...     metrics={
        ...         "gpu_id": 0,
        ...         "utilization": 85.5,
        ...         "memory_used": 7168,
        ...         "memory_total": 11264,
        ...         "memory_percent": 63.6,
        ...         "temperature": 72,
        ...         "power_usage": 245.3,
        ...         "timestamp": "2025-10-22T12:34:56Z"
        ...     }
        ... )
        True
    """
    channel = f"system/gpu/{gpu_id}"
    # Add namespace prefix to event name for proper WebSocket routing
    return emit_progress(channel, "system:metrics", metrics)


def emit_cpu_metrics(
    metrics: Dict[str, Any],
) -> bool:
    """
    Emit CPU utilization metrics.

    Convenience function for CPU monitoring that automatically constructs
    the channel name for system-wide CPU metrics.

    Args:
        metrics: CPU metrics payload including utilization, core count, etc.

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        system/cpu

    Examples:
        >>> emit_cpu_metrics(
        ...     metrics={
        ...         "cpu_percent": 45.2,
        ...         "cpu_count": 16,
        ...         "cpu_freq": 3.5,
        ...         "timestamp": "2025-10-22T12:34:56Z"
        ...     }
        ... )
        True
    """
    channel = "system/cpu"
    # Add namespace prefix to event name for proper WebSocket routing
    return emit_progress(channel, "system:metrics", metrics)


def emit_memory_metrics(
    metrics: Dict[str, Any],
) -> bool:
    """
    Emit RAM and Swap memory metrics.

    Convenience function for memory monitoring that automatically constructs
    the channel name for system-wide memory metrics.

    Args:
        metrics: Memory metrics payload including RAM and Swap usage

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        system/memory

    Examples:
        >>> emit_memory_metrics(
        ...     metrics={
        ...         "ram_used_gb": 24.5,
        ...         "ram_total_gb": 64.0,
        ...         "ram_percent": 38.3,
        ...         "swap_used_gb": 0.5,
        ...         "swap_total_gb": 8.0,
        ...         "swap_percent": 6.25,
        ...         "timestamp": "2025-10-22T12:34:56Z"
        ...     }
        ... )
        True
    """
    channel = "system/memory"
    # Add namespace prefix to event name for proper WebSocket routing
    return emit_progress(channel, "system:metrics", metrics)


def emit_disk_metrics(
    metrics: Dict[str, Any],
) -> bool:
    """
    Emit disk I/O metrics.

    Convenience function for disk monitoring that automatically constructs
    the channel name for system-wide disk I/O metrics.

    Args:
        metrics: Disk I/O metrics payload including read/write bytes

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        system/disk

    Examples:
        >>> emit_disk_metrics(
        ...     metrics={
        ...         "disk_read_mb": 1523.4,
        ...         "disk_write_mb": 892.1,
        ...         "timestamp": "2025-10-22T12:34:56Z"
        ...     }
        ... )
        True
    """
    channel = "system/disk"
    # Add namespace prefix to event name for proper WebSocket routing
    return emit_progress(channel, "system:metrics", metrics)


def emit_network_metrics(
    metrics: Dict[str, Any],
) -> bool:
    """
    Emit network I/O metrics.

    Convenience function for network monitoring that automatically constructs
    the channel name for system-wide network I/O metrics.

    Args:
        metrics: Network I/O metrics payload including sent/received bytes

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        system/network

    Examples:
        >>> emit_network_metrics(
        ...     metrics={
        ...         "network_sent_mb": 234.5,
        ...         "network_recv_mb": 567.8,
        ...         "timestamp": "2025-10-22T12:34:56Z"
        ...     }
        ... )
        True
    """
    channel = "system/network"
    # Add namespace prefix to event name for proper WebSocket routing
    return emit_progress(channel, "system:metrics", metrics)


def emit_system_metrics(
    metrics_type: str,
    metrics: Dict[str, Any],
) -> bool:
    """
    Generic system metrics emission function.

    This is a lower-level function for emitting any type of system metrics.
    Prefer using the specific convenience functions (emit_gpu_metrics, emit_cpu_metrics, etc.)
    for better type safety and clarity.

    Args:
        metrics_type: Type of metrics (gpu, cpu, memory, disk, network)
        metrics: Metrics data payload

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        system/{metrics_type}  OR  system/{metrics_type}/{id} for per-device metrics

    Examples:
        >>> emit_system_metrics(
        ...     metrics_type="cpu",
        ...     metrics={"cpu_percent": 45.2, "cpu_count": 16}
        ... )
        True
    """
    channel = f"system/{metrics_type}"
    return emit_progress(channel, "metrics", metrics)


# ============================================================================
# Feature Labeling Emission Functions
# ============================================================================


def emit_labeling_progress(
    labeling_job_id: str,
    event: str,
    data: Dict[str, Any],
) -> bool:
    """
    Emit progress update for a feature labeling operation.

    Convenience function that automatically constructs the channel name
    for labeling progress updates.

    Args:
        labeling_job_id: Labeling job ID
        event: Event type (e.g., "progress", "completed", "failed")
        data: Event data payload

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        labeling/{labeling_job_id}/progress

    Examples:
        >>> emit_labeling_progress(
        ...     labeling_job_id="label_extr_abc123_20250108",
        ...     event="progress",
        ...     data={
        ...         "labeling_job_id": "label_extr_abc123_20250108",
        ...         "extraction_job_id": "extr_abc123",
        ...         "progress": 0.45,
        ...         "features_labeled": 450,
        ...         "total_features": 1000,
        ...         "labeling_method": "openai",
        ...         "status": "labeling"
        ...     }
        ... )
        True
    """
    channel = f"labeling/{labeling_job_id}/progress"
    return emit_progress(channel, event, data)


def emit_labeling_result(
    labeling_job_id: str,
    feature_data: Dict[str, Any],
) -> bool:
    """
    Emit individual feature labeling result in real-time.

    Emits each labeled feature as it's completed, allowing frontend
    to display a live feed of labeling results with full activation context.

    Args:
        labeling_job_id: Labeling job ID
        feature_data: Feature labeling result containing:
            - feature_id: Neuron index (e.g., 23319)
            - label: Assigned feature name
            - category: Feature category (e.g., "semantic", "syntactic")
            - description: Feature description (optional)
            - examples: List of activation examples (up to 10) with structure:
                * prefix_tokens: List of tokens before prime token
                * prime_token: The token with highest activation
                * suffix_tokens: List of tokens after prime token
                * max_activation: Activation strength value

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        labeling/{labeling_job_id}/results

    Examples:
        >>> emit_labeling_result(
        ...     labeling_job_id="label_extr_abc123_20250108",
        ...     feature_data={
        ...         "feature_id": 23319,
        ...         "label": "common_prepositions",
        ...         "category": "semantic",
        ...         "description": "Detects common prepositions like 'of', 'to', 'at'",
        ...         "examples": [
        ...             {
        ...                 "prefix_tokens": ["the", "end"],
        ...                 "prime_token": "▁of",
        ...                 "suffix_tokens": ["the", "day"],
        ...                 "max_activation": 8.5
        ...             }
        ...         ]
        ...     }
        ... )
        True
    """
    channel = f"labeling/{labeling_job_id}/results"
    return emit_progress(channel, "result", feature_data)


def emit_deletion_progress(
    training_id: str,
    task: str,
    status: str = "completed",
    message: Optional[str] = None,
    count: Optional[int] = None,
) -> bool:
    """
    Emit deletion progress update for a training operation.

    This function emits real-time progress updates as different deletion
    sub-tasks complete (checkpoints, metrics, features, files, etc.).

    Args:
        training_id: Training job ID being deleted
        task: Deletion task identifier (e.g., "checkpoints", "metrics", "features", "files")
        status: Task status ("in_progress" or "completed")
        message: Optional human-readable status message
        count: Optional count of items deleted (e.g., number of features)

    Returns:
        True if emission succeeded, False otherwise

    Examples:
        >>> # Mark task as in progress
        >>> emit_deletion_progress(
        ...     training_id="train_abc123",
        ...     task="features",
        ...     status="in_progress",
        ...     message="Deleting features..."
        ... )
        True

        >>> # Mark task as completed
        >>> emit_deletion_progress(
        ...     training_id="train_abc123",
        ...     task="features",
        ...     status="completed",
        ...     message="Deleted 23,308 features",
        ...     count=23308
        ... )
        True
    """
    channel = f"trainings/{training_id}/deletion"
    data = {
        "training_id": training_id,
        "task": task,
        "status": status,
    }

    if message:
        data["message"] = message
    if count is not None:
        data["count"] = count

    return emit_progress(channel, "task_update", data)


# ============================================================================
# SAE Download/Upload Emission Functions
# ============================================================================


def emit_sae_download_progress(
    sae_id: str,
    progress: float,
    status: str,
    message: Optional[str] = None,
    stage: Optional[str] = None,
) -> bool:
    """
    Emit SAE download progress update.

    Args:
        sae_id: SAE UUID
        progress: Progress percentage (0-100)
        status: Status string (downloading, converting, ready, failed)
        message: Optional status message
        stage: Optional stage (download, convert)

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        sae/{sae_id}/download
    """
    channel = f"sae/{sae_id}/download"
    data = {
        "sae_id": sae_id,
        "progress": progress,
        "status": status,
    }

    if message:
        data["message"] = message
    if stage:
        data["stage"] = stage

    return emit_progress(channel, "sae:download", data)


def emit_sae_upload_progress(
    sae_id: str,
    progress: float,
    status: str,
    message: Optional[str] = None,
    stage: Optional[str] = None,
    repo_url: Optional[str] = None,
) -> bool:
    """
    Emit SAE upload progress update.

    Args:
        sae_id: SAE UUID
        progress: Progress percentage (0-100)
        status: Status string (converting, uploading, completed, failed)
        message: Optional status message
        stage: Optional stage (convert, upload)
        repo_url: Optional repository URL (on success)

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        sae/{sae_id}/upload
    """
    channel = f"sae/{sae_id}/upload"
    data = {
        "sae_id": sae_id,
        "progress": progress,
        "status": status,
    }

    if message:
        data["message"] = message
    if stage:
        data["stage"] = stage
    if repo_url:
        data["repo_url"] = repo_url

    return emit_progress(channel, "sae:upload", data)


def emit_sae_extraction_progress(
    sae_id: str,
    extraction_id: str,
    progress: float,
    status: str,
    message: str,
    features_extracted: Optional[int] = None,
    total_features: Optional[int] = None,
) -> bool:
    """
    Emit progress update for SAE feature extraction.

    Convenience function for SAE-based feature extraction progress with
    standardized payload structure.

    Args:
        sae_id: External SAE ID being extracted from
        extraction_id: Unique extraction job ID
        progress: Progress percentage (0-100)
        status: Current status (starting, extracting, processing, completed, failed)
        message: Human-readable status message
        features_extracted: Number of features processed so far
        total_features: Total number of features to process

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        sae/{sae_id}/extraction

    Examples:
        >>> emit_sae_extraction_progress(
        ...     sae_id="sae_abc123",
        ...     extraction_id="extr_20250128_120000_sae_abc123",
        ...     progress=60.0,
        ...     status="extracting",
        ...     message="Processing batch 6/10",
        ...     features_extracted=6000,
        ...     total_features=10000
        ... )
        True
    """
    channel = f"sae/{sae_id}/extraction"
    data = {
        "sae_id": sae_id,
        "extraction_id": extraction_id,
        "progress": progress,
        "status": status,
        "message": message,
    }

    if features_extracted is not None:
        data["features_extracted"] = features_extracted
    if total_features is not None:
        data["total_features"] = total_features

    return emit_progress(channel, "sae:extraction", data)


# ============================================================================
# Neuronpedia Export Emission Functions
# ============================================================================


def emit_export_progress(
    job_id: str,
    progress: float,
    stage: str,
    status: str,
    message: Optional[str] = None,
    feature_count: Optional[int] = None,
    output_path: Optional[str] = None,
) -> bool:
    """
    Emit Neuronpedia export progress update.

    Args:
        job_id: Export job UUID
        progress: Progress percentage (0-100)
        stage: Current stage (initializing, computing_logit_lens, computing_histograms,
               computing_top_tokens, generating_json, packaging, completed)
        status: Job status (pending, computing, packaging, completed, failed, cancelled)
        message: Optional status message
        feature_count: Optional total feature count
        output_path: Optional output archive path (when completed)

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        neuronpedia/{job_id}/export
    """
    channel = f"neuronpedia/{job_id}/export"
    data = {
        "job_id": job_id,
        "progress": progress,
        "stage": stage,
        "status": status,
    }

    if message:
        data["message"] = message
    if feature_count is not None:
        data["feature_count"] = feature_count
    if output_path:
        data["output_path"] = output_path

    return emit_progress(channel, "export:progress", data)


def emit_extraction_job_progress(
    extraction_id: str,
    training_id: Optional[str] = None,
    sae_id: Optional[str] = None,
    current_batch: int = 0,
    total_batches: int = 0,
    samples_processed: int = 0,
    total_samples: int = 0,
    progress: float = 0.0,
    samples_per_second: float = 0.0,
    eta_seconds: float = 0.0,
    status: str = "extracting",
    message: Optional[str] = None,
    features_in_heap: int = 0,
    heap_examples_count: int = 0,
    features_extracted: Optional[int] = None,
    total_features: Optional[int] = None,
) -> bool:
    """
    Emit detailed extraction job progress update (similar to training progress).

    This provides comprehensive progress metrics for extraction jobs, allowing
    the frontend to display live metrics similar to training "Show Metrics" modal.

    Args:
        extraction_id: Extraction job ID
        training_id: Training ID (for training-based extraction)
        sae_id: External SAE ID (for SAE-based extraction)
        current_batch: Current batch number
        total_batches: Total number of batches
        samples_processed: Number of samples processed so far
        total_samples: Total samples to process
        progress: Progress percentage (0.0 - 1.0)
        samples_per_second: Processing speed (samples/sec)
        eta_seconds: Estimated time remaining in seconds
        status: Current status (extracting, processing, completed, failed)
        message: Optional status message
        features_in_heap: Number of features with activations in heap
        heap_examples_count: Total examples stored in heap

    Returns:
        True if emission succeeded, False otherwise

    Channel Convention:
        extraction/{extraction_id}

    Examples:
        >>> emit_extraction_job_progress(
        ...     extraction_id="extr_20250128_120000_train_abc",
        ...     training_id="train_abc123",
        ...     current_batch=125,
        ...     total_batches=2500,
        ...     samples_processed=1000,
        ...     total_samples=20000,
        ...     progress=0.05,
        ...     samples_per_second=8.5,
        ...     eta_seconds=2235.0,
        ...     features_in_heap=15234,
        ...     heap_examples_count=76170
        ... )
        True
    """
    channel = f"extraction/{extraction_id}"
    data = {
        "extraction_id": extraction_id,
        "current_batch": current_batch,
        "total_batches": total_batches,
        "samples_processed": samples_processed,
        "total_samples": total_samples,
        "progress": progress,
        "samples_per_second": samples_per_second,
        "eta_seconds": eta_seconds,
        "status": status,
        "features_in_heap": features_in_heap,
        "heap_examples_count": heap_examples_count,
    }

    if training_id:
        data["training_id"] = training_id
    if sae_id:
        data["sae_id"] = sae_id
    if message:
        data["message"] = message
    if features_extracted is not None:
        data["features_extracted"] = features_extracted
    if total_features is not None:
        data["total_features"] = total_features

    return emit_progress(channel, "extraction:progress", data)


# Export public API
__all__ = [
    "emit_progress",
    "emit_dataset_progress",
    "emit_model_progress",
    "emit_extraction_progress",
    "emit_extraction_failed",
    "emit_extraction_deleted",
    "emit_extraction_deletion_progress",
    "emit_extraction_job_progress",
    "emit_training_progress",
    "emit_checkpoint_created",
    "emit_deletion_progress",
    # System monitoring functions
    "emit_system_metrics",
    "emit_gpu_metrics",
    "emit_cpu_metrics",
    "emit_memory_metrics",
    "emit_disk_metrics",
    "emit_network_metrics",
    # Feature labeling functions
    "emit_labeling_progress",
    "emit_labeling_result",
    # SAE download/upload/extraction functions
    "emit_sae_download_progress",
    "emit_sae_upload_progress",
    "emit_sae_extraction_progress",
    # Neuronpedia export functions
    "emit_export_progress",
]
