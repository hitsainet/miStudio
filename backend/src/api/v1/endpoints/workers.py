"""
Worker and queue monitoring API endpoints.

This module provides endpoints for monitoring Celery workers, queues,
and task execution status.
"""

from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from ....core.celery_app import (
    get_queue_lengths,
    get_active_tasks,
    get_worker_stats,
)

router = APIRouter(prefix="/workers", tags=["workers"])


@router.get("/queues", response_model=Dict[str, int])
async def get_queues():
    """
    Get the number of pending tasks in each queue.

    Returns:
        Dictionary mapping queue names to task counts

    Example response:
        ```json
        {
            "high_priority": 0,
            "datasets": 2,
            "processing": 1,
            "training": 1,
            "extraction": 0,
            "low_priority": 0
        }
        ```
    """
    try:
        return get_queue_lengths()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get queue lengths: {str(e)}"
        )


@router.get("/active", response_model=Dict[str, Any])
async def get_active():
    """
    Get currently active (running) tasks across all workers.

    Returns:
        Dictionary mapping worker names to lists of active tasks

    Example response:
        ```json
        {
            "celery@worker-datasets": [
                {
                    "id": "task-uuid-123",
                    "name": "src.workers.dataset_tasks.download_dataset_task",
                    "args": ["dataset-id", "repo/name"],
                    "kwargs": {},
                    "time_start": 1234567890.0,
                    "worker_pid": 12345
                }
            ],
            "celery@worker-training": []
        }
        ```
    """
    try:
        return get_active_tasks()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get active tasks: {str(e)}"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """
    Get statistics for all connected workers.

    Returns:
        Dictionary with worker statistics including:
        - Connected workers
        - Queue assignments
        - Resource usage
        - Task processing stats

    Example response:
        ```json
        {
            "celery@worker-datasets": {
                "queues": ["datasets", "processing"],
                "queue_details": [
                    {
                        "name": "datasets",
                        "exchange": {"name": "datasets"},
                        "routing_key": "datasets"
                    }
                ],
                "stats": {
                    "total": {"src.workers.dataset_tasks.download_dataset_task": 5},
                    "pool": {"max-concurrency": 4}
                }
            }
        }
        ```
    """
    try:
        return get_worker_stats()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get worker stats: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def get_health():
    """
    Get overall health status of the worker system.

    Returns:
        Health check information including:
        - Number of connected workers
        - Queue status
        - Active task count
        - Any warnings or issues

    Example response:
        ```json
        {
            "status": "healthy",
            "workers_connected": 3,
            "total_queued": 3,
            "total_active": 2,
            "queues": {
                "high_priority": {"pending": 0, "status": "ok"},
                "datasets": {"pending": 2, "status": "ok"},
                "processing": {"pending": 1, "status": "ok"},
                "training": {"pending": 0, "status": "ok"},
                "extraction": {"pending": 0, "status": "ok"},
                "low_priority": {"pending": 0, "status": "ok"}
            },
            "warnings": []
        }
        ```
    """
    try:
        queue_lengths = get_queue_lengths()
        active_tasks = get_active_tasks()
        worker_stats = get_worker_stats()

        total_queued = sum(queue_lengths.values())
        total_active = sum(len(tasks) for tasks in active_tasks.values())
        workers_connected = len(worker_stats)

        # Check for issues
        warnings = []
        if workers_connected == 0:
            warnings.append("No workers connected")

        # Check for stuck queues (many pending, no active tasks)
        for queue_name, pending in queue_lengths.items():
            if pending > 10:
                # Check if any worker is handling this queue
                queue_assigned = any(
                    queue_name in worker["queues"]
                    for worker in worker_stats.values()
                )
                if not queue_assigned:
                    warnings.append(
                        f"Queue '{queue_name}' has {pending} pending tasks but no worker assigned"
                    )

        # Determine overall status
        if warnings:
            status = "warning" if workers_connected > 0 else "critical"
        else:
            status = "healthy"

        return {
            "status": status,
            "workers_connected": workers_connected,
            "total_queued": total_queued,
            "total_active": total_active,
            "queues": {
                name: {
                    "pending": count,
                    "status": "ok" if count < 100 else "high_load"
                }
                for name, count in queue_lengths.items()
            },
            "warnings": warnings,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get health status: {str(e)}"
        )
