"""
Celery application configuration for background task processing.

This module initializes Celery with Redis broker for distributed task queue
processing, including dataset downloads, tokenization, and training jobs.

⚠️ IMPORTANT: Worker Startup Configuration
===========================================
Workers MUST be started with explicit queue configuration using the -Q flag!

❌ WRONG (will only listen to default "celery" queue):
    celery -A src.core.celery_app worker --loglevel=info

✅ CORRECT (listens to all required queues):
    celery -A src.core.celery_app worker -Q high_priority,datasets,processing,training,extraction,low_priority -c 8 --loglevel=info

OR use the startup script:
    ./backend/start-celery-worker.sh

See backend/CELERY_WORKERS.md for full documentation.
"""

from celery import Celery
from celery.signals import task_failure, task_success, worker_ready

from .config import settings

# Apply transformers compatibility patches BEFORE any task imports
# This prevents import errors during autodiscovery
from ..ml.transformers_compat import patch_transformers_compatibility
patch_transformers_compatibility()

# Initialize Celery app
celery_app = Celery(
    "mistudio",
    broker=str(settings.celery_broker_url),
    backend=str(settings.celery_result_backend),
)

# Celery configuration
celery_app.conf.update(
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task routing - Multi-queue architecture for optimal resource allocation
    task_routes={
        # High priority: Quick operations, metadata updates
        "src.workers.quick_tasks.*": {
            "queue": "high_priority",
            "priority": 10,
        },

        # Dataset operations: I/O bound, medium concurrency
        "src.workers.dataset_tasks.download_dataset_task": {
            "queue": "datasets",
            "priority": 7,
        },
        "src.workers.dataset_tasks.tokenize_dataset_task": {
            "queue": "datasets",  # Changed from "processing" to "datasets" for consistency
            "priority": 7,
        },

        # Model operations: GPU bound, high priority
        "src.workers.model_tasks.download_and_load_model": {
            "queue": "high_priority",
            "priority": 8,
        },

        # Training operations: GPU bound, low concurrency
        "src.workers.training_tasks.*": {
            "queue": "training",
            "priority": 5,
        },

        # Extraction operations: GPU bound, medium concurrency
        "src.workers.extraction_tasks.*": {
            "queue": "extraction",
            "priority": 5,
        },

        # Labeling operations: LLM bound, medium priority
        "src.workers.labeling_tasks.*": {
            "queue": "processing",
            "priority": 6,
        },

        # Maintenance operations: Background tasks
        "src.workers.maintenance_tasks.*": {
            "queue": "low_priority",
            "priority": 3,
        },

        # System monitoring operations: Background metrics collection
        "src.workers.system_monitor_tasks.*": {
            "queue": "low_priority",
            "priority": 2,
        },

        # Cleanup operations: Periodic maintenance tasks
        "src.workers.cleanup_stuck_extractions.*": {
            "queue": "low_priority",
            "priority": 3,
        },
    },

    # Task priority queues (higher priority = processed first)
    task_queue_max_priority=10,
    task_default_priority=5,

    # Queue configuration
    task_default_queue="datasets",  # Default queue for unrouted tasks
    task_create_missing_queues=True,  # Auto-create queues if they don't exist

    # Task time limits (soft/hard)
    # Training tasks can take 5-10 hours for 100k steps, so set generous limits
    task_soft_time_limit=36000,  # 10 hour soft limit (training tasks need this)
    task_time_limit=43200,  # 12 hour hard limit (safety margin)

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        "master_name": "mistudio",
    },

    # Task execution options
    task_acks_late=True,  # Acknowledge after task completion
    task_reject_on_worker_lost=True,  # Reject tasks if worker crashes
    worker_prefetch_multiplier=1,  # Disable prefetching for fair distribution

    # Beat scheduler settings (for periodic tasks)
    beat_schedule={
        # System metrics monitoring - runs every N seconds (configurable via SYSTEM_MONITOR_INTERVAL_SECONDS)
        "monitor-system-metrics": {
            "task": "workers.monitor_system_metrics",
            "schedule": settings.system_monitor_interval_seconds,  # Run every 2 seconds (default)
            "options": {
                "queue": "low_priority",
                "priority": 2,
            },
        },
        # Cleanup stuck extraction jobs - runs every 10 minutes
        "cleanup-stuck-extractions": {
            "task": "cleanup_stuck_extractions",
            "schedule": 600.0,  # Run every 10 minutes (600 seconds)
            "options": {
                "queue": "low_priority",
                "priority": 3,
            },
        },
    },

    # Worker settings
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks (memory cleanup)
    worker_disable_rate_limits=False,

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Task autodiscovery - automatically discover tasks in these modules
celery_app.autodiscover_tasks(
    [
        "src.workers.dataset_tasks",
        "src.workers.model_tasks",
        "src.workers.training_tasks",
        "src.workers.extraction_tasks",
        "src.workers.labeling_tasks",
        "src.workers.system_monitor_tasks",
        "src.workers.cleanup_stuck_extractions",
    ],
    force=True,
)


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """
    Signal handler called when Celery worker is ready.

    Args:
        sender: Worker instance
        **kwargs: Additional arguments
    """
    print(f"Celery worker ready: {sender.hostname}")


@task_success.connect
def on_task_success(sender=None, result=None, **kwargs):
    """
    Signal handler called when task completes successfully.

    Args:
        sender: Task instance
        result: Task result
        **kwargs: Additional arguments
    """
    if settings.is_development:
        print(f"Task success: {sender.name} - Result: {result}")


@task_failure.connect
def on_task_failure(sender=None, task_id=None, exception=None, **kwargs):
    """
    Signal handler called when task fails.

    Args:
        sender: Task instance
        task_id: Task ID
        exception: Exception that caused failure
        **kwargs: Additional arguments
    """
    print(f"Task failure: {sender.name} (ID: {task_id}) - Error: {exception}")


def create_task_signature(task_name: str, args: tuple = (), kwargs: dict = None, **options):
    """
    Create a task signature for delayed execution.

    Args:
        task_name: Full task name (e.g., 'app.workers.dataset_tasks.download_dataset_task')
        args: Positional arguments for task
        kwargs: Keyword arguments for task
        **options: Additional Celery options (countdown, eta, priority, etc.)

    Returns:
        celery.canvas.Signature: Task signature

    Usage:
        ```python
        from app.core.celery_app import create_task_signature

        # Create signature
        sig = create_task_signature(
            'app.workers.dataset_tasks.download_dataset_task',
            args=('ds_123', 'roneneldan/TinyStories'),
            priority=9
        )

        # Execute task
        result = sig.apply_async()
        ```
    """
    kwargs = kwargs or {}
    return celery_app.signature(task_name, args=args, kwargs=kwargs, **options)


def get_task_status(task_id: str) -> dict:
    """
    Get status of a Celery task.

    Args:
        task_id: Task ID returned from apply_async()

    Returns:
        dict: Task status information
            - state: PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED
            - result: Task result (if completed)
            - traceback: Error traceback (if failed)
            - info: Additional task information

    Usage:
        ```python
        from app.core.celery_app import get_task_status

        status = get_task_status('task-uuid-123')
        if status['state'] == 'SUCCESS':
            print(f"Result: {status['result']}")
        elif status['state'] == 'FAILURE':
            print(f"Error: {status['traceback']}")
        ```
    """
    result = celery_app.AsyncResult(task_id)
    return {
        "state": result.state,
        "result": result.result if result.successful() else None,
        "traceback": result.traceback if result.failed() else None,
        "info": result.info,
    }


def revoke_task(task_id: str, terminate: bool = False) -> None:
    """
    Revoke (cancel) a Celery task.

    Args:
        task_id: Task ID to revoke
        terminate: If True, terminate task immediately (sends SIGTERM)
                  If False, task won't start if not already running

    Usage:
        ```python
        from app.core.celery_app import revoke_task

        # Stop task gracefully (if not started)
        revoke_task('task-uuid-123')

        # Force terminate running task
        revoke_task('task-uuid-123', terminate=True)
        ```

    Notes:
        - Terminated tasks cannot be recovered
        - Use terminate=True only when necessary
        - Task status will be set to REVOKED
    """
    celery_app.control.revoke(task_id, terminate=terminate)


def get_queue_lengths() -> dict:
    """
    Get the length of all Celery queues.

    Returns:
        dict: Queue names mapped to number of pending tasks

    Usage:
        ```python
        from app.core.celery_app import get_queue_lengths

        lengths = get_queue_lengths()
        # {'high_priority': 0, 'datasets': 2, 'processing': 1, ...}
        ```
    """
    from kombu import Connection

    queue_names = [
        "high_priority",
        "datasets",
        "processing",
        "training",
        "extraction",
        "low_priority",
    ]

    with Connection(str(settings.celery_broker_url)) as conn:
        lengths = {}
        for queue_name in queue_names:
            try:
                queue = conn.SimpleQueue(queue_name)
                lengths[queue_name] = queue.qsize()
                queue.close()
            except Exception as e:
                # Queue doesn't exist yet or error accessing it
                lengths[queue_name] = 0

        return lengths


def get_active_tasks() -> dict:
    """
    Get currently active (running) tasks across all workers.

    Returns:
        dict: Worker names mapped to list of active tasks

    Usage:
        ```python
        from app.core.celery_app import get_active_tasks

        active = get_active_tasks()
        # {
        #     'celery@worker1': [
        #         {'id': 'task-123', 'name': 'download_dataset_task', ...}
        #     ]
        # }
        ```
    """
    inspect = celery_app.control.inspect()
    active = inspect.active()
    return active or {}


def get_worker_stats() -> dict:
    """
    Get statistics for all connected workers.

    Returns:
        dict: Worker statistics including queue assignments and resource usage

    Usage:
        ```python
        from app.core.celery_app import get_worker_stats

        stats = get_worker_stats()
        ```
    """
    inspect = celery_app.control.inspect()

    stats = {}
    active_queues = inspect.active_queues()
    stats_data = inspect.stats()

    if active_queues:
        for worker, queues in active_queues.items():
            worker_stats = {
                "queues": [q["name"] for q in queues],
                "queue_details": queues,
            }

            # Add additional stats if available
            if stats_data and worker in stats_data:
                worker_stats["stats"] = stats_data[worker]

            stats[worker] = worker_stats

    return stats


# Export commonly used objects
__all__ = [
    "celery_app",
    "create_task_signature",
    "get_task_status",
    "revoke_task",
    "get_queue_lengths",
    "get_active_tasks",
    "get_worker_stats",
]
