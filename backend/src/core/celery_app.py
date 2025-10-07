"""
Celery application configuration for background task processing.

This module initializes Celery with Redis broker for distributed task queue
processing, including dataset downloads, tokenization, and training jobs.
"""

from celery import Celery
from celery.signals import task_failure, task_success, worker_ready

from .config import settings

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

    # Task routing
    task_routes={
        "app.workers.dataset_tasks.*": {"queue": "datasets"},
        "app.workers.training_tasks.*": {"queue": "training"},
        "app.workers.extraction_tasks.*": {"queue": "extraction"},
    },

    # Task priority queues (higher priority = processed first)
    task_queue_max_priority=10,
    task_default_priority=5,

    # Task time limits (soft/hard)
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=7200,  # 2 hour hard limit

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
    beat_schedule={},  # To be populated with periodic tasks

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
        "app.workers.dataset_tasks",
        "app.workers.training_tasks",
        "app.workers.extraction_tasks",
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


# Export commonly used objects
__all__ = [
    "celery_app",
    "create_task_signature",
    "get_task_status",
    "revoke_task",
]
