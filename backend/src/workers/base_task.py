"""
Base task class for Celery workers with database session management.

This module provides a base class for all Celery tasks that need database access.
It automatically manages sync database sessions and provides standardized error handling.
"""

import logging
from typing import Any, Dict, Optional
from celery import Task
from sqlalchemy.orm import Session

from ..core.database import get_sync_db

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """
    Base class for Celery tasks that need database access.

    This class provides:
    - Automatic sync database session management
    - Standardized error handling and logging
    - WebSocket progress emission utilities (via subclasses)

    Usage:
        ```python
        from app.workers.base_task import DatabaseTask
        from app.workers.celery import celery_app

        @celery_app.task(base=DatabaseTask, bind=True)
        def process_dataset(self, dataset_id: str):
            # Database session available via self.get_db()
            with self.get_db() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                # ... process dataset ...
        ```

    Notes:
        - Use `bind=True` in task decorator to access self
        - Use `get_db()` context manager for database access
        - Database sessions are automatically committed/rolled back
        - Exceptions are logged with task context
    """

    _db_session: Optional[Session] = None

    def get_db(self):
        """
        Get a database session for this task.

        Returns:
            Context manager that yields a sync database session

        Usage:
            ```python
            with self.get_db() as db:
                result = db.query(Model).filter_by(id=some_id).first()
            ```
        """
        return get_sync_db()

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """
        Error handler called when the task fails.

        Args:
            exc: The exception raised by the task
            task_id: Unique ID of the failed task
            args: Positional arguments passed to the task
            kwargs: Keyword arguments passed to the task
            einfo: Exception information (traceback)

        Notes:
            - Override this method to add custom error handling
            - Always call super().on_failure() to maintain base behavior
        """
        logger.error(
            f"Task {self.name}[{task_id}] failed with exception: {exc}",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "args": args,
                "kwargs": kwargs,
                "exception": str(exc),
            },
            exc_info=einfo,
        )
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_success(self, retval, task_id, args, kwargs):
        """
        Success handler called when the task completes successfully.

        Args:
            retval: Return value of the task
            task_id: Unique ID of the successful task
            args: Positional arguments passed to the task
            kwargs: Keyword arguments passed to the task

        Notes:
            - Override this method to add custom success handling
            - Always call super().on_success() to maintain base behavior
        """
        logger.info(
            f"Task {self.name}[{task_id}] completed successfully",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "args": args,
                "kwargs": kwargs,
            },
        )
        super().on_success(retval, task_id, args, kwargs)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """
        Retry handler called when the task is retried.

        Args:
            exc: The exception that caused the retry
            task_id: Unique ID of the task being retried
            args: Positional arguments passed to the task
            kwargs: Keyword arguments passed to the task
            einfo: Exception information (traceback)

        Notes:
            - Override this method to add custom retry handling
            - Always call super().on_retry() to maintain base behavior
        """
        logger.warning(
            f"Task {self.name}[{task_id}] is being retried due to: {exc}",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "args": args,
                "kwargs": kwargs,
                "exception": str(exc),
            },
        )
        super().on_retry(exc, task_id, args, kwargs, einfo)

    def update_progress(
        self,
        db: Session,
        model_class: Any,
        record_id: str,
        progress: float,
        status: Optional[str] = None,
        error_message: Optional[str] = None,
        **extra_fields,
    ):
        """
        Update progress and status for a database record.

        Args:
            db: Database session
            model_class: SQLAlchemy model class (Dataset, Model, etc.)
            record_id: ID of the record to update
            progress: Progress percentage (0-100)
            status: Optional status string
            error_message: Optional error message
            **extra_fields: Additional fields to update

        Usage:
            ```python
            self.update_progress(
                db=db,
                model_class=Dataset,
                record_id=dataset_id,
                progress=50.0,
                status="processing",
            )
            ```

        Notes:
            - Commits the update immediately
            - Handles exceptions gracefully
        """
        try:
            record = db.query(model_class).filter_by(id=record_id).first()
            if not record:
                logger.error(f"Record {record_id} not found in {model_class.__name__}")
                return

            # Update progress
            record.progress = progress

            # Update optional fields
            if status is not None:
                record.status = status
            if error_message is not None:
                record.error_message = error_message

            # Update any extra fields
            for field_name, field_value in extra_fields.items():
                if hasattr(record, field_name):
                    setattr(record, field_name, field_value)
                else:
                    logger.warning(
                        f"Field {field_name} not found on {model_class.__name__}"
                    )

            db.commit()
            db.refresh(record)

        except Exception as e:
            logger.error(f"Failed to update progress for {record_id}: {e}")
            db.rollback()
