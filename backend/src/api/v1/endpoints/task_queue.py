"""
Task Queue API endpoints.

This module provides endpoints for viewing and managing background task operations,
including failed tasks that can be manually retried.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ....db.session import get_db
from ....services.task_queue_service import TaskQueueService
from ....schemas.task_queue import (
    TaskQueueResponse,
    TaskQueueListResponse,
    TaskQueueRetryRequest,
    TaskQueueRetryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=TaskQueueListResponse)
async def list_tasks(
    status: Optional[str] = None,
    entity_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all task queue entries with optional filtering.

    Query Parameters:
        status: Filter by status (queued, running, failed, completed, cancelled)
        entity_type: Filter by entity type (model, dataset, training)

    Returns:
        List of task queue entries with entity information
    """
    tasks = await TaskQueueService.get_all_tasks(db, status=status, entity_type=entity_type)

    # Enrich tasks with entity information
    enriched_tasks = []
    for task in tasks:
        entity_info = await TaskQueueService.get_entity_info(
            db, task.entity_id, task.entity_type
        )

        task_dict = {
            "id": task.id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "entity_id": task.entity_id,
            "entity_type": task.entity_type,
            "status": task.status,
            "progress": task.progress,
            "error_message": task.error_message,
            "retry_params": task.retry_params,
            "retry_count": task.retry_count,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "entity_info": entity_info,
        }
        enriched_tasks.append(task_dict)

    return {"data": enriched_tasks}


@router.get("/failed", response_model=TaskQueueListResponse)
async def list_failed_tasks(db: AsyncSession = Depends(get_db)):
    """
    List all failed task queue entries.

    Returns:
        List of failed tasks with entity information
    """
    tasks = await TaskQueueService.get_failed_tasks(db)

    enriched_tasks = []
    for task in tasks:
        entity_info = await TaskQueueService.get_entity_info(
            db, task.entity_id, task.entity_type
        )

        task_dict = {
            "id": task.id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "entity_id": task.entity_id,
            "entity_type": task.entity_type,
            "status": task.status,
            "progress": task.progress,
            "error_message": task.error_message,
            "retry_params": task.retry_params,
            "retry_count": task.retry_count,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "entity_info": entity_info,
        }
        enriched_tasks.append(task_dict)

    return {"data": enriched_tasks}


@router.get("/active", response_model=TaskQueueListResponse)
async def list_active_tasks(db: AsyncSession = Depends(get_db)):
    """
    List all active (queued or running) task queue entries.

    Returns:
        List of active tasks with entity information
    """
    tasks = await TaskQueueService.get_active_tasks(db)

    enriched_tasks = []
    for task in tasks:
        entity_info = await TaskQueueService.get_entity_info(
            db, task.entity_id, task.entity_type
        )

        task_dict = {
            "id": task.id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "entity_id": task.entity_id,
            "entity_type": task.entity_type,
            "status": task.status,
            "progress": task.progress,
            "error_message": task.error_message,
            "retry_params": task.retry_params,
            "retry_count": task.retry_count,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "entity_info": entity_info,
        }
        enriched_tasks.append(task_dict)

    return {"data": enriched_tasks}


@router.get("/{task_queue_id}", response_model=TaskQueueResponse)
async def get_task(
    task_queue_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific task queue entry by ID.

    Args:
        task_queue_id: Task queue entry ID

    Returns:
        Task queue entry with entity information

    Raises:
        HTTPException: If task not found
    """
    task = await TaskQueueService.get_task_by_id(db, task_queue_id)

    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task queue entry '{task_queue_id}' not found"
        )

    entity_info = await TaskQueueService.get_entity_info(
        db, task.entity_id, task.entity_type
    )

    return {
        "data": {
            "id": task.id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "entity_id": task.entity_id,
            "entity_type": task.entity_type,
            "status": task.status,
            "progress": task.progress,
            "error_message": task.error_message,
            "retry_params": task.retry_params,
            "retry_count": task.retry_count,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "entity_info": entity_info,
        }
    }


@router.post("/{task_queue_id}/retry", response_model=TaskQueueRetryResponse)
async def retry_task(
    task_queue_id: str,
    request: TaskQueueRetryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Retry a failed task with optional parameter overrides.

    This endpoint:
    1. Verifies the task exists and is in failed state
    2. Updates retry count and status
    3. Dispatches a new Celery task with retry parameters
    4. Returns the new task information

    Args:
        task_queue_id: Task queue entry ID
        request: Optional parameter overrides for retry

    Returns:
        Retry status and new task information

    Raises:
        HTTPException: If task not found or not in failed state
    """
    task = await TaskQueueService.get_task_by_id(db, task_queue_id)

    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task queue entry '{task_queue_id}' not found"
        )

    if task.status != "failed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is in '{task.status}' state, can only retry failed tasks"
        )

    # Merge retry_params with any overrides from request
    retry_params = task.retry_params.copy() if task.retry_params else {}
    if request.param_overrides:
        retry_params.update(request.param_overrides)

    # Increment retry count
    await TaskQueueService.increment_retry_count(db, task_queue_id)

    # Dispatch appropriate Celery task based on task type
    if task.task_type == "download" and task.entity_type == "model":
        from ....workers.model_tasks import download_and_load_model

        celery_task = download_and_load_model.delay(
            model_id=task.entity_id,
            repo_id=retry_params.get("repo_id"),
            quantization=retry_params.get("quantization"),
            access_token=retry_params.get("access_token"),
            trust_remote_code=retry_params.get("trust_remote_code", False),
        )

        # Update task_queue entry with new Celery task ID
        task.task_id = celery_task.id
        await db.commit()

        logger.info(f"Retried model download task {task_queue_id}, new Celery task: {celery_task.id}")

    elif task.task_type == "download" and task.entity_type == "dataset":
        from ....workers.dataset_tasks import download_dataset_task

        celery_task = download_dataset_task.delay(
            dataset_id=task.entity_id,
            repo_id=retry_params.get("repo_id"),
            access_token=retry_params.get("access_token"),
            split=retry_params.get("split"),
            config=retry_params.get("config"),
        )

        task.task_id = celery_task.id
        await db.commit()

        logger.info(f"Retried dataset download task {task_queue_id}, new Celery task: {celery_task.id}")

    elif task.task_type == "tokenization" and task.entity_type == "dataset":
        from ....workers.dataset_tasks import tokenize_dataset_task

        celery_task = tokenize_dataset_task.delay(
            dataset_id=task.entity_id,
            tokenizer_name=retry_params.get("tokenizer_name"),
            max_length=retry_params.get("max_length", 512),
            stride=retry_params.get("stride", 0),
            padding=retry_params.get("padding", "max_length"),
            truncation=retry_params.get("truncation", "longest_first"),
            add_special_tokens=retry_params.get("add_special_tokens", True),
            text_column=retry_params.get("text_column"),
        )

        task.task_id = celery_task.id
        await db.commit()

        logger.info(f"Retried tokenization task {task_queue_id}, new Celery task: {celery_task.id}")

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task type '{task.task_type}' for entity type '{task.entity_type}'"
        )

    return {
        "success": True,
        "message": f"Task retry initiated (attempt {task.retry_count + 1})",
        "task_queue_id": task_queue_id,
        "celery_task_id": task.task_id,
        "retry_count": task.retry_count,
    }


@router.delete("/{task_queue_id}", status_code=204)
async def delete_task(
    task_queue_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a task queue entry.

    This removes the task from the queue. For active tasks, you should
    cancel them first using the cancel endpoint.

    Args:
        task_queue_id: Task queue entry ID

    Raises:
        HTTPException: If task not found
    """
    success = await TaskQueueService.delete_task(db, task_queue_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Task queue entry '{task_queue_id}' not found"
        )

    logger.info(f"Deleted task queue entry {task_queue_id}")
