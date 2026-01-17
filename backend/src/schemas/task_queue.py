"""
Task Queue Pydantic schemas.

This module defines the request and response schemas for task queue operations.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict, Field


class TaskQueueBase(BaseModel):
    """Base schema for task queue entries."""
    task_type: str = Field(..., description="Type of task (download, training, extraction, tokenization)")
    entity_id: str = Field(..., description="ID of the entity being processed")
    entity_type: str = Field(..., description="Type of entity (model, dataset, training, extraction)")
    status: str = Field(..., description="Task status (queued, running, failed, completed, cancelled)")


class TaskQueueCreate(TaskQueueBase):
    """Schema for creating a new task queue entry."""
    task_id: Optional[str] = Field(None, description="Celery task ID")
    retry_params: Optional[Dict[str, Any]] = Field(None, description="Parameters for retry")


class TaskQueueUpdate(BaseModel):
    """Schema for updating a task queue entry."""
    status: Optional[str] = Field(None, description="New status")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Progress percentage")
    error_message: Optional[str] = Field(None, description="Error message")


class TaskQueueData(BaseModel):
    """Schema for task queue entry data."""
    id: str
    task_id: Optional[str]
    task_type: str
    entity_id: str
    entity_type: str
    status: str
    progress: Optional[float]
    error_message: Optional[str]
    retry_params: Optional[Dict[str, Any]]
    retry_count: int
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    updated_at: Optional[str]
    entity_info: Optional[Dict[str, Any]] = Field(None, description="Information about the associated entity")

    model_config = ConfigDict(from_attributes=True)


class TaskQueueResponse(BaseModel):
    """Schema for single task queue entry response."""
    data: TaskQueueData


class TaskQueueListResponse(BaseModel):
    """Schema for list of task queue entries response."""
    data: List[TaskQueueData]


class TaskQueueRetryRequest(BaseModel):
    """Schema for retry request."""
    param_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional parameter overrides for retry (e.g., change quantization, batch size)"
    )


class TaskQueueRetryResponse(BaseModel):
    """Schema for retry response."""
    success: bool
    message: str
    task_queue_id: str
    celery_task_id: str
    retry_count: int
