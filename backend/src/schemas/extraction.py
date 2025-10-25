"""
Pydantic schemas for feature extraction.

This module defines request and response schemas for feature extraction
from trained SAE models.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class ExtractionConfigRequest(BaseModel):
    """
    Request schema for feature extraction configuration.

    Validates extraction parameters before starting a feature extraction job.
    """
    model_config = ConfigDict(from_attributes=True)

    evaluation_samples: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Number of dataset samples to evaluate (1,000-100,000)"
    )
    top_k_examples: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of top-activating examples to store per feature (10-1,000)"
    )

    # Optional resource configuration
    batch_size: Optional[int] = Field(
        default=None,
        ge=8,
        le=256,
        description="Batch size for processing (8-256). If not provided, will use recommended value based on system resources"
    )
    num_workers: Optional[int] = Field(
        default=None,
        ge=1,
        le=32,
        description="Number of CPU workers for parallel processing (1-32). If not provided, will use recommended value"
    )
    db_commit_batch: Optional[int] = Field(
        default=None,
        ge=500,
        le=5000,
        description="Number of features to commit at once (500-5000). If not provided, will use recommended value"
    )


class ExtractionStatusResponse(BaseModel):
    """
    Response schema for extraction job status.

    Returns current status and progress of a feature extraction job.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    training_id: str
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    status: str
    progress: Optional[float] = None
    features_extracted: Optional[int] = None
    total_features: Optional[int] = None
    error_message: Optional[str] = None
    config: Dict[str, Any]
    statistics: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


class ExtractionListResponse(BaseModel):
    """
    Response schema for list of extraction jobs.

    Returns paginated list of extraction jobs with metadata.
    """
    model_config = ConfigDict(from_attributes=True)

    data: List[ExtractionStatusResponse]
    meta: Dict[str, Any] = Field(
        description="Metadata including total count, limit, offset"
    )
