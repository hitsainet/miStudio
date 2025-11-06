"""
Pydantic schemas for Model API endpoints.

These schemas define the structure for request/response validation
and serialization for all model-related API operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field, field_validator

from ..models.model import ModelStatus, QuantizationFormat


class ModelBase(BaseModel):
    """Base schema with common model fields."""

    name: str = Field(..., min_length=1, max_length=500, description="Model name")


class ModelCreate(ModelBase):
    """Schema for creating a new model."""

    repo_id: Optional[str] = Field(None, max_length=255, description="HuggingFace repository ID")
    quantization: str = Field("FP32", max_length=20, description="Quantization type")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ModelUpdate(BaseModel):
    """Schema for updating an existing model."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[str] = Field(None, pattern="^(downloading|loading|quantizing|ready|error)$")
    progress: Optional[float] = Field(None, ge=0, le=100)
    error_message: Optional[str] = None
    file_path: Optional[str] = Field(None, max_length=512)
    quantized_path: Optional[str] = Field(None, max_length=512)
    architecture: Optional[str] = Field(None, max_length=100)
    params_count: Optional[int] = Field(None, ge=0)
    architecture_config: Optional[Dict[str, Any]] = None
    memory_required_bytes: Optional[int] = Field(None, ge=0, alias="memory_req_bytes")
    disk_size_bytes: Optional[int] = Field(None, ge=0)
    num_layers: Optional[int] = Field(None, ge=0)
    hidden_dim: Optional[int] = Field(None, ge=0)
    num_heads: Optional[int] = Field(None, ge=0)
    metadata: Optional[Dict[str, Any]] = None

    model_config = {
        "populate_by_name": True,  # Allow using either field name or alias
    }


class ModelResponse(ModelBase):
    """Schema for model response."""

    id: str = Field(..., description="Unique model identifier (format: m_{uuid})")
    repo_id: Optional[str] = Field(None, description="HuggingFace repository ID")
    architecture: str = Field(..., description="Model architecture")
    params_count: int = Field(..., description="Number of parameters")
    quantization: QuantizationFormat = Field(..., description="Quantization format")
    status: ModelStatus = Field(..., description="Current processing status")
    progress: Optional[float] = Field(None, description="Download/loading progress (0-100)")
    error_message: Optional[str] = Field(None, description="Error message if status is ERROR")
    file_path: Optional[str] = Field(None, description="Path to raw model files")
    quantized_path: Optional[str] = Field(None, description="Path to quantized model files")
    architecture_config: Optional[Dict[str, Any]] = Field(None, description="Architecture configuration")
    memory_required_bytes: Optional[int] = Field(None, description="Estimated memory requirement")
    disk_size_bytes: Optional[int] = Field(None, description="Disk size in bytes")
    has_completed_extractions: bool = Field(False, description="Whether model has any completed extraction jobs")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record last update timestamp")

    model_config = {
        "from_attributes": True,
        "populate_by_name": True,
    }


class ModelListResponse(BaseModel):
    """Schema for paginated list of models."""

    data: list[ModelResponse] = Field(..., description="List of models")
    pagination: Dict[str, Any] = Field(..., description="Pagination metadata")


class ModelDownloadRequest(BaseModel):
    """Schema for HuggingFace model download request."""

    repo_id: str = Field(..., min_length=1, description="HuggingFace repository ID (e.g., 'TinyLlama/TinyLlama-1.1B')")
    quantization: QuantizationFormat = Field(QuantizationFormat.FP16, description="Quantization format")
    access_token: Optional[str] = Field(None, description="HuggingFace access token for gated models")
    trust_remote_code: bool = Field(False, description="Trust remote code in model repository (required for some models like Phi-4)")

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, v: str) -> str:
        """Validate HuggingFace repository ID format."""
        if "/" not in v:
            raise ValueError("repo_id must be in format 'username/model-name'")
        return v


class ActivationExtractionRequest(BaseModel):
    """Schema for activation extraction request."""

    dataset_id: str = Field(..., min_length=1, description="Dataset ID (UUID format)")
    layer_indices: List[int] = Field(..., min_length=1, description="List of layer indices to extract from (e.g., [0, 5, 11])")
    hook_types: List[str] = Field(..., min_length=1, description="List of hook types (residual, mlp, attention)")
    max_samples: int = Field(..., ge=1, le=1000000, description="Maximum number of samples to process")
    batch_size: Optional[int] = Field(8, ge=1, le=512, description="Batch size for processing (1, 8, 16, 32, 64, 128, 256, 512)")
    micro_batch_size: Optional[int] = Field(None, ge=1, le=512, description="GPU micro-batch size for memory efficiency (defaults to batch_size if not specified)")
    top_k_examples: Optional[int] = Field(10, ge=1, le=100, description="Number of top activating examples to save")

    @field_validator("hook_types")
    @classmethod
    def validate_hook_types(cls, v: List[str]) -> List[str]:
        """Validate hook types."""
        valid_types = {"residual", "mlp", "attention"}
        for hook_type in v:
            if hook_type not in valid_types:
                raise ValueError(f"Invalid hook type: {hook_type}. Must be one of: {valid_types}")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: Optional[int]) -> Optional[int]:
        """Validate batch size is a power of 2 or 1."""
        if v is None:
            return v
        # Allow 1 or powers of 2 up to 512
        valid_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        if v not in valid_sizes:
            raise ValueError(
                f"Batch size must be one of: {valid_sizes}. "
                f"Got: {v}. Use powers of 2 for optimal GPU performance."
            )
        return v

    @field_validator("micro_batch_size")
    @classmethod
    def validate_micro_batch_size(cls, v: Optional[int]) -> Optional[int]:
        """Validate micro_batch_size is a power of 2 or 1."""
        if v is None:
            return v
        # Allow 1 or powers of 2 up to 512
        valid_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        if v not in valid_sizes:
            raise ValueError(
                f"Micro-batch size must be one of: {valid_sizes}. "
                f"Got: {v}. Use powers of 2 for optimal GPU performance."
            )
        return v


class ActiveExtractionResponse(BaseModel):
    """Schema for active extraction response."""

    extraction_id: str = Field(..., description="Unique extraction identifier")
    model_id: str = Field(..., description="Model ID this extraction belongs to")
    dataset_id: str = Field(..., description="Dataset ID used for extraction")
    celery_task_id: Optional[str] = Field(None, description="Celery task ID")
    status: str = Field(..., description="Current extraction status (queued, loading, extracting, saving, completed, failed)")
    progress: float = Field(..., description="Progress percentage (0-100)")
    samples_processed: int = Field(..., description="Number of samples processed so far")
    max_samples: int = Field(..., description="Total number of samples to process")
    layer_indices: List[int] = Field(..., description="Layer indices being extracted")
    hook_types: List[str] = Field(..., description="Hook types being used")
    batch_size: int = Field(..., description="Batch size for processing")
    created_at: str = Field(..., description="Extraction start time (ISO format)")
    updated_at: str = Field(..., description="Last update time (ISO format)")
    error_message: Optional[str] = Field(None, description="Error message if extraction failed")

    model_config = {
        "json_schema_extra": {
            "example": {
                "extraction_id": "ext_m_50874ca7_20251014_100249",
                "model_id": "m_50874ca7",
                "dataset_id": "81939572-73d9-4c3c-bde6-7b14bde589ea",
                "celery_task_id": "497e2ba2-7f18-4154-b143-abcb023669ec",
                "status": "extracting",
                "progress": 45.5,
                "samples_processed": 450,
                "max_samples": 1000,
                "layer_indices": [10, 21, 32],
                "hook_types": ["residual"],
                "batch_size": 32,
                "created_at": "2025-10-14T10:02:49Z",
                "updated_at": "2025-10-14T10:05:23Z",
                "error_message": None
            }
        }
    }


class ExtractionHistoryItem(BaseModel):
    """Schema for a single extraction in history."""

    extraction_id: str = Field(..., description="Unique extraction identifier")
    status: str = Field(..., description="Extraction status")
    progress: float = Field(..., description="Progress percentage (0-100)")
    samples_processed: Optional[int] = Field(None, description="Number of samples processed")
    max_samples: Optional[int] = Field(None, description="Total samples to process")
    layer_indices: List[int] = Field(default_factory=list, description="Layer indices extracted")
    hook_types: List[str] = Field(default_factory=list, description="Hook types used")
    created_at: Optional[str] = Field(None, description="Extraction start time")
    completed_at: Optional[str] = Field(None, description="Extraction completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    statistics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extraction statistics")
    saved_files: List[str] = Field(default_factory=list, description="List of saved activation files")
    num_samples_processed: Optional[int] = Field(None, description="Actual samples processed (from metadata)")
    dataset_path: Optional[str] = Field(None, description="Dataset path used")
    architecture: Optional[str] = Field(None, description="Model architecture")
    quantization: Optional[str] = Field(None, description="Model quantization")


class ExtractionHistoryResponse(BaseModel):
    """Schema for extraction history response."""

    model_id: str = Field(..., description="Model ID")
    model_name: str = Field(..., description="Model name")
    extractions: List[ExtractionHistoryItem] = Field(..., description="List of extractions")
    count: int = Field(..., description="Total number of extractions")

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "m_50874ca7",
                "model_name": "Fathom-Search-4B",
                "count": 2,
                "extractions": [
                    {
                        "extraction_id": "ext_m_50874ca7_20251014_100249",
                        "status": "completed",
                        "progress": 100.0,
                        "samples_processed": 1000,
                        "max_samples": 1000,
                        "layer_indices": [10, 21, 32],
                        "hook_types": ["residual"],
                        "created_at": "2025-10-14T10:02:49Z",
                        "completed_at": "2025-10-14T10:25:33Z",
                        "statistics": {
                            "layer_10_residual": {
                                "mean_magnitude": 0.234,
                                "max_activation": 12.5
                            }
                        },
                        "saved_files": ["layer_10_residual.npy", "layer_21_residual.npy"]
                    }
                ]
            }
        }
    }


class ExtractionRetryRequest(BaseModel):
    """Schema for extraction retry request."""

    batch_size: Optional[int] = Field(None, ge=1, le=512, description="Override batch size for retry")
    micro_batch_size: Optional[int] = Field(None, ge=1, le=512, description="Override micro-batch size for retry")
    max_samples: Optional[int] = Field(None, ge=1, le=1000000, description="Override max samples for retry")

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: Optional[int]) -> Optional[int]:
        """Validate batch size is a power of 2 or 1."""
        if v is None:
            return v
        valid_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        if v not in valid_sizes:
            raise ValueError(
                f"Batch size must be one of: {valid_sizes}. "
                f"Use powers of 2 for optimal GPU performance."
            )
        return v

    @field_validator("micro_batch_size")
    @classmethod
    def validate_micro_batch_size(cls, v: Optional[int]) -> Optional[int]:
        """Validate micro_batch_size is a power of 2 or 1."""
        if v is None:
            return v
        valid_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        if v not in valid_sizes:
            raise ValueError(
                f"Micro-batch size must be one of: {valid_sizes}. "
                f"Use powers of 2 for optimal GPU performance."
            )
        return v


class ExtractionCancelResponse(BaseModel):
    """Schema for extraction cancellation response."""

    extraction_id: str = Field(..., description="Extraction ID that was cancelled")
    status: str = Field(..., description="New status (cancelled)")
    message: str = Field(..., description="Cancellation confirmation message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "extraction_id": "ext_m_50874ca7_20251015_031430",
                "status": "cancelled",
                "message": "Extraction cancelled successfully"
            }
        }
    }


class ExtractionRetryResponse(BaseModel):
    """Schema for extraction retry response."""

    original_extraction_id: str = Field(..., description="Original extraction ID")
    new_extraction_id: str = Field(..., description="New extraction ID for retry")
    job_id: str = Field(..., description="Celery task ID for the retry job")
    status: str = Field(..., description="Initial status (queued)")
    message: str = Field(..., description="Retry confirmation message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "original_extraction_id": "ext_m_50874ca7_20251015_031430",
                "new_extraction_id": "ext_m_50874ca7_20251015_032145",
                "job_id": "497e2ba2-7f18-4154-b143-abcb023669ec",
                "status": "queued",
                "message": "Extraction retry queued successfully"
            }
        }
    }


class ExtractionDeleteRequest(BaseModel):
    """Schema for batch extraction deletion request."""

    extraction_ids: List[str] = Field(..., min_length=1, description="List of extraction IDs to delete")

    model_config = {
        "json_schema_extra": {
            "example": {
                "extraction_ids": [
                    "ext_m_50874ca7_20251015_031430",
                    "ext_m_50874ca7_20251015_032145"
                ]
            }
        }
    }


class ExtractionDeleteResponse(BaseModel):
    """Schema for batch extraction deletion response."""

    model_id: str = Field(..., description="Model ID")
    deleted_count: int = Field(..., description="Number of extractions successfully deleted")
    failed_count: int = Field(..., description="Number of extractions that failed to delete")
    deleted_ids: List[str] = Field(..., description="List of successfully deleted extraction IDs")
    failed_ids: List[str] = Field(default_factory=list, description="List of extraction IDs that failed to delete")
    errors: Dict[str, str] = Field(default_factory=dict, description="Map of extraction_id to error message for failures")
    message: str = Field(..., description="Summary message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "m_50874ca7",
                "deleted_count": 2,
                "failed_count": 0,
                "deleted_ids": [
                    "ext_m_50874ca7_20251015_031430",
                    "ext_m_50874ca7_20251015_032145"
                ],
                "failed_ids": [],
                "errors": {},
                "message": "Successfully deleted 2 extraction(s)"
            }
        }
    }
