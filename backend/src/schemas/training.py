"""
Pydantic schemas for Training API endpoints.

These schemas define the structure for request/response validation
and serialization for all SAE training-related API operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from ..models.training import TrainingStatus


class SAEArchitectureType(str, Enum):
    """SAE architecture types."""
    STANDARD = "standard"
    SKIP = "skip"
    TRANSCODER = "transcoder"


class TrainingHyperparameters(BaseModel):
    """Training hyperparameters schema."""

    # SAE Architecture
    hidden_dim: int = Field(..., gt=0, description="Hidden dimension (input/output size)")
    latent_dim: int = Field(..., gt=0, description="Latent dimension (SAE width)")
    architecture_type: SAEArchitectureType = Field(
        SAEArchitectureType.STANDARD,
        description="SAE architecture: standard, skip, or transcoder"
    )

    # Layer configuration
    training_layers: List[int] = Field(
        default=[0],
        min_length=1,
        description="List of layer indices to train SAEs on (e.g., [0, 6, 12])"
    )

    # Sparsity
    l1_alpha: float = Field(
        ...,
        gt=0.00001,
        le=100.0,
        description="L1 sparsity penalty coefficient (SAELens standard: 1.0-10.0 with activation normalization)"
    )
    target_l0: Optional[float] = Field(
        None,
        gt=0,
        le=0.2,
        description="Target L0 sparsity (fraction of active features, typically 0.01-0.05)"
    )
    top_k_sparsity: Optional[float] = Field(
        None,
        gt=0,
        le=1.0,
        description="Top-K sparsity (fraction of neurons to keep active, e.g., 0.05 for 5%). Guarantees exact sparsity level."
    )
    normalize_activations: Optional[str] = Field(
        "constant_norm_rescale",
        description="Activation normalization method: 'constant_norm_rescale' (SAELens standard) or 'none'"
    )

    # Training
    learning_rate: float = Field(..., gt=0, description="Initial learning rate")
    batch_size: int = Field(..., gt=0, description="Training batch size")
    total_steps: int = Field(..., gt=0, description="Total training steps")
    warmup_steps: int = Field(0, ge=0, description="Linear warmup steps")

    # Optimization
    weight_decay: float = Field(0.0, ge=0, description="Weight decay (L2 regularization)")
    grad_clip_norm: Optional[float] = Field(None, gt=0, description="Gradient clipping norm")

    # Checkpointing
    checkpoint_interval: int = Field(1000, gt=0, description="Save checkpoint every N steps")
    log_interval: int = Field(100, gt=0, description="Log metrics every N steps")

    # Dead neuron handling
    dead_neuron_threshold: int = Field(1000, gt=0, description="Steps before a neuron is considered dead")
    resample_dead_neurons: bool = Field(True, description="Resample dead neurons during training")
    resample_interval: int = Field(5000, gt=0, description="Resample dead neurons every N steps")

    @field_validator("training_layers")
    @classmethod
    def validate_training_layers(cls, v: List[int]) -> List[int]:
        """Validate training_layers array."""
        if not v:
            raise ValueError("training_layers must contain at least one layer")
        if any(layer < 0 for layer in v):
            raise ValueError("All layer indices must be non-negative")
        if len(v) != len(set(v)):
            raise ValueError("training_layers must not contain duplicate layer indices")
        return sorted(v)  # Return sorted list for consistency

    model_config = {
        "json_schema_extra": {
            "example": {
                "hidden_dim": 768,
                "latent_dim": 16384,
                "architecture_type": "standard",
                "training_layers": [0, 6, 12],
                "l1_alpha": 0.001,
                "target_l0": 0.05,
                "normalize_activations": "constant_norm_rescale",
                "learning_rate": 0.0003,
                "batch_size": 4096,
                "total_steps": 100000,
                "warmup_steps": 1000,
                "checkpoint_interval": 5000,
                "log_interval": 100
            }
        }
    }


class TrainingCreate(BaseModel):
    """Schema for creating a new training job."""

    model_id: str = Field(..., min_length=1, description="Model ID to train SAE on")
    dataset_id: str = Field(..., min_length=1, description="Dataset ID for training data")
    extraction_id: Optional[str] = Field(None, description="Activation extraction ID (if using pre-extracted activations)")
    hyperparameters: TrainingHyperparameters = Field(..., description="Training hyperparameters")

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        """Validate model ID format."""
        if not v.startswith("m_"):
            raise ValueError("model_id must start with 'm_'")
        return v

    @field_validator("extraction_id")
    @classmethod
    def validate_extraction_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate extraction ID format."""
        if v is not None and not v.startswith("ext_m_"):
            raise ValueError("extraction_id must start with 'ext_m_'")
        return v


class TrainingUpdate(BaseModel):
    """Schema for updating a training job (primarily for status changes)."""

    status: Optional[TrainingStatus] = Field(None, description="Training status")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Training progress")
    current_step: Optional[int] = Field(None, ge=0, description="Current training step")
    current_loss: Optional[float] = Field(None, description="Current reconstruction loss")
    current_l0_sparsity: Optional[float] = Field(None, description="Current L0 sparsity")
    current_dead_neurons: Optional[int] = Field(None, ge=0, description="Current dead neuron count")
    current_learning_rate: Optional[float] = Field(None, ge=0, description="Current learning rate")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_traceback: Optional[str] = Field(None, description="Error traceback for debugging")


class TrainingResponse(BaseModel):
    """Schema for training job response."""

    id: str = Field(..., description="Training job ID (format: train_{uuid})")
    model_id: str = Field(..., description="Model ID")
    dataset_id: str = Field(..., description="Dataset ID")
    extraction_id: Optional[str] = Field(None, description="Extraction ID if using pre-extracted activations")

    status: TrainingStatus = Field(..., description="Current training status")
    progress: float = Field(..., description="Training progress (0-100)")
    current_step: int = Field(..., description="Current training step")
    total_steps: int = Field(..., description="Total training steps")

    hyperparameters: Dict[str, Any] = Field(..., description="Training hyperparameters")

    # Current metrics
    current_loss: Optional[float] = Field(None, description="Current reconstruction loss")
    current_l0_sparsity: Optional[float] = Field(None, description="Current L0 sparsity")
    current_dead_neurons: Optional[int] = Field(None, description="Current dead neuron count")
    current_learning_rate: Optional[float] = Field(None, description="Current learning rate")

    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Paths
    checkpoint_dir: Optional[str] = Field(None, description="Checkpoint directory path")
    logs_path: Optional[str] = Field(None, description="Logs file path")

    # Celery
    celery_task_id: Optional[str] = Field(None, description="Celery task ID")

    # Timestamps
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    started_at: Optional[datetime] = Field(None, description="Training start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Training completion timestamp")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": "train_abc123",
                "model_id": "m_model123",
                "dataset_id": "ds_dataset456",
                "extraction_id": None,
                "status": "running",
                "progress": 45.5,
                "current_step": 45500,
                "total_steps": 100000,
                "hyperparameters": {
                    "hidden_dim": 768,
                    "latent_dim": 16384,
                    "l1_alpha": 0.001,
                    "learning_rate": 0.0003
                },
                "current_loss": 0.0234,
                "current_l0_sparsity": 0.05,
                "current_dead_neurons": 42,
                "current_learning_rate": 0.00028
            }
        }
    }


class TrainingListResponse(BaseModel):
    """Schema for paginated list of training jobs."""

    data: List[TrainingResponse] = Field(..., description="List of training jobs")
    pagination: Dict[str, Any] = Field(..., description="Pagination metadata")
    status_counts: Dict[str, int] = Field(..., description="Count of trainings by status (all, running, completed, failed)")


class TrainingMetricResponse(BaseModel):
    """Schema for a single training metric record."""

    id: int = Field(..., description="Metric record ID")
    training_id: str = Field(..., description="Training job ID")
    step: int = Field(..., description="Training step")
    timestamp: datetime = Field(..., description="Timestamp")

    # Loss metrics
    loss: float = Field(..., description="Total reconstruction loss")
    loss_reconstructed: Optional[float] = Field(None, description="Reconstruction component")
    loss_zero: Optional[float] = Field(None, description="Zero ablation loss")

    # Sparsity metrics
    l0_sparsity: Optional[float] = Field(None, description="L0 sparsity")
    l1_sparsity: Optional[float] = Field(None, description="L1 sparsity penalty")
    dead_neurons: Optional[int] = Field(None, description="Dead neuron count")

    # Training dynamics
    learning_rate: Optional[float] = Field(None, description="Learning rate")
    grad_norm: Optional[float] = Field(None, description="Gradient norm")

    # Resource metrics
    gpu_memory_used_mb: Optional[float] = Field(None, description="GPU memory usage (MB)")
    samples_per_second: Optional[float] = Field(None, description="Training throughput")

    model_config = {
        "from_attributes": True
    }


class TrainingMetricsListResponse(BaseModel):
    """Schema for list of training metrics."""

    data: List[TrainingMetricResponse] = Field(..., description="List of training metrics")
    pagination: Optional[Dict[str, Any]] = Field(None, description="Pagination metadata")


class CheckpointResponse(BaseModel):
    """Schema for checkpoint response."""

    id: str = Field(..., description="Checkpoint ID (format: ckpt_{uuid})")
    training_id: str = Field(..., description="Training job ID")
    step: int = Field(..., description="Training step at checkpoint")

    loss: float = Field(..., description="Loss at checkpoint")
    l0_sparsity: Optional[float] = Field(None, description="L0 sparsity at checkpoint")

    storage_path: str = Field(..., description="Path to .safetensors file")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")

    is_best: bool = Field(..., description="Whether this is the best checkpoint")
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    created_at: datetime = Field(..., description="Checkpoint creation timestamp")

    model_config = {
        "from_attributes": True
    }


class CheckpointListResponse(BaseModel):
    """Schema for list of checkpoints."""

    data: List[CheckpointResponse] = Field(..., description="List of checkpoints")
    pagination: Optional[Dict[str, Any]] = Field(None, description="Pagination metadata")


class TrainingControlRequest(BaseModel):
    """Schema for training control operations (pause/resume/stop)."""

    action: Literal["pause", "resume", "stop"] = Field(..., description="Control action to perform")


class TrainingControlResponse(BaseModel):
    """Schema for training control response."""

    success: bool = Field(..., description="Whether the action succeeded")
    training_id: str = Field(..., description="Training job ID")
    action: str = Field(..., description="Action that was performed")
    status: TrainingStatus = Field(..., description="New training status")
    message: Optional[str] = Field(None, description="Additional message")
