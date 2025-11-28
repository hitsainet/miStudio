"""
Pydantic schemas for SAE management API endpoints.

These schemas define the structure for request/response validation
and serialization for SAE download, upload, and management operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field, field_validator

from ..models.external_sae import SAESource, SAEStatus, SAEFormat


# ============================================================================
# Base Schemas
# ============================================================================

class SAEBase(BaseModel):
    """Base schema with common SAE fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Display name for the SAE")
    description: Optional[str] = Field(None, max_length=2000, description="Optional description")


# ============================================================================
# HuggingFace Preview/Download Schemas
# ============================================================================

class HFRepoPreviewRequest(BaseModel):
    """Schema for previewing a HuggingFace SAE repository."""

    repo_id: str = Field(..., min_length=1, max_length=500, description="HuggingFace repository ID (e.g., 'google/gemma-scope-2b-pt-res')")
    access_token: Optional[str] = Field(None, description="HuggingFace access token for private repos")

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, v: str) -> str:
        """Validate HuggingFace repository ID format."""
        if "/" not in v:
            raise ValueError("repo_id must be in format 'username/repo-name'")
        return v


class HFFileInfo(BaseModel):
    """Information about a file in a HuggingFace repository."""

    filepath: str = Field(..., description="File path within repository")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    is_sae: bool = Field(False, description="Whether this appears to be an SAE file")


class HFRepoPreviewResponse(BaseModel):
    """Schema for HuggingFace repository preview response."""

    repo_id: str = Field(..., description="Repository ID")
    repo_type: str = Field("model", description="Repository type (model, dataset, space)")
    description: Optional[str] = Field(None, description="Repository description")
    files: List[HFFileInfo] = Field(..., description="List of files in repository")
    sae_files: List[HFFileInfo] = Field(default_factory=list, description="List of detected SAE files")
    sae_paths: List[str] = Field(..., description="Detected SAE file paths")
    model_name: Optional[str] = Field(None, description="Detected target model name")
    total_size_bytes: Optional[int] = Field(None, description="Total repository size")


class SAEDownloadRequest(BaseModel):
    """Schema for downloading an SAE from HuggingFace."""

    repo_id: str = Field(..., min_length=1, max_length=500, description="HuggingFace repository ID")
    filepath: str = Field(..., min_length=1, max_length=1000, description="Path to SAE within repository")
    name: Optional[str] = Field(None, max_length=255, description="Display name (defaults to repo/path)")
    description: Optional[str] = Field(None, max_length=2000, description="Optional description")
    revision: Optional[str] = Field(None, max_length=255, description="Git revision/branch (defaults to main)")
    access_token: Optional[str] = Field(None, description="HuggingFace access token for private repos")
    model_name: Optional[str] = Field(None, max_length=255, description="Target model name for compatibility checking")

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, v: str) -> str:
        """Validate HuggingFace repository ID format."""
        if "/" not in v:
            raise ValueError("repo_id must be in format 'username/repo-name'")
        return v


# ============================================================================
# SAE Upload Schemas
# ============================================================================

class SAEUploadRequest(BaseModel):
    """Schema for uploading an SAE to HuggingFace."""

    sae_id: str = Field(..., description="Local SAE ID to upload")
    repo_id: str = Field(..., min_length=1, max_length=500, description="Target HuggingFace repository ID")
    filepath: str = Field(..., min_length=1, max_length=1000, description="Path within repository to upload to")
    access_token: str = Field(..., description="HuggingFace access token with write permissions")
    create_repo: bool = Field(False, description="Create repository if it doesn't exist")
    private: bool = Field(False, description="Make repository private (only if creating)")
    commit_message: Optional[str] = Field("Upload SAE", max_length=500, description="Commit message")

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, v: str) -> str:
        """Validate HuggingFace repository ID format."""
        if "/" not in v:
            raise ValueError("repo_id must be in format 'username/repo-name'")
        return v


class SAEUploadResponse(BaseModel):
    """Schema for SAE upload response."""

    sae_id: str = Field(..., description="Local SAE ID that was uploaded")
    repo_id: str = Field(..., description="HuggingFace repository ID")
    filepath: str = Field(..., description="Path within repository")
    url: str = Field(..., description="URL to uploaded SAE")
    commit_hash: Optional[str] = Field(None, description="Git commit hash")


# ============================================================================
# Local SAE Import Schemas
# ============================================================================

class SAEImportFromTrainingRequest(BaseModel):
    """Schema for importing an SAE from a completed training job."""

    training_id: str = Field(..., description="Training job ID to import from")
    name: Optional[str] = Field(None, max_length=255, description="Display name (defaults to training name)")
    description: Optional[str] = Field(None, max_length=2000, description="Optional description")


class SAEImportFromFileRequest(BaseModel):
    """Schema for importing an SAE from local file."""

    file_path: str = Field(..., min_length=1, max_length=1000, description="Local path to SAE file")
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    description: Optional[str] = Field(None, max_length=2000, description="Optional description")
    format: SAEFormat = Field(SAEFormat.COMMUNITY_STANDARD, description="SAE file format")
    model_name: Optional[str] = Field(None, max_length=255, description="Target model name")
    layer: Optional[int] = Field(None, ge=0, description="Target layer")


# ============================================================================
# SAE Response Schemas
# ============================================================================

class SAEResponse(SAEBase):
    """Schema for SAE response."""

    id: str = Field(..., description="Unique SAE identifier")
    source: SAESource = Field(..., description="Source type (huggingface, local, trained)")
    status: SAEStatus = Field(..., description="Current status")

    # HuggingFace source info
    hf_repo_id: Optional[str] = Field(None, description="HuggingFace repository ID")
    hf_filepath: Optional[str] = Field(None, description="Path within HF repository")
    hf_revision: Optional[str] = Field(None, description="Git revision")

    # Training source info
    training_id: Optional[str] = Field(None, description="Source training job ID")

    # Model compatibility
    model_name: Optional[str] = Field(None, description="Target model name")
    model_id: Optional[str] = Field(None, description="Linked model ID")

    # SAE architecture info
    layer: Optional[int] = Field(None, description="Target layer")
    n_features: Optional[int] = Field(None, description="Number of features (latent dim)")
    d_model: Optional[int] = Field(None, description="Model dimension")
    architecture: Optional[str] = Field(None, description="SAE architecture type")

    # Format and storage
    format: SAEFormat = Field(..., description="SAE file format")
    local_path: Optional[str] = Field(None, description="Local storage path")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")

    # Progress and status
    progress: float = Field(0.0, description="Download/upload progress (0-100)")
    error_message: Optional[str] = Field(None, description="Error message if status is ERROR")

    # Metadata
    sae_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Timestamps
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record last update timestamp")
    downloaded_at: Optional[datetime] = Field(None, description="Download completion timestamp")

    model_config = {
        "from_attributes": True,
    }


class SAEListResponse(BaseModel):
    """Schema for paginated list of SAEs."""

    data: List[SAEResponse] = Field(..., description="List of SAEs")
    pagination: Dict[str, Any] = Field(..., description="Pagination metadata")


class SAEDownloadProgressResponse(BaseModel):
    """Schema for SAE download progress update."""

    sae_id: str = Field(..., description="SAE ID")
    status: SAEStatus = Field(..., description="Current status")
    progress: float = Field(..., description="Download progress (0-100)")
    bytes_downloaded: Optional[int] = Field(None, description="Bytes downloaded so far")
    total_bytes: Optional[int] = Field(None, description="Total bytes to download")
    speed_bytes_per_sec: Optional[int] = Field(None, description="Download speed")


# ============================================================================
# SAE Feature Browser Schemas (for Steering integration)
# ============================================================================

class SAEFeatureSummary(BaseModel):
    """Summary of a single SAE feature for the feature browser."""

    feature_idx: int = Field(..., description="Feature index")
    layer: int = Field(..., description="Layer this feature is from")
    label: Optional[str] = Field(None, description="Feature label from labeling")
    activation_count: Optional[int] = Field(None, description="Number of activations")
    mean_activation: Optional[float] = Field(None, description="Mean activation value")
    max_activation: Optional[float] = Field(None, description="Maximum activation value")
    top_tokens: List[str] = Field(default_factory=list, description="Top activating tokens")
    neuronpedia_url: Optional[str] = Field(None, description="Neuronpedia link if available")
    feature_id: Optional[str] = Field(None, description="Database feature ID if extracted")


class SAEFeatureBrowserResponse(BaseModel):
    """Schema for browsing features in an SAE."""

    sae_id: str = Field(..., description="SAE ID")
    n_features: int = Field(..., description="Total number of features")
    features: List[SAEFeatureSummary] = Field(..., description="List of feature summaries")
    pagination: Dict[str, Any] = Field(..., description="Pagination metadata")


# ============================================================================
# SAE Delete Schemas
# ============================================================================

class SAEDeleteRequest(BaseModel):
    """Schema for batch SAE deletion request."""

    sae_ids: List[str] = Field(..., min_length=1, description="List of SAE IDs to delete")


class SAEDeleteResponse(BaseModel):
    """Schema for batch SAE deletion response."""

    deleted_count: int = Field(..., description="Number of SAEs successfully deleted")
    failed_count: int = Field(..., description="Number of SAEs that failed to delete")
    deleted_ids: List[str] = Field(..., description="List of successfully deleted SAE IDs")
    failed_ids: List[str] = Field(default_factory=list, description="List of SAE IDs that failed to delete")
    errors: Dict[str, str] = Field(default_factory=dict, description="Map of sae_id to error message for failures")
    message: str = Field(..., description="Summary message")
