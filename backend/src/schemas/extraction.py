"""
Pydantic schemas for feature extraction.

This module defines request and response schemas for feature extraction
from trained SAE models.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
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

    # Token filtering configuration (matches labeling filter structure)
    # These filters control which tokens are stored in FeatureActivation records during extraction
    filter_special: bool = Field(True, description="Filter special tokens (<s>, </s>, etc.)")
    filter_single_char: bool = Field(True, description="Filter single character tokens")
    filter_punctuation: bool = Field(True, description="Filter pure punctuation")
    filter_numbers: bool = Field(True, description="Filter pure numeric tokens")
    filter_fragments: bool = Field(True, description="Filter word fragments (BPE subwords)")
    filter_stop_words: bool = Field(False, description="Filter common stop words (the, and, is, etc.)")

    # Context window configuration
    # Captures tokens before and after the prime token (max activation) for better interpretability
    # Based on Anthropic/OpenAI research showing asymmetric windows improve feature understanding
    context_prefix_tokens: int = Field(5, ge=0, le=50, description="Number of tokens before the prime token (0-50)")
    context_suffix_tokens: int = Field(3, ge=0, le=50, description="Number of tokens after the prime token (0-50)")

    # Dead neuron filtering
    # Neurons that fire on less than this fraction of samples are considered "dead" and skipped
    # Default 0.001 = 0.1% means neurons must fire on at least 1 in 1000 samples
    min_activation_frequency: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Minimum activation frequency to keep a feature (0-0.1). Features firing less often are filtered as 'dead neurons'. Default 0.001 = 0.1%"
    )

    # NOTE: Labeling configuration has been removed from extraction
    # Features are created unlabeled and can be labeled separately via LabelingService
    # This allows re-labeling without re-extraction


class ExtractionStatusResponse(BaseModel):
    """
    Response schema for extraction job status.

    Returns current status and progress of a feature extraction job.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    training_id: Optional[str] = None  # Nullable for external SAE extractions
    external_sae_id: Optional[str] = None  # Set for external SAE extractions
    source_type: Literal["training", "external_sae"] = "training"  # Source type indicator
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    sae_name: Optional[str] = None  # Name of external SAE (when applicable)
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

    # Token filtering configuration (matches labeling filter structure)
    filter_special: Optional[bool] = True
    filter_single_char: Optional[bool] = True
    filter_punctuation: Optional[bool] = True
    filter_numbers: Optional[bool] = True
    filter_fragments: Optional[bool] = True
    filter_stop_words: Optional[bool] = False

    # Context window configuration
    context_prefix_tokens: Optional[int] = 5
    context_suffix_tokens: Optional[int] = 3


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
