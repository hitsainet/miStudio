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

    # Layer selection for multi-layer trainings
    layer_index: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Layer index to extract from (for multi-layer trainings). If not specified, uses the first available layer."
    )
    # Hook type selection for multi-hook trainings
    hook_type: Optional[str] = Field(
        default=None,
        description="Hook type to extract from (for multi-hook trainings). Valid values: 'residual', 'mlp', 'attention'. If not specified, uses the first available hook type."
    )

    evaluation_samples: int = Field(
        default=10000,
        ge=1000,
        le=1000000,
        description="Number of dataset samples to evaluate (1,000-1,000,000)"
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
    context_prefix_tokens: int = Field(25, ge=0, le=50, description="Number of tokens before the prime token (0-50)")
    context_suffix_tokens: int = Field(25, ge=0, le=50, description="Number of tokens after the prime token (0-50)")

    # Dead neuron filtering
    # Neurons that fire on less than this fraction of samples are considered "dead" and skipped
    # Default 0.001 = 0.1% means neurons must fire on at least 1 in 1000 samples
    min_activation_frequency: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Minimum activation frequency to keep a feature (0-0.1). Features firing less often are filtered as 'dead neurons'. Default 0.001 = 0.1%"
    )

    # NLP Analysis configuration
    # Auto-NLP triggers NLP analysis automatically when extraction completes
    # This pre-computes POS tagging, NER, context patterns for feature labels
    auto_nlp: bool = Field(
        default=True,
        description="Automatically start NLP analysis after extraction completes (default: True)"
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
    training_id: Optional[str] = None  # Legacy: kept for backward compatibility with existing extractions
    external_sae_id: Optional[str] = None  # Set for external SAE extractions
    source_type: Literal["training", "external_sae"] = "external_sae"  # Source type (training is legacy)
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    sae_name: Optional[str] = None  # Name of external SAE (when applicable)
    layer_index: Optional[int] = None  # Layer index for multi-layer trainings
    hook_type: Optional[str] = None  # Hook type for multi-hook trainings
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
    context_prefix_tokens: Optional[int] = 25
    context_suffix_tokens: Optional[int] = 25

    # NLP configuration
    auto_nlp: Optional[bool] = Field(
        default=True,
        description="Whether NLP was/will be auto-triggered after extraction"
    )

    # NLP Processing status (separate from feature extraction)
    nlp_status: Optional[str] = Field(
        default=None,
        description="NLP processing status: null, pending, processing, completed, failed"
    )
    nlp_progress: Optional[float] = Field(
        default=None,
        description="NLP processing progress 0.0-1.0"
    )
    nlp_processed_count: Optional[int] = Field(
        default=None,
        description="Number of features with NLP analysis completed"
    )
    nlp_error_message: Optional[str] = Field(
        default=None,
        description="Error message if NLP processing failed"
    )

    # Batch extraction fields
    batch_id: Optional[str] = Field(
        default=None,
        description="Batch ID if this job is part of a batch extraction"
    )
    batch_position: Optional[int] = Field(
        default=None,
        description="Position in batch (1-indexed, e.g., 1 of 5)"
    )
    batch_total: Optional[int] = Field(
        default=None,
        description="Total jobs in batch"
    )


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


class BatchExtractionRequest(BaseModel):
    """
    Request schema for batch feature extraction from multiple SAEs.

    Allows starting extraction for multiple SAEs with the same dataset
    and configuration in a single API call.
    """
    model_config = ConfigDict(from_attributes=True)

    sae_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of SAE IDs to extract features from (1-50 SAEs per batch)"
    )
    dataset_id: str = Field(
        ...,
        min_length=1,
        description="Dataset ID to use for all extractions"
    )

    # Extraction configuration (applied to all SAEs in batch)
    evaluation_samples: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Number of dataset samples to evaluate per SAE (1,000-100,000)"
    )
    top_k_examples: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of top-activating examples to store per feature (10-500)"
    )

    # Token filtering configuration
    filter_special: bool = Field(True, description="Filter special tokens")
    filter_single_char: bool = Field(True, description="Filter single character tokens")
    filter_punctuation: bool = Field(True, description="Filter pure punctuation")
    filter_numbers: bool = Field(True, description="Filter pure numeric tokens")
    filter_fragments: bool = Field(True, description="Filter word fragments")
    filter_stop_words: bool = Field(False, description="Filter common stop words")

    # Context window configuration
    context_prefix_tokens: int = Field(25, ge=0, le=50, description="Tokens before prime token")
    context_suffix_tokens: int = Field(25, ge=0, le=50, description="Tokens after prime token")

    # Dead neuron filtering
    min_activation_frequency: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Minimum activation frequency to keep a feature"
    )

    # NLP Analysis
    auto_nlp: bool = Field(
        default=True,
        description="Auto-start NLP analysis after each extraction completes"
    )


class BatchExtractionJobInfo(BaseModel):
    """Information about a single job in a batch extraction."""
    model_config = ConfigDict(from_attributes=True)

    sae_id: str = Field(..., description="SAE ID")
    sae_name: Optional[str] = Field(None, description="SAE name for display")
    job_id: str = Field(..., description="Extraction job ID")
    position: int = Field(..., description="Position in batch (1-indexed)")
    status: str = Field(default="queued", description="Initial job status")


class BatchExtractionSkippedInfo(BaseModel):
    """Information about a skipped SAE in a batch extraction."""
    model_config = ConfigDict(from_attributes=True)

    sae_id: str = Field(..., description="SAE ID that was skipped")
    reason: str = Field(..., description="Reason why the SAE was skipped")


class BatchExtractionResponse(BaseModel):
    """
    Response schema for batch extraction creation.

    Returns information about the batch extraction jobs that were created.
    """
    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(..., description="Unique batch ID for tracking all jobs in this batch")
    created_jobs: List[BatchExtractionJobInfo] = Field(
        ...,
        description="List of extraction jobs that were created"
    )
    skipped_saes: List[BatchExtractionSkippedInfo] = Field(
        default_factory=list,
        description="List of SAEs that were skipped (not ready, already extracting, etc.)"
    )
    total_requested: int = Field(..., description="Total SAEs requested in batch")
    total_created: int = Field(..., description="Number of extraction jobs created")
    total_skipped: int = Field(..., description="Number of SAEs skipped")
    dataset_id: str = Field(..., description="Dataset ID used for all extractions")
    dataset_name: Optional[str] = Field(None, description="Dataset name for display")
