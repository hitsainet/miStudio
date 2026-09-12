"""
External SAE database model.

This module defines the SQLAlchemy model for external SAEs downloaded from
HuggingFace or converted from local training. Supports the SAEs tab for
managing SAEs used in steering.
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import Column, String, Integer, BigInteger, Float, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class SAESource(str, Enum):
    """Source type for external SAE."""
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    TRAINED = "trained"


class SAEStatus(str, Enum):
    """Status of external SAE."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    CONVERTING = "converting"
    READY = "ready"
    ERROR = "error"
    DELETED = "deleted"


class SAEFormat(str, Enum):
    """Format of SAE weights."""
    COMMUNITY_STANDARD = "community_standard"
    GEMMA_SCOPE = "gemma_scope"  # Legacy: same as community_standard but from Gemma Scope repos
    MISTUDIO = "mistudio"
    CUSTOM = "custom"


class ExternalSAE(Base):
    """
    External SAE database model.

    Stores metadata and status for SAEs downloaded from HuggingFace,
    imported from local files, or exported from training jobs.
    Used for the SAEs tab and steering functionality.
    """

    __tablename__ = "external_saes"

    # Primary identifier
    id = Column(String(255), primary_key=True)  # Format: sae_{uuid}

    # Display info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Source type
    source = Column(String(50), nullable=False)  # huggingface, local, trained

    # Status
    status = Column(String(50), nullable=False, default=SAEStatus.PENDING.value)

    # HuggingFace source fields
    hf_repo_id = Column(String(500), nullable=True)  # e.g., "google/gemma-scope-2b-pt-res"
    hf_filepath = Column(String(1000), nullable=True)  # e.g., "layer_12/width_16k/canonical"
    hf_revision = Column(String(255), nullable=True)  # Git revision/branch

    # Trained source reference
    training_id = Column(
        String(255),
        ForeignKey("trainings.id", ondelete="SET NULL"),
        nullable=True
    )

    # Model compatibility
    model_name = Column(String(255), nullable=True)  # e.g., "gemma-2-2b", "gpt2-small"
    model_id = Column(
        String(255),
        ForeignKey("models.id", ondelete="SET NULL"),
        nullable=True
    )

    # SAE architecture info
    layer = Column(Integer, nullable=True)
    hook_type = Column(String(100), nullable=True)  # e.g., hook_resid_pre, hook_resid_post, hook_mlp_out
    n_features = Column(Integer, nullable=True)  # Number of features (latent dim)
    d_model = Column(Integer, nullable=True)  # Model dimension
    architecture = Column(String(100), nullable=True)  # standard, gated, etc.

    # Format
    format = Column(String(50), nullable=False, default=SAEFormat.COMMUNITY_STANDARD.value)

    # Local storage
    local_path = Column(String(1000), nullable=True)  # Path to downloaded/converted SAE
    file_size_bytes = Column(BigInteger, nullable=True)

    # Progress (for download/upload tracking)
    progress = Column(Float, nullable=False, default=0.0)  # 0-100

    # Error handling
    error_message = Column(Text, nullable=True)

    # Flexible metadata storage (named sae_metadata to avoid SQLAlchemy reserved name)
    sae_metadata = Column(JSONB, nullable=False, default=dict)
    # Can contain: activation_stats, neuronpedia_url, top_features, etc.

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    downloaded_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    training = relationship("Training", foreign_keys=[training_id])
    model = relationship("Model", foreign_keys=[model_id])
    extraction_jobs = relationship("ExtractionJob", back_populates="external_sae", cascade="all, delete-orphan")
    features = relationship("Feature", back_populates="external_sae", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ExternalSAE(id={self.id}, name={self.name}, source={self.source}, status={self.status})>"

    @property
    def is_ready(self) -> bool:
        """Check if SAE is ready for use."""
        return self.status == SAEStatus.READY.value

    @property
    def is_downloading(self) -> bool:
        """Check if SAE is currently downloading."""
        return self.status == SAEStatus.DOWNLOADING.value

    @property
    def display_name(self) -> str:
        """Get display name, falling back to HF repo if no name set."""
        if self.name:
            return self.name
        if self.hf_repo_id and self.hf_filepath:
            return f"{self.hf_repo_id}/{self.hf_filepath}"
        return self.id
