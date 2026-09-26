"""
ExtractionTemplate database model.

This module defines the SQLAlchemy model for extraction templates,
which allow users to save and reuse activation extraction configurations.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from ..core.database import Base


class ExtractionTemplate(Base):
    """
    ExtractionTemplate database model for managing activation extraction configurations.

    Stores named templates with layer indices, hook types, batch sizes, and other
    extraction parameters that users can save, load, favorite, and reuse.
    """

    __tablename__ = "extraction_templates"

    # Primary identifiers
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Extraction configuration
    layer_indices = Column(ARRAY(Integer), nullable=False)  # e.g., [0, 5, 11]
    hook_types = Column(ARRAY(String(50)), nullable=False)  # e.g., ["residual", "mlp", "attention"]
    max_samples = Column(Integer, nullable=False)  # Maximum samples to process
    batch_size = Column(Integer, nullable=False)  # Processing batch size
    micro_batch_size = Column(Integer, nullable=True)  # GPU micro-batch size for memory efficiency
    top_k_examples = Column(Integer, nullable=False)  # Number of top activating examples

    # Context window configuration
    # Based on Anthropic/OpenAI research showing asymmetric windows improve interpretability
    context_prefix_tokens = Column(Integer, nullable=False, default=25, server_default='25')  # Tokens before prime token
    context_suffix_tokens = Column(Integer, nullable=False, default=25, server_default='25')  # Tokens after prime token

    # User preferences
    is_favorite = Column(Boolean, nullable=False, default=False, server_default="false", index=True)

    # Additional metadata (flexible JSONB for future extensions)
    # Note: Using 'extra_metadata' instead of 'metadata' as 'metadata' is reserved by SQLAlchemy
    extra_metadata = Column(JSONB, nullable=True, default=dict)
    # Contains: tags, author, version, export_source, etc.

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<ExtractionTemplate(id={self.id}, name={self.name}, is_favorite={self.is_favorite})>"
