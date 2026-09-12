"""
PromptTemplate database model.

This module defines the SQLAlchemy model for prompt templates,
which allow users to save and reuse prompt series for steering experiments.
"""

from uuid import uuid4

from sqlalchemy import Column, String, Boolean, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from ..core.database import Base


class PromptTemplate(Base):
    """
    PromptTemplate database model for managing steering prompt series.

    Stores named templates with arrays of prompts that users can save, load,
    favorite, and reuse across different steering experiments.
    """

    __tablename__ = "prompt_templates"

    # Primary identifiers
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Prompt content - array of prompt strings stored as JSONB
    prompts = Column(JSONB, nullable=False, default=list)
    # Example: ["What is the capital of France?", "Tell me a story about a cat"]

    # User preferences
    is_favorite = Column(Boolean, nullable=False, default=False, server_default="false", index=True)

    # Tags for organization (stored as JSONB array)
    tags = Column(JSONB, nullable=True, default=list)
    # Example: ["humor", "creative", "testing"]

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        prompt_count = len(self.prompts) if self.prompts else 0
        return f"<PromptTemplate(id={self.id}, name={self.name}, prompts={prompt_count}, is_favorite={self.is_favorite})>"
