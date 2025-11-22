"""
Labeling Prompt Template database model.

This model stores customizable prompt templates for semantic feature labeling.
Users can create, customize, and reuse prompts with different API parameters.
"""

from sqlalchemy import Column, String, Text, Float, Integer, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class LabelingPromptTemplate(Base):
    """
    Labeling Prompt Template database model.

    Stores customizable prompts and API parameters for semantic feature labeling.
    Each template can define the system message, user prompt template, and API
    parameters (temperature, max_tokens, top_p) to control LLM behavior.
    """
    __tablename__ = "labeling_prompt_templates"

    # Primary key
    id = Column(String(255), primary_key=True)  # Format: "lpt_{uuid}"

    # Basic information
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Prompt content
    system_message = Column(Text, nullable=False)
    user_prompt_template = Column(Text, nullable=False)  # Uses {examples_block} placeholder

    # API parameters
    temperature = Column(Float, nullable=False, default=0.3)  # 0.0-2.0
    max_tokens = Column(Integer, nullable=False, default=50)  # 10-500
    top_p = Column(Float, nullable=False, default=0.9)  # 0.0-1.0

    # Template configuration
    template_type = Column(String(50), nullable=False, default='legacy')  # 'legacy', 'mistudio_context', 'anthropic_logit', 'eleutherai_detection'
    max_examples = Column(Integer, nullable=False, default=10)  # K value for top-K examples

    # Context window configuration
    include_prefix = Column(Boolean, nullable=False, default=True)  # Include prefix tokens
    include_suffix = Column(Boolean, nullable=False, default=True)  # Include suffix tokens
    prime_token_marker = Column(String(20), nullable=False, default='<<>>')  # Marker format for prime token

    # Logit effects configuration (for Anthropic-style template)
    include_logit_effects = Column(Boolean, nullable=False, default=False)  # Include promoted/suppressed tokens
    top_promoted_tokens_count = Column(Integer, nullable=True)  # Number of top promoted tokens (default: 10)
    top_suppressed_tokens_count = Column(Integer, nullable=True)  # Number of top suppressed tokens (default: 10)

    # Detection/scoring template flag (for EleutherAI-style template)
    is_detection_template = Column(Boolean, nullable=False, default=False)  # For binary classification/scoring

    # Metadata
    is_default = Column(Boolean, nullable=False, default=False)
    is_system = Column(Boolean, nullable=False, default=False)  # System templates can't be deleted
    created_by = Column(String(255), nullable=True)  # Future: user ID

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    labeling_jobs = relationship("LabelingJob", back_populates="prompt_template")

    def __repr__(self) -> str:
        return f"<LabelingPromptTemplate(id={self.id}, name={self.name}, is_default={self.is_default})>"
