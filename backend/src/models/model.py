"""
Model database model.

This module defines the SQLAlchemy model for models (ML models).
"""

from datetime import datetime
from enum import Enum
from uuid import uuid4

from sqlalchemy import Column, String, Integer, BigInteger, Float, DateTime, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from ..core.database import Base


class ModelStatus(str, Enum):
    """Model processing status."""
    DOWNLOADING = "downloading"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class Model(Base):
    """Model database model."""

    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    repo_id = Column(String(255), nullable=True)
    architecture = Column(String(100), nullable=False)
    params_count = Column(BigInteger, nullable=True)
    quantization = Column(String(20), nullable=False, default="FP32")
    memory_req_bytes = Column(BigInteger, nullable=True)
    status = Column(SQLEnum(ModelStatus), nullable=False, default=ModelStatus.DOWNLOADING)
    progress = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    file_path = Column(String(512), nullable=True)
    num_layers = Column(Integer, nullable=True)
    hidden_dim = Column(Integer, nullable=True)
    num_heads = Column(Integer, nullable=True)
    extra_metadata = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
