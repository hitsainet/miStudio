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
    QUANTIZING = "quantizing"
    READY = "ready"
    ERROR = "error"


class QuantizationFormat(str, Enum):
    """Model quantization format."""
    FP32 = "FP32"
    FP16 = "FP16"
    Q8 = "Q8"
    Q4 = "Q4"
    Q2 = "Q2"


class Model(Base):
    """
    Model database model for managing language models.

    Stores metadata about downloaded models including architecture details,
    quantization format, file paths, and processing status.
    """

    __tablename__ = "models"

    # Primary identifiers
    id = Column(String(255), primary_key=True)  # Format: m_{uuid}
    name = Column(String(500), nullable=False)  # Model display name
    repo_id = Column(String(500), nullable=True)  # HuggingFace repo ID (e.g., "meta-llama/Llama-2-7b-hf")
    architecture = Column(String(100), nullable=False)  # llama, gpt2, phi, pythia

    # Model specifications
    params_count = Column(BigInteger, nullable=False)  # Total parameter count
    quantization = Column(SQLEnum(QuantizationFormat), nullable=False, default=QuantizationFormat.FP32)

    # Processing status
    status = Column(SQLEnum(ModelStatus), nullable=False, default=ModelStatus.DOWNLOADING)
    progress = Column(Float, nullable=True)  # 0-100 for download/quantization progress
    error_message = Column(Text, nullable=True)

    # File system paths
    file_path = Column(String(1000), nullable=True)  # /data/models/raw/{id}/
    quantized_path = Column(String(1000), nullable=True)  # /data/models/quantized/{id}_{format}/

    # Architecture configuration (flexible JSONB for model-specific details)
    architecture_config = Column(JSONB, nullable=True, default=dict)
    # Contains: model_type, num_hidden_layers, hidden_size, num_attention_heads,
    #           intermediate_size, max_position_embeddings, vocab_size, etc.

    # Resource requirements
    memory_required_bytes = Column(BigInteger, nullable=True)
    disk_size_bytes = Column(BigInteger, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, name={self.name}, status={self.status})>"
