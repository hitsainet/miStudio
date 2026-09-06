"""
AppSetting database model for persistent application configuration.

Stores key-value settings with optional encryption for sensitive values.
Uses upsert semantics — each key is unique and can be created or updated.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Boolean, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from ..core.database import Base


class AppSetting(Base):
    """Application settings stored in the database.

    Each row is a key-value pair with metadata about sensitivity and category.
    Sensitive values (API keys, tokens) are stored encrypted via AES-256-GCM.
    """

    __tablename__ = "app_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    key = Column(String(255), nullable=False, unique=True, index=True)
    value = Column(Text, nullable=False)  # Plaintext or encrypted (base64)
    is_sensitive = Column(Boolean, nullable=False, default=False, server_default="false")
    category = Column(String(50), nullable=False, default="general", server_default="general", index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<AppSetting(key={self.key}, category={self.category}, sensitive={self.is_sensitive})>"
