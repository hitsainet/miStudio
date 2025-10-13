"""
Pydantic schemas for API request/response validation.

This module exports all schema classes for use in API endpoints.
"""

from .dataset import (
    DatasetBase,
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetListResponse,
    DatasetDownloadRequest,
)
from .extraction_template import (
    ExtractionTemplateBase,
    ExtractionTemplateCreate,
    ExtractionTemplateUpdate,
    ExtractionTemplateResponse,
    ExtractionTemplateListResponse,
    ExtractionTemplateExport,
    ExtractionTemplateImport,
)

__all__ = [
    "DatasetBase",
    "DatasetCreate",
    "DatasetUpdate",
    "DatasetResponse",
    "DatasetListResponse",
    "DatasetDownloadRequest",
    "ExtractionTemplateBase",
    "ExtractionTemplateCreate",
    "ExtractionTemplateUpdate",
    "ExtractionTemplateResponse",
    "ExtractionTemplateListResponse",
    "ExtractionTemplateExport",
    "ExtractionTemplateImport",
]
