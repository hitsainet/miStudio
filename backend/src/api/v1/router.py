"""
API v1 router.

This module aggregates all v1 API endpoints into a single router.
"""

from fastapi import APIRouter

from .endpoints import datasets, models, workers

api_router = APIRouter(prefix="/v1")

# Include all endpoint routers
api_router.include_router(datasets.router)
api_router.include_router(models.router)
api_router.include_router(workers.router)
