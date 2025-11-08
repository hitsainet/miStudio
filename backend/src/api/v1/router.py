"""
API v1 router.

This module aggregates all v1 API endpoints into a single router.
"""

from fastapi import APIRouter

from .endpoints import datasets, models, workers, extraction_templates, training_templates, system, trainings, task_queue, features, labeling

api_router = APIRouter(prefix="/v1")

# Include all endpoint routers
api_router.include_router(datasets.router)
api_router.include_router(models.router)
api_router.include_router(workers.router)
api_router.include_router(extraction_templates.router)
api_router.include_router(training_templates.router)
api_router.include_router(system.router)
api_router.include_router(trainings.router)
api_router.include_router(task_queue.router, prefix="/task-queue", tags=["task-queue"])
api_router.include_router(features.router, tags=["features"])
api_router.include_router(labeling.router, tags=["labeling"])
