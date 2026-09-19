"""
API v1 router.

This module aggregates all v1 API endpoints into a single router.
"""

from fastapi import APIRouter

from .endpoints import jlens, datasets, models, workers, extraction_templates, training_templates, prompt_templates, system, trainings, task_queue, features, labeling, labeling_prompt_templates, saes, steering, neuronpedia, settings, version, enhanced_labeling, feature_groups, mcp_approvals, cluster_profiles, circuits, circuit_discovery, circuit_validation

api_router = APIRouter(prefix="/v1")

# Include all endpoint routers
api_router.include_router(datasets.router)
api_router.include_router(models.router)
api_router.include_router(workers.router)
api_router.include_router(extraction_templates.router)
api_router.include_router(training_templates.router)
api_router.include_router(prompt_templates.router)
api_router.include_router(system.router)
api_router.include_router(trainings.router)
api_router.include_router(task_queue.router, prefix="/task-queue", tags=["task-queue"])
api_router.include_router(features.router, tags=["features"])
api_router.include_router(labeling.router, tags=["labeling"])
api_router.include_router(labeling_prompt_templates.router)
api_router.include_router(saes.router)
api_router.include_router(steering.router)
api_router.include_router(neuronpedia.router)
api_router.include_router(settings.router)
api_router.include_router(version.router)
api_router.include_router(enhanced_labeling.router, tags=["enhanced-labeling"])
api_router.include_router(feature_groups.router, tags=["feature-groups"])
api_router.include_router(cluster_profiles.router)
api_router.include_router(circuits.router)
api_router.include_router(circuit_discovery.router)
api_router.include_router(jlens.router)
api_router.include_router(circuit_validation.router)
api_router.include_router(mcp_approvals.router)
