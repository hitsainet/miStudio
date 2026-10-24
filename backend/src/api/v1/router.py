"""
API v1 router.

This module aggregates all v1 API endpoints into a single router.
"""

from fastapi import APIRouter

from .endpoints import jlens, datasets, models, workers, extraction_templates, training_templates, prompt_templates, system, trainings, task_queue, features, labeling, labeling_prompt_templates, saes, steering, neuronpedia, settings, version, enhanced_labeling, feature_groups, mcp_approvals, cluster_profiles, circuits, circuit_discovery, circuit_validation, probe_monitors, probe_judge_prompts, environment

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
api_router.include_router(probe_monitors.router)
# The judge-prompt library (PADR IDL-57): its own router under the same prefix.
api_router.include_router(probe_judge_prompts.router)
api_router.include_router(mcp_approvals.router)
# Feature 036: runtime environment settings (D3). Removing this line fails
# test_environment_api.py::test_the_routes_are_served.
api_router.include_router(environment.router)

# MCP gap G9 (2026-10-21): every active job of every kind, and live per-card GPU state.
# Removing this line fails test_operations_api.py::test_the_routes_are_served.
from .endpoints import operations  # noqa: E402

api_router.include_router(operations.router)

# Feature 037: the behaviour catalogue (PADR IDL-58). Removing this line fails
# test_behaviour_catalogue.py::test_behaviour_routes_in_openapi.
from .endpoints import behaviours  # noqa: E402

api_router.include_router(behaviours.router)


# Feature 038: session spot-checks (PADR IDL-59). Removing this line fails
# test_probe_spot_check.py::test_spot_check_routes_in_openapi.
from .endpoints import probe_spot_checks  # noqa: E402

api_router.include_router(probe_spot_checks.router)

# Feature 040 S2: behaviour-detector reports (PADR IDL-61). Removing this line fails
# test_behaviour_detectors_api.py::test_routes_in_openapi.
from .endpoints import behaviour_detectors  # noqa: E402

api_router.include_router(behaviour_detectors.router)

# Feature 041 S2: blend reports (FR-6, FR-16 – FR-19). Removing this line fails
# test_blend_reports_routes.py::test_routes_in_openapi.
from .endpoints import probe_blends  # noqa: E402

api_router.include_router(probe_blends.router)
