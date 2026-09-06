"""
MCP tool modules, one per gated category (Feature 010, BR-5.2).

Each module exposes ``register(mcp, client, settings)``; ``server.py`` calls
it only when the module's category is enabled via ``MCP_TOOL_CATEGORIES``.
"""

from . import (
    admin,
    circuits,
    discovery,
    experiments,
    features,
    groups,
    howto,
    jlens,
    jobs,
    labeling,
    millm_clusters,
    millm_runtime,
    millm_circuits,
    millm_sensing,
    profiles,
    steering,
)

# category name → module (registration order = tools/list order)
CATEGORY_MODULES = {
    "read": [discovery, features, howto],
    "groups": [groups],
    "steering": [steering],
    "experiments": [experiments],
    "profiles": [profiles],
    "circuits": [circuits],
    "jlens": [jlens],
    "labeling": [labeling],
    "jobs": [jobs],
    "admin": [admin],
}

# miLLM categories (Unified MCP): registered with (mcp, millm_client, gate)
# instead of (mcp, client, settings) — server.py wires them separately and
# skips them when MILLM_API_URL is unset.
MILLM_CATEGORY_MODULES = {
    "millm_runtime": [millm_runtime],
    "millm_clusters": [millm_clusters],
    "millm_circuits": [millm_circuits],
    "millm_sensing": [millm_sensing],
}
