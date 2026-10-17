"""
MCP tool modules, one per gated category (Feature 010, BR-5.2).

Native modules expose ``register(mcp, client, settings)``; external-product modules expose
``register(mcp, product_client, gate, ctx)`` (Feature 034, TD8). ``server.py`` calls each only when
its category is enabled via ``MCP_TOOL_CATEGORIES``.

⚠ "EVERY CATEGORY" IS `all_categories()`, NEVER A UNION WRITTEN AT THE CALL SITE. Before 034 ten
sites each spelled `{**CATEGORY_MODULES, **MILLM_CATEGORY_MODULES}`, so a second external product
would have been invisible to whichever site was missed — including the guards meant to catch it.
`test_no_direct_category_union.py` refuses a hand-written union anywhere in `src/` or `tests/`.
"""

from dataclasses import dataclass
from types import ModuleType

from . import (
    admin,
    circuits,
    dataworks,
    discovery,
    experiments,
    features,
    groups,
    howto,
    jlens,
    jobs,
    labeling,
    models,
    millm_clusters,
    millm_models,
    millm_runtime,
    millm_batches,
    millm_circuits,
    millm_inference,
    millm_probes,
    millm_sensing,
    probe_monitors,
    profiles,
    steering,
)
from ..products import DATAWORKS, MILLM, ProductSpec

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
    # 033: build, export and publish a portable probe definition, plus 032's readers.
    "probe_monitors": [probe_monitors],
    "jobs": [jobs],
    "models": [models],
    "admin": [admin],
}

# miLLM categories (Unified MCP): registered with (mcp, millm_client, gate, ctx)
# instead of (mcp, client, settings) — server.py wires them through the product
# loop and skips them when MILLM_API_URL is unset. The miLLM row's map.
MILLM_CATEGORY_MODULES = {
    "millm_runtime": [millm_runtime],
    "millm_clusters": [millm_clusters],
    "millm_circuits": [millm_circuits],
    "millm_sensing": [millm_sensing],
    "millm_models": [millm_models],
    # Feature 24 / 033 phase 7 — CO-RELEASED with miLLM's /api/probes routes.
    "millm_probes": [millm_probes],
    # Feature 034 (FR-16): scoring and generation against the served model, never swapping it.
    "millm_inference": [millm_inference, millm_batches],
}

# miDataworks categories (Feature 034 Part 1). The row existed, empty, from phase 1 so every guard
# iterated ROWS rather than names before any dataworks tool existed (FTID §3.1).
DATAWORKS_CATEGORY_MODULES: dict[str, list[ModuleType]] = {
    # Feature 034 phase 7: the curated miDataworks proxy (FR-10).
    "dataworks": [dataworks],
}


@dataclass(frozen=True)
class ExternalProduct:
    """A product row (`products.ProductSpec`) joined to its category map."""

    spec: ProductSpec
    categories: dict[str, list[ModuleType]]


EXTERNAL_PRODUCTS: tuple[ExternalProduct, ...] = (
    ExternalProduct(MILLM, MILLM_CATEGORY_MODULES),
    ExternalProduct(DATAWORKS, DATAWORKS_CATEGORY_MODULES),
)


def all_categories() -> dict[str, list[ModuleType]]:
    """Every category this build can register — native plus every product's. A fresh dict."""
    out: dict[str, list[ModuleType]] = dict(CATEGORY_MODULES)
    for product in EXTERNAL_PRODUCTS:
        for category, modules in product.categories.items():
            if category in out:
                raise RuntimeError(
                    f"category {category!r} is claimed twice ({product.spec.name}) — a "
                    "category name must identify one product"
                )
            out[category] = modules
    return out


def external_categories() -> dict[str, ExternalProduct]:
    """`{category: product}` for every external-product category."""
    return {
        category: product
        for product in EXTERNAL_PRODUCTS
        for category in product.categories
    }


def product_for(category: str) -> "ExternalProduct | None":
    """The product a category belongs to, or `None` for a native category."""
    return external_categories().get(category)
