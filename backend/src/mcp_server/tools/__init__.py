"""
MCP tool modules, one per gated category (Feature 010, BR-5.2).

Native modules expose ``register(mcp, client, settings)``; external-product modules expose
``register(mcp, product_client, gate, ctx)`` (Feature 034, TD8). ``server.py`` calls each only when
its category is enabled via ``MCP_TOOL_CATEGORIES``.

⚠ "EVERY CATEGORY" IS `all_categories()`, NEVER A UNION WRITTEN AT THE CALL SITE. Before 034 ten
sites each spelled `{**CATEGORY_MODULES, **MISERVER_CATEGORY_MODULES}`, so a second external product
would have been invisible to whichever site was missed — including the guards meant to catch it.
`test_no_direct_category_union.py` refuses a hand-written union anywhere in `src/` or `tests/`.
"""

from dataclasses import dataclass
from types import ModuleType

from . import (
    admin,
    approvals,
    behaviour_detectors,
    behaviours,
    circuits,
    dataworks,
    discovery,
    environment,
    experiments,
    features,
    gpu_ops,
    groups,
    howto,
    jlens,
    jobs,
    judge_prompts,
    labeling,
    models,
    miserver_approvals,
    miserver_clusters,
    miserver_models,
    miserver_runtime,
    miserver_batches,
    miserver_behaviours,
    miserver_circuits,
    miserver_inference,
    miserver_probes,
    miserver_sensing,
    probe_families,
    probe_monitors,
    profiles,
    rule_sets,
    steering,
)
from ..products import DATAWORKS, MISERVER, ProductSpec

# category name → module (registration order = tools/list order)
CATEGORY_MODULES = {
    # Feature 036 (ENH-003 D5): the runtime environment settings, read-only.
    "read": [discovery, features, howto, environment, approvals],
    "groups": [groups],
    "steering": [steering],
    "experiments": [experiments],
    "profiles": [profiles],
    "circuits": [circuits],
    "jlens": [jlens],
    "labeling": [labeling],
    # 033: build, export and publish a portable probe definition, plus 032's readers.
    # + the judge-prompt library (PADR IDL-57), under the same switch: a judge prompt only means
    # something to the probe lifecycle it measures.
    # + the behaviour catalogue (Feature 037, IDL-58): what each judge prompt's and probe's key means.
    # + the behaviour-detector reports (Feature 040 S2): breakdowns, success reports, model comparison.
    # + probe families and blend reports (Feature 041): appended after 040's module.
    # Feature 040 S3: `rule_sets` (rule-set validation, single-text rules, rules ∪ probe evaluations).
    "probe_monitors": [probe_monitors, judge_prompts, behaviours, behaviour_detectors, probe_families, rule_sets],
    "jobs": [jobs, gpu_ops],
    "models": [models],
    "admin": [admin],
}

# miServer categories (Unified MCP): registered with (mcp, millm_client, gate, ctx)
# instead of (mcp, client, settings) — server.py wires them through the product
# loop and skips them when MISERVER_API_URL is unset. The miServer row's map.
MISERVER_CATEGORY_MODULES = {
    "miserver_runtime": [miserver_runtime],
    "miserver_clusters": [miserver_clusters],
    "miserver_circuits": [miserver_circuits],
    "miserver_sensing": [miserver_sensing],
    # ENH-008: miServer's own approval queue (`millm_list_approvals`, `millm_get_approval_status`), under the
    # category of the model tools it gates — no new category, so no deployment change.
    "miserver_models": [miserver_models, miserver_approvals],
    # Feature 24 / 033 phase 7 — CO-RELEASED with miServer's /api/probes routes.
    # miServer 039 (§4j, §4k): behaviour probe sets and rule sets, under the same category (no new category, so no
    # deployment change: `k8s/base/mcp.yaml` already enables `millm_probes`).
    "miserver_probes": [miserver_probes, miserver_behaviours],
    # Feature 034 (FR-16): scoring and generation against the served model, never swapping it.
    "miserver_inference": [miserver_inference, miserver_batches],
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
    ExternalProduct(MISERVER, MISERVER_CATEGORY_MODULES),
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
