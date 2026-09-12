"""MIS-E2E-135 / -100 — `Feature.training_id` is NULL by design.

Features are extracted against an entry in the SAE registry, so a feature row
carries `external_sae_id` and the training — when there was one at all — lives
one hop away on `ExternalSAE.training_id`. A downloaded community SAE has no
training, and downloading one from HuggingFace is a documented workflow.

Three consumers read the column directly, and every consequence was silent:

  * the Logit Lens tab's training branch never ran for a real feature;
  * Correlations scoped on `Feature.training_id == feature.training_id`, which
    with a NULL on both sides compiles to `IS NULL` — matching every OTHER
    training-less feature in the database, i.e. features of other models
    entirely, then freezing that for 7 days in the cache;
  * `browse_sae_features` resolved only through `sae.training_id`, so every
    feature of an external SAE rendered with no label, no statistics and **no
    `activation_frequency`** — which the frequency auto-baseline computes from,
    so each silently took the default strength of 10 instead of its measured
    one.

`Feature.source_id` already existed and nothing used it. The sweep found the
other four candidate sites (`logit_lens_service`, `neuronpedia_local_service`,
`histogram_service` ×2) were ALREADY correct — they `or_` over both links — so
this is two fixes, not six. Checked rather than assumed.
"""

import ast
import inspect

import pytest


class _Feature:
    def __init__(self, fid="f1", training_id=None, external_sae_id=None):
        self.id = fid
        self.training_id = training_id
        self.external_sae_id = external_sae_id


# ── The scope clause ───────────────────────────────────────────────────────

def test_a_registry_feature_scopes_to_its_own_dictionary():
    from src.models.feature import Feature
    from src.services.feature_provenance import feature_scope_clause

    clause = feature_scope_clause(_Feature(external_sae_id="sae_a"))
    rendered = str(clause.compile(compile_kwargs={"literal_binds": True}))
    assert "external_sae_id" in rendered
    assert "sae_a" in rendered
    assert Feature is not None


def test_a_training_feature_still_scopes_by_training():
    """Negative control for the direction: the old path must keep working."""
    from src.services.feature_provenance import feature_scope_clause

    clause = feature_scope_clause(_Feature(training_id="train_1"))
    rendered = str(clause.compile(compile_kwargs={"literal_binds": True}))
    assert "training_id" in rendered and "train_1" in rendered


def test_a_feature_with_neither_scopes_to_itself_not_to_every_null():
    """The actual defect, as SQL.

    `col == None` compiles to `IS NULL`, so the old comparison matched every
    other source-less feature rather than none — "correlated features" drawn
    from other models entirely.
    """
    from src.services.feature_provenance import feature_scope_clause

    clause = feature_scope_clause(_Feature(fid="f-lonely"))
    rendered = str(clause.compile(compile_kwargs={"literal_binds": True}))
    assert "IS NULL" not in rendered.upper(), (
        "the scope compiles to IS NULL, which matches every other feature "
        "without a source instead of none"
    )
    assert "f-lonely" in rendered


def test_the_bare_comparison_really_does_compile_to_is_null():
    """Negative control for the claim above — measured, not asserted.

    If SQLAlchemy did not do this, the finding would be wrong and the fix
    unnecessary. It does.
    """
    from src.models.feature import Feature

    rendered = str(
        (Feature.training_id == None).compile(  # noqa: E711 — that is the point
            compile_kwargs={"literal_binds": True}
        )
    )
    assert "IS NULL" in rendered.upper()


# ── The training hop ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_resolver_follows_the_sae_hop():
    from src.services.feature_provenance import resolve_training_id

    class _Result:
        @staticmethod
        def scalar_one_or_none():
            return "train_behind_the_sae"

    class _DB:
        async def execute(self, *a, **k):
            return _Result()

    got = await resolve_training_id(_DB(), _Feature(external_sae_id="sae_a"))
    assert got == "train_behind_the_sae"


@pytest.mark.asyncio
async def test_a_direct_training_id_short_circuits():
    from src.services.feature_provenance import resolve_training_id

    class _DB:
        async def execute(self, *a, **k):
            raise AssertionError("should not have queried — the column was set")

    assert await resolve_training_id(_DB(), _Feature(training_id="t1")) == "t1"


@pytest.mark.asyncio
async def test_no_training_is_a_legitimate_answer():
    """A downloaded SAE has none. `None` must mean "no checkpoint to load",
    not "error" — reading it as failure is what produced "Feature not found"
    for a feature that had just loaded successfully."""
    from src.services.feature_provenance import resolve_training_id

    class _DB:
        async def execute(self, *a, **k):
            raise AssertionError("should not have queried")

    assert await resolve_training_id(_DB(), _Feature()) is None


# ── The consumers are on it ────────────────────────────────────────────────

def test_the_logit_lens_path_resolves_rather_than_reading_the_column():
    from src.services.analysis_service import AnalysisService

    src = inspect.getsource(AnalysisService)
    assert "resolve_training_id(self.db, feature)" in src, (
        "the training branch reads feature.training_id directly, so it never "
        "runs for a registry-sourced feature"
    )


def test_browse_sae_features_scopes_by_the_sae_not_only_its_training():
    """MIS-E2E-100, the one still open at audit close."""
    from src.api.v1.endpoints import saes

    tree = ast.parse(inspect.getsource(saes))
    target = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "browse_sae_features"
    )
    body = ast.unparse(target)
    assert "Feature.external_sae_id == sae.id" in body, (
        "features of a downloaded SAE never match, so they render with no "
        "label, no statistics and no activation_frequency"
    )
    # And the old link must survive for features predating the registry.
    assert "Feature.training_id == training_id" in body


@pytest.mark.parametrize(
    "module",
    [
        "src.services.logit_lens_service",
        "src.services.neuronpedia_local_service",
        "src.services.histogram_service",
    ],
)
def test_the_already_correct_sites_stay_correct(module):
    """These four sites were verified correct during the sweep — they `or_`
    over both links. Pinned so a later "cleanup" cannot narrow them back."""
    import importlib

    src = inspect.getsource(importlib.import_module(module))
    assert "Feature.external_sae_id" in src, (
        f"{module} no longer matches features by their registry link"
    )
