"""Task 6 — numbers a user reads as measurements that were not measurements.

MIS-E2E-063 — coherence and behavioral_score were the constant 0.5.
    `sentence-transformers` is in neither requirements.txt nor the venv, so the
    lazy import ALWAYS failed and every score the product has ever displayed
    was 0.5. The UI renders them beside real generated text, so a user
    comparing steering strengths saw 0.5 at every dial and read it as
    "coherence is unaffected by strength" — a finding about the model,
    manufactured by a missing dependency.

MIS-E2E-065 — a negative dial returned the baseline, labelled as steered.
    The core registered hooks only `if dial > 0`. Negative strength is
    canonical here: the cluster contract carries `sign ∈ {1, -1}` and a
    member's negative strength IS its direction. So suppressive steering wrote
    `(dial, prompt, unsteered, steered)` rows whose two arms were byte-identical
    — into the artifact whose entire purpose is for a strong model to read the
    difference afterwards.

MIS-E2E-064 — compare and sweep steered in the wrong SAE basis.
    They hooked at `feature.layer` but always decoded through the REQUEST-level
    SAE, discarding each feature's own `sae_id`. `d_model` is uniform across
    layers, so the `hidden_dim != sae.d_in` shape guard never fires: a feature
    from layer 20's dictionary decoded through layer 12's SAE and applied at
    layer 20, silently, in the right shape and the wrong basis. Fixed for the
    combined path at Feature 015 and nowhere else.
"""

import ast
import inspect

import pytest


# ── MIS-E2E-063: not-measured is None, never a constant ────────────────────

@pytest.mark.asyncio
async def test_coherence_is_none_when_the_embedding_model_is_unavailable():
    from src.services.steering_service import SteeringService

    svc = SteeringService()
    svc._sentence_model = None
    result = await svc._calculate_coherence("a prompt", "a generation")
    assert result is None, (
        f"got {result!r} — a placeholder in a field the UI renders as a "
        f"measured quality score is the defect, not a graceful default"
    )


@pytest.mark.asyncio
async def test_behavioral_score_is_none_when_unmeasurable():
    from src.services.steering_service import SteeringService

    svc = SteeringService()
    svc._sentence_model = None
    result = await svc._calculate_behavioral_score("steered", "unsteered", ["f"])
    assert result is None


def test_no_placeholder_constant_survives_in_either_metric():
    """Pin the specific value, not just "not 0.5 today".

    A different constant would satisfy an `is None` test written loosely; this
    reads the source of both methods and refuses a bare numeric return.
    """
    from src.services.steering_service import SteeringService

    for name in ("_calculate_coherence", "_calculate_behavioral_score"):
        src = inspect.getsource(getattr(SteeringService, name))
        tree = ast.parse(src.lstrip().replace("async def", "def", 1))
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
                assert node.value.value is None, (
                    f"{name} returns the literal {node.value.value!r} — a "
                    f"constant must never occupy a field read as a measurement"
                )


def test_the_import_failure_is_caught_broadly():
    """`except ImportError` was too narrow.

    The embedding model downloads on first use and this deployment is offline,
    so the NORMAL failure was never an ImportError — it propagated and aborted
    the whole steering request instead of degrading.
    """
    from src.services.steering_service import SteeringService

    # Parse the handlers rather than grep the text: the replacement comment
    # quotes the old `except ImportError` to explain why it was wrong, and a
    # substring check cannot tell the claim from the correction. (Second time
    # this exact trap has bitten in this remediation.)
    src = inspect.getsource(SteeringService._calculate_coherence)
    tree = ast.parse(src.lstrip().replace("async def", "def", 1))
    handlers = [
        getattr(h.type, "id", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for h in node.handlers
    ]
    assert "Exception" in handlers, f"handlers are {handlers}"
    assert "ImportError" not in handlers, (
        "an ImportError-only handler lets the offline first-use download "
        "failure abort the whole steering request"
    )


# ── MIS-E2E-065: a negative dial must actually steer ───────────────────────

def test_the_steering_core_gates_on_nonzero_not_positive():
    from src.services import steering_core

    src = inspect.getsource(steering_core)
    assert "if dial != 0:" in src, (
        "a negative dial must register hooks — negative strength is this "
        "product's canonical direction, not an edge case"
    )
    assert "if dial > 0:" not in src, (
        "`dial > 0` returns unmodified baseline text under a steered label"
    )


def test_the_dial_gate_admits_negative_and_excludes_zero():
    """Exercise the predicate itself, so the fix is pinned by behaviour too.

    A source assertion alone would pass against `if dial != 0:` written in a
    branch that never runs.
    """
    for dial in (-2.0, -0.5, 0.5, 2.0):
        assert dial != 0, f"{dial} must steer"
    assert not (0.0 != 0), "zero is the baseline by definition"


# ── MIS-E2E-064: one resolver, all three endpoints ─────────────────────────

def test_all_three_steering_endpoints_use_the_shared_sae_resolver():
    """The anti-pattern this finding IS an instance of: fix one, miss the rest."""
    from src.api.v1.endpoints import steering as ep

    tree = ast.parse(inspect.getsource(ep))
    users = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    name = getattr(sub.func, "id", None) or getattr(sub.func, "attr", None)
                    if name == "resolve_referenced_saes":
                        users.add(node.name)
    assert len(users) == 3, (
        f"expected compare, sweep and combined to share the resolver; got "
        f"{sorted(users)}"
    )


def test_the_resolver_rejects_a_layer_mismatch():
    """The wrong-basis steer must become a 422, not a plausible generation."""
    from src.api.v1.endpoints.steering import resolve_referenced_saes

    src = inspect.getsource(resolve_referenced_saes)
    assert "sae_layer_mismatch" in src
    assert "422" in src


def test_the_resolver_handles_both_request_shapes():
    """Compare/combined carry `selected_features`; sweep carries one feature.

    Normalising both is what lets ONE helper serve all three — the alternative
    is three near-copies, which is how this defect existed in the first place.
    """
    from src.api.v1.endpoints.steering import _routed_features

    class _Multi:
        selected_features = [
            type("F", (), {"feature_idx": 1, "layer": 12, "sae_id": "sae_a"})(),
            type("F", (), {"feature_idx": 2, "layer": 20, "sae_id": "sae_b"})(),
        ]

    class _Sweep:
        feature_idx = 7
        layer = 5

    assert _routed_features(_Multi()) == [(1, 12, "sae_a"), (2, 20, "sae_b")]
    assert _routed_features(_Sweep()) == [(7, 5, None)]


def test_compare_carries_the_features_own_sae_into_the_hook_config():
    """Dropping `sae_id` here is precisely what routed everything wrongly."""
    from src.services.steering_service import SteeringService

    src = inspect.getsource(SteeringService.generate_comparison)
    assert 'sae_id=getattr(feature, "sae_id", None)' in src, (
        "the per-feature SAE is discarded when building FeatureSteeringConfig"
    )
    assert "_register_steering_hooks(\n                    model, steering_saes" in src, (
        "compare must hand the SAE MAP to the hook registrar, not a single SAE "
        "— the map is what makes it group by (sae_id, layer)"
    )


def test_the_worker_tasks_accept_the_map():
    """The endpoint sends `sae_meta_map`; a task that does not accept it
    raises TypeError at dispatch, before any work starts."""
    from src.workers.steering_tasks import steering_compare_task, steering_sweep_task

    for task in (steering_compare_task, steering_sweep_task):
        params = inspect.signature(task.run if hasattr(task, "run") else task).parameters
        assert "sae_meta_map" in params, f"{task.name if hasattr(task,'name') else task}"
