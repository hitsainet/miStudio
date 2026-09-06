"""MIS-E2E-083 — the circuit plane ran the SAE off-distribution.

`encode()` does NOT normalize. `forward()` does — `x_normalized, _ =
self.normalize(x)` and then `encode` — so a caller reaching for `encode()`
directly hands the dictionary raw activations when it was trained on normalized
ones.

The extraction path knew this and normalized inline, with a comment saying why.
Circuit **capture**, **attribution**, **faithfulness** and **intervention** all
called `encode()` bare. Every circuit discovered from a capture was therefore
mined from activations the dictionary was not trained to decode: the features
fire, the numbers are plausible, and the basis is wrong. Co-activation
statistics, attribution and edge validation all inherit it.

Compounding it, `_load_sae_sync` loaded the community-format `config` — which
carries `normalize_activations` — and **discarded** it, so every SAE it built
took the constructor default rather than its trained mode. Nothing downstream
could have recovered the right convention even if it had asked.

Two of the four sites (faithfulness, intervention) were NOT named in the
finding; they came out of the sibling sweep.
"""

import ast
import inspect

import pytest
import torch

from src.ml.sparse_autoencoder import (
    create_sae,
    encode_with_training_normalization,
)


# ── The helper does what forward() does ────────────────────────────────────

def test_the_helper_matches_forwards_normalization():
    """`forward` is the reference: normalize, then encode."""
    torch.manual_seed(0)
    sae = create_sae("standard", hidden_dim=32, latent_dim=64,
                     normalize_activations="constant_norm_rescale").eval()
    x = torch.randn(8, 32) * 4.0

    with torch.no_grad():
        expected = sae.encode(sae.normalize(x)[0])
        got = encode_with_training_normalization(sae, x)

    assert torch.allclose(got, expected, atol=0, rtol=0)


def test_the_helper_differs_from_a_bare_encode_when_normalization_is_on():
    """If these agreed, the fix would be measuring nothing.

    This is the finding, stated as a test: a bare `encode` on raw activations
    is a DIFFERENT computation, and the difference is the wrong basis.
    """
    torch.manual_seed(1)
    sae = create_sae("standard", hidden_dim=32, latent_dim=64,
                     normalize_activations="constant_norm_rescale").eval()
    x = torch.randn(8, 32) * 4.0

    with torch.no_grad():
        bare = sae.encode(x)
        correct = encode_with_training_normalization(sae, x)

    assert not torch.allclose(bare, correct, atol=1e-4), (
        "raw and normalized encodes agree — the fixture is not exercising "
        "normalization, so every other test here is vacuous"
    )


def test_the_helper_is_a_no_op_when_the_sae_trained_without_normalization():
    torch.manual_seed(2)
    sae = create_sae("standard", hidden_dim=16, latent_dim=32,
                     normalize_activations="none").eval()
    x = torch.randn(4, 16) * 3.0
    with torch.no_grad():
        assert torch.equal(encode_with_training_normalization(sae, x), sae.encode(x))


def test_the_helper_preserves_the_gradient_path():
    """Attribution encodes IN-GRAPH; normalization must not detach it."""
    sae = create_sae("standard", hidden_dim=16, latent_dim=32,
                     normalize_activations="constant_norm_rescale")
    x = torch.randn(4, 16, requires_grad=True)
    z = encode_with_training_normalization(sae, x)
    z.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ── Every consumer is on it ────────────────────────────────────────────────

_SAE_ENCODERS = [
    "src.services.circuit_capture_service",
    "src.services.circuit_attribution_service",
    "src.services.circuit_faithfulness_service",
    "src.services.circuit_intervention_service",
    "src.services.extraction_service",
]


@pytest.mark.parametrize("modname", _SAE_ENCODERS)
def test_no_module_calls_a_bare_sae_encode(modname):
    """An AST walk for `<something>.encode(...)` on an SAE-shaped receiver.

    Names the receiver explicitly rather than matching every `.encode(` — the
    tokenizer's, `str.encode`, and the sentence-transformer's are all legitimate
    and unrelated.
    """
    import importlib

    tree = ast.parse(inspect.getsource(importlib.import_module(modname)))
    sae_receivers = {"sae", "sae_d", "sae_e", "model", "sae_l"}
    offenders = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "encode"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id.lower() in sae_receivers
        ):
            offenders.append(f"{node.func.value.id}.encode(...) at line {node.lineno}")
    assert not offenders, (
        f"{modname} encodes without the training-time normalization — the SAE "
        f"sees activations it was not fitted on: {offenders}"
    )


def test_the_ast_scan_would_catch_the_original_defect():
    """Negative control — a source-derived guard that matches nothing asserts nothing."""
    tree = ast.parse("def f(sae, acts):\n    z = sae.encode(acts)\n    return z\n")
    hits = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "encode"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id.lower() in {"sae"}
    ]
    assert len(hits) == 1, "the scan no longer detects a bare sae.encode call"


def test_the_capture_loader_carries_the_trained_normalization():
    """The mode must survive the load, or no consumer can apply the right one."""
    from src.services import circuit_capture_service as m

    src = inspect.getsource(m._load_sae_sync)
    assert "normalize_activations" in src, (
        "_load_sae_sync loads `config` and discards it, so every SAE it builds "
        "takes the constructor default instead of its trained mode"
    )
    assert "getattr(config" in src, "the mode must come from the loaded config"


def test_create_sae_actually_honours_the_mode_it_is_given():
    """Negative control for the loader test above.

    Passing `normalize_activations` through is worthless if the factory drops
    it — which would leave the loader test asserting a keyword that goes
    nowhere.
    """
    sae = create_sae("standard", hidden_dim=8, latent_dim=16,
                     normalize_activations="constant_norm_rescale")
    assert sae.normalize_activations == "constant_norm_rescale"
    other = create_sae("standard", hidden_dim=8, latent_dim=16,
                       normalize_activations="none")
    assert other.normalize_activations == "none"
