"""The readout must find the final norm on a NESTED (unified) architecture.

gemma-4-12B is `Gemma4UnifiedForConditionalGeneration`: text, vision and audio
towers under one root, with the decoder stack at
`model.model.language_model.layers` and its final norm beside it at
`model.model.language_model.norm`. `_resolve_final_norm` searched only `model`
and `model.model`, so it found nothing and `_normalize` fell back to plain RMS.

**THAT FALLBACK CANNOT BE SEEN IN THE OUTPUT.** Plain RMS is a scalar divide,
so it leaves token rankings bit-identical to applying no norm at all — the
readout is not obviously broken, it has merely dropped the learned per-channel
gain that carries the signal. Measured on the cluster 2026-09-05, identical
residuals and identical W_U:

    without the norm   ['𒅘', '𒉼', '𒈿', '𒉘', '𒂷', ...]   (and the SAME
                        rare cuneiform for every prompt)
    with the norm      [' Paris', 'Paris', '巴黎', ' París', ' Париж', ...]

A 53-minute gemma-4-12B fit was rejected by the SEMANTIC validation class over
this while the fitted artifact itself was fine.
"""

import logging

import pytest
import torch
from torch import nn

from src.services.jlens_readout_service import ReadoutService


class _Layer(nn.Module):
    def forward(self, x):  # pragma: no cover - never called
        return x


class _Tower(nn.Module):
    """A language-model tower: the decoder list and its final norm, together."""

    def __init__(self, n_layers=4, d_model=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, d_model)
        self.layers = nn.ModuleList(_Layer() for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)


class _UnifiedInner(nn.Module):
    """The shape that broke it: the tower is a CHILD, not the root."""

    def __init__(self):
        super().__init__()
        self.language_model = _Tower()
        self.vision_embedder = nn.Linear(4, 8)


class _UnifiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _UnifiedInner()


class _FlatInner(nn.Module):
    """The ordinary Llama/Granite shape, which always resolved."""

    def __init__(self, d_model=8):
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(4))
        self.norm = nn.LayerNorm(d_model)


class _FlatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FlatInner()


class _Structure:
    def __init__(self, layers_module):
        self.layers_module = layers_module


def test_final_norm_is_found_on_a_nested_unified_model():
    model = _UnifiedModel()
    tower = model.model.language_model
    structure = _Structure(tower.layers)

    found = ReadoutService._resolve_final_norm(model, structure)

    assert found is tower.norm, (
        "the final norm was not found beside the decoder layers; the readout "
        "will fall back to plain RMS, which cannot change a ranking and so "
        "produces a silently wrong readout"
    )


def test_the_nested_norm_is_unreachable_from_the_old_search_roots():
    """Negative control for the premise.

    If `model` or `model.model` exposed a `norm` of their own, the test above
    would pass without the parent-of-layers search doing any work — the fixture
    would agree by construction and pin nothing. This asserts the fixture
    really does model the shape that broke.
    """
    model = _UnifiedModel()
    for owner in (model, model.model):
        for name in ("norm", "ln_f", "final_layernorm", "final_layer_norm",
                     "embedding_norm"):
            assert not isinstance(getattr(owner, name, None), nn.Module), (
                f"fixture is not the broken shape: {name} is reachable directly"
            )


def test_the_flat_case_still_resolves_the_same_module():
    """The fix must not move the answer for architectures that already worked."""
    model = _FlatModel()
    structure = _Structure(model.model.layers)

    assert ReadoutService._resolve_final_norm(model, structure) is model.model.norm
    # ...and with no structure at all, the old two-root search still applies.
    assert ReadoutService._resolve_final_norm(model, None) is model.model.norm


def test_a_model_with_no_final_norm_says_so_out_loud(caplog):
    """The RMS fallback must announce itself, FROM THE REAL CONSTRUCTOR.

    `_normalize`'s docstring claimed the fallback was "recorded" and nothing
    recorded it. A silent fallback is why a whole fit was spent before anyone
    noticed, so the warning is the fix, not decoration.

    This drives `ReadoutService.__init__` rather than emitting the log line
    itself. A test that logs its own message and then asserts the message was
    logged passes against a service that warns about nothing — the exact shape
    of assertion this repo keeps catching.
    """

    class _NoNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(_Layer() for _ in range(2))

    model = _NoNorm()
    structure = _Structure(model.layers)
    assert ReadoutService._resolve_final_norm(model, structure) is None

    with caplog.at_level(logging.WARNING, logger="src.services.jlens_readout_service"):
        svc = ReadoutService(
            model=model,
            tokenizer=object(),
            structure=structure,
            unembedding=torch.zeros(5, 8),
            model_name="fixture/no-norm",
        )

    assert svc._final_norm is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("plain RMS" in w for w in warnings), (
        f"the RMS fallback was taken silently; warnings seen: {warnings}"
    )
    assert any("fixture/no-norm" in w for w in warnings), (
        "the warning does not name the model, so an operator cannot tell which "
        "readout is untrustworthy"
    )


def test_a_model_that_resolves_its_norm_warns_about_nothing(caplog):
    """Negative control for the warning: it must not fire on a healthy model."""
    model = _FlatModel()
    with caplog.at_level(logging.WARNING, logger="src.services.jlens_readout_service"):
        svc = ReadoutService(
            model=model,
            tokenizer=object(),
            structure=_Structure(model.model.layers),
            unembedding=torch.zeros(5, 8),
            model_name="fixture/flat",
        )
    assert svc._final_norm is model.model.norm
    assert not [r for r in caplog.records if "plain RMS" in r.getMessage()], (
        "a model WITH a final norm warned about the fallback — the guard fires "
        "on everything and therefore says nothing"
    )


def test_plain_rms_cannot_change_a_ranking():
    """Why the fallback is invisible — the premise of this whole file.

    If RMS-only reordered tokens, a broken resolution would show up as obvious
    garbage and be caught immediately. It does not: it is a positive scalar
    divide, so argsort is preserved exactly. Computed, not asserted by eye.
    """
    torch.manual_seed(0)
    x = torch.randn(64)
    W = torch.randn(200, 64)

    raw_rank = (W @ x).argsort(descending=True)
    rms = x.pow(2).mean().sqrt().clamp_min(1e-6)
    rms_rank = (W @ (x / rms)).argsort(descending=True)

    assert torch.equal(raw_rank, rms_rank), (
        "if this ever fails the RMS fallback is no longer rank-preserving and "
        "the reasoning in this file needs revisiting"
    )

    # ...whereas a LEARNED per-channel gain does reorder, which is the signal
    # the fallback throws away.
    gain = torch.rand(64) * 4.0
    gained_rank = (W @ ((x / rms) * gain)).argsort(descending=True)
    assert not torch.equal(raw_rank, gained_rank)
