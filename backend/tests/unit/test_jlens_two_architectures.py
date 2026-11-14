"""
Phase 8.1: the same code path over TWO architectures.

A suite that only ever sees one architecture is how `SUPPORTED_ARCHITECTURES`
survived as long as it did. So every assertion here runs twice — once over a
HYBRID stack that interleaves attending and non-attending layers, and once over
a DENSE stack where every layer attends — and the hybrid case is the one that
catches the errors, because it is the shape where "freeze Q/K" is undefined on
some layers and where a per-layer property cannot be inferred from a model-level
one.

The two fixtures deliberately differ in layer COUNT as well as in kind, so
nothing can pass by agreeing with a hardcoded shape.

MUTATION CONTROLS (each must turn this file red):
  * return False instead of None for an inapplicable layer -> "absent" fails
  * infer layer kind from the model instead of per layer   -> "hybrid" fails
  * assume a fixed layer count anywhere on the path        -> "axis" fails
"""

from __future__ import annotations

import pytest
import torch

from src.services.jlens_readout_service import (
    IdentityTransport,
    ReadoutService,
    build_layer_applicability,
)
from src.schemas.jlens import LensMetaMessage, LensTokenMessage

D_MODEL = 8
N_VOCAB = 61


class Block(torch.nn.Module):
    def forward(self, hidden, **_):
        return (hidden * 1.02,)


class Stack(torch.nn.Module):
    def __init__(self, n_layers: int, layer_types=None):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(n_layers)])
        self.embed = torch.nn.Embedding(64, D_MODEL)

        class Config:
            pass

        self.config = Config()
        if layer_types is not None:
            self.config.layer_types = layer_types

    def forward(self, input_ids=None, **_):
        hidden = self.embed(input_ids)
        for b in self.blocks:
            hidden = b(hidden)[0]
        return hidden


class Structure:
    def __init__(self, stack: Stack, attention: bool):
        self.layers_module = stack.blocks
        self.num_layers = len(stack.blocks)
        self.attention_module = object() if attention else None
        self.residual_norm_module = None


class Tok:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids] if isinstance(ids, list) else f"t{ids}"

    def decode(self, ids, **_):
        return "t"


# A 16-layer hybrid: 10 conv, 6 attention — the reference model's shape.
HYBRID_TYPES = [
    "conv", "conv", "full_attention", "conv", "conv", "full_attention",
    "conv", "conv", "full_attention", "conv", "conv", "full_attention",
    "conv", "full_attention", "conv", "full_attention",
]
# A 26-layer dense model publishes no per-layer kind at all.
DENSE_LAYERS = 26


def make(kind: str):
    if kind == "hybrid":
        stack = Stack(len(HYBRID_TYPES), layer_types=HYBRID_TYPES)
        return stack, Structure(stack, attention=True)
    stack = Stack(DENSE_LAYERS)
    return stack, Structure(stack, attention=True)


def service_for(kind: str) -> ReadoutService:
    stack, structure = make(kind)
    return ReadoutService(
        model=stack,
        tokenizer=Tok(),
        structure=structure,
        unembedding=torch.randn(N_VOCAB, D_MODEL),
        model_name=f"{kind}-model",
    )


@pytest.mark.parametrize("kind,expected_layers", [("hybrid", 16), ("dense", 26)])
def test_the_readout_axis_follows_the_model_not_a_constant(kind, expected_layers):
    """The two fixtures differ in layer count, so a fixed axis fails one of them."""
    messages = list(service_for(kind).stream("abcd", [IdentityTransport()], top_n=3))

    meta = next(m for m in messages if isinstance(m, LensMetaMessage))
    assert len(meta.layers_by_type["LOGIT_LENS"]) == expected_layers

    tokens = [m for m in messages if isinstance(m, LensTokenMessage)]
    assert tokens, "no token messages produced"
    for token in tokens:
        assert len(token.results[0].top_tokens) == expected_layers


@pytest.mark.parametrize("kind", ["hybrid", "dense"])
def test_a_well_formed_stream_on_both_architectures(kind):
    messages = list(service_for(kind).stream("abcd", [IdentityTransport()], top_n=3))
    meta = next(m for m in messages if isinstance(m, LensMetaMessage))

    assert meta.types == ["LOGIT_LENS"]
    assert meta.top_n == 3
    for token in (m for m in messages if isinstance(m, LensTokenMessage)):
        for row, probs in zip(token.results[0].top_tokens, token.results[0].top_probs):
            assert len(row) == len(probs) == 3
            # Decoded strings, never ids — ids type-check and render as
            # unreadable cells.
            assert all(isinstance(t, str) for t in row)


def test_the_hybrid_records_inapplicable_layers_as_ABSENT():
    """10 of 16 layers do not attend, so frozen-Q/K is UNDEFINED there.

    None, not False. A False is averaged by a consumer and reads as "we checked
    and it does not apply"; absent forces the consumer to decide. This is the
    assertion the dense fixture cannot make, which is why both exist.
    """
    stack, structure = make("hybrid")
    applicability = build_layer_applicability(structure, stack.config)

    assert len(applicability) == 16
    attending = [a for a in applicability if a.has_attention]
    assert len(attending) == 6, "the hybrid fixture is not actually hybrid"

    for entry in applicability:
        if entry.has_attention:
            assert entry.frozen_qk_applicable is True
            assert entry.broadcast_metrics_applicable is True
        else:
            assert entry.frozen_qk_applicable is None
            assert entry.broadcast_metrics_applicable is None


def test_the_dense_model_applies_everywhere():
    stack, structure = make("dense")
    applicability = build_layer_applicability(structure, stack.config)

    assert len(applicability) == DENSE_LAYERS
    assert all(a.has_attention for a in applicability)
    assert all(a.frozen_qk_applicable is True for a in applicability)


def test_applicability_is_never_averaged_into_a_model_level_claim():
    """"This artifact is frozen-Q/K" is false when the treatment reached 6 of 16.

    Asserted as a counting property because the tempting simplification —
    reducing per-layer applicability to one boolean — produces a recipe that
    describes an artifact nobody built.
    """
    stack, structure = make("hybrid")
    applicability = build_layer_applicability(structure, stack.config)

    applied = sum(1 for a in applicability if a.frozen_qk_applicable is True)
    inapplicable = sum(1 for a in applicability if a.frozen_qk_applicable is None)

    assert applied == 6
    assert inapplicable == 10
    assert applied + inapplicable == len(applicability)
    # The model-level shorthand would be `all(...)`, which is False here, and
    # `any(...)`, which is True — neither describes the artifact.
    assert not all(a.frozen_qk_applicable for a in applicability)
