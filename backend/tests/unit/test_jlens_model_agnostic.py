"""The J-space readout must work for ANY model the workbench can load.

This is the point of the increment, and it is a promise this codebase has had to
make once already: the SUPPORTED_ARCHITECTURES whitelist was deleted and every
hardcoded architecture branch replaced with `discover_transformer_structure`.
J-space must not reintroduce one.

TWO PITFALLS ARE PINNED HERE, both of which have cost real time in this repo:

  1. HOOK TARGET. `TransformerStructure.residual_norm_module` sounds like the
     residual stream and is not. On a hybrid model it is a post-attention
     RMSNorm. In the steering work a vector applied there was renormalised away
     and steered output was BYTE-IDENTICAL to unsteered at every dial
     (steering_core.py:230, PADR IDL-38). A readout captured there fails the
     same way but is harder to notice: plausibly-shaped numbers with the signal
     scaled out.

  2. HYBRID LAYERS. The reference model interleaves 10 conv with 6 attention
     layers over 16. "Freeze Q/K" is undefined on a conv layer, so applicability
     is per-layer and inapplicable means ABSENT (None), never False — a False
     gets averaged and silently understates.

MUTATION CONTROLS:
  * hook residual_norm_module instead of layers_module[L] -> hook test fails
  * make applicability model-level instead of per-layer   -> hybrid test fails
  * record frozen_qk=False on a conv layer                -> absent test fails
  * add an architecture name to the service               -> name guard fails
"""

import ast
import inspect
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.services import jlens_readout_service as mod
from src.services.jlens_readout_service import (
    IdentityTransport,
    ReadoutService,
    build_layer_applicability,
)

SERVICE_PATH = Path(mod.__file__)


# --------------------------------------------------------------------- fakes


class FakeBlock(torch.nn.Module):
    """A decoder block whose OUTPUT is the residual stream (resid_post)."""

    def __init__(self, d_model: int, scale: float):
        super().__init__()
        self.scale = scale
        # A module that a naive search would call "residual"/norm.
        self.ffn_norm = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        return (x * self.scale,)


class FakeModel(torch.nn.Module):
    def __init__(self, d_model, n_layers, vocab, layer_types=None):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [FakeBlock(d_model, 1.0 + i) for i in range(n_layers)]
        )
        self.embed = torch.nn.Embedding(vocab, d_model)
        self.config = SimpleNamespace(
            hidden_size=d_model, num_hidden_layers=n_layers, vocab_size=vocab
        )
        if layer_types is not None:
            self.config.layer_types = layer_types

    def forward(self, input_ids=None, **_):
        x = self.embed(input_ids)
        for blk in self.layers:
            x = blk(x)[0]
        return SimpleNamespace(logits=x)


class FakeTokenizer:
    """Ids stay strictly inside [0, vocab) — an id == vocab indexes past the
    embedding table and raises IndexError from inside the fake model, which
    reads as a service bug rather than a fixture bug."""

    def __init__(self, vocab: int = 50):
        self.vocab = vocab

    def __call__(self, text, return_tensors=None):
        ids = [ord(c) % self.vocab for c in text[:6]]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids):
        return f"t{int(ids[0])}"

    def encode(self, s, add_special_tokens=False):
        return [ord(s[0]) % self.vocab]


def make_structure(model, attention_module="attn"):
    return SimpleNamespace(
        layers_module=model.layers,
        num_layers=len(model.layers),
        attention_module=attention_module,
        mlp_module="mlp",
        residual_norm_module="ffn_norm",
    )


def make_service(layer_types=None, d_model=8, n_layers=4, vocab=50):
    model = FakeModel(d_model, n_layers, vocab, layer_types)
    W_U = torch.randn(vocab, d_model)
    return ReadoutService(
        model=model,
        tokenizer=FakeTokenizer(vocab),
        structure=make_structure(model),
        unembedding=W_U,
        model_name="fake",
    )


# ------------------------------------------------------------- architecture


class TestNoArchitectureNames:
    def test_service_has_no_architecture_branch(self):
        """A name in the executable path is how the old whitelist survived."""
        src = SERVICE_PATH.read_text()
        code_lines = []
        in_doc = False
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_doc = not in_doc if stripped.count('"""') % 2 else in_doc
                continue
            if in_doc or stripped.startswith("#"):
                continue
            code_lines.append(line)

        offenders = [
            l for l in code_lines
            if re.search(r"\b(lfm2|gemma|llama|granite|mistral|qwen|neox)\b", l, re.I)
        ]
        assert not offenders, (
            f"architecture name in the executable path: {offenders}. "
            "Resolve structure through discover_transformer_structure instead."
        )

    def test_two_architectures_produce_well_formed_streams(self):
        """A suite that only sees one architecture proves nothing.

        Hybrid (mixed conv/attention) and dense must both work with no code
        change between them.
        """
        hybrid = make_service(
            layer_types=["conv", "conv", "full_attention", "conv"]
        )
        dense = make_service(layer_types=None)

        for svc in (hybrid, dense):
            msgs = list(svc.stream("abc", [IdentityTransport()], top_n=3))
            meta = msgs[0]
            tokens = [m for m in msgs if getattr(m, "kind", None) == "token"]
            assert meta.kind == "meta"
            assert msgs[-1].kind == "done"
            assert tokens, "no token messages"
            for t in tokens:
                for sl in t.results:
                    assert len(sl.top_tokens) == len(meta.layers_by_type[sl.type])


# -------------------------------------------------------------- hook target


class TestHookTarget:
    def test_capture_hooks_the_decoder_layer_not_the_norm(self):
        """Check EXECUTABLE lines only.

        The function's docstring names residual_norm_module precisely to say
        what not to hook; a naive substring search over the whole source flags
        that explanation as the defect it warns about.
        """
        src = inspect.getsource(ReadoutService._capture_residuals)
        body = ast.parse(textwrap.dedent(src)).body[0]
        code = ast.unparse(
            ast.Module(
                body=[n for n in body.body if not (
                    isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str)
                )],
                type_ignores=[],
            )
        )
        assert "layers_module[layer_idx]" in code, (
            "capture must hook the decoder layer output (resid_post)"
        )
        assert "residual_norm_module" not in code, (
            "capture hooks the norm module in executable code — on a hybrid "
            "model that is a post-attention RMSNorm and the signal is "
            "renormalised away"
        )

    def test_hook_target_is_recorded_for_diagnosis(self):
        svc = make_service()
        ids = torch.tensor([[1, 2, 3]])
        cap = svc._capture_residuals(ids, [0, 1])
        assert cap.hook_target == "layers_module[L]"

    def test_norm_capture_would_degrade_the_signal(self):
        """Negative control for the pitfall.

        The blocks scale by (1+i), so resid_post magnitudes differ per layer. A
        LayerNorm strips exactly that scale — which is why hooking it destroyed
        steering. If capturing at the norm produced the same values as capturing
        at the layer output, this whole guard would be pointless.
        """
        svc = make_service()
        ids = torch.tensor([[1, 2, 3]])
        correct = svc._capture_residuals(ids, [2]).by_layer[2]

        # What the wrong hook would have yielded.
        model = svc.model
        normed = model.layers[2].ffn_norm(correct)

        assert not torch.allclose(correct, normed, atol=1e-3), (
            "normalised and un-normalised captures are indistinguishable in "
            "this fixture, so the negative control proves nothing"
        )


# ------------------------------------------------------- hybrid applicability


class TestPerLayerApplicability:
    def test_hybrid_reports_only_attention_layers_as_applicable(self):
        types = ["conv", "conv", "full_attention", "conv", "full_attention", "conv"]
        model = FakeModel(8, 6, 50, types)
        appl = build_layer_applicability(make_structure(model), model.config)

        assert len(appl) == 6
        assert sum(1 for a in appl if a.has_attention) == 2

    def test_inapplicable_is_absent_not_false(self):
        """None forces a consumer to decide; False gets averaged."""
        types = ["conv", "full_attention"]
        model = FakeModel(8, 2, 50, types)
        appl = build_layer_applicability(make_structure(model), model.config)

        conv, attn = appl[0], appl[1]
        assert conv.frozen_qk_applicable is None, (
            "conv layer records frozen_qk as False rather than absent; False "
            "will be averaged and will silently understate"
        )
        assert conv.broadcast_metrics_applicable is None
        assert attn.frozen_qk_applicable is True

    def test_dense_model_without_layer_types_marks_all_applicable(self):
        model = FakeModel(8, 3, 50, layer_types=None)
        appl = build_layer_applicability(make_structure(model), model.config)
        assert all(a.has_attention for a in appl)

    def test_applicability_travels_on_the_meta_message(self):
        """A consumer must never infer homogeneity from the layer count."""
        svc = make_service(layer_types=["conv", "full_attention", "conv", "conv"])
        meta = next(iter(svc.stream("ab", [IdentityTransport()], top_n=2)))
        assert meta.layer_applicability is not None
        assert sum(1 for a in meta.layer_applicability if a.has_attention) == 1
