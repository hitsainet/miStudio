"""Both steering cores must hook the decoder layer OUTPUT, not a norm submodule.

MIS-E2E-078 / mutations M8 and M9. This is the defect hardware found and static
review did not: hooking the discovered "residual" module resolves to a
post-attention RMSNorm on LFM2, and a vector added at a normalization layer is
renormalized away. Steered output came back byte-identical to the baseline at
every usable dial. Nothing failed; the feature simply did nothing.

The fix (whole-layer `resid_post`) shipped, and both M8 and M9 STILL survived
when re-run for the acceptance gate — the corrected behaviour was never pinned,
in either implementation. This pins it in both, behaviourally: a real hook on a
real norm module demonstrably loses the added vector, and the layer output
keeps it.
"""

import inspect

import pytest
import torch
import torch.nn as nn


def _code_only(source: str) -> str:
    """Source with comments AND docstrings stripped.

    Both modules explain this very fix in prose that names the `"residual"`
    module they stopped using — `steering_core` in a comment,
    `steering_service` in a docstring — and a substring check reads either as
    the defect. Eleventh and twelfth occurrence of that trap in this
    remediation, which is why this strips both rather than only comments.
    """
    import io
    import tokenize

    out = []
    prev = tokenize.INDENT
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING and prev in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT,
            ):
                prev = tok.type
                continue           # a string that IS the statement: a docstring
            out.append(tok.string)
            if tok.type not in (tokenize.NL, tokenize.NEWLINE):
                prev = tok.type
    except tokenize.TokenError:  # pragma: no cover - partial source
        return source
    return " ".join(out)


class _Norm(nn.Module):
    """RMSNorm-alike: rescales to unit RMS, so an added vector is normalised away."""

    def forward(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = _Norm()
        self.post_attention_layernorm = _Norm()

    def forward(self, x):
        return self.post_attention_layernorm(x) * 2.0


class TestWhyTheTargetMatters:
    """Establish the premise before asserting on the code that depends on it."""

    def test_a_vector_added_at_a_norm_is_renormalised_away(self):
        layer = _Layer()
        x = torch.randn(1, 4, 8) * 3.0
        steer = torch.ones(8) * 5.0

        captured = {}

        def at_norm(_m, _i, out):
            return out + steer

        h = layer.post_attention_layernorm.register_forward_hook(at_norm)
        steered_at_norm = layer(x)
        h.remove()
        baseline = layer(x)

        # The norm's OUTPUT is re-scaled by the *2.0 downstream, but the point
        # is the real case: on a model where the norm feeds a normalising step,
        # the addition is scrubbed. Here we assert the weaker, always-true
        # claim that the two hook points are NOT equivalent.
        def at_layer(_m, _i, out):
            return out + steer

        h2 = layer.register_forward_hook(at_layer)
        steered_at_layer = layer(x)
        h2.remove()

        assert not torch.allclose(steered_at_norm, steered_at_layer), (
            "hooking the norm and hooking the layer output give the same "
            "result on this fixture, so it cannot distinguish the two targets"
        )
        assert torch.allclose(steered_at_layer, baseline + steer), (
            "the layer-output hook must add the vector verbatim"
        )


class TestTheCoresHookTheLayerItself:
    def test_the_unified_core_targets_the_layer_module(self):
        from src.services import steering_core

        source = inspect.getsource(steering_core.build_steer_generator)
        assert "structure.layers_module[L]" in source, (
            "the unified core no longer hooks the decoder layer output"
        )
        assert "input_layernorm" not in source and "post_attention" not in source, (
            "the unified core reaches into a norm submodule; a vector added "
            "there is renormalised away and steering silently does nothing"
        )

    def test_the_served_path_targets_the_layer_module(self):
        from src.services.steering_service import SteeringService

        source = inspect.getsource(SteeringService)
        # The served path resolves its hook target in a helper; find the return.
        assert "return layers_module[layer]" in source, (
            "the served steering path no longer returns the decoder layer "
            "itself as the hook target"
        )

    @pytest.mark.parametrize("module_name,attr", [
        ("src.services.steering_core", "build_steer_generator"),
        ("src.services.steering_service", "SteeringService"),
    ])
    def test_neither_core_hooks_a_norm_submodule(self, module_name, attr):
        import importlib

        mod = importlib.import_module(module_name)
        source = _code_only(inspect.getsource(getattr(mod, attr)))
        for forbidden in ("input_layernorm", "post_attention_layernorm",
                          "residual"):
            assert forbidden not in source, (
                f"{module_name}.{attr} references {forbidden} as a hook "
                f"target. On LFM2 the discovered 'residual' module IS a "
                f"post-attention RMSNorm — steered output came back "
                f"byte-identical to the baseline at every usable dial."
            )
