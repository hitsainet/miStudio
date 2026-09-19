"""The "residual" hook must capture the decoder layer's OUTPUT, on real architectures.

FOUND ON A LIVE EXTRACTION, 2026-09-12. A 390 GB LFM2.5-1.2B-Instruct
extraction stored, under the name "residual", the output of each layer's
`ffn_norm` — an RMS-normalised signal read before the MLP:

  * dividing every stored token by `ffn_norm.weight` gave RMS 0.96-0.996 with
    about 1% spread (a raw residual stream varies far more than that);
  * re-running a row through the extraction's own loader reproduced the file
    EXACTLY at `ffn_norm`, and matched the decoder-layer output at cosine 0.74.

`get_hookable_module(layer, "residual", ...)` preferred the discovered norm
module (`post_attention_layernorm` on Llama, `ffn_norm` on LFM2) and fell back to
the layer only when no norm existed. Every SAE trained here learned that signal,
while miLLM encodes and steers at `model.layers[L]`'s output and the Neuronpedia
export labels the SAE `hook_resid_post`.

WHY REAL ARCHITECTURES. The previous unit test built a toy layer and asserted
`isinstance(module, nn.LayerNorm)` — it pinned the defect. A hand-made fixture
decides what "the layer output" means by construction; transformers' own
`output_hidden_states` does not, so it is the oracle here.
"""

import ast
import inspect

import pytest
import torch

from src.ml.forward_hooks import HookManager, HookType
from src.ml.layer_discovery import discover_transformer_structure, get_hookable_module


def _lfm2():
    from transformers import Lfm2Config, Lfm2ForCausalLM

    torch.manual_seed(0)
    config = Lfm2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
    )
    return Lfm2ForCausalLM(config).eval(), "lfm2", "ffn_norm"


def _llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2,
    )
    return LlamaForCausalLM(config).eval(), "llama", "post_attention_layernorm"


ARCHITECTURES = [pytest.param(_lfm2, id="lfm2"), pytest.param(_llama, id="llama")]
#: Not the last layer: some models append the FINAL-norm output as the last
#: hidden state, which is not any decoder layer's raw output.
LAYERS = [0, 1, 2]


def _run(model, arch):
    input_ids = torch.randint(0, 64, (2, 12), generator=torch.Generator().manual_seed(1))
    manager = HookManager(model)
    manager.register_hooks(LAYERS, [HookType.RESIDUAL], arch)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        captured = {name: list(tensors) for name, tensors in manager.activations.items()}
    finally:
        manager.remove_hooks()
    return captured, out.hidden_states


@pytest.mark.parametrize("build", ARCHITECTURES)
class TestTheCapturedSignalIsTheLayerOutput:
    def test_equals_transformers_own_hidden_state_for_that_layer(self, build):
        model, arch, _norm = build()
        captured, hidden_states = _run(model, arch)

        for L in LAYERS:
            got = torch.cat(captured[f"layer_{L}_residual"], dim=0)
            torch.testing.assert_close(
                got, hidden_states[L + 1], rtol=0, atol=0,
                msg=f"layer {L}: the residual hook did not capture the decoder layer's output",
            )

    def test_is_not_the_norm_submodule_output(self, build):
        """NEGATIVE CONTROL: the signal extraction used to store is different,
        so the test above cannot pass by coincidence."""
        model, arch, norm_name = build()
        norm_out = {}
        handles = [
            getattr(model.model.layers[L], norm_name).register_forward_hook(
                lambda m, i, o, L=L: norm_out.__setitem__(L, o.detach())
            )
            for L in LAYERS
        ]
        try:
            captured, _ = _run(model, arch)
        finally:
            for h in handles:
                h.remove()

        for L in LAYERS:
            got = torch.cat(captured[f"layer_{L}_residual"], dim=0)
            assert not torch.allclose(got, norm_out[L], atol=1e-5), (
                f"layer {L}: the residual hook captured the {norm_name} output — "
                f"a normalised pre-MLP signal, not the residual stream"
            )

    def test_get_hookable_module_returns_the_decoder_layer(self, build):
        model, arch, _norm = build()
        structure = discover_transformer_structure(model, architecture_hint=arch)
        for L in LAYERS:
            layer = structure.layers_module[L]
            assert get_hookable_module(layer, "residual", structure) is layer


class TestTheExtractionRecordsWhereItRead:
    def test_metadata_names_the_hook_point(self):
        from src.services.activation_service import ActivationService

        tree = ast.parse(_dedented(ActivationService.extract_activations))
        values = [
            v.value
            for node in ast.walk(tree) if isinstance(node, ast.Dict)
            for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant) and k.value == "hook_point" and isinstance(v, ast.Constant)
        ]
        assert values == ["resid_post"], (
            "metadata.json must say where activations were read; 'residual' alone "
            "meant a normalisation module until 2026-09-12"
        )


def _dedented(fn):
    import textwrap

    return textwrap.dedent(inspect.getsource(fn))
