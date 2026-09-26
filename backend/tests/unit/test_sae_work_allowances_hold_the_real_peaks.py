"""The SAE working allowances a circuit pass is placed and mapped with hold what the pass really allocates.

Review round 3 (2026-09-14). A split keeps SHARD_RESERVE_MB (1 GiB) free per card, and each
layer's SAE sits on its layer's card (plan decision D5). Round 2 charged each layer's SAE
WEIGHTS there and nothing for what the pass does with them:

* capture encodes a batch of 8 x 512 tokens per layer. Its codes, pre-activations and event
  mask on a 65,536-feature SAE are 2-3 GiB, beside the previous layer's codes and events
  still held by the loop: up to 5 GiB on the layer's card, five times the reserve;
* attribution keeps every hooked layer's codes, their gradient and the graph's saved tensors
  until its backward: about 1 GiB a layer at 512 tokens.

`base_model_budget.sae_encode_working_mb` and `sae_backward_retained_mb` size them. Their
unit counts are measured, and these tests keep them honest: each runs the REAL per-layer
path (capture's encode, decode, threshold and nonzero; attribution's passthrough hook and
`attribute_prompt`) on small SAEs of every architecture and requires the allowance to hold
the measured peak. A test that asserted a hand-worked figure would pass a unit count the
code outgrew.

FOUND BY THIS TEST: the first unit count, 4 (2-3 measured for one encode, plus the previous
layer's codes), was below the real peak once the previous layer's EVENTS were held too (86 MiB
against 64 for a standard SAE at the fixture's size). It is 5, stated for SAEs firing on up to
10% of their features.

MUTATION CONTROLS (review round 3, 2026-09-14; scratchpad p2-r3/mutate.py + mutations_r3.json,
each alone in a private copy of backend/, restored by sha256). Both killed:
  N21 SAE_ENCODE_WORKING_UNITS 5 -> 4   -> test_the_encode_allowance_holds_two_capture_layers...[jumprelu]
  N22 SAE_BACKWARD_UNITS 8 -> 5         -> test_the_backward_allowance_holds_what_attribution_keeps...[jumprelu]
"""

from __future__ import annotations

import copy
import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from transformers import AutoModelForCausalLM, LlamaConfig

from src.ml.sparse_autoencoder import create_sae
from src.services.base_model_budget import sae_backward_retained_mb, sae_encode_working_mb
from src.services.circuit_attribution_service import PassThroughState, attribute_prompt, make_passthrough_hook
from src.services.circuit_capture_service import _encode_layer, _event_threshold

MIB = 1024 * 1024
ARCHITECTURES = ["standard", "jumprelu", "topk"]


class _PeakStorage(TorchDispatchMode):
    """Peak live CPU tensor storage allocated inside the block, each storage counted once."""

    def __init__(self):
        super().__init__()
        self.live = self.peak = 0
        self._sizes = {}

    def _free(self, key):
        self.live -= self._sizes.pop(key, 0)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for tensor in tree_flatten(out)[0]:
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                storage = tensor.untyped_storage()
                key = storage.data_ptr()
                if key and key not in self._sizes:
                    self._sizes[key] = storage.nbytes()
                    self.live += storage.nbytes()
                    weakref.finalize(storage, self._free, key)
        self.peak = max(self.peak, self.live)
        return out


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_the_encode_allowance_holds_two_capture_layers_encoding_one_after_the_other(architecture):
    """Two layers on one card, as capture's loop runs them: the second encodes while the
    first layer's codes and events are still held. Each SAE's threshold selects 10% of its
    features, the density the allowance is stated for (a random SAE's zero threshold
    selects about half, which no trained SAE a capture would be run with does)."""
    tokens, d_model, d_sae = 1024, 64, 4096
    torch.manual_seed(0)
    saes = [create_sae(architecture, hidden_dim=d_model, latent_dim=d_sae, normalize_activations="none").eval()
            for _ in range(2)]
    flat = torch.randn(tokens, d_model)
    floors = []
    for sae in saes:
        with torch.no_grad():
            codes = _encode_layer(sae, flat).flatten()
        floors.append(float(codes.kthvalue(int(0.9 * codes.numel())).values))
    held = []
    tracker = _PeakStorage()
    with tracker:
        for sae, floor in zip(saes, floors):
            with torch.no_grad():
                z = _encode_layer(sae, flat)
                recon = sae.decode(z)
                err = (flat - recon).norm(dim=-1)
            thresh = _event_threshold(z, None, 0.0, floor)
            hits = (z > thresh.unsqueeze(0)).nonzero(as_tuple=False)
            values = z[hits[:, 0], hits[:, 1]]
            held = [z, recon, err, thresh, hits, values]
    del held

    assert tracker.peak <= sae_encode_working_mb(d_sae, tokens) * MIB, (
        architecture, tracker.peak / MIB, sae_encode_working_mb(d_sae, tokens))


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_the_backward_allowance_holds_what_attribution_keeps_beside_each_layer(architecture):
    """Two hooked layers, one prompt: the codes and graph attribution keeps, per layer, beyond
    the same run with 128-feature SAEs (the model's own activations)."""
    seq_len, layers = 128, [0, 2]

    def peak(d_sae):
        config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                             num_key_value_heads=4, vocab_size=500, tie_word_embeddings=False)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(copy.deepcopy(config), dtype=torch.float32).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        saes = {}
        for L in layers:
            saes[L] = create_sae(architecture, hidden_dim=64, latent_dim=d_sae, normalize_activations="none").eval()
            for p in saes[L].parameters():
                p.requires_grad_(False)
        input_ids = torch.randint(0, 500, (1, seq_len))
        tracker = _PeakStorage()
        with tracker:
            state = PassThroughState()
            handles = [model.model.layers[L].register_forward_hook(make_passthrough_hook(L, saes[L], state))
                       for L in layers]
            try:
                model(input_ids=input_ids)
                attribute_prompt(state, {(layers[-1], 0): [(layers[0], 1)]})
            finally:
                for handle in handles:
                    handle.remove()
            del state
        return tracker.peak

    kept = peak(8192) - peak(128)

    assert kept <= len(layers) * sae_backward_retained_mb(8192, seq_len) * MIB, (
        architecture, kept / MIB, len(layers) * sae_backward_retained_mb(8192, seq_len))
