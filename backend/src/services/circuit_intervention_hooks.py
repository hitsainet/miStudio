"""
Circuit intervention hooks (Feature 017, IDL-34 — FTDD §1 / A.2/A.5 normative).

Directional suppression: at a hooked layer, remove the contribution of an
upstream feature u by SUBTRACTING (a_u(t) − a_base)·W_dec[:, u] from the
residual stream — the feature's decoder direction scaled by how far above
baseline it fired.

THE CARDINAL RULE (A.2): we NEVER re-decode. We subtract from the residual we
were handed; the SAE reconstruction error ε is untouched BY CONSTRUCTION, so
the measurement is a real causal intervention, not an SAE artifact. (The
classic mistake — reconstruct x̂ = decode(encode(x)) and steer that — drops ε
and turns the whole model into its SAE approximation.)

Pure of DB/GPU-orchestration concerns: it needs only a decoder-weight tensor
and per-feature activation getter, so ε-invariance and subtraction-exactness
are pinned on toy tensors without a model.
"""

import logging
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)


def suppress_directional(hidden, decoder_weight, feature_idx, a_u, a_base=0.0,
                         positions=None):
    """Subtract (a_u − a_base)·d_i from the residual, in place, returning it.

    hidden: [batch, seq, d_model] residual activations (modified in place).
    decoder_weight: [d_model, d_sae] (resolve_decoder_weight orientation).
    feature_idx: the upstream feature u.
    a_u: [batch, seq] the feature's activation at each token (same-pass encode).
    a_base: scalar baseline (0 or corpus-mean of u).
    positions: optional [batch, seq] bool mask — suppress only where True
        (v1: all captured-fire positions); None ⇒ everywhere.

    ε is never touched: we operate on `hidden` (the residual), never on a
    re-decoded reconstruction.
    """
    import torch

    d_i = decoder_weight[:, feature_idx]            # [d_model]
    delta = (a_u - a_base)                          # [batch, seq]
    if positions is not None:
        delta = delta * positions.to(delta.dtype)
    # outer product delta ⊗ d_i → [batch, seq, d_model]
    hidden.sub_(delta.unsqueeze(-1) * d_i.to(hidden.dtype))
    return hidden


def suppress_feature_list(hidden, decoder_weight, feature_indices, a_us,
                          a_bases=None, positions=None):
    """Faithfulness path: subtract the SUM of several features' contributions
    in ONE pass (per-layer member suppression). Summing in a single op keeps
    float ordering stable vs N separate hooks (FTID pitfall)."""
    import torch

    feature_indices = list(feature_indices)
    if a_bases is None:
        a_bases = [0.0] * len(feature_indices)
    # accumulate the combined delta·d_i then subtract once
    total = torch.zeros_like(hidden)
    for idx, a_u, a_base in zip(feature_indices, a_us, a_bases):
        d_i = decoder_weight[:, idx]
        delta = (a_u - a_base)
        if positions is not None:
            delta = delta * positions.to(delta.dtype)
        total = total + delta.unsqueeze(-1) * d_i.to(hidden.dtype)
    hidden.sub_(total)
    return hidden


def make_suppression_hook(decoder_weight, feature_idx, a_base=0.0,
                          positions=None, encode_fn=None) -> Callable:
    """A forward hook (house convention, mirrors _create_steering_hook) that
    suppresses `feature_idx`. `encode_fn(hidden) -> a_u [batch,seq]` computes
    the feature's activation from the SAME pass (cheap encode; never a
    re-decode). In-place modification, original tuple returned (Gemma-2 safe).
    """
    def hook(module, inp, output):
        import torch

        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if hidden.dim() != 3:
            return output
        with torch.no_grad():
            a_u = (encode_fn(hidden) if encode_fn is not None
                   else torch.zeros(hidden.shape[:2], device=hidden.device))
            suppress_directional(hidden, decoder_weight, feature_idx, a_u,
                                 a_base=a_base, positions=positions)
        return output  # in place — same references (LFM2/Gemma-2 safe)

    return hook
