"""Weight-space readouts of attention Q/K/V/O, per head (BR-015, 024 task 4.2).

A head's matrices act on the residual stream on one side: Q, K and V READ from it
(weight [head_dim, d_model] per head — their right singular vectors are directions in
residual space), O WRITES into it (weight [d_model, head_dim] per head — its left
singular vectors are). Projecting those residual-space directions through the lens gives
a component interpretation of what a head attends with, reads, and writes.

MODEL-AGNOSTIC, SHAPE-DRIVEN (BR-032). The structure comes from instantiating the
model on the META device (no memory, no weights) and finding the layer's attention module
structurally (`find_attention_module`). Its linear children are classified by SHAPE —
`in_features == d_model` reads, `out_features == d_model` writes — so a fused QKV or an
unfamiliar naming scheme is reported under its own name rather than refused. Only a SQUARE
projection (both d_model) is ambiguous by shape; the output projection's conventional
names (`_WRITE_NAME`) decide that case. Weights are
then read tensor by tensor from the checkpoint shards; nothing is loaded whole.

A SINGULAR VECTOR HAS NO SIGN, so both ends are read out: `toward_tokens` for +v and
`away_tokens` for −v. Reporting one end would invent a polarity the matrix does not have.

Logit lens only (J = I): a Jacobian artifact is fitted for residual STATES at a layer,
and these are weight directions; reading them through J is not what the artifact
measured. RUNG 0 — a component interpretation, not a claim about behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch import nn

MAX_COMPONENTS = 3

import re as _re

#: Output-projection names across families (llama/mistral/gemma `o_proj`, lfm2/granite
#: `out_proj`, gpt-neox/phi `dense`, gpt2 `c_proj`, t5-style `o`/`wo`).
_WRITE_NAME = _re.compile(r"^(o_proj|out_proj|dense|c_proj|o|wo|out)$")


def _meta_model(config: Any) -> nn.Module:
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    with init_empty_weights():
        return AutoModelForCausalLM.from_config(config)


def attention_projections(model: nn.Module, layer: int, d_model: int) -> Dict[str, Dict[str, Any]]:
    """`{name: {"key": checkpoint key, "side": "read"|"write", "out", "in"}}` for one layer."""
    from ..ml.layer_discovery import discover_transformer_structure, find_attention_module

    structure = discover_transformer_structure(model)
    if layer < 0 or layer >= structure.num_layers:
        raise ValueError(f"layer {layer} outside range 0..{structure.num_layers - 1}")
    attention = find_attention_module(structure.layers_module[layer])
    if attention is None:
        raise ValueError(
            f"layer {layer} has no attention module (a convolutional or state-space mixer "
            "has no Q/K/V/O to read)"
        )
    names = {id(m): n for n, m in model.named_modules()}
    prefix = names[id(attention)]
    found: Dict[str, Dict[str, Any]] = {}
    for child_name, child in attention.named_children():
        if not isinstance(child, nn.Linear):
            continue
        reads = child.in_features == d_model
        writes = child.out_features == d_model
        if reads and writes:
            # SQUARE (heads x head_dim == d_model, the common case): shape cannot say
            # which side it is on, so the output projection's conventional names decide.
            # An unrecognised square projection is reported as `read` AND named, so a
            # misclassification is visible rather than silent.
            side = "write" if _WRITE_NAME.search(child_name) else "read"
        elif reads:
            side = "read"
        elif writes:
            side = "write"
        else:
            continue
        found[child_name] = {
            "key": f"{prefix}.{child_name}.weight",
            "side": side,
            "out": child.out_features,
            "in": child.in_features,
        }
    if not found:
        raise ValueError(f"layer {layer}'s attention exposes no linear projection to read")
    return found


def read_tensor(snapshot: Path, key: str) -> torch.Tensor:
    """One tensor from a safetensors checkpoint, sharded or not, matched by key suffix
    when the checkpoint prefixes differ from the model's module names."""
    from safetensors import safe_open

    index = snapshot / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.is_file() else [p.name for p in snapshot.glob("*.safetensors")]
    )
    for fname in files:
        with safe_open(str(snapshot / fname), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if key in keys:
                return f.get_tensor(key)
            matches = [k for k in keys if k.endswith(key) or key.endswith(k)]
            if len(matches) == 1:
                return f.get_tensor(matches[0])
    raise KeyError(f"{key} is not in the checkpoint at {snapshot}")


def head_directions(weight: torch.Tensor, side: str, head_dim: int, head: int,
                    components: int) -> List[Dict[str, Any]]:
    """The top residual-space singular directions of one head's slice."""
    w = weight.to(torch.float32)
    lo, hi = head * head_dim, (head + 1) * head_dim
    if side == "read":
        block = w[lo:hi, :]                      # [head_dim, d_model]
        _u, s, vh = torch.linalg.svd(block, full_matrices=False)
        dirs = vh                                # rows in d_model space
    else:
        block = w[:, lo:hi]                      # [d_model, head_dim]
        u, s, _vh = torch.linalg.svd(block, full_matrices=False)
        dirs = u.T
    total = float((s ** 2).sum()) or 1.0
    return [
        {"direction": dirs[i], "singular_value": float(s[i]),
         "energy_fraction": float(s[i] ** 2) / total}
        for i in range(min(components, dirs.shape[0]))
    ]


def readout(
    config: Any,
    snapshot: Path,
    unembedding: torch.Tensor,
    decode: Callable[[Sequence[int]], List[str]],
    *,
    layer: int,
    heads: Optional[Sequence[int]] = None,
    matrices: Optional[Sequence[str]] = None,
    components: int = 1,
    top_k: int = 8,
    model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    if not 1 <= components <= MAX_COMPONENTS:
        raise ValueError(f"components must be 1..{MAX_COMPONENTS}")
    W_U = unembedding.to(torch.float32)
    d_model = int(W_U.shape[1])
    model = model if model is not None else _meta_model(config)
    projections = attention_projections(model, layer, d_model)
    if matrices:
        unknown = sorted(set(matrices) - set(projections))
        if unknown:
            raise ValueError(f"no projection named {unknown}; this layer has {sorted(projections)}")
        projections = {k: v for k, v in projections.items() if k in matrices}

    n_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    head_dim = int(getattr(config, "head_dim", 0) or (d_model // n_heads if n_heads else 0))
    if not head_dim:
        raise ValueError("the config records no head count or head_dim; heads are never guessed")

    out: Dict[str, Any] = {"layer": layer, "head_dim": head_dim, "lens_type": "LOGIT_LENS",
                           "matrices": {}, "evidence_rung": 0}
    for name, proj in sorted(projections.items()):
        weight = read_tensor(snapshot, proj["key"])
        width = proj["out"] if proj["side"] == "read" else proj["in"]
        n_here = width // head_dim
        wanted = list(heads) if heads is not None else list(range(n_here))
        bad = [h for h in wanted if h < 0 or h >= n_here]
        if bad:
            raise ValueError(f"{name} has {n_here} heads; {bad} do not exist")
        rows = []
        for h in wanted:
            for rank, comp in enumerate(head_directions(weight, proj["side"], head_dim, h, components)):
                logits = W_U @ comp["direction"]
                k = min(top_k, logits.numel())
                rows.append({
                    "head": h,
                    "component": rank,
                    "singular_value": comp["singular_value"],
                    "energy_fraction": comp["energy_fraction"],
                    "toward_tokens": decode(torch.topk(logits, k).indices.tolist()),
                    "away_tokens": decode(torch.topk(-logits, k).indices.tolist()),
                })
        out["matrices"][name] = {"side": proj["side"], "checkpoint_key": proj["key"],
                                 "n_heads": n_here, "heads": rows}
    return out
