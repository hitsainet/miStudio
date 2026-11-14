"""Where a circuit job's tensors go when its model may be split across GPUs.

Multi-GPU Phase 2 (``0xcc/plans/Multi-GPU-Plan.md``) loads a model that fits no
single card across several, with the placement's accelerate device map and
per-card budget. The circuit services were
written for one card: every SAE went to the placed device, input ids went to
``model.device``, and hook math converted a decoder column's dtype but not its
device. On a split each of those is a device-mismatch error at the first layer
that landed on another card — or, for input ids, whenever the first registered
parameter is not the embedding.

The rules applied here:

* An SAE lives on its LAYER's device (plan decision D5). Its hook then computes
  where the hidden state already is, and SAE memory is spread over the cards the
  way the layers are, instead of piling onto the first card's 1 GB reserve.
* Tensors an SAE reads or produces are moved to the device of the thing they
  meet — the SAE for an encode, the hidden state for a subtraction — so the math
  stays correct even if an SAE were placed elsewhere.
* Input ids go to the embedding's device (``model_devices.input_device``).

Nothing here chooses a card; ``gpu_placement`` does.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import torch

from ..ml.model_devices import detach_dispatch_hooks, module_device


def model_load_kwargs(
    device: Any,
    placement: Any = None,
    *,
    db: Any = None,
    sae_ids_by_layer: Optional[Mapping[int, Any]] = None,
    encode_tokens: int = 0,
    backward_tokens: int = 0,
) -> Dict[str, Any]:
    """The ``device_map``/``max_memory`` a circuit job's model load takes.

    A split placement loads with ITS OWN ``device_map`` and per-card budget (no
    ``"cpu"`` key — GPUs only), passed through unchanged: which accelerate map
    strategy honours the budgets is the placement's decision, not this
    service's. Anything else loads onto
    ``device`` exactly as a single-card job always has, with no ``max_memory``
    argument at all, so a single-card call is unchanged.

    ``sae_ids_by_layer`` names the SAE the job will put on each layer's card
    (D5). For a split they are sized from ``db`` and passed as
    ``extra_mb_by_layer``, so the split is mapped to hold each SAE beside its
    layer: the budget keeps only ``SHARD_RESERVE_MB`` free on a card, less than a
    12B-class SAE (``ml/split_load.py``).

    WHAT THE PASS DOES WITH THEM, TOO (review round 3). ``encode_tokens``: the
    tokens one encode covers, whose codes are allocated beside a layer while it
    runs, one layer at a time (``working_mb_by_layer``) — for capture's batch of
    4,096 tokens and a 65,536-feature SAE that is 5 GiB, five times the reserve.
    ``backward_tokens``: the tokens whose codes an attribution pass keeps beside
    EVERY hooked layer until its backward, added to each layer's allowance.
    """
    if placement is not None and placement.is_shard:
        kwargs = {"device_map": placement.device_map, "max_memory": placement.max_memory}
        if sae_ids_by_layer and db is not None:
            sized = sae_mb_by_layer(db, sae_ids_by_layer, backward_tokens=backward_tokens)
            if sized:
                kwargs["extra_mb_by_layer"] = sized
            working = sae_working_mb_by_layer(db, sae_ids_by_layer, tokens=encode_tokens)
            if working:
                kwargs["working_mb_by_layer"] = working
        return kwargs
    return {"device_map": device}


def _sae_dims_by_layer(db: Any, sae_ids_by_layer: Mapping[int, Any]) -> Dict[int, tuple]:
    """``{layer: (d_model, n_features)}`` from each layer's ``ExternalSAE`` row; rows without both are left out."""
    from ..models.external_sae import ExternalSAE

    dims: Dict[int, tuple] = {}
    for layer, sae_id in sae_ids_by_layer.items():
        if not sae_id:
            continue
        row = db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
        d_model = getattr(row, "d_model", None)
        n_features = getattr(row, "n_features", None)
        if isinstance(d_model, int) and isinstance(n_features, int) and d_model > 0 and n_features > 0:
            dims[int(layer)] = (d_model, n_features)
    return dims


def sae_mb_by_layer(
    db: Any,
    sae_ids_by_layer: Mapping[int, Any],
    *,
    bytes_per_value: float = 4,
    decoder_only: bool = False,
    backward_tokens: int = 0,
) -> Dict[int, float]:
    """MB of the SAE each layer's card will hold, by layer index, from the ``ExternalSAE`` rows.

    fp32 by default: ``circuit_capture_service._load_sae_sync`` builds every SAE the
    circuit passes and the steering core load at fp32 and never halves it. A layer
    whose row is missing, or records no dimensions, is left out — there is nothing
    to charge, and the job fails on that SAE where it always did.

    ``decoder_only``: the steering core keeps only each decoder on the card.
    ``backward_tokens``: an attribution pass also keeps its codes beside the layer
    until the backward (``base_model_budget.sae_backward_retained_mb``).
    """
    from .base_model_budget import sae_backward_retained_mb, sae_decoder_mb, sae_weights_mb

    held = sae_decoder_mb if decoder_only else sae_weights_mb
    return {
        layer: held(d_model, n_features, bytes_per_value)
        + (sae_backward_retained_mb(n_features, backward_tokens) if backward_tokens else 0.0)
        for layer, (d_model, n_features) in _sae_dims_by_layer(db, sae_ids_by_layer).items()
    }


def sae_working_mb_by_layer(db: Any, sae_ids_by_layer: Mapping[int, Any], *, tokens: int) -> Dict[int, float]:
    """MB an encode of ``tokens`` needs beside each layer while it runs; empty for no tokens."""
    from .base_model_budget import sae_encode_working_mb

    if not tokens:
        return {}
    return {
        layer: sae_encode_working_mb(n_features, tokens)
        for layer, (_d_model, n_features) in _sae_dims_by_layer(db, sae_ids_by_layer).items()
    }


def job_devices(device: Any, placement: Any = None) -> tuple:
    """Every device a job may allocate on: a split's cards, or its one device."""
    if placement is not None:
        return tuple(placement.all_devices)
    return (torch.device(device),)


def layer_device(structure: Any, layer: int, fallback: Any) -> Any:
    """The device decoder layer ``layer`` runs on; ``fallback`` when it holds no tensors."""
    device = module_device(structure.layers_module[layer])
    return fallback if device is None else device


def layer_devices(structure: Any, layers: Iterable[int], fallback: Any) -> Dict[int, Any]:
    return {int(L): layer_device(structure, int(L), fallback) for L in layers}


def to_module(tensor: "torch.Tensor", module: Any, *, dtype: Optional[torch.dtype] = None) -> "torch.Tensor":
    """``tensor`` on ``module``'s device (and ``dtype`` when given); unmoved for a tensorless module."""
    device = module_device(module) if isinstance(module, torch.nn.Module) else None
    if device is None:
        return tensor if dtype is None else tensor.to(dtype=dtype)
    return tensor.to(device=device, dtype=dtype if dtype is not None else tensor.dtype)


def release_model(model: Any) -> None:
    """Drop accelerate's dispatch hooks before a model is freed.

    A split model's ``AlignDevicesHook``s hold references a cleanup expects to
    have dropped, and the cleanup moves the model with ``.cpu()``, which the
    hooks were not written to survive. A single-card model has none.
    """
    if isinstance(model, torch.nn.Module):
        detach_dispatch_hooks(model)
