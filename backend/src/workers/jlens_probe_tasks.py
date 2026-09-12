"""
Probe mode as a background task (BR-008).

Bound the same way as the readout and for the same measured reason: probing
needs the whole model resident for a forward pass, and loading a real model
takes about a minute — well past nginx's 60s. Sharing the worker also shares
the worker's single-entry model cache, so a probe after a readout on the same
model pays no load at all.

PROBE IS NOT A CHEAPER READOUT. It scores named directions without ranking the
vocabulary, and the two can DISAGREE: the full ranking applies a data-dependent
normalisation (the model's own final norm) that scoring a raw direction does
not. Which mode is canonical must therefore be recorded per analysis, which is
why the response carries `mode` rather than leaving the caller to assume.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..core.celery_app import celery_app
from . import jlens_progress
from .task_heartbeat import beat

logger = logging.getLogger(__name__)


@celery_app.task(
    name="src.workers.jlens_probe_tasks.compute_probe",
    bind=True,
    max_retries=0,
)
@jlens_progress.owns_its_failure
def compute_probe(
    self,
    model_id: str,
    prompt: str,
    tokens: List[str],
    layers: Optional[List[int]] = None,
    artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Score `tokens` at every (layer, position). Returns rows + the mode used.

    `max_retries=0` for the readout's reason: every way this fails fails the
    same way on a retry, and retrying a minute-long load on a shared box makes
    things worse.
    """
    import torch

    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..services.jlens_model_registry import ModelNotAvailable, load_for_readout
    from ..services.jlens_readout_service import (
        IdentityTransport,
        JacobianTransport,
        ReadoutService,
        check_readout_budget,
    )

    with get_sync_db() as db:
        record = db.query(Model).filter(Model.id == model_id).first()
        if record is None:
            raise ValueError(f"No model with id {model_id!r}")

        self.update_state(state="PROGRESS", meta=beat({"stage": "loading_model"}))
        try:
            # None = any resident copy. A fit may have left this model on the GPU;
            # capturing there is free, and the readout maths runs on
            # READOUT_DEVICE either way.
            loaded = load_for_readout(record, capture_device=None)
        except ModelNotAvailable as exc:
            raise ValueError(str(exc)) from exc

    if artifact_id:
        # Same definition of "serviceable" as the readout path — imported
        # rather than restated, so an artifact cannot be serviceable for one
        # mode and not the other.
        from ..api.v1.endpoints.jlens import _jacobian_transport

        transport = _jacobian_transport(loaded, artifact_id)
    else:
        transport = IdentityTransport()

    service = ReadoutService(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        structure=loaded.structure,
        unembedding=loaded.unembedding,
        model_name=loaded.name,
    )

    n_layers = int(loaded.structure.num_layers)
    selected = list(layers) if layers is not None else list(range(n_layers))
    out_of_range = [l for l in selected if l < 0 or l >= n_layers]
    if out_of_range:
        raise ValueError(f"layers {out_of_range} outside range 0..{n_layers - 1}")

    encoded = service.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(service.capture_device)
    ids = [int(i) for i in input_ids[0]]

    # The SAME envelope bound as the readout. A probe is cheaper per cell but
    # captures identical residuals, so an unbounded probe reaches the same
    # 8.4 GB the readout's per-field limits once permitted.
    check_readout_budget(len(ids), len(selected), service.d_model)

    self.update_state(state="PROGRESS", meta=beat({"stage": "probing"}))
    residuals = service._capture_residuals(input_ids, selected)

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for layer in selected:
            h_layer = residuals.by_layer[layer]  # [positions, d_model]
            for position in range(h_layer.shape[0]):
                scores = service.probe(
                    h_layer[position], layer, tokens, transport
                )
                for token, score in scores.items():
                    rows.append(
                        {
                            "layer": layer,
                            "position": position,
                            "token": token,
                            "score": score,
                        }
                    )

    return {
        "scores": rows,
        # Recorded, never inferred: probe and full-ranking scores can disagree,
        # so an analysis that does not say which produced its numbers cannot be
        # compared against one that does (BR-008).
        "mode": "probe",
        "lens_type": transport.lens_type,
        "model": loaded.name,
        "prompt_len": len(ids),
        "hook_target": residuals.hook_target,
    }
