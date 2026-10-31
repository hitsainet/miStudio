"""
Probe mode as a background task (BR-008).

Bound the same way as the readout and for the same measured reason: probing
needs the whole model resident for a forward pass, and loading a real model
takes about a minute — well past nginx's 60s.

ON THE CARD THE CALLER CHOSE, released when the probe ends — see
`jlens_readout_tasks` for why a GPU copy is not kept across tasks.

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
from ..services.gpu_placement import AUTO
from . import jlens_progress
from .gpu_job import gpu_job
from .task_heartbeat import beat

logger = logging.getLogger(__name__)


@celery_app.task(
    name="src.workers.jlens_probe_tasks.compute_probe",
    bind=True,
    max_retries=0,
)
@gpu_job("jlens_probe")
@jlens_progress.owns_its_failure
def compute_probe(
    self,
    model_id: str,
    prompt: str,
    tokens: List[str],
    layers: Optional[List[int]] = None,
    artifact_id: Optional[str] = None,
    gpu_request: str = AUTO,
) -> Dict[str, Any]:
    """Score `tokens` at every (layer, position). Returns rows + the mode used.

    `gpu_request` is what the endpoint resolved: "auto", or the UUID of the card
    the caller named. A card that cannot be used fails the task with the reason.

    `max_retries=0` for the readout's reason: every way this fails fails the
    same way on a retry, and retrying a minute-long load on a shared box makes
    things worse.
    """
    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..services.jlens_model_registry import (
        ModelNotAvailable,
        estimate_weights_mb,
        load_for_readout,
    )

    placement = None
    loaded = None
    try:
        with get_sync_db() as db:
            record = db.query(Model).filter(Model.id == model_id).first()
            if record is None:
                raise ValueError(f"No model with id {model_id!r}")

            self.update_state(state="PROGRESS", meta=beat({"stage": "loading_model"}))
            jlens_progress.update_row(self.request.id, status="running", progress=1.0)
            # MAY SPLIT: a probe captures through `ReadoutService`, which copies
            # each residual off its card and sends ids to the embedding's card.
            placement = jlens_progress.place_on_card(
                self.request.id,
                gpu_request,
                required_mb=estimate_weights_mb(record),
                allow_shard=True,
            )
            try:
                loaded = load_for_readout(record, placement=placement)
            except ModelNotAvailable as exc:
                raise ValueError(str(exc)) from exc

        result = _probe(
            self,
            loaded=loaded,
            prompt=prompt,
            tokens=tokens,
            layers=layers,
            artifact_id=artifact_id,
        )
    finally:
        loaded = None  # noqa: F841 - the assignment IS the release
        jlens_progress.release_card(placement)

    jlens_progress.update_row(self.request.id, status="completed", progress=100.0)
    return result


def _probe(
    self,
    loaded,
    prompt: str,
    tokens: List[str],
    layers: Optional[List[int]],
    artifact_id: Optional[str],
) -> Dict[str, Any]:
    """The probe itself, split out so the task can guarantee the release."""
    import torch

    from ..services.jlens_readout_service import (
        IdentityTransport,
        ReadoutService,
        check_readout_budget,
    )

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


# ── sparse decomposition (BR-008 mode c, 022 task 5.3) ───────────────────────


@celery_app.task(
    name="src.workers.jlens_probe_tasks.compute_decomposition",
    bind=True,
    max_retries=0,
)
@gpu_job("jlens_decompose")
@jlens_progress.owns_its_failure
def compute_decomposition(
    self,
    model_id: str,
    prompt: str,
    layers: List[int],
    positions: Optional[List[int]] = None,
    k: int = 8,
    max_iterations: Optional[int] = None,
    control_seed: int = 0,
    artifact_id: Optional[str] = None,
    gpu_request: str = AUTO,
) -> Dict[str, Any]:
    """Decompose the residual at each (layer, position) by gradient pursuit.

    Same lifecycle as the probe — load, place, capture, release — because it needs the
    same thing: the whole model resident for one forward pass. The pursuit itself runs
    on CPU against the readout's unembedding (``READOUT_DEVICE``), off the serving card.
    """
    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..services.jlens_model_registry import (
        ModelNotAvailable,
        estimate_weights_mb,
        load_for_readout,
    )

    placement = None
    loaded = None
    try:
        with get_sync_db() as db:
            record = db.query(Model).filter(Model.id == model_id).first()
            if record is None:
                raise ValueError(f"No model with id {model_id!r}")
            self.update_state(state="PROGRESS", meta=beat({"stage": "loading_model"}))
            jlens_progress.update_row(self.request.id, status="running", progress=1.0)
            placement = jlens_progress.place_on_card(
                self.request.id,
                gpu_request,
                required_mb=estimate_weights_mb(record),
                allow_shard=True,
            )
            try:
                loaded = load_for_readout(record, placement=placement)
            except ModelNotAvailable as exc:
                raise ValueError(str(exc)) from exc

        result = _decompose(
            self,
            loaded=loaded,
            prompt=prompt,
            layers=layers,
            positions=positions,
            k=k,
            max_iterations=max_iterations,
            control_seed=control_seed,
            artifact_id=artifact_id,
        )
    finally:
        loaded = None  # noqa: F841 - the assignment IS the release
        jlens_progress.release_card(placement)

    jlens_progress.update_row(self.request.id, status="completed", progress=100.0)
    return result


def _decompose(
    self,
    loaded,
    prompt: str,
    layers: List[int],
    positions: Optional[List[int]],
    k: int,
    max_iterations: Optional[int],
    control_seed: int,
    artifact_id: Optional[str],
) -> Dict[str, Any]:
    """The decomposition itself, split out so the task can guarantee the release."""
    import torch

    from ..schemas.jlens import DecompositionResult
    from ..services.jlens_decomposition import LensDictionary, decompose
    from ..services.jlens_readout_service import (
        IdentityTransport,
        ReadoutService,
        check_readout_budget,
    )

    if artifact_id:
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
    out_of_range = [l for l in layers if l < 0 or l >= n_layers]
    if out_of_range:
        raise ValueError(f"layers {out_of_range} outside range 0..{n_layers - 1}")
    uncovered = [l for l in layers if l not in transport.covers(layers)]
    if uncovered:
        raise ValueError(f"the lens artifact has no Jacobian for layers {uncovered}")

    encoded = service.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(service.capture_device)
    n_positions = int(input_ids.shape[1])
    wanted = positions if positions else [-1]
    resolved = sorted({p if p >= 0 else n_positions + p for p in wanted})
    bad = [p for p in resolved if p < 0 or p >= n_positions]
    if bad:
        raise ValueError(f"positions {bad} outside range 0..{n_positions - 1}")
    check_readout_budget(n_positions, len(layers), service.d_model)

    self.update_state(state="PROGRESS", meta=beat({"stage": "capturing"}))
    residuals = service._capture_residuals(input_ids, list(layers))

    rows: List[Dict[str, Any]] = []
    total = len(layers) * len(resolved)
    with torch.no_grad():
        for layer in layers:
            dictionary = LensDictionary(service.W_U, transport, layer)
            h_layer = residuals.by_layer[layer]
            for position in resolved:
                row = decompose(
                    h_layer[position].to("cpu"),
                    dictionary,
                    k=k,
                    control_seed=control_seed,
                    decode=lambda i: service._decode_batch([i])[0],
                    position=position,
                    max_iterations=max_iterations,
                )
                # Validated against the contract BEFORE it leaves the worker: a result
                # without its control seed is invalid, not merely undocumented (5.5).
                DecompositionResult.model_validate(row)
                rows.append(row)
                done = len(rows)
                self.update_state(
                    state="PROGRESS",
                    meta=beat({"stage": "decomposing", "cells_done": done, "cells_total": total}),
                )
                jlens_progress.update_row(
                    self.request.id, progress=5.0 + 90.0 * done / max(1, total)
                )

    return {
        "decompositions": rows,
        "mode": "decomposition",
        "lens_type": transport.lens_type,
        "model": loaded.name,
        "prompt_len": n_positions,
        "hook_target": residuals.hook_target,
    }


# ── template lens (BR-023, 027 task 4.3) ─────────────────────────────────────


@celery_app.task(
    name="src.workers.jlens_probe_tasks.compute_template_lens",
    bind=True,
    max_retries=0,
)
@gpu_job("jlens_template_lens")
@jlens_progress.owns_its_failure
def compute_template_lens(
    self,
    model_id: str,
    vocabulary: Dict[str, List[str]],
    baseline_contexts: List[str],
    layers: List[int],
    ridge: float = 0.1,
    filter_words: Optional[List[str]] = None,
    gpu_request: str = AUTO,
) -> Dict[str, Any]:
    """Build template-lens directions for multi-token phrases. FORWARD PASSES ONLY."""
    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..services.jlens_model_registry import (
        ModelNotAvailable,
        estimate_weights_mb,
        load_for_readout,
    )

    placement = None
    loaded = None
    try:
        with get_sync_db() as db:
            record = db.query(Model).filter(Model.id == model_id).first()
            if record is None:
                raise ValueError(f"No model with id {model_id!r}")
            self.update_state(state="PROGRESS", meta=beat({"stage": "loading_model"}))
            jlens_progress.update_row(self.request.id, status="running", progress=1.0)
            placement = jlens_progress.place_on_card(
                self.request.id,
                gpu_request,
                required_mb=estimate_weights_mb(record),
                allow_shard=True,
            )
            try:
                loaded = load_for_readout(record, placement=placement)
            except ModelNotAvailable as exc:
                raise ValueError(str(exc)) from exc
        result = _template_lens(
            self, loaded=loaded, vocabulary=vocabulary, baseline_contexts=baseline_contexts,
            layers=layers, ridge=ridge, filter_words=filter_words,
        )
    finally:
        loaded = None  # noqa: F841 - the assignment IS the release
        jlens_progress.release_card(placement)

    jlens_progress.update_row(self.request.id, status="completed", progress=100.0)
    return result


def _template_lens(self, loaded, vocabulary, baseline_contexts, layers, ridge, filter_words):
    import torch

    from ..services.jlens_readout_service import ReadoutService
    from ..services.jlens_template_lens import LIMITATIONS, build_templates, contexts_digest

    service = ReadoutService(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        structure=loaded.structure,
        unembedding=loaded.unembedding,
        model_name=loaded.name,
    )
    n_layers = int(loaded.structure.num_layers)
    bad = [l for l in layers if l < 0 or l >= n_layers]
    if bad:
        raise ValueError(f"layers {bad} outside range 0..{n_layers - 1}")

    total = sum(len(v) for v in vocabulary.values()) + len(baseline_contexts)
    done = {"n": 0}

    def final_states(texts):
        """[n_texts, d_model] per layer: the residual at each context's LAST position."""
        rows = {L: [] for L in layers}
        for text in texts:
            ids = service.tokenizer(text, return_tensors="pt")["input_ids"].to(service.capture_device)
            captured = service._capture_residuals(ids, list(layers))
            for L in layers:
                rows[L].append(captured.by_layer[L][-1].to("cpu", torch.float32))
            done["n"] += 1
            jlens_progress.update_row(self.request.id, progress=5.0 + 90.0 * done["n"] / max(1, total))
        return {L: torch.stack(v) for L, v in rows.items()}

    self.update_state(state="PROGRESS", meta=beat({"stage": "baseline"}))
    with torch.no_grad():
        baseline = final_states(baseline_contexts)
        self.update_state(state="PROGRESS", meta=beat({"stage": "phrases"}))
        phrase_states = {phrase: final_states(ctxs) for phrase, ctxs in vocabulary.items()}

    templates = build_templates(phrase_states, baseline, ridge=ridge, filter_words=filter_words)
    by_phrase: Dict[str, Dict[str, Any]] = {}
    for t in templates:
        entry = by_phrase.setdefault(t.phrase, {
            # The day-one contract fields (TemplateLensSpec), now COMPUTED.
            "phrase": t.phrase,
            "n_contexts": t.n_contexts,
            "baseline_set": f"{len(baseline_contexts)} contexts, sha256 {contexts_digest(baseline_contexts)[:16]}",
            "whitening": f"(cov + {ridge} x mean-diag x I)^-1, baseline final positions",
            "direction_available": True,
            "contexts_sha256": contexts_digest(vocabulary[t.phrase]),
            "layers": {},
        })
        entry["layers"][str(t.layer)] = {
            "direction": [float(x) for x in t.direction],
            "separation_in_baseline_std": t.separation,
            "raw_norm": t.raw_norm,
        }
    return {
        "templates": list(by_phrase.values()),
        "mode": "template_lens",
        "model": loaded.name,
        "hook_target": "layers_module[L] (resid_post), final position",
        "limitations": list(LIMITATIONS),
        "evidence_rung": 0,
    }
