"""Where a circuit GPU job runs: one helper shared by every circuit task.

Capture, attribution, validation, faithfulness, calibration and the steered
transcript recorder each load a model. Each asks for ``"auto"``, ``"all"`` or a
GPU UUID when it is submitted; the task calls :func:`place_circuit_job` before
any model load, so the choice is made against the memory free when the job
STARTS, and a named card that cannot be used fails the run with the placement's
own message — never a silent swap to another card, never a retry.

SPLITTING (Multi-GPU Phase 2). A run type passes ``allow_shard=True`` only once
its code runs correctly on a model split across cards: capture, attribution,
validation and faithfulness read the capture; calibration and the recorder
generate through ``steering_core``, which loads with the placement's budget and
puts each layer's decoder on that layer's card. A split also needs a SIZE:
without ``required_mb`` Auto always takes the single most-free card, so
:func:`circuit_required_mb` estimates the model and its SAEs from the database —
from the capture for a pass over it, from :func:`steering_sizing_manifest` for a
job that steers.

The returned ``gpu`` record is what a run writes into its result structure
(report or manifest) for the tables that have no ``gpu_uuid`` column, so a
reader can see which card produced a number. ``request`` is kept too: a
reproduction reuses the request its original was submitted with.

Plan: ``0xcc/plans/Multi-GPU-Plan.md``, Phases 1 and 2.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from ..services.gpu_placement import AUTO, Placement, place_job

logger = logging.getLogger(__name__)

#: fp32 encoder + decoder, the dominant tensors of an SAE as `_load_sae_sync` builds it.
_SAE_BYTES_PER_WEIGHT = 4
_SAE_MATRICES = 2


def gpu_record(gpu_request: Optional[str], placement: Placement) -> Dict[str, Any]:
    """What a run records about its card: the request, the UUID and the name.

    A split also records ``uuids``, every card most free first as the placement
    chose them (a split still fills in CUDA index order). ``uuid`` stays the
    first card, as ``gpu_uuid`` does on the job tables; the key is absent for a
    single-card run so its record is exactly what it was before splitting.
    """
    record = {
        "request": gpu_request or AUTO,
        "uuid": placement.uuid,
        "name": None if placement.card is None else placement.card.name,
    }
    if placement.is_shard:
        record["uuids"] = placement.uuids
    return record


def place_circuit_job(
    gpu_request: Optional[str],
    *,
    kind: str,
    run_id: str,
    allow_shard: bool = False,
    required_mb: Optional[float] = None,
) -> Tuple[Placement, Dict[str, Any]]:
    """Choose the card(s) for one circuit job and make the first current.

    Args:
        gpu_request: ``"auto"``, ``"all"``, a GPU UUID, or None (a row or
            message from before GPUs could be chosen — treated as auto).
        kind: The job kind, for the log line.
        run_id: The run, circuit or manifest id, for the log line.
        allow_shard: The run's code is split-safe; see the module docstring.
        required_mb: The job's estimated need. Only with it can Auto decide a
            model fits no single card and split it.

    Raises:
        GpuPlacementError: no card, or set of cards, can take the job as asked.
    """
    if allow_shard and required_mb is None:
        logger.warning(
            "Circuit %s %s has no size estimate: Auto will choose one card, and a model "
            "larger than any card is refused rather than split", kind, run_id,
        )
    placement = place_job(gpu_request or AUTO, required_mb=required_mb, allow_shard=allow_shard)
    logger.info(
        "Circuit %s %s runs on %s as %s (requested %s)",
        kind, run_id, placement.describe(), placement.device, gpu_request or AUTO,
    )
    return placement, gpu_record(gpu_request, placement)


def release_circuit_job(placement: Optional[Placement], *, kind: str, run_id: str) -> None:
    """Return a steering-core job's GPU memory to the driver, on every card it ran on.

    Calibration (run and reproduce) and the steered transcript recorder load a
    model through ``steering_core`` and used to return without releasing it,
    unlike capture, attribution, validation and faithfulness. They run on the
    shared ``extraction`` worker: the freed tensors' blocks stayed reserved by
    PyTorch's caching allocator on every card of the job, NVML went on counting
    them, and the next job's placement judged those cards fuller than they were
    — choosing another card, splitting a model that fits, or refusing it.

    Called from the task's ``finally``, after the service has returned and its
    references are gone. On a FAILURE the traceback still holds the service's
    frames, and the model in their locals, so those frames are cleared first;
    ``sys.exc_info()`` alone only reads them.

    A CPU placement (no card) releases nothing.
    """
    if placement is None or not placement.all_cards:
        return
    import sys
    import traceback

    from ..services.extraction_service import cleanup_gpu_memory

    pending = sys.exc_info()[1]
    if pending is not None and pending.__traceback__ is not None:
        traceback.clear_frames(pending.__traceback__)
    cleanup_gpu_memory(None, context=f"circuit_{kind}:{run_id}", device=list(placement.all_devices))


def circuit_required_mb(
    db,
    manifest: Optional[Dict[str, Any]],
    *,
    encode_tokens: int = 0,
    backward_tokens: int = 0,
) -> Optional[float]:
    """MB a circuit job needs on the GPUs: model weights, activation headroom, and its SAEs.

    The weights and headroom use the load preflight's own figures
    (``resource_config``), so placement and preflight agree about what fits.
    The SAEs are added because they sit on the GPUs beside the model and a split
    keeps only ``SHARD_RESERVE_MB`` free on each card. A job that steers keeps
    only each decoder there (``manifest["sae_matrices"] == 1``,
    :func:`steering_sizing_manifest`).

    And what the pass does with them (review round 3), the same figures its split
    is mapped with (``circuit_devices.model_load_kwargs``): ``encode_tokens``, one
    encode's codes, the largest layer's, since the layers encode one at a time;
    ``backward_tokens``, the codes an attribution pass keeps beside every hooked
    layer until its backward. Capture's 4,096-token encode on a 65,536-feature
    SAE is 5 GiB, two and a half times the activation headroom.

    None when the model row or its parameter count is unknown: the job is then
    placed without a size, as before splitting existed.
    """
    from ..models.external_sae import ExternalSAE
    from ..models.model import Model
    from ..services.resource_config import _ACTIVATION_HEADROOM_GB, _BYTES_PER_PARAM

    if not isinstance(manifest, dict) or not manifest.get("model_id"):
        return None
    from ..services.base_model_budget import params_for_sizing

    model = db.query(Model).filter(Model.id == manifest["model_id"]).first()
    quantization = getattr(model, "quantization", None)
    # A Q4 row's count is the packed count; its description is not.
    params = params_for_sizing(
        getattr(model, "params_count", None), quantization, getattr(model, "architecture_config", None)
    )
    if params is None:
        return None
    quantization = str(getattr(quantization, "value", quantization)).upper()
    weights_mb = params * _BYTES_PER_PARAM.get(quantization, 2.0) / 2**20

    from ..services.base_model_budget import sae_backward_retained_mb, sae_encode_working_mb

    matrices = manifest.get("sae_matrices", _SAE_MATRICES)
    saes_mb = 0.0
    working_mb = 0.0
    for entry in manifest.get("layers") or []:
        sae = db.query(ExternalSAE).filter(ExternalSAE.id == entry.get("sae_id")).first()
        d_model = getattr(sae, "d_model", None)
        n_features = getattr(sae, "n_features", None)
        if isinstance(d_model, int) and isinstance(n_features, int):
            saes_mb += matrices * d_model * n_features * _SAE_BYTES_PER_WEIGHT / 2**20
            saes_mb += sae_backward_retained_mb(n_features, backward_tokens) if backward_tokens else 0.0
            working_mb = max(working_mb, sae_encode_working_mb(n_features, encode_tokens) if encode_tokens else 0.0)
    return weights_mb + _ACTIVATION_HEADROOM_GB * 1024 + saes_mb + working_mb


def steering_sizing_manifest(model_id, sae_ids) -> Optional[Dict[str, Any]]:
    """What :func:`circuit_required_mb` sizes for a job that steers.

    ``sae_ids`` is one entry per steered layer: ``steering_core`` puts a decoder
    on the GPUs per layer, so an SAE named at two layers is there twice. Only the
    DECODER (``sae_matrices``: 1): the core builds each SAE on the CPU and moves
    its decoder alone (review round 3). None without a model id, which places the
    job without a size.
    """
    if not isinstance(model_id, str) or not model_id:
        return None
    return {"model_id": model_id, "layers": [{"sae_id": sid} for sid in sae_ids if sid], "sae_matrices": 1}


def circuit_steering_manifest(db, circuit_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The model and per-layer SAEs a circuit steers with, as calibration loads them.

    Not the capture: a circuit keeps the layers its members sit on, which can be
    fewer than the capture recorded. None when the circuit cannot be found.
    """
    from ..models.circuit import Circuit

    if not circuit_id:
        return None
    circuit = db.query(Circuit).filter(Circuit.id == circuit_id).first()
    saes = getattr(circuit, "saes", None) or []
    # The same selection `resolve_circuit_members` makes.
    return steering_sizing_manifest(
        getattr(circuit, "model_id", None),
        [s.get("mistudio_sae_id") for s in saes
         if isinstance(s, dict) and s.get("layer") is not None])


def discovery_capture_manifest(db, run_id: str) -> Optional[Dict[str, Any]]:
    """The capture manifest behind a discovery run, or None when it cannot be found."""
    from ..models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun

    run = db.query(CircuitDiscoveryRun).filter(CircuitDiscoveryRun.id == run_id).first()
    capture_id = getattr(run, "capture_run_id", None)
    if not capture_id:
        return None
    capture = db.query(CircuitCaptureRun).filter(CircuitCaptureRun.id == capture_id).first()
    manifest = getattr(capture, "manifest", None)
    return manifest if isinstance(manifest, dict) else None


def circuit_capture_manifest(db, circuit_id: str) -> Optional[Dict[str, Any]]:
    """The capture manifest a circuit was mined from, or None when it cannot be found."""
    from ..models.circuit import Circuit

    circuit = db.query(Circuit).filter(Circuit.id == circuit_id).first()
    discovery_id = getattr(circuit, "discovery_run_id", None)
    if not isinstance(discovery_id, str) or not discovery_id:
        return None
    return discovery_capture_manifest(db, discovery_id)


def circuit_faithfulness_manifest(db, circuit_id: str) -> Optional[Dict[str, Any]]:
    """What :func:`circuit_required_mb` sizes for a faithfulness pass over a circuit.

    The capture manifest's model, with only the layers the pass loads an SAE for:
    the layers the circuit's members expand to (``expand_circuit_members``, the
    pass's own expansion). The capture records every layer it captured, and a
    circuit keeps only those its members sit on, so a size read from the capture
    counted SAEs the pass never loads — and a split keeps only
    ``SHARD_RESERVE_MB`` free on each card, so an over-count can split or refuse
    a job one card holds. None when the capture cannot be found.
    """
    from ..models.circuit import Circuit
    from ..services.circuit_faithfulness_service import expand_circuit_members

    manifest = circuit_capture_manifest(db, circuit_id)
    if manifest is None:
        return None
    circuit = db.query(Circuit).filter(Circuit.id == circuit_id).first()
    layers = set(expand_circuit_members(db, circuit)) if circuit is not None else set()
    return {**manifest,
            "layers": [e for e in manifest.get("layers") or [] if e.get("layer") in layers]}
