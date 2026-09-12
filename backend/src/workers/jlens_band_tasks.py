"""
Band-report computation as a background task (BR-002, BR-003).

WHY THIS FILE EXISTS. `compute_band_report`, `save_band_report`, `decide_gate`
and `save_gate` were fully implemented and unit-tested with ZERO production
callers — the same shape as the 16 MCP tools this project once shipped
registered with nothing. Every test passed by importing the module directly, so
the suite was green while no user or agent could produce a band report at all,
and the panel's band rendering was permanently unreachable as a result.

BANDS ARE MEASURED HERE OR NOWHERE (BR-002). The published sensory / workspace /
motor boundaries come from one specific model, and there is no default band
constant anywhere in this product by construction. A model without a report
computed against ITS OWN kurtosis profile draws no bands — which is why this
task is the only thing that can ever make bands appear, and why it refuses to
invent them: `derive_boundaries` returns None when the profile does not support
a three-way split, and None is stored and rendered as "no bands" rather than
being softened into a guess.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..core.celery_app import celery_app
from .task_heartbeat import beat
from . import jlens_progress

logger = logging.getLogger(__name__)


@celery_app.task(
    name="src.workers.jlens_band_tasks.compute_band_report",
    bind=True,
    max_retries=0,
)
@jlens_progress.owns_its_failure
def compute_band_report_task(
    self,
    model_id: str,
    prompts: List[str],
    control_seed: int,
    layers: Optional[List[int]] = None,
    use_artifact: bool = True,
) -> Dict[str, Any]:
    """Measure a model's per-layer profile and derive its own boundaries.

    `control_seed` is REQUIRED rather than defaulted, all the way up through
    the API: the autocorrelation null is drawn from it, and a report whose
    control cannot be reproduced is not evidence.

    `use_artifact` supplies the fitted lens dictionary when one exists.
    Effective dimensionality is a property of that dictionary, not of the
    residual stream — for the logit lens the dictionary is the identity, whose
    effective dimensionality is d_model at every layer and says nothing, so it
    is recorded ABSENT rather than as a number that looks like a measurement.

    `max_retries=0` for the readout's reason: every way this fails fails the
    same way on a retry, and it holds a model for minutes on a shared card.
    """
    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..services.jlens_band_report import build_band_report
    from ..services.jlens_band_service import save_band_report
    from ..services.jlens_model_registry import ModelNotAvailable, load_for_readout
    from ..services.jlens_readout_service import ReadoutService

    with get_sync_db() as db:
        record = db.query(Model).filter(Model.id == model_id).first()
        if record is None:
            raise ValueError(f"No model with id {model_id!r}")

        self.update_state(state="PROGRESS", meta=beat({"stage": "loading_model"}))
        try:
            loaded = load_for_readout(record, capture_device=None)
        except ModelNotAvailable as exc:
            raise ValueError(str(exc)) from exc

    from ..api.v1.endpoints.jlens import _service

    service = _service()
    ref = service.find(loaded.name)
    if ref is None:
        raise ValueError(
            f"No J-lens artifact directory for {loaded.name}. The band report is "
            "stored beside the artifact, so one must exist to hold it."
        )

    jacobians = None
    if use_artifact:
        payload = service._load_payload(ref)  # noqa: SLF001 - same package concern
        if isinstance(payload, dict):
            jacobians = {int(k): v for k, v in payload.items()}

    # The layer set is the ARTIFACT's when a lens is being used, because
    # effective dimensionality is undefined at a layer it does not cover — and
    # a profile silently missing that field on some layers reads as a measured
    # absence rather than an unfitted one.
    n_layers = int(loaded.structure.num_layers)
    if layers is not None:
        selected = list(layers)
    elif jacobians:
        selected = sorted(jacobians)
    else:
        selected = list(range(n_layers))

    out_of_range = [l for l in selected if l < 0 or l >= n_layers]
    if out_of_range:
        raise ValueError(f"layers {out_of_range} outside range 0..{n_layers - 1}")

    self.update_state(state="PROGRESS", meta=beat({"stage": "profiling"}))
    readout_service = ReadoutService(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        structure=loaded.structure,
        unembedding=loaded.unembedding,
        model_name=loaded.name,
    )

    from ..services.jlens_band_service import compute_band_report

    def on_prompt(index: int, total: int) -> None:
        jlens_progress.update_row(
            self.request.id, status="running", progress=100.0 * index / max(total, 1)
        )
        # Inside the loop, not at its edges: this stage runs for tens of
        # minutes and a beat only at its start would be indistinguishable from
        # a worker that died at its start.
        self.update_state(
            state="PROGRESS",
            meta=beat({"stage": "profiling", "prompt": index + 1, "of": total}),
        )

    report = compute_band_report(
        readout_service=readout_service,
        on_progress=on_prompt,
        prompts=prompts,
        layers=selected,
        control_seed=control_seed,
        model_id=loaded.name,
        jacobians=jacobians,
    )

    # `compute_band_report` returns a BandReport already; rebuild through
    # `build_band_report` ONLY if it returned raw profiles. Guarded by shape so
    # the two entry points cannot drift into disagreeing about derivation text.
    if not hasattr(report, "boundaries"):
        report = build_band_report(
            model_id=loaded.name,
            profiles=report,
            control_seed=control_seed,
        )

    self.update_state(state="PROGRESS", meta=beat({"stage": "saving"}))
    path = save_band_report(ref.directory, report)
    logger.info("Wrote band report for %s to %s", loaded.name, path)

    jlens_progress.update_row(self.request.id, status="completed", progress=100.0)
    return {
        "model_id": loaded.name,
        "slug": ref.slug,
        "layers": selected,
        "used_artifact": jacobians is not None,
        "control_seed": control_seed,
        # NULL is the honest answer, not a missing value: this model's profile
        # did not support a three-way split, and no boundary from another model
        # may be substituted (BR-002).
        "has_bands": report.has_bands,
        "boundaries": report.boundaries,
        "derivation": report.derivation,
        "path": str(path),
    }


@celery_app.task(
    name="src.workers.jlens_band_tasks.record_gate",
    bind=True,
    max_retries=0,
)
@jlens_progress.owns_its_failure
def record_gate_task(
    self,
    model_id: str,
    claim_set_replicated: bool,
    larger_scale_indicated: bool,
    rationale: str,
    replication_report_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Record the Phase-0 GO / NO-GO decision against a computed band report.

    THE INPUTS ARE FINDINGS, NOT SCORES, and they stay that way through the
    API: `claim_set_replicated` is the question BR-003 actually asks and is
    supplied by the analysis. There is deliberately no numeric criterion — a
    threshold on any single metric would become the definition of the gate, and
    the metric most likely to be reached for is next-token agreement, which
    BR-004 forbids being scored on at all.
    """
    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..services.jlens_band_report import build_band_report, decide_gate
    from ..services.jlens_band_service import load_band_report, save_gate
    from ..ml.jlens_metrics import LayerProfile

    with get_sync_db() as db:
        record = db.query(Model).filter(Model.id == model_id).first()
        if record is None:
            raise ValueError(f"No model with id {model_id!r}")
        repo_id = record.repo_id

    from ..api.v1.endpoints.jlens import _service

    service = _service()
    ref = service.find(repo_id)
    if ref is None:
        raise ValueError(f"No J-lens artifact directory for {repo_id}")

    stored = load_band_report(ref.directory)
    if stored is None:
        # REFUSED, not defaulted. A gate recorded without the report it is
        # supposed to weigh is a decision with no evidence behind it, which is
        # precisely what BR-003 exists to prevent.
        raise ValueError(
            f"No band report for {repo_id}; compute one before recording a gate. "
            "A gate decision without the evidence it weighs is not a record."
        )

    band_report = build_band_report(
        model_id=stored.get("model_id", repo_id),
        profiles=[LayerProfile(**p) for p in stored.get("profiles", [])],
        control_seed=stored.get("control_seed") or 0,
    )

    gate = decide_gate(
        model_id=repo_id,
        band_report=band_report,
        claim_set_replicated=claim_set_replicated,
        larger_scale_indicated=larger_scale_indicated,
        rationale=rationale,
        replication_report_id=replication_report_id,
    )
    path = save_gate(ref.directory, gate)
    logger.info("Recorded gate %s for %s at %s", gate.decision.value, repo_id, path)

    return {
        "model_id": repo_id,
        "slug": ref.slug,
        "decision": gate.decision.value,
        "blocking": gate.is_blocking(),
        "rationale": gate.rationale,
        "has_bands": band_report.has_bands,
        "path": str(path),
    }
