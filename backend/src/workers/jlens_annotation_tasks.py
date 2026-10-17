"""J-space annotation SWEEP as a bounded, queued job (024 task 4.4, BR-028).

Annotates a contiguous range of one SAE's features through the LOGIT lens, on one
surface, persists each row, and reports the distributional shape check (BR-014) as a
RESULT rather than raising — "most features are workspace" means a mis-scaled
threshold, and the sweep says so instead of failing.

CPU, NOT GPU. The sweep needs W_U, the tokenizer and the SAE's weights — no forward
pass — so it runs on the `processing` queue and never holds a card.

BOUNDED. `max_features` is capped at `MAX_SWEEP_FEATURES` at the route and again here;
a whole dictionary is several sweeps with offsets, each priced in advance by the
`annotation_sweep` cost class.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..core.celery_app import celery_app
from . import jlens_progress
from .task_heartbeat import beat

logger = logging.getLogger(__name__)

#: Rows written per database commit: a crash mid-sweep keeps what was done.
COMMIT_EVERY = 256


@celery_app.task(
    name="src.workers.jlens_annotation_tasks.sweep_annotations",
    bind=True,
    max_retries=0,
)
@jlens_progress.owns_its_failure
def sweep_annotations(
    self,
    sae_id: str,
    surface: str = "decoder",
    offset: int = 0,
    max_features: int = 1024,
    top_k: int = 8,
) -> Dict[str, Any]:
    import torch

    from ..core.database import get_sync_db
    from ..models.external_sae import ExternalSAE
    from ..models.feature import Feature
    from ..models.model import Model
    from ..services.circuit_capture_service import _load_sae_sync
    from ..services.jlens_annotation import (
        FeatureAnnotation,
        WorkspaceClass,
        distribution_is_plausible,
        summarise_distribution,
    )
    from ..services.jlens_annotation_store import (
        MAX_SWEEP_FEATURES,
        SURFACES,
        annotate_rows,
        sae_surfaces,
        unembedding_and_tokenizer,
        upsert_annotation,
    )
    from ..services.jlens_readout_service import IdentityTransport

    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}, not {surface!r}")
    max_features = max(1, min(int(max_features), MAX_SWEEP_FEATURES))

    self.update_state(state="PROGRESS", meta=beat({"stage": "loading_weights"}))
    jlens_progress.update_row(self.request.id, status="running", progress=1.0)

    with get_sync_db() as db:
        sae_row = db.query(ExternalSAE).filter(ExternalSAE.id == sae_id).first()
        if sae_row is None:
            raise ValueError(f"No SAE {sae_id!r}")
        if sae_row.layer is None or not sae_row.model_id:
            raise ValueError(
                f"SAE {sae_id!r} records no layer or model; annotating against a guessed "
                "layer produces a complete, plausible, wrong answer"
            )
        model_row = db.query(Model).filter(Model.id == sae_row.model_id).first()
        if model_row is None:
            raise ValueError(f"No model {sae_row.model_id!r} for SAE {sae_id!r}")
        sae = _load_sae_sync(sae_row, "cpu")
        layer = int(sae_row.layer)

    directions = sae_surfaces(sae)[surface]
    if directions is None:
        raise ValueError(f"SAE {sae_id!r} exposes no {surface} weight to annotate")
    unembedding, tokenizer = unembedding_and_tokenizer(model_row)
    if directions.shape[1] != unembedding.shape[1]:
        raise ValueError(
            f"{surface} width {directions.shape[1]} != model d_model {unembedding.shape[1]}"
        )

    end = min(directions.shape[0], int(offset) + max_features)
    indices = list(range(int(offset), end))
    if not indices:
        raise ValueError(f"offset {offset} is past this SAE's {directions.shape[0]} features")

    with get_sync_db() as db:
        labels = {
            int(n): name
            for n, name in db.query(Feature.neuron_index, Feature.name)
            .filter(Feature.external_sae_id == sae_id)
            .filter(Feature.neuron_index.in_(indices))
            .all()
        }

    written = []
    transport = IdentityTransport()
    with torch.no_grad():
        for start in range(0, len(indices), COMMIT_EVERY):
            chunk = indices[start:start + COMMIT_EVERY]
            rows = annotate_rows(
                directions=directions,
                indices=chunk,
                labels=labels,
                unembedding=unembedding,
                tokenizer=tokenizer,
                transport=transport,
                layer=layer,
                sae_id=sae_id,
                surface=surface,
                top_k=top_k,
                sweep_task_id=self.request.id,
            )
            with get_sync_db() as db:
                for row in rows:
                    upsert_annotation(db, row)
                db.commit()
            written.extend(rows)
            done = len(written)
            self.update_state(
                state="PROGRESS",
                meta=beat({"stage": "annotating", "done": done, "total": len(indices)}),
            )
            jlens_progress.update_row(
                self.request.id, progress=5.0 + 90.0 * done / len(indices)
            )

    summary = summarise_distribution([
        FeatureAnnotation(
            feature_id=str(r["feature_index"]),
            layer=r["layer"],
            lens_kurtosis=r["lens_kurtosis"],
            workspace_class=WorkspaceClass(r["workspace_class"]),
            top_tokens=r["top_tokens"],
        )
        for r in written
    ])
    jlens_progress.update_row(self.request.id, status="completed", progress=100.0)
    return {
        "sae_id": sae_id,
        "surface": surface,
        "lens_type": transport.lens_type,
        "offset": int(offset),
        "annotated": len(written),
        "flagged": sum(1 for r in written if r["has_disagreement"]),
        "next_offset": end if end < directions.shape[0] else None,
        "distribution": summary,
        # BR-014: a RESULT, not an exception. None = cannot say (no band report).
        "distribution_plausible": distribution_is_plausible(summary),
        "evidence_rung": 0,
    }
