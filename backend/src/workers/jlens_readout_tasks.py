"""
Readout as a background task.

WHY THIS EXISTS, measured rather than assumed. The readout was bound
synchronously and 502'd at the ingress twice on a real model:

    POST /jlens/readout (gemma-2-2b-it, CPU)  ->  502 after 64.9s
    POST /jlens/readout (retry, 2 layers)     ->  502 after 54.0s

nginx gives up at 60s. The work itself is fine — a J-space readout needs the
whole model resident for its forward pass, and loading gemma-2-2b takes about a
minute on CPU. Raising the proxy timeout would be a bandaid: readout cost is
O(positions x layers x top_n) on top of the load, so no fixed timeout bounds it.

So the readout follows the pattern every other model-bound operation in this
codebase already uses — steering, extraction, calibration all queue and poll.
The API returns a task id immediately; the worker holds the loaded model in its
own single-entry cache across tasks, so the FIRST readout for a model pays the
load and subsequent ones do not.

THE WORKER IS THE RIGHT PLACE TO CACHE, and the API is not. A Celery worker is
a separate process: warming a cache inside the API process cannot help the
worker, and warming one inside the worker cannot help a synchronous API
handler. Putting the readout in the worker is what makes the cache useful at
all — which is why this is a redesign rather than a timeout bump.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..core.celery_app import celery_app
from . import jlens_progress
from .task_heartbeat import beat

logger = logging.getLogger(__name__)


@celery_app.task(
    name="src.workers.jlens_readout_tasks.compute_readout",
    bind=True,
    max_retries=0,
)
@jlens_progress.owns_its_failure
def compute_readout(
    self,
    model_id: str,
    prompt: str,
    types: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    top_n: int = 8,
    artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Produce a wire-format readout. Returns the serialised meta + tokens.

    `max_retries=0`: a readout that failed for a real reason (model missing,
    artifact unvalidated, request too large) fails the same way on a retry, and
    retrying a minute-long model load on a shared box makes things worse.
    """
    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..services.jlens_model_registry import ModelNotAvailable, load_for_readout
    from ..services.jlens_readout_service import (
        IdentityTransport,
        JacobianTransport,
        ReadoutService,
    )
    from ..schemas.jlens import LensMetaMessage, LensTokenMessage

    requested = types or ["LOGIT_LENS"]

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
            # Surfaced as the task's failure message rather than a retry: the
            # model is not in a state a readout can use and waiting will not
            # change that.
            raise ValueError(str(exc)) from exc

    transports = []
    for lens_type in requested:
        if lens_type == "LOGIT_LENS":
            transports.append(IdentityTransport())
            continue
        # Imported here so the API's validation logic stays the single
        # definition of what a serviceable artifact is.
        from ..api.v1.endpoints.jlens import _jacobian_transport

        transports.append(_jacobian_transport(loaded, artifact_id))

    self.update_state(state="PROGRESS", meta=beat({"stage": "reading_out"}))
    service = ReadoutService(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        structure=loaded.structure,
        unembedding=loaded.unembedding,
        model_name=loaded.name,
    )

    meta = None
    tokens = []
    for message in service.stream(prompt, transports, layers=layers, top_n=top_n):
        if isinstance(message, LensMetaMessage):
            meta = message.model_dump()
        elif isinstance(message, LensTokenMessage):
            tokens.append(message.model_dump())

    if meta is None:
        # Never returned as an empty success: an empty readout is
        # indistinguishable from a real one with no content.
        raise ValueError("readout produced no meta message")

    return {"meta": meta, "tokens": tokens}
