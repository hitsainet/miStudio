"""Celery task: re-run the post-run evaluation of a completed training.

SAE TRAINING REMEDIATION, ITEM 6. The post-run step inside ``train_sae`` evaluates
every new run; this task evaluates one that already finished — train_6247e768
predates the step — from ``POST /api/v1/trainings/{id}/evaluate``.

A GPU JOB LIKE ANY OTHER. It loads the base model, so it runs under ``@gpu_job``:
dispatched through ``dispatch_gpu_task`` to the card the request names (or
``gpu.auto``), placed against live free memory with ``place_job``, and holding a
lease until it returns. It may split a model no one card holds, as the on-the-fly
training path does: the evaluation feeds each SAE on its own card and returns the
reconstruction to the layer's.

The SAEs come from ``community_format/`` — the export downstream consumers read —
not from a checkpoint, which can predate the final weights by up to a checkpoint
interval.
"""

import logging
from typing import Any, Dict, Optional

from ..core.celery_app import celery_app
from ..core.config import settings
from ..ml.model_loader import load_model_from_hf
from ..models.activation_extraction import ActivationExtraction
from ..models.model import Model
from ..models.training import Training, TrainingStatus
from ..services.base_model_budget import base_model_mb
from ..services.gpu_job_claim import JobHandoff
from ..services.gpu_placement import AUTO, GpuPlacementError, place_job
from ..services.training_evaluation import (
    STATUS_FAILED,
    base_model_loader,
    evaluation_working_mb,
    exported_sae_dir,
    load_exported_sae,
    run_evaluation,
    sources_from_extractions,
    unspliceable_reason,
    vocab_size_of,
    write_evaluation,
)
from .base_task import DatabaseTask
from .gpu_job import gpu_job

logger = logging.getLogger(__name__)

#: Working memory the evaluation needs beside the weights: a batch's logits and
#: two log-probability copies over the vocabulary, plus activations.
EVALUATION_WORKING_MB = 3_072


def _lease_lost() -> Optional[str]:
    """Why this job no longer holds its GPU lease, or None: the re-run's only stop signal."""
    from ..services.gpu_job_claim import lease_lost_reason

    return lease_lost_reason()


def _layer_hook_combinations(hp: Dict[str, Any]) -> list:
    layers = hp.get("training_layers", [0])
    if not isinstance(layers, list):
        layers = [layers]
    hooks = hp.get("hook_types", hp.get("hook_type", ["residual"])) or ["residual"]
    if isinstance(hooks, str):
        hooks = [hooks]
    return [(int(layer), hook) for layer in layers for hook in hooks]


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    # Fully qualified: Celery routes on the TASK NAME.
    name="src.workers.training_evaluation_tasks.evaluate_training",
)
@gpu_job("training_evaluation")
def evaluate_training_task(
    self,
    training_id: str,
    gpu_request: str = AUTO,
    token_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a completed training's exported SAEs on blocks it never read."""
    from .training_tasks import extraction_ids_of

    task_id = getattr(getattr(self, "request", None), "id", None)
    extra = {"task_id": task_id, "gpu_request": gpu_request}

    def fail(reason: str) -> Dict[str, Any]:
        logger.error("Evaluation of %s could not start: %s", training_id, reason)
        write_evaluation(self.get_db, training_id, {
            "version": 1, "status": STATUS_FAILED, "trigger": "rerun", "reason": reason, **extra,
        })
        return {"status": STATUS_FAILED, "reason": reason}

    with self.get_db() as db:
        training = db.query(Training).filter(Training.id == training_id).first()
        if training is None:
            return {"status": "missing"}
        if training.status != TrainingStatus.COMPLETED.value:
            return fail(f"the training is {training.status}, not completed")
        hp = dict(training.hyperparameters or {})
        extraction_ids = extraction_ids_of(training) or []
        extractions = [
            db.query(ActivationExtraction).filter(ActivationExtraction.id == ext_id).first()
            for ext_id in extraction_ids
        ]
        model_row = db.query(Model).filter(Model.id == training.model_id).first()
        model_fields = None if model_row is None else {
            "repo_id": model_row.repo_id,
            "quantization": model_row.quantization,
            "file_path": model_row.file_path,
            "params_count": model_row.params_count,
            "architecture_config": model_row.architecture_config,
        }
        # Plain values: the session closes before the job places.
        extraction_refs = [
            None if row is None else type("ExtractionRef", (), {"id": row.id, "output_path": row.output_path})
            for row in extractions
        ]

    if any(ref is None for ref in extraction_refs):
        return fail("an extraction this training read no longer exists")

    community_dir = settings.data_dir / "trainings" / training_id / "community_format"
    combinations = _layer_hook_combinations(hp)
    sae_dirs = {key: exported_sae_dir(community_dir, *key) for key in combinations}
    missing = [str(path) for path in sae_dirs.values() if not (path / "sae_weights.safetensors").exists()]
    if missing:
        return fail(f"no Community Standard export at {missing}")
    sae_mb = sum((path / "sae_weights.safetensors").stat().st_size for path in sae_dirs.values()) / 1024 ** 2

    base_mb = 0.0
    working_mb = float(EVALUATION_WORKING_MB)
    if model_fields is not None:
        base_mb = base_model_mb(
            model_fields.get("params_count"), model_fields.get("quantization"),
            model_fields.get("architecture_config"),
        ) or 0.0
        # SIZED BY THE VOCABULARY (review R1-C): a batch holds two logit tensors
        # of it. The flat 3,072 MB is the floor, and the whole reservation when the
        # row records no vocabulary.
        working_mb = max(working_mb, evaluation_working_mb(
            vocab_size_of(model_fields.get("architecture_config") or {})
        ) or 0.0)

    # ONLY A RESIDUAL SAE CAN BE SPLICED, so only those are loaded (review R1-C).
    # A transcoder's export does not carry its input centring bias, so loading it
    # strictly refused and the whole re-run was recorded as FAILED where the
    # post-run step records the same training's transcoder as skipped.
    architecture_type = hp.get("architecture_type")
    reasons = {key: unspliceable_reason(architecture_type, key[1]) for key in sae_dirs}
    skipped = [
        {"layer": layer, "hook_type": hook, "reason": reason}
        for (layer, hook), reason in reasons.items() if reason
    ]
    spliceable = {key: path for key, path in sae_dirs.items() if not reasons[key]}
    try:
        placement = place_job(
            gpu_request, required_mb=base_mb + sae_mb + working_mb, allow_shard=True,
        )
    except JobHandoff:
        raise  # the claim re-queues it; not a failure
    except GpuPlacementError as exc:
        return fail(str(exc))

    try:
        saes = {key: load_exported_sae(hp, path).to(placement.device) for key, path in spliceable.items()}
    except Exception as exc:  # noqa: BLE001 - recorded, never silent
        return fail(f"could not load the exported SAEs: {type(exc).__name__}: {exc}")

    document = run_evaluation(
        get_db=self.get_db,
        training_id=training_id,
        hp=hp,
        saes=saes,
        sources=lambda: (
            sources_from_extractions(extraction_refs, settings.resolve_data_path, hp.get("dataset_weights"))
            if extraction_refs else []
        ),
        load_base_model=base_model_loader(
            model_fields=model_fields,
            placement=placement,
            sae_mb=sae_mb,
            loader=load_model_from_hf,
            resolve_path=settings.resolve_data_path,
        ),
        trigger="rerun",
        token_budget=token_budget,
        extra={**extra, "placement": placement.describe()},
        skipped_saes=skipped,
        # A re-run has no Stop of its own; a lost GPU lease stops it between batches.
        should_stop=_lease_lost,
    )
    return {"status": document.get("status"), "reason": document.get("reason")}
