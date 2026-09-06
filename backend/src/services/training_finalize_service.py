"""
Finalize a training from a saved checkpoint.

WHY THIS EXISTS
---------------
The training loop only writes the Community Standard export
(``{data_dir}/trainings/{id}/community_format/``) on its success path
(``training_tasks.py`` finalize block). Cancelling a run skips that block, so a
stopped training leaves behind perfectly good ``checkpoint.safetensors`` files
that NOTHING downstream can consume — ``sae_manager_service`` scans
``community_format/``, and circuit capture / Neuronpedia export / analysis all
load through it. The UI therefore offers "Retry" (which starts a brand-new run
from step 0) instead of "Import to SAEs".

This service closes that gap: it rebuilds the SAE modules from a checkpoint and
runs the SAME community writer the success path uses, so a finalized-early run
is interchangeable with a normally-completed one for every downstream consumer.
(One deliberate difference: sparsity.safetensors is not written, because the
per-feature sparsity statistics only exist inside the training loop.)

It is CPU-only and never touches the GPU — a finalize can run while another
training owns the device.
"""

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from sqlalchemy.exc import SQLAlchemyError

from ..core.config import settings
from ..models.checkpoint import Checkpoint
from ..ml.sparse_autoencoder import create_sae
from ..models.model import Model
from ..models.training import Training
from ..services.checkpoint_service import CheckpointService

logger = logging.getLogger(__name__)

# Directory name written by save_multilayer_checkpoint: layer_{idx}_{hook} (current)
# or layer_{idx} (legacy, single hook type).
_LAYER_DIR_RE = re.compile(r"^layer_(\d+)(?:_(.+))?$")
_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint_(\d+)$")

CHECKPOINT_FILENAME = "checkpoint.safetensors"


class FinalizeError(Exception):
    """Raised when a training cannot be finalized from its checkpoints."""


def _checkpoints_root(training_id: str) -> Path:
    return Path(settings.data_dir) / "trainings" / training_id / "checkpoints"


def community_format_dir(training_id: str) -> Path:
    """Where the Community Standard export lives. Mirrors training_tasks.py."""
    return Path(settings.data_dir) / "trainings" / training_id / "community_format"


def list_checkpoint_steps(training_id: str) -> List[int]:
    """Return every step that has an on-disk checkpoint directory, ascending."""
    root = _checkpoints_root(training_id)
    if not root.is_dir():
        return []
    steps = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_DIR_RE.match(child.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def resolve_layer_dirs(step_dir: Path) -> Dict[Tuple[int, str], Path]:
    """Map (layer_idx, hook_type) -> the checkpoint file inside ``step_dir``.

    Reads what is actually ON DISK rather than trusting hyperparameters: a run
    whose config was edited mid-flight, or a legacy ``layer_{idx}/`` layout,
    still resolves correctly. Legacy directories (no hook suffix) are reported
    with hook_type "residual", matching the training default.
    """
    found: Dict[Tuple[int, str], Path] = {}
    if not step_dir.is_dir():
        return found
    for child in sorted(step_dir.iterdir()):
        if not child.is_dir():
            continue
        match = _LAYER_DIR_RE.match(child.name)
        if not match:
            continue
        ckpt_file = child / CHECKPOINT_FILENAME
        if not ckpt_file.is_file():
            continue
        layer_idx = int(match.group(1))
        hook_type = match.group(2) or "residual"
        found[(layer_idx, hook_type)] = ckpt_file
    return found


def read_dims_from_checkpoint(ckpt_file: Path) -> Tuple[int, int]:
    """Return (hidden_dim, latent_dim) read from the checkpoint's tensor shapes.

    CRITICAL: do NOT take these from ``training.hyperparameters``. The training
    task OVERWRITES ``hp['hidden_dim']`` in memory after peeking at the actual
    activation file (training_tasks.py "HIDDEN_DIM MISMATCH DETECTED") and never
    writes the corrected value back to the database. Rebuilding a model from the
    stored hyperparameters can therefore construct the wrong input width and
    fail ``load_state_dict``. The checkpoint tensors are the ground truth.

    Both supported encoder layouts are [latent_dim, hidden_dim]:
      * JumpReLU/TopK: ``model.W_enc``
      * Standard/Skip/Transcoder: ``model.encoder.weight``
    """
    with safe_open(str(ckpt_file), framework="pt") as f:
        keys = set(f.keys())
        for candidate in ("model.W_enc", "model.encoder.weight"):
            if candidate in keys:
                shape = f.get_slice(candidate).get_shape()
                if len(shape) != 2:
                    raise FinalizeError(
                        f"{candidate} in {ckpt_file} has rank {len(shape)}, expected 2"
                    )
                latent_dim, hidden_dim = int(shape[0]), int(shape[1])
                return hidden_dim, latent_dim
    raise FinalizeError(
        f"Checkpoint {ckpt_file} has neither 'model.W_enc' nor "
        f"'model.encoder.weight'; cannot determine SAE dimensions"
    )


def read_architecture_from_checkpoint(ckpt_file: Path) -> Optional[str]:
    """Read the SAE class name recorded by save_checkpoint, if present."""
    with safe_open(str(ckpt_file), framework="pt") as f:
        metadata = f.metadata() or {}
    return metadata.get("architecture")


# Maps the class name stored in checkpoint metadata back to a create_sae type.
_CLASS_TO_ARCHITECTURE = {
    "JumpReLUSAE": "jumprelu",
    "TopKSAE": "topk",
    "SkipAutoencoder": "skip",
    "Transcoder": "transcoder",
    "SparseAutoencoder": "standard_saelens",
}


def _architecture_type(hp: Dict[str, Any], ckpt_file: Path) -> str:
    """Pick the architecture, preferring what the checkpoint actually contains.

    Hyperparameters can drift from the artifact (see read_dims_from_checkpoint);
    the class name baked into the checkpoint is authoritative when we recognise
    it. Falls back to hyperparameters, then to the training-task default.
    """
    recorded = read_architecture_from_checkpoint(ckpt_file)
    if recorded and recorded in _CLASS_TO_ARCHITECTURE:
        return _CLASS_TO_ARCHITECTURE[recorded]

    architecture_type = hp.get("architecture_type", "standard")
    if architecture_type == "standard":
        # Same backward-compat mapping the training task applies.
        architecture_type = "standard_saelens"
    return architecture_type


def build_model_for_checkpoint(
    hp: Dict[str, Any], ckpt_file: Path
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Construct an SAE module shaped to match ``ckpt_file``, on CPU.

    Returns:
        (model, resolved) where ``resolved`` carries the hidden_dim/latent_dim/
        architecture_type actually taken from the checkpoint. The caller MUST
        write these into the hyperparameters handed to the community writer:
        cfg.json derives ``d_in``/``d_sae``/``architecture`` from that dict, so
        passing the stored (possibly stale) hyperparameters would emit a config
        that contradicts the weights sitting beside it.
    """
    from ..core.framework_defaults import get_framework_defaults

    hidden_dim, latent_dim = read_dims_from_checkpoint(ckpt_file)
    architecture_type = _architecture_type(hp, ckpt_file)
    fw = get_framework_defaults(architecture_type)
    resolved = {
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "architecture_type": architecture_type,
    }

    model = create_sae(
        architecture_type=architecture_type,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        # "or" would swallow a legitimate l1_alpha of 0 (valid for topk/jumprelu).
        l1_alpha=(
            hp["l1_alpha"] if hp.get("l1_alpha") is not None
            else fw.get("default_l1_alpha", 5e-4)
        ),
        ghost_gradient_penalty=hp.get("ghost_gradient_penalty", 0.0),
        normalize_activations=hp.get("normalize_activations", fw["normalize_activations"]),
        top_k_sparsity=hp.get("top_k_sparsity", None),
        top_k=hp.get("top_k"),
        aux_k=hp.get("aux_k"),
        aux_loss_alpha=hp.get("aux_loss_alpha"),
        initial_threshold=hp.get("initial_threshold", 0.5),
        bandwidth=hp.get("bandwidth", 0.01),
        sparsity_coeff=hp.get("sparsity_coeff"),
        normalize_decoder=hp.get("normalize_decoder", fw["normalize_decoder"]),
    )
    return model, resolved


def resolve_finalize_step(training_id: str, checkpoint_step: Optional[int]) -> int:
    """Choose which step to finalize from, validating it exists on disk."""
    steps = list_checkpoint_steps(training_id)
    if not steps:
        raise FinalizeError(
            f"Training {training_id} has no on-disk checkpoints to finalize from"
        )
    if checkpoint_step is None:
        return steps[-1]
    if checkpoint_step not in steps:
        raise FinalizeError(
            f"Training {training_id} has no checkpoint at step {checkpoint_step} "
            f"(available: {steps})"
        )
    return checkpoint_step


def expected_sae_keys(db, training_id: str, step: int, hp: Dict[str, Any]) -> set:
    """The (layer, hook) set this step SHOULD contain, or empty if unknowable.

    Prefers the checkpoint rows' recorded metadata (the training loop writes
    training_layers/hook_types into extra_metadata) and falls back to the run's
    hyperparameters. Returning an empty set means "cannot tell" — the caller
    then skips the completeness check rather than blocking a valid finalize.
    """
    try:
        rows = (
            db.query(Checkpoint)
            .filter(Checkpoint.training_id == training_id, Checkpoint.step == step)
            .all()
        )
    except SQLAlchemyError as e:
        # Fail CLOSED. Returning an empty set makes the caller SKIP the
        # completeness check entirely — disabling the one guard that stops a torn
        # step being exported as a whole run, in response to the single failure
        # mode it cannot distinguish from "no metadata recorded".
        # Also roll back: after a SQLAlchemyError the session is in a failed
        # transaction, so the caller's later db.query(Model) would raise
        # PendingRollbackError only AFTER every file had been written.
        try:
            db.rollback()
        except SQLAlchemyError:
            pass
        raise FinalizeError(
            f"Could not read checkpoint rows for {training_id} step {step}; the "
            f"step cannot be verified as complete: {e}"
        ) from e

    keys = set()
    for row in rows:
        meta = row.extra_metadata or {}
        layer_idx, hook_type = meta.get("layer_idx"), meta.get("hook_type")
        if layer_idx is not None and hook_type:
            keys.add((int(layer_idx), str(hook_type)))
    if keys:
        return keys

    layers = hp.get("training_layers")
    hooks = hp.get("hook_types", hp.get("hook_type"))
    if isinstance(layers, int):
        layers = [layers]
    if isinstance(hooks, str):
        hooks = [hooks]
    if not layers or not hooks:
        return set()
    return {(int(l), str(h)) for l in layers for h in hooks}


def _consistent_resolved_config(
    resolved_configs: Dict[Tuple[int, str], Dict[str, Any]]
) -> Dict[str, Any]:
    """Collapse per-layer resolved configs into the one cfg.json will carry.

    All layers of a run share dims/architecture in practice; if they ever differ
    we log loudly and take the first, because a single cfg.json cannot describe
    two shapes.
    """
    if not resolved_configs:
        return {}
    items = sorted(resolved_configs.items())
    first_key, first = items[0]
    for key, other in items[1:]:
        if other != first:
            logger.warning(
                "finalize: layer %s resolved config %s differs from layer %s %s; "
                "cfg.json will describe the latter",
                key, other, first_key, first,
            )
    return dict(first)


def _prune_stale_layer_dirs(output_dir: Path, keep: List[Tuple[int, str]]) -> None:
    """Delete community_format layer dirs not produced by this finalize."""
    if not output_dir.is_dir():
        return
    wanted = {f"layer_{layer}_{hook}" for layer, hook in keep}
    for child in output_dir.iterdir():
        if not child.is_dir() or not _LAYER_DIR_RE.match(child.name):
            continue
        if child.name in wanted:
            continue
        try:
            shutil.rmtree(child)
            logger.info("finalize: removed stale community_format dir %s", child)
        except OSError as e:
            logger.warning("finalize: could not remove stale dir %s: %s", child, e)


def finalize_from_checkpoint(
    db,
    training_id: str,
    checkpoint_step: Optional[int] = None,
) -> Dict[str, Any]:
    """Write the Community Standard export for ``training_id`` from a checkpoint.

    Args:
        db: SYNC SQLAlchemy session (this runs inside a Celery worker).
        training_id: Training to finalize.
        checkpoint_step: Step to finalize from; defaults to the newest on disk.

    Returns:
        Summary dict: step used, per-SAE output paths, community_format dir.

    Raises:
        FinalizeError: no training, no checkpoints, or unreadable checkpoints.
    """
    training = db.query(Training).filter_by(id=training_id).first()
    if not training:
        raise FinalizeError(f"Training not found: {training_id}")

    step = resolve_finalize_step(training_id, checkpoint_step)
    step_dir = _checkpoints_root(training_id) / f"checkpoint_{step}"
    layer_files = resolve_layer_dirs(step_dir)
    if not layer_files:
        raise FinalizeError(
            f"Checkpoint directory {step_dir} contains no readable layer checkpoints"
        )

    hp: Dict[str, Any] = dict(training.hyperparameters or {})

    # Model name for the community config — same lookup the success path uses.
    model_record = db.query(Model).filter_by(id=training.model_id).first()
    model_name = (model_record.repo_id if model_record else None) or "unknown"

    # A stopped worker can be SIGTERMed midway through writing a step's layer
    # directories, so verify the step is COMPLETE before exporting from it.
    # Silently exporting 1 of 3 layers would present a partial run as a whole one.
    expected = expected_sae_keys(db, training_id, step, hp)
    if expected:
        missing = expected - set(layer_files)
        if missing and checkpoint_step is None:
            # The caller did not pin a step, so fall back to the newest COMPLETE
            # one rather than failing outright. A worker SIGTERMed mid-write (the
            # normal case for stop_and_finalize) leaves exactly this: a newest
            # step with only some layers on disk, and a perfectly good one behind it.
            for candidate in reversed(list_checkpoint_steps(training_id)[:-1]):
                candidate_dir = _checkpoints_root(training_id) / f"checkpoint_{candidate}"
                candidate_files = resolve_layer_dirs(candidate_dir)
                if expected - set(candidate_files):
                    continue
                logger.warning(
                    "finalize: step %s is incomplete (missing %s); falling back to "
                    "the newest complete step %s",
                    step, sorted(missing), candidate,
                )
                step, step_dir, layer_files, missing = (
                    candidate, candidate_dir, candidate_files, set()
                )
                break

        if missing:
            raise FinalizeError(
                f"Checkpoint step {step} for training {training_id} is incomplete: "
                f"missing {sorted(missing)}. No earlier complete step is available "
                f"(steps on disk: {list_checkpoint_steps(training_id)})."
            )

    models: Dict[Tuple[int, str], torch.nn.Module] = {}
    resolved_configs: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for sae_key, ckpt_file in sorted(layer_files.items()):
        layer_idx, hook_type = sae_key
        model, resolved = build_model_for_checkpoint(hp, ckpt_file)
        # load_state_dict is strict, so a shape or key mismatch surfaces here
        # rather than producing a silently wrong export. Translate the raw torch
        # error into a FinalizeError that names the layer.
        try:
            CheckpointService.load_checkpoint(str(ckpt_file), model=model, device="cpu")
        except FinalizeError:
            raise
        except Exception as e:
            raise FinalizeError(
                f"Could not load checkpoint for layer {layer_idx}/{hook_type} "
                f"({ckpt_file}): {e}"
            ) from e
        model.eval()
        models[sae_key] = model
        resolved_configs[sae_key] = resolved
        logger.info(
            "finalize: loaded layer=%s hook=%s from %s", layer_idx, hook_type, ckpt_file
        )

    output_dir = community_format_dir(training_id)
    layer_hook_combinations = sorted(models.keys())

    # cfg.json takes d_in/d_sae/architecture from these hyperparameters, so use
    # the values READ FROM THE CHECKPOINTS. Passing the stored hyperparameters
    # would write a config contradicting the weights next to it (the training
    # task corrects hidden_dim in memory and never persists it).
    export_hp = dict(hp)
    export_hp.update(_consistent_resolved_config(resolved_configs))

    # ATOMICITY: write into a scratch directory and swap it in only once every
    # layer is on disk. Writing in place meant a per-layer failure (empty
    # state_dict, ENOSPC, a too-small file) left community_format/ holding this
    # step's layer 7 beside the PREVIOUS step's layers 14/18 — a chimeric SAE
    # spanning two training steps, which sae_manager_service would happily scan.
    staging_dir = output_dir.parent / f"{output_dir.name}.tmp-{step}"
    shutil.rmtree(staging_dir, ignore_errors=True)

    try:
        written = CheckpointService.save_multilayer_community_checkpoint(
            models=models,
            base_output_dir=str(staging_dir),
            model_name=model_name,
            layer_hook_combinations=layer_hook_combinations,
            hyperparams=export_hp,
            training_id=training_id,
            checkpoint_step=step,
            tied_weights=hp.get("tied_weights", False),
        )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    # Swap the completed export in. The previous one survives untouched until
    # this point, so a failure above leaves the old SAEs intact rather than
    # mixing steps. This also makes a stale-layer prune unnecessary: the new
    # directory only ever contains the layers this step produced.
    previous_dir = output_dir.parent / f"{output_dir.name}.prev-{step}"
    shutil.rmtree(previous_dir, ignore_errors=True)
    try:
        if output_dir.exists():
            output_dir.rename(previous_dir)
        staging_dir.rename(output_dir)
    finally:
        shutil.rmtree(previous_dir, ignore_errors=True)

    # Re-point the returned paths at the final location.
    written = {
        key: str(output_dir / Path(path).relative_to(staging_dir))
        for key, path in written.items()
    }

    logger.info(
        "finalize: wrote %d SAE(s) for training %s from step %s to %s",
        len(written), training_id, step, output_dir,
    )

    return {
        "training_id": training_id,
        "checkpoint_step": step,
        "community_format_dir": str(output_dir),
        "sae_count": len(written),
        "outputs": {f"layer_{k[0]}_{k[1]}": str(v) for k, v in written.items()},
    }
