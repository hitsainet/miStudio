"""
SAE Manager service layer.

This module contains business logic for SAE management operations,
including listing, creating, importing from training, and deleting SAEs.
"""

import json
import logging
import re
import shutil
from typing import Optional, Tuple, List, Dict, Any, Mapping, NamedTuple
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..models.external_sae import ExternalSAE, SAESource, SAEStatus, SAEFormat
from ..models.training import Training, TrainingStatus
from ..models.dataset import Dataset
from ..schemas.sae import (
    SAEDownloadRequest,
    SAEImportFromTrainingRequest,
    SAEImportFromFileRequest,
    SAEResponse,
    AvailableSAEInfo,
    ImportedSAEInfo,
    TrainingAvailableSAEsResponse,
    SAEImportFromTrainingResponse,
)
from .huggingface_sae_service import HuggingFaceSAEService
from ..core.clock import utc_now

logger = logging.getLogger(__name__)


# ============================================================================
# The hook an SAE was trained at, recorded where its row is written (review R2, B5)
# ============================================================================
#
# Feature extraction, the Neuronpedia export and the local push refuse an MLP or
# attention SAE by reading ``external_saes.hook_type``, and a NULL there reads as
# residual. Only SAEs imported from a miStudio training ever recorded a hook, so every
# HuggingFace download and local import passed as residual -- a Gemma Scope MLP SAE
# included -- and its features described the wrong activations without a word.
#
# Sources, in order of authority. Nothing is guessed:
#   1. the SAE's own cfg.json: ``hook_name`` (SAELens), ``hook_point`` (older SAELens),
#      or either under ``metadata`` (SAELens 6), stored exactly as written;
#   2. a Gemma Scope repository or directory name ending in -res, -mlp or -att
#      (optionally -canonical), stored in miStudio's hook vocabulary;
#   3. nothing, which stays NULL.
# The loaders in ml/community_format.py invent ``blocks.L.hook_resid_post`` for Gemma
# Scope and cfg-less files; that is a default, not a record, and is never consulted here.

HOOK_SOURCE_CFG = "cfg.json"
HOOK_SOURCE_GEMMA_SCOPE_NAME = "gemma_scope_name"
HOOK_SOURCE_SPARSIFY_DIRECTORY = "sparsify_hookpoint"

# REVIEW R3-B. Real configs fetched from HuggingFace showed that reading cfg.json alone
# recorded NULL -- which reads as residual -- for whole published families:
#   * Gemma Scope 2 (google/gemma-scope-2-*): config.json with ``type`` (sae / transcoder /
#     clt / crosscoder) and ``hf_hook_point_out`` (model.layers.L.output,
#     ...post_feedforward_layernorm.output, ...self_attn.o_proj.input); the repository id
#     names no kind, the folder does (resid_post/, mlp_out/, attn_out/, transcoder/, clt/,
#     crosscoder/, each optionally _all);
#   * Llama Scope (fnlp/*): hyperparams.json ``hook_point_in`` / ``hook_point_out``;
#   * Qwen Scope: config.json ``hook_point``; dictionary_learning (andyrdt, canrager):
#     config.json ``trainer.submodule_name`` (resid_post_layer_3);
#   * EleutherAI sparsify: cfg.json with no hook at all -- the hookpoint IS the directory
#     name (layers.10, layers.0.mlp, layers.0.attention, embed_tokens);
#   * Gemma Scope 1 transcoders (-transcoders) and embedding SAEs (the -res sets'
#     embedding/ folders, which the name rule used to record as ``residual``).
# A transcoder reads one point and writes another, so it is recorded as ``transcoder``,
# never by either name. SAELens 5 writes ``hook_layer`` and SAELens 6 no layer at all, so
# the layer is resolved here too (``resolve_sae_layer``); a NULL layer turns off every
# consumer's own-layer check and feature extraction hooks layer 0.

#: The config files an SAE's hook can be read from, in order of authority.
SAE_CONFIG_FILENAMES = ("cfg.json", "config.json", "hyperparams.json")

_GEMMA_SCOPE_KIND = re.compile(
    r"(?:^|/|--)gemma-scope-[a-z0-9.\-]*?-(res|mlp|att|transcoders)(?:-canonical)?(?:/|$)"
)
_GEMMA_SCOPE_KIND_TO_HOOK = {"res": "residual", "mlp": "mlp", "att": "attention", "transcoders": "transcoder"}
_GEMMA_SCOPE_2_KIND = re.compile(
    r"(?:^|/|--)gemma-scope-2-[a-z0-9.\-]+/(?:.*/)?(resid_post|mlp_out|attn_out|transcoder|clt|crosscoder)(?:_all)?(?:/|$)"
)
_GEMMA_SCOPE_2_KIND_TO_HOOK = {
    "resid_post": "residual", "mlp_out": "mlp", "attn_out": "attention",
    "transcoder": "transcoder", "clt": "transcoder", "crosscoder": "crosscoder",
}
_SPARSIFY_HOOKPOINT = re.compile(r"^(?:[A-Za-z_]\w*\.)*(?:layers|h)\.\d+(?:\.[A-Za-z_]\w*)*$|^embed_tokens$")
_LAYER_IN_HOOK_NAME = re.compile(r"(?:^|[._])(?:blocks|layers|h|layer)[._](\d+)(?:[._]|$)")
_LAYER_IN_PATH = re.compile(r"(?:^|/|_)layer_(\d+)(?:/|_|$)|(?:^|/)layer(\d+)\.")


class RecordedHook(NamedTuple):
    """The hook to store on an ExternalSAE row and where it came from (both None when unknown)."""

    hook_type: Optional[str]
    source: Optional[str]


def _text(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def hook_name_from_sae_config(cfg: Mapping[str, Any]) -> Optional[str]:
    """The hook an SAE config names, exactly as written, or None when it names none.

    A transcoder (a config ``type`` of transcoder or clt, sparsify's ``transcode``, or an
    input point that differs from the output point) is ``transcoder``; a Gemma Scope 2
    crosscoder is ``crosscoder``.
    """
    # miStudio's own Neuronpedia export writes a hook name because SAELens and Neuronpedia
    # both require one, and says HERE whether the SAE actually recorded it. Without this,
    # re-importing an export turned an unrecorded hook into a claimed resid_post -- a NULL
    # laundered into a fact by a round trip through our own artifact (review R3-B, R3B-13).
    if cfg.get("hook_point_recorded") is False:
        return None
    metadata = cfg.get("metadata") if isinstance(cfg.get("metadata"), Mapping) else {}
    trainer = cfg.get("trainer") if isinstance(cfg.get("trainer"), Mapping) else {}
    kind = (_text(cfg.get("type")) or "").lower()
    pairs = [
        (cfg.get("hook_point_in"), cfg.get("hook_point_out")),
        (cfg.get("hf_hook_point_in"), cfg.get("hf_hook_point_out")),
        (cfg.get("hook_name"), cfg.get("hook_name_out")),
        (metadata.get("hook_name"), metadata.get("hook_name_out")),
    ]
    if kind in ("transcoder", "clt") or cfg.get("transcode") is True or any(
        _text(a) and _text(b) and _text(a) != _text(b) for a, b in pairs
    ):
        return "transcoder"
    if kind == "crosscoder":
        return "crosscoder"
    candidates = [
        cfg.get("hook_name"), cfg.get("hook_point"), metadata.get("hook_name"), metadata.get("hook_point"),
        cfg.get("hook_point_out"), cfg.get("hf_hook_point_out"), trainer.get("submodule_name"),
    ]
    for candidate in candidates:
        if _text(candidate):
            return _text(candidate)
    return None


def _config_directory(location: Optional[Path]) -> Optional[Path]:
    if location is None:
        return None
    location = Path(location)
    return location if location.is_dir() else location.parent


def _sae_configs(location: Optional[Path]) -> List[Tuple[str, Mapping[str, Any]]]:
    """Every readable config (file name, contents) in ``location`` or beside it, in authority order."""
    directory = _config_directory(location)
    found: List[Tuple[str, Mapping[str, Any]]] = []
    if directory is None:
        return found
    for filename in SAE_CONFIG_FILENAMES:
        path = directory / filename
        if not path.is_file():
            continue
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read the SAE config at {path}: {e}")
            continue
        if isinstance(cfg, Mapping):
            found.append((filename, cfg))
    return found


def _read_config_hook(location: Optional[Path]) -> Tuple[Optional[str], Optional[str]]:
    for filename, cfg in _sae_configs(location):
        hook = hook_name_from_sae_config(cfg)
        if hook:
            return hook, filename
    return None, None


def read_sae_config_hook(location: Optional[Path]) -> Optional[str]:
    """The hook named by a config (cfg.json, config.json, hyperparams.json) in ``location`` or beside it."""
    return _read_config_hook(location)[0]


def sparsify_hookpoint(location: Optional[Path]) -> Optional[str]:
    """The hookpoint a sparsify SAE's directory is named for, when its cfg.json names no hook.

    sparsify saves each SAE as ``<hookpoint>/cfg.json`` + ``sae.safetensors`` and writes no
    hook into the config, so the directory name is the only record (``layers.0.mlp``).
    """
    directory = _config_directory(location)
    if directory is None or not _SPARSIFY_HOOKPOINT.match(directory.name):
        return None
    for filename, cfg in _sae_configs(location):
        if filename == "cfg.json" and "d_in" in cfg and "k" in cfg and (
            "expansion_factor" in cfg or "num_latents" in cfg
        ):
            return directory.name
    return None


def gemma_scope_hook_kind(origin: Optional[str]) -> Optional[str]:
    """The kind a Gemma Scope set's name gives: residual, mlp, attention, transcoder, crosscoder
    or embedding; None when ``origin`` names no Gemma Scope set.

    Gemma Scope 1 names the kind in the repository (``-res``, ``-mlp``, ``-att``,
    ``-transcoders``); its -res sets also hold ``embedding/`` SAEs. Gemma Scope 2 names it in
    the folder under ``gemma-scope-2-*`` (``mlp_out/``). ``--`` admits a HuggingFace cache path.
    """
    if not origin:
        return None
    text = str(origin).replace("\\", "/").lower()
    match = _GEMMA_SCOPE_2_KIND.search(text)
    if match:
        return _GEMMA_SCOPE_2_KIND_TO_HOOK[match.group(1)]
    match = _GEMMA_SCOPE_KIND.search(text)
    if not match:
        return None
    if match.group(1) == "res" and re.search(r"(?:^|/)embedding(?:/|$)", text[match.end() - 1:]):
        return "embedding"
    return _GEMMA_SCOPE_KIND_TO_HOOK[match.group(1)]


def resolve_sae_hook(location: Optional[Path], *origins: Optional[str]) -> RecordedHook:
    """The hook to record for an SAE at ``location`` known by the names in ``origins``.

    In order: a config's own hook; a sparsify hookpoint directory; a Gemma Scope set's name
    (the origins are read together, since Gemma Scope 2 needs the repository AND the folder).
    """
    hook, filename = _read_config_hook(location)
    if hook:
        return RecordedHook(hook, filename)
    hookpoint = sparsify_hookpoint(location)
    if hookpoint:
        return RecordedHook(hookpoint, HOOK_SOURCE_SPARSIFY_DIRECTORY)
    kind = gemma_scope_hook_kind("/".join(str(origin) for origin in origins if origin))
    if kind:
        return RecordedHook(kind, HOOK_SOURCE_GEMMA_SCOPE_NAME)
    return RecordedHook(None, None)


def _integer(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def resolve_sae_layer(location: Optional[Path], hook_type: Optional[str], *origins: Optional[str]) -> Optional[int]:
    """The decoder layer an SAE reads, or None when nothing records it.

    In order: the layer in the recorded hook's own name (``blocks.25.hook_resid_post``,
    ``model.layers.17.output``, ``layers.10``, ``resid_post_layer_3`` -- the point the
    recorded hook names, so the layer and hook_type cannot disagree); a config's integer
    (SAELens ``hook_point_layer`` / ``hook_layer`` / ``metadata.hook_layer``,
    dictionary_learning ``trainer.layer``); a ``layer_N`` folder in the origins (Gemma Scope).
    A config integer that disagrees with the hook name is logged. A crosscoder reads several
    layers and has no single one (its folder, ``layer_5_9_12_15_width_1m``, would say 5).
    """
    if hook_type == "crosscoder":
        return None
    from_name = None
    match = _LAYER_IN_HOOK_NAME.search(hook_type) if hook_type else None
    if match:
        from_name = int(match.group(1))
    from_config = None
    for _, cfg in _sae_configs(location):
        metadata = cfg.get("metadata") if isinstance(cfg.get("metadata"), Mapping) else {}
        trainer = cfg.get("trainer") if isinstance(cfg.get("trainer"), Mapping) else {}
        for value in (cfg.get("hook_point_layer"), cfg.get("hook_layer"), metadata.get("hook_layer"), trainer.get("layer")):
            if _integer(value) is not None:
                from_config = _integer(value)
                break
        if from_config is not None:
            break
    if from_name is not None:
        if from_config is not None and from_config != from_name:
            logger.warning(
                f"SAE config at {location} gives layer {from_config} but its hook {hook_type!r} names "
                f"layer {from_name}; recording {from_name}"
            )
        return from_name
    if from_config is not None:
        return from_config
    for origin in origins:
        match = _LAYER_IN_PATH.search(str(origin).replace("\\", "/")) if origin else None
        if match:
            return int(match.group(1) or match.group(2))
    return None


class SAEManagerService:
    """Service class for SAE management operations."""

    @staticmethod
    def generate_sae_id() -> str:
        """
        Generate a unique SAE ID.

        Returns:
            SAE ID in format sae_{uuid_hex[:12]}
        """
        return f"sae_{uuid4().hex[:12]}"

    @staticmethod
    async def get_sae(db: AsyncSession, sae_id: str) -> Optional[ExternalSAE]:
        """
        Get an SAE by ID.

        Args:
            db: Database session
            sae_id: SAE ID

        Returns:
            ExternalSAE if found, None otherwise
        """
        result = await db.execute(
            select(ExternalSAE).where(ExternalSAE.id == sae_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_saes(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        source: Optional[SAESource] = None,
        status: Optional[SAEStatus] = None,
        model_name: Optional[str] = None,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> Tuple[List[ExternalSAE], int]:
        """
        List SAEs with filtering, pagination, and sorting.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            search: Search query for name or description
            source: Filter by source type
            status: Filter by status
            model_name: Filter by model name
            sort_by: Column to sort by
            order: Sort order (asc or desc)

        Returns:
            Tuple of (list of SAEs, total count)
        """
        # Build base query
        query = select(ExternalSAE).where(ExternalSAE.status != SAEStatus.DELETED.value)

        # Apply filters
        if search:
            search_filter = or_(
                ExternalSAE.name.ilike(f"%{search}%"),
                ExternalSAE.description.ilike(f"%{search}%"),
                ExternalSAE.hf_repo_id.ilike(f"%{search}%")
            )
            query = query.where(search_filter)

        if source:
            query = query.where(ExternalSAE.source == source.value)

        if status:
            query = query.where(ExternalSAE.status == status.value)

        if model_name:
            query = query.where(ExternalSAE.model_name.ilike(f"%{model_name}%"))

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar_one()

        # Apply sorting
        sort_column = getattr(ExternalSAE, sort_by, ExternalSAE.created_at)
        if order.lower() == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Apply pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await db.execute(query)
        saes = list(result.scalars().all())

        return saes, total

    @staticmethod
    async def initiate_download(
        db: AsyncSession,
        request: SAEDownloadRequest
    ) -> ExternalSAE:
        """
        Initiate an SAE download from HuggingFace.

        Creates an SAE database record in PENDING state.
        The actual download should be handled by a Celery task.

        Args:
            db: Database session
            request: Download request

        Returns:
            Created ExternalSAE in PENDING state
        """
        sae_id = SAEManagerService.generate_sae_id()

        # Generate name if not provided
        name = request.name
        if not name:
            name = f"{request.repo_id.split('/')[-1]}/{request.filepath}"

        # Create local storage path
        local_path = HuggingFaceSAEService.get_sae_storage_path(sae_id)

        # Nothing is downloaded yet, so only the repository's name can say (review R2, B5).
        # The download task records the hook again once the SAE's own config is on disk.
        recorded = resolve_sae_hook(None, request.repo_id, request.filepath)

        # Create SAE record
        db_sae = ExternalSAE(
            id=sae_id,
            name=name,
            description=request.description,
            source=SAESource.HUGGINGFACE.value,
            status=SAEStatus.PENDING.value,
            hf_repo_id=request.repo_id,
            hf_filepath=request.filepath,
            hf_revision=request.revision,
            model_name=request.model_name,
            model_id=request.model_id,  # Link to local model for steering
            hook_type=recorded.hook_type,
            format=SAEFormat.COMMUNITY_STANDARD.value,
            local_path=str(local_path),
            progress=0.0,
            sae_metadata={"hook_source": recorded.source}
        )

        db.add(db_sae)
        await db.commit()
        await db.refresh(db_sae)

        logger.info(f"Initiated SAE download {sae_id} from {request.repo_id}/{request.filepath}")

        return db_sae

    @staticmethod
    async def get_available_saes_from_training(
        db: AsyncSession,
        training_id: str
    ) -> TrainingAvailableSAEsResponse:
        """
        Get list of available SAEs from a completed training.

        Scans the community_format directory for layer subdirectories
        and extracts layer/hook_type information. Filters out SAEs that
        have already been imported to the SAE repository.

        Directory naming conventions:
        - Single hook: layer_{idx}/
        - Multi-hook: layer_{idx}_{hook_type}/

        Args:
            db: Database session
            training_id: Training job ID

        Returns:
            TrainingAvailableSAEsResponse with list of available SAEs (excluding already imported)
        """
        # Get the training job
        training_result = await db.execute(
            select(Training).where(Training.id == training_id)
        )
        training = training_result.scalar_one_or_none()

        if not training:
            raise ValueError(f"Training job not found: {training_id}")

        if training.status != TrainingStatus.COMPLETED.value:
            raise ValueError(f"Training job is not completed: {training.status}")

        # Get already-imported SAEs for this training
        # These will be shown as greyed out (unselectable) in the UI
        imported_result = await db.execute(
            select(ExternalSAE)
            .where(ExternalSAE.training_id == training_id)
            .where(ExternalSAE.status != SAEStatus.DELETED.value)
        )
        imported_sae_set = set()  # For quick lookup
        imported_sae_list: List[ImportedSAEInfo] = []  # For returning to frontend
        for sae in imported_result.scalars():
            # Create a key for already-imported SAEs
            imported_sae_set.add((sae.layer, sae.hook_type))
            # Add to the list for display
            imported_sae_list.append(ImportedSAEInfo(
                layer=sae.layer,
                hook_type=sae.hook_type,
                sae_id=str(sae.id),
                sae_name=sae.name,
                imported_at=sae.created_at.isoformat() if sae.created_at else None
            ))

        # Check for Community Standard format
        training_base_dir = settings.data_dir / "trainings" / training_id
        community_format_dir = training_base_dir / "community_format"

        available_saes: List[AvailableSAEInfo] = []

        if community_format_dir.exists():
            # Scan for layer directories
            for item in community_format_dir.iterdir():
                if not item.is_dir():
                    continue

                dir_name = item.name

                # Parse directory name: layer_{idx} or layer_{idx}_{hook_type}
                if not dir_name.startswith("layer_"):
                    continue

                parts = dir_name.split("_", 2)  # ["layer", "{idx}", "{hook_type}"] or ["layer", "{idx}"]

                if len(parts) < 2:
                    continue

                try:
                    layer_idx = int(parts[1])
                except ValueError:
                    continue

                # Determine hook_type from directory name or cfg.json
                if len(parts) > 2:
                    # Multi-hook directory naming: layer_{idx}_{hook_type}
                    hook_type = parts[2]
                else:
                    # Legacy single-hook naming: layer_{idx}
                    # Try to read hook_type from cfg.json if available
                    hook_type = "residual"  # Default fallback (matches training config naming)
                    cfg_path = item / "cfg.json"
                    if cfg_path.exists():
                        try:
                            with open(cfg_path, "r") as f:
                                cfg = json.load(f)
                            hook_point = cfg.get("hook_point", "")
                            # Parse hook_point like "blocks.13.hook_resid_post" to training config names
                            # Training config uses: "residual", "mlp", "attention"
                            if "resid" in hook_point:  # hook_resid_pre or hook_resid_post
                                hook_type = "residual"
                            elif "mlp" in hook_point:  # hook_mlp_out
                                hook_type = "mlp"
                            elif "attn" in hook_point:  # hook_attn_out
                                hook_type = "attention"
                            logger.debug(f"Inferred hook_type '{hook_type}' from cfg.json hook_point: {hook_point}")
                        except Exception as e:
                            logger.warning(f"Failed to read cfg.json for {item}: {e}")

                # Skip if this SAE has already been imported
                if (layer_idx, hook_type) in imported_sae_set:
                    logger.debug(f"Skipping already-imported SAE: layer {layer_idx}, hook_type {hook_type}")
                    continue

                # Calculate size
                total_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())

                available_saes.append(AvailableSAEInfo(
                    layer=layer_idx,
                    hook_type=hook_type,
                    path=str(item.relative_to(training_base_dir)),
                    size_bytes=total_size
                ))

        # Sort by layer, then hook_type
        available_saes.sort(key=lambda x: (x.layer, x.hook_type))
        imported_sae_list.sort(key=lambda x: (x.layer, x.hook_type))

        return TrainingAvailableSAEsResponse(
            training_id=training_id,
            available_saes=available_saes,
            imported_saes=imported_sae_list,
            total_count=len(available_saes) + len(imported_sae_list)
        )

    @staticmethod
    async def _import_single_sae(
        db: AsyncSession,
        training: Training,
        source_dir: Path,
        layer: int,
        hook_type: str,
        name_prefix: Optional[str],
        description: Optional[str],
    ) -> ExternalSAE:
        """
        Import a single SAE from a source directory.

        Args:
            db: Database session
            training: Training record
            source_dir: Directory containing SAE files
            layer: Layer index
            hook_type: Hook type string
            name_prefix: Optional name prefix
            description: Optional description

        Returns:
            Created ExternalSAE record
        """
        hyperparams = training.hyperparameters or {}

        sae_id = SAEManagerService.generate_sae_id()

        # Generate name with layer/hook suffix
        if name_prefix:
            name = f"{name_prefix} (L{layer}-{hook_type})"
        else:
            name = f"SAE from {training.id} (L{layer}-{hook_type})"

        # Copy checkpoint to SAE storage
        local_path = HuggingFaceSAEService.get_sae_storage_path(sae_id)
        local_path.mkdir(parents=True, exist_ok=True)

        # Copy the files
        for item in source_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, local_path / item.name)
            elif item.is_dir():
                shutil.copytree(item, local_path / item.name)

        # Calculate total size
        total_size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())

        # Get dataset names from dataset_ids
        dataset_names = []
        dataset_ids = training.dataset_ids or []
        if dataset_ids:
            dataset_result = await db.execute(
                select(Dataset.name).where(Dataset.id.in_(dataset_ids))
            )
            dataset_names = [row[0] for row in dataset_result.fetchall()]

        # Build metadata with training info
        sae_metadata = {
            "training_hyperparameters": hyperparams,
            "training_status": training.status,
            "final_loss": training.current_loss,
            "final_l0_sparsity": training.current_l0_sparsity,
            "format_source": "community_format",
            # New fields for SAE tile display
            "training_completed_at": training.completed_at.isoformat() if training.completed_at else None,
            "training_total_steps": training.total_steps,
            "training_steps_completed": training.current_step,
            "training_dataset_names": dataset_names,
        }

        # Create SAE record
        db_sae = ExternalSAE(
            id=sae_id,
            name=name,
            description=description,
            source=SAESource.TRAINED.value,
            status=SAEStatus.READY.value,
            training_id=training.id,
            model_id=training.model_id,
            model_name=None,
            layer=layer,
            hook_type=hook_type,
            n_features=hyperparams.get("latent_dim"),
            d_model=hyperparams.get("hidden_dim"),
            architecture=hyperparams.get("architecture_type", "standard"),
            format=SAEFormat.COMMUNITY_STANDARD.value,
            local_path=str(local_path),
            file_size_bytes=total_size,
            progress=100.0,
            sae_metadata=sae_metadata,
            downloaded_at=utc_now()
        )

        db.add(db_sae)

        logger.info(f"Imported SAE {sae_id} from training {training.id} (L{layer}-{hook_type})")

        return db_sae

    @staticmethod
    async def import_from_training(
        db: AsyncSession,
        request: SAEImportFromTrainingRequest
    ) -> SAEImportFromTrainingResponse:
        """
        Import SAE(s) from a completed training job.

        Supports importing multiple SAEs from multi-layer/multi-hook trainings.
        Uses Community Standard format if available.

        Args:
            db: Database session
            request: Import request with training_id and optional filters

        Returns:
            SAEImportFromTrainingResponse with list of created SAEs
        """
        # Get the training job
        training_result = await db.execute(
            select(Training).where(Training.id == request.training_id)
        )
        training = training_result.scalar_one_or_none()

        if not training:
            raise ValueError(f"Training job not found: {request.training_id}")

        if training.status != TrainingStatus.COMPLETED.value:
            raise ValueError(f"Training job is not completed: {training.status}")

        # Get available SAEs
        available_response = await SAEManagerService.get_available_saes_from_training(
            db, request.training_id
        )
        available_saes = available_response.available_saes

        if not available_saes:
            raise ValueError("No SAEs found in training checkpoint")

        # Filter if not importing all
        if not request.import_all:
            filtered = []
            for sae_info in available_saes:
                # Check layer filter
                if request.layers and sae_info.layer not in request.layers:
                    continue
                # Check hook_type filter
                if request.hook_types and sae_info.hook_type not in request.hook_types:
                    continue
                filtered.append(sae_info)
            available_saes = filtered

        if not available_saes:
            raise ValueError("No SAEs match the specified filters")

        # Import each SAE
        training_base_dir = settings.data_dir / "trainings" / request.training_id
        created_saes: List[ExternalSAE] = []

        for sae_info in available_saes:
            source_dir = training_base_dir / sae_info.path

            db_sae = await SAEManagerService._import_single_sae(
                db=db,
                training=training,
                source_dir=source_dir,
                layer=sae_info.layer,
                hook_type=sae_info.hook_type,
                name_prefix=request.name,
                description=request.description,
            )
            created_saes.append(db_sae)

        # Commit all at once
        await db.commit()

        # Refresh all SAEs
        for sae in created_saes:
            await db.refresh(sae)

        # Build response
        sae_responses = [SAEResponse.model_validate(sae) for sae in created_saes]

        logger.info(
            f"Imported {len(created_saes)} SAE(s) from training {request.training_id}"
        )

        return SAEImportFromTrainingResponse(
            imported_count=len(created_saes),
            sae_ids=[sae.id for sae in created_saes],
            saes=sae_responses,
            training_id=request.training_id,
            message=f"Successfully imported {len(created_saes)} SAE(s)"
        )

    @staticmethod
    async def import_from_file(
        db: AsyncSession,
        request: SAEImportFromFileRequest
    ) -> ExternalSAE:
        """
        Import an SAE from a local file.

        Args:
            db: Database session
            request: Import request with file path

        Returns:
            Created ExternalSAE
        """
        # request.file_path is user-supplied — use the strict resolver that
        # rejects paths escaping the trusted data roots (prevents path injection
        # like ../../../etc/passwd or /etc/shadow).
        source_path = settings.resolve_user_path(request.file_path)

        if not source_path.exists():
            raise ValueError(f"File not found: {request.file_path}")

        # RESOLVED AND VALIDATED BEFORE ANYTHING IS COPIED. This ran after the copytree, so
        # a refused import left its bytes in SAE storage with no row to find them by.
        #
        # The hook comes from the SAE's cfg.json (in the directory, or beside the file) or a
        # Gemma Scope set's name in the path, else NULL (review R2, B5); a layer the request
        # does not give comes from the SAE's own record (review R3-B).
        recorded = resolve_sae_hook(source_path, str(source_path))
        recorded_layer = resolve_sae_layer(source_path, recorded.hook_type, str(source_path))
        # A REQUESTED layer that contradicts the SAE's own record is REFUSED, not logged
        # (review R3-B, R3B-14). The layer decides where every consumer reads: extraction
        # hooks it, the export and the push publish and key by it, steering steers at it.
        # One of the two values is wrong; a warning in a worker log is read by nobody, and
        # this repo's rule is to refuse rather than pick. Omitting the layer takes the SAE's.
        if request.layer is not None and recorded_layer is not None and request.layer != recorded_layer:
            raise ValueError(
                f"Refusing to import {request.file_path} at layer {request.layer}: the SAE's own "
                f"files record layer {recorded_layer} (hook {recorded.hook_type!r}). Omit the layer "
                f"to import it at {recorded_layer}, or correct the SAE's config if {request.layer} "
                "is the right one."
            )
        layer = request.layer if request.layer is not None else recorded_layer

        sae_id = SAEManagerService.generate_sae_id()

        # Copy to SAE storage
        local_path = HuggingFaceSAEService.get_sae_storage_path(sae_id)
        local_path.mkdir(parents=True, exist_ok=True)

        if source_path.is_file():
            shutil.copy2(source_path, local_path / source_path.name)
            total_size = source_path.stat().st_size
        else:
            shutil.copytree(source_path, local_path, dirs_exist_ok=True)
            total_size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())

        # Create SAE record
        db_sae = ExternalSAE(
            id=sae_id,
            name=request.name,
            description=request.description,
            source=SAESource.LOCAL.value,
            status=SAEStatus.READY.value,
            model_name=request.model_name,
            layer=layer,
            hook_type=recorded.hook_type,
            format=request.format.value,
            local_path=str(local_path),
            file_size_bytes=total_size,
            progress=100.0,
            sae_metadata={
                "original_path": str(source_path),
                "hook_source": recorded.source,
            },
            downloaded_at=utc_now()
        )

        db.add(db_sae)
        await db.commit()
        await db.refresh(db_sae)

        logger.info(f"Imported SAE {sae_id} from {request.file_path}")

        return db_sae

    @staticmethod
    async def update_download_progress(
        db: AsyncSession,
        sae_id: str,
        progress: float,
        status: Optional[SAEStatus] = None,
        error_message: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> Optional[ExternalSAE]:
        """
        Update SAE download progress.

        Args:
            db: Database session
            sae_id: SAE ID
            progress: Download progress (0-100)
            status: New status (optional)
            error_message: Error message if failed
            metadata_updates: Additional metadata to merge

        Returns:
            Updated ExternalSAE if found
        """
        sae = await SAEManagerService.get_sae(db, sae_id)
        if not sae:
            return None

        sae.progress = progress

        if status:
            sae.status = status.value

        if error_message:
            sae.error_message = error_message

        if metadata_updates:
            current_metadata = sae.sae_metadata or {}
            current_metadata.update(metadata_updates)
            sae.sae_metadata = current_metadata

        if status == SAEStatus.READY:
            sae.downloaded_at = utc_now()

        await db.commit()
        await db.refresh(sae)

        return sae

    @staticmethod
    async def update_sae_info(
        db: AsyncSession,
        sae_id: str,
        layer: Optional[int] = None,
        n_features: Optional[int] = None,
        d_model: Optional[int] = None,
        architecture: Optional[str] = None,
        model_name: Optional[str] = None,
        file_size_bytes: Optional[int] = None
    ) -> Optional[ExternalSAE]:
        """
        Update SAE architecture info after download/conversion.

        Args:
            db: Database session
            sae_id: SAE ID
            layer: Target layer
            n_features: Number of features
            d_model: Model dimension
            architecture: SAE architecture type
            model_name: Target model name
            file_size_bytes: File size

        Returns:
            Updated ExternalSAE if found
        """
        sae = await SAEManagerService.get_sae(db, sae_id)
        if not sae:
            return None

        if layer is not None:
            sae.layer = layer
        if n_features is not None:
            sae.n_features = n_features
        if d_model is not None:
            sae.d_model = d_model
        if architecture is not None:
            sae.architecture = architecture
        if model_name is not None:
            sae.model_name = model_name
        if file_size_bytes is not None:
            sae.file_size_bytes = file_size_bytes

        await db.commit()
        await db.refresh(sae)

        return sae

    @staticmethod
    async def delete_sae(
        db: AsyncSession,
        sae_id: str,
        delete_files: bool = True
    ) -> bool:
        """
        Delete an SAE.

        Args:
            db: Database session
            sae_id: SAE ID
            delete_files: Whether to delete local files

        Returns:
            True if deleted, False if not found
        """
        sae = await SAEManagerService.get_sae(db, sae_id)
        if not sae:
            return False

        # Delete local files if requested
        if delete_files and sae.local_path:
            try:
                local_path = settings.resolve_deletable_path(sae.local_path)
            except ValueError as e:
                logger.error(f"Refusing to delete SAE local_path: {e}")
                local_path = None
            if local_path is not None and local_path.exists():
                try:
                    if local_path.is_dir():
                        shutil.rmtree(local_path)
                    else:
                        local_path.unlink()
                    logger.info(f"Deleted SAE files at {local_path}")
                except Exception as e:
                    logger.warning(f"Error deleting SAE files: {e}")

        if delete_files:
            # Hard delete: the files are gone, so keeping the row would orphan
            # any Features extracted from this SAE (they reference a live row
            # whose files no longer exist). The features.external_sae_id and
            # extraction_jobs.external_sae_id FKs are ON DELETE CASCADE, so the
            # DATABASE cleans up derived jobs, features and activations. That
            # holds only while those relationships keep passive_deletes=True:
            # without it this line SELECTs every child into the API process
            # first (113 GB on 2026-09-12). test_delete_does_not_load_children.py
            await db.delete(sae)
        else:
            # Metadata-only removal that preserves files → reversible soft delete.
            sae.status = SAEStatus.DELETED.value
        await db.commit()

        logger.info(f"Deleted SAE {sae_id} (hard={delete_files})")

        return True

    @staticmethod
    async def delete_saes_batch(
        db: AsyncSession,
        sae_ids: List[str],
        delete_files: bool = True
    ) -> Dict[str, Any]:
        """
        Delete multiple SAEs.

        Args:
            db: Database session
            sae_ids: List of SAE IDs to delete
            delete_files: Whether to delete local files

        Returns:
            Dict with deleted_count, failed_count, deleted_ids, failed_ids, errors
        """
        deleted_ids = []
        failed_ids = []
        errors = {}

        # Local import avoids a service-layer import cycle (profiles → saes).
        from src.services.cluster_profile_service import ClusterProfileService

        for sae_id in sae_ids:
            try:
                # Feature 014 guard: bound cluster profiles block deletion
                # (explicit check — the FK RESTRICT would otherwise poison the
                # session mid-batch with an IntegrityError).
                profile_count = await ClusterProfileService.count_for_sae(db, sae_id)
                if profile_count > 0:
                    failed_ids.append(sae_id)
                    errors[sae_id] = (
                        f"{profile_count} cluster profile(s) bound — delete them or "
                        "delete the SAE individually with force=true"
                    )
                    continue
                success = await SAEManagerService.delete_sae(db, sae_id, delete_files)
                if success:
                    deleted_ids.append(sae_id)
                else:
                    failed_ids.append(sae_id)
                    errors[sae_id] = "SAE not found"
            except Exception as e:
                await db.rollback()
                failed_ids.append(sae_id)
                errors[sae_id] = str(e)

        return {
            "deleted_count": len(deleted_ids),
            "failed_count": len(failed_ids),
            "deleted_ids": deleted_ids,
            "failed_ids": failed_ids,
            "errors": errors
        }

    # ``get_ready_saes_for_steering`` used to sit here and was REMOVED (review R3-B, the
    # steering-picker item). It had no caller anywhere -- not in the backend, not in the MCP
    # tools, not behind any route -- while looking exactly like the steering picker's source.
    # A fix aimed at "what the picker offers" would have landed in code nobody runs, which is
    # this repo's signature failure. The picker is the frontend filtering the SAE list by
    # what steering will accept: see `frontend/src/utils/saeSteerability.ts`.
