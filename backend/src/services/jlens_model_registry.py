"""
Model resolution for J-space readouts (feature 022 Phase 4.5).

WHY A CACHE AND NOT A LOAD PER REQUEST. Unlike the SAE logit lens — which reads
`W_U` alone out of the safetensors shard and never instantiates the model — a
J-space readout needs a FORWARD PASS to capture residuals, so the whole model
has to be resident. Loading it per request would make every readout cost tens of
seconds and would thrash memory under any real use.

ONE MODEL AT A TIME, DELIBERATELY. The cache holds exactly one entry and evicts
on a miss. A larger cache means two models resident at once, and this workbench
shares a card with a serving process — the previous logit-lens implementation
failed outright the moment miLLM occupied the GPU, which is why the readout
itself is CPU-only (`jlens_readout_service.READOUT_DEVICE`). Capture is the one
GPU-touching step and its device is chosen explicitly, never inherited.

MODEL-AGNOSTIC (BR-032). Structure comes from `discover_transformer_structure`;
there is no architecture name in this module.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from ..core.config import settings

logger = logging.getLogger(__name__)


class ModelNotAvailable(RuntimeError):
    """The model cannot be loaded for a readout, with the reason stated.

    Distinct from a generic failure because the caller turns it into a 4xx with
    an actionable message rather than a 500 — "this model is not downloaded" is
    a different thing to tell a user than "the readout crashed".
    """


@dataclass
class LoadedModel:
    key: str
    model: Any
    tokenizer: Any
    structure: Any
    unembedding: torch.Tensor
    name: str
    d_model: int
    n_layers: int
    n_vocab: int


class _SingleEntryCache:
    """Holds at most one loaded model.

    Guarded by a lock: FastAPI serves requests concurrently and two readouts for
    different models arriving together would otherwise both load, putting two
    full models in memory — the exact failure this cache exists to prevent.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entry: Optional[LoadedModel] = None

    def get_or_load(self, key: str, loader) -> LoadedModel:
        with self._lock:
            if self._entry is not None and self._entry.key == key:
                return self._entry
            if self._entry is not None:
                logger.info("Evicting J-lens model %s to load %s", self._entry.key, key)
                self._entry = None
                _release_memory()
            self._entry = loader()
            return self._entry

    def clear(self) -> None:
        with self._lock:
            self._entry = None
            _release_memory()

    def peek(self) -> Optional[LoadedModel]:
        """The resident entry, whatever it is. For callers that accept any
        device rather than requiring a specific one."""
        with self._lock:
            return self._entry

    @property
    def loaded_key(self) -> Optional[str]:
        entry = self._entry
        return entry.key if entry else None


def _release_memory() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_CACHE = _SingleEntryCache()


_TOKENIZER_CACHE: Dict[str, Any] = {}


def tokenizer_for(model_record: Any) -> Any:
    """The model's OWN tokenizer, without loading its weights.

    THE SAME TOKENIZER THE INTERVENTION WILL USE. A direction is resolved by
    `service.tokenizer.encode(...)`, and whether a string is one token is a
    property of THAT vocabulary — ' Rome' is a single token on one model and two
    on another. Checking against any other tokenizer would answer a different
    question and would be wrong exactly where it matters, on the unusual strings
    a user types by hand.

    Weights are NOT loaded: a tokenizer is a few megabytes of JSON and a
    single-token check must not cost a model load on a single-GPU box, behind a
    possible 45-minute fit.
    """
    from transformers import AutoTokenizer

    # `repo_id`, THE SAME FIELD `load_for_readout` USES. The first version read
    # `name`, which is the DISPLAY name — "LFM2.5-1.2B-Instruct" rather than
    # "LiquidAI/LFM2.5-1.2B-Instruct" — so `from_pretrained` looked for a
    # snapshot that does not exist and the endpoint 500'd on every call. Every
    # test stubbed this function whole, so nothing exercised the derivation.
    repo_id, resolved = locate_weights(model_record)
    if repo_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[repo_id]

    try:
        tok = AutoTokenizer.from_pretrained(
            repo_id, cache_dir=resolved, local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001 - reported, never a 500
        raise ModelNotAvailable(
            f"Could not load the tokenizer for {repo_id}: {exc}"
        ) from exc
    _TOKENIZER_CACHE[repo_id] = tok
    return tok


def locate_weights(model_record: Any) -> Tuple[str, Any]:
    """The repo id and local cache directory for a Model row.

    ONE DEFINITION, because two drifted. `load_for_readout` reads `repo_id` and
    resolves `file_path`; the tokenizer path re-derived both and got the field
    wrong, which is invisible until something actually tries to load.
    """
    repo_id = getattr(model_record, "repo_id", None)
    if not repo_id:
        raise ModelNotAvailable(
            f"Model {getattr(model_record, 'id', '?')} has no repo_id, so its "
            "weights cannot be located."
        )
    raw_path = getattr(model_record, "file_path", None)
    resolved = settings.resolve_data_path(raw_path) if raw_path else None
    if not (resolved and resolved.exists()):
        raise ModelNotAvailable(
            f"{repo_id} is not downloaded locally, so its vocabulary is not "
            "available to check a token against."
        )
    return str(repo_id), resolved


def clear_cache() -> None:
    """Drop the resident model. Called by tests and by an explicit unload."""
    _CACHE.clear()
    # THE VOCABULARY GOES WITH IT. A model re-downloaded under the same name can
    # carry a different tokenizer, and a stale one would answer "is this a
    # single token" for weights that are no longer there.
    _TOKENIZER_CACHE.clear()


def loaded_model_key() -> Optional[str]:
    return _CACHE.loaded_key


def load_for_readout(model_record: Any, capture_device: str = "cpu") -> LoadedModel:
    """Resolve a Model row to everything `ReadoutService` needs.

    `capture_device` defaults to CPU rather than "auto". A readout is an
    analysis operation that must never contend with serving for VRAM, and
    "auto" silently takes the GPU the moment one exists.

    THE CACHE KEY CARRIES THE DEVICE. It used to be the model id alone, so a
    caller asking for CPU received whatever device the last caller happened to
    load on. That is not a preference being overridden — it is a different
    object than the one requested, and it produced a device-mismatch crash in
    the readout after a fit had loaded the same model onto CUDA. Consumers that
    are happy with any resident copy should say so by passing None, which is a
    request this function can honour honestly rather than a silent substitution
    it performs behind the caller's back.
    """
    from ..ml.layer_discovery import discover_transformer_structure
    from ..ml.model_loader import load_model_from_hf
    from ..models.model import QuantizationFormat
    from .analysis_service import load_unembedding_matrix, resolve_snapshot_dir

    repo_id = getattr(model_record, "repo_id", None)
    if not repo_id:
        raise ModelNotAvailable(
            f"Model {getattr(model_record, 'id', '?')} has no repo_id, so its "
            "weights cannot be located for a readout."
        )

    model_key = str(getattr(model_record, "id", repo_id))

    # None = "any resident copy of this model will do". Used by the readout,
    # which can capture on whatever device the model already occupies and does
    # its own maths on READOUT_DEVICE regardless — so evicting a GPU-resident
    # model just to reload it on CPU would cost a minute and free nothing.
    if capture_device is None:
        resident = _CACHE.loaded_key
        if resident is not None and resident.rsplit("@", 1)[0] == model_key:
            entry = _CACHE.peek()
            if entry is not None:
                return entry
        capture_device = "cpu"

    key = f"{model_key}@{capture_device}"

    def _load() -> LoadedModel:
        raw_path = getattr(model_record, "file_path", None)
        resolved = settings.resolve_data_path(raw_path) if raw_path else None
        downloaded = bool(resolved and resolved.exists())
        if not downloaded:
            raise ModelNotAvailable(
                f"{repo_id} is not downloaded locally. A J-space readout runs a "
                "forward pass, so the weights must be present — download the "
                "model first."
            )

        logger.info("Loading %s for J-space readout on %s", repo_id, capture_device)

        # LOAD IN THE CHECKPOINT'S OWN DTYPE, not a forced one.
        #
        # Forcing fp16 onto a checkpoint whose weights are bfloat16 leaves the
        # model internally MIXED, and the forward pass then dies with
        # "expected scalar type BFloat16 but found Half" before any readout
        # arithmetic happens. That is what gemma-2-2b-it did on the cluster.
        #
        # A readout does not need a particular dtype — it needs the model to
        # RUN — so the right precision is whatever the checkpoint was saved in.
        # The readout's own matvec casts to fp32 separately, which is about
        # ranking stability rather than about making the model work.
        # ...AND HONOUR THE QUANTIZATION THE MODEL ROW ASKS FOR.
        #
        # This loaded at native dtype unconditionally, so a row configured Q8
        # was silently ignored for every fit and readout. On gemma-4-12B that
        # is not a fidelity question, it is a hard stop: ~12.3B bf16 parameters
        # are ~24.6 GB against a 23.56 GB card, and the fit OOM'd during a
        # forward pass with the model already resident. Observed 2026-09-05.
        #
        # THIS IS NOT THE BUG THE COMMENT ABOVE GUARDS AGAINST. That one was
        # FORCING fp16 onto a bf16 checkpoint, which leaves the model
        # internally mixed. bitsandbytes is a different mechanism: it replaces
        # the linear layers with quantized ones and leaves everything else in
        # the checkpoint's own dtype, which `dtype="auto"` still selects. The
        # two are compatible, and only the forcing was ever the problem.
        #
        # FP16/FP32 rows still get `None` here, so the native-dtype path is
        # unchanged for every model that does not ask to be quantized.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..ml.model_loader import get_quantization_config

        quant_name = getattr(model_record, "quantization", None)
        quant_config = None
        if quant_name:
            try:
                quant_config = get_quantization_config(
                    QuantizationFormat(
                        getattr(quant_name, "value", quant_name)
                    )
                )
            except ValueError:
                logger.warning(
                    "Unrecognised quantization %r on %s; loading at native dtype",
                    quant_name, repo_id,
                )
        if quant_config is not None:
            logger.info(
                "Loading %s with %s quantization (the model row asks for it)",
                repo_id, getattr(quant_name, "value", quant_name),
            )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                cache_dir=resolved,
                local_files_only=True,
                dtype="auto",
                device_map=capture_device,
                quantization_config=quant_config,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                repo_id, cache_dir=resolved, local_files_only=True
            )
        except Exception as exc:  # noqa: BLE001 - fall back, reporting why
            logger.warning(
                "Native-dtype load of %s failed (%s); falling back to the "
                "shared loader, which may force a dtype the checkpoint does "
                "not use",
                repo_id,
                exc,
            )
            quant = getattr(model_record, "quantization", None)
            model, tokenizer, _config, _meta = load_model_from_hf(
                repo_id=repo_id,
                quant_format=(
                    QuantizationFormat(quant) if quant else QuantizationFormat.FP16
                ),
                cache_dir=resolved,
                device_map=capture_device,
                local_files_only=True,
            )
        model.eval()

        structure = discover_transformer_structure(model)

        # W_U read from the shard rather than off the model, so the readout
        # holds one CPU copy regardless of where the model itself landed.
        snapshot = resolve_snapshot_dir(resolved, repo_id)
        unembedding = None
        if snapshot is not None:
            try:
                unembedding = load_unembedding_matrix(snapshot, device="cpu")
            except Exception as exc:  # noqa: BLE001 - falls back, reports why
                logger.warning("Could not read W_U from %s: %s", snapshot, exc)

        if unembedding is None:
            # The model is already resident, so taking its output embedding is
            # not a second copy — this is a fallback, not the primary path.
            head = getattr(model, "lm_head", None)
            weight = getattr(head, "weight", None)
            if weight is None:
                embed = model.get_input_embeddings()
                weight = getattr(embed, "weight", None)  # tied embeddings
            if weight is None:
                raise ModelNotAvailable(
                    f"Could not locate an unembedding matrix for {repo_id}."
                )
            unembedding = weight.detach().to("cpu")

        n_vocab, d_model = int(unembedding.shape[0]), int(unembedding.shape[1])
        return LoadedModel(
            key=key,
            model=model,
            tokenizer=tokenizer,
            structure=structure,
            unembedding=unembedding,
            name=repo_id,
            d_model=d_model,
            n_layers=int(structure.num_layers),
            n_vocab=n_vocab,
        )

    return _CACHE.get_or_load(key, _load)
