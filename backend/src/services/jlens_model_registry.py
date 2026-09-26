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
from typing import Any, Dict, Iterable, Optional, Tuple

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
                self._drop()
            self._entry = loader()
            return self._entry

    def clear(self) -> None:
        with self._lock:
            self._drop()

    def _drop(self) -> None:
        """Drop the entry and return its memory on EVERY card it held.

        A split copy holds weights on several cards. `torch.cuda.empty_cache()`
        with no device acts on the CURRENT device, which is the split's first
        card, so the other cards' cached blocks would stay reserved: NVML goes
        on counting them, and the next placement judges those cards fuller than
        they are. The cards come from the entry's key, which names them all.
        """
        entry, self._entry = self._entry, None
        devices = parse_device_spec(entry.key.rpartition("@")[2]) if entry is not None else ()
        model = getattr(entry, "model", None)
        if len(devices) > 1 and isinstance(model, torch.nn.Module):
            # accelerate's dispatch hooks wrap every module's forward in a
            # partial that references the module, so a split model is a web of
            # cycles only the cycle collector can free. Removing the hooks first
            # makes the release plain reference counting.
            from ..ml.model_devices import detach_dispatch_hooks

            detach_dispatch_hooks(model)
        del entry, model
        _release_memory(devices)

    def peek(self) -> Optional[LoadedModel]:
        """The resident entry, whatever it is. For callers that accept any
        device rather than requiring a specific one."""
        with self._lock:
            return self._entry

    @property
    def loaded_key(self) -> Optional[str]:
        entry = self._entry
        return entry.key if entry else None


def _is_out_of_memory(exc: BaseException) -> bool:
    """A load that ran out of GPU memory, as the shared loader recognises one."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _plan_split(repo_id: str, cache_dir: Any, quantization_config: Any, max_memory: dict):
    """The split's budget verified against transformers' own map; None when it cannot be mapped here.

    Mapped with the kwargs this registry's load passes: its config as
    ``from_pretrained`` reads it from the snapshot, the row's bitsandbytes config,
    and ``dtype="auto"`` resolved as transformers resolves it — the config's own
    ``dtype``. A config that names none leaves transformers reading the weights'
    dtype, which is not known here, so the load maps that split itself.

    Raises:
        ModelNotAvailable: transformers would map part of the model off the split's GPUs.
    """
    from transformers import AutoConfig

    from ..ml.split_load import SplitDoesNotFit, plan_split_load

    try:
        config = AutoConfig.from_pretrained(repo_id, cache_dir=cache_dir, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - no config: the load maps the split itself
        logger.warning("No local config for %s, so its split is mapped only by the load: %s", repo_id, exc)
        return None
    dtype = getattr(config, "dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
    if not isinstance(dtype, torch.dtype):
        logger.warning(
            "%s's config names no dtype, so the checkpoint's is read at load and its split is "
            "mapped only by the load", repo_id,
        )
        return None
    try:
        return plan_split_load(
            config,
            max_memory=max_memory,
            dtype=dtype,
            quantization_config=quantization_config,
            model_name=repo_id,
        )
    except SplitDoesNotFit as refusal:
        raise ModelNotAvailable(str(refusal)) from refusal


def _release_memory(devices: Iterable[str] = ()) -> None:
    """Collect, then return cached allocator memory on each GPU in `devices`.

    With no GPU named (a CPU copy, or nothing was cached) the current device is
    emptied, which is what this always did.
    """
    import gc

    gc.collect()
    if not torch.cuda.is_available():
        return
    cards = [torch.device(name) for name in devices if _is_gpu(name)]
    if cards:
        from ..ml.model_devices import empty_cache_on

        empty_cache_on(cards)
    else:
        torch.cuda.empty_cache()


#: Joins the devices of a SPLIT copy in its cache key: "cuda:0+cuda:1".
DEVICE_SEPARATOR = "+"


def _is_gpu(name: str) -> bool:
    try:
        return torch.device(name).type == "cuda"
    except (RuntimeError, TypeError, ValueError):
        return False


def device_spec(devices: Iterable[Any]) -> str:
    """ONE spelling for the device(s) a copy of a model occupies.

    "cpu", "cuda:1", or a split's cards joined in index order ("cuda:0+cuda:1").
    A `torch.device` and its string name are one device, and the same cards in
    another order are one copy: accelerate fills a split's cards in index order
    whatever order the placement listed them in, so both orders load the same
    layout and must not become two cache entries holding two copies.
    """
    names = {str(torch.device(device)) for device in devices}
    if not names:
        raise ValueError("A model copy must occupy at least one device")

    def order(name: str):
        parsed = torch.device(name)
        return (parsed.type != "cuda", parsed.type, -1 if parsed.index is None else parsed.index)

    return DEVICE_SEPARATOR.join(sorted(names, key=order))


def parse_device_spec(spec: Optional[str]) -> Tuple[str, ...]:
    """The devices a `device_spec` names; empty for None or an empty string."""
    if not spec:
        return ()
    return tuple(part for part in str(spec).split(DEVICE_SEPARATOR) if part)


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


def _model_key(model_record: Any) -> str:
    """The model half of a cache key. ONE derivation, shared by the load and the lookup.

    THE QUANTIZATION IS PART OF IT. Switching a model's precision is a
    Re-download on the SAME model id, in another worker, so an id-only key let a
    readout reuse the copy loaded at the OLD precision — Q4 and full-precision
    activations agree at only ~0.93 cosine — and report it as the new one. A
    readout kept on its card (`unload_after=False`) made that the common path.
    """
    base = str(getattr(model_record, "id", getattr(model_record, "repo_id", None)))
    quantization = getattr(model_record, "quantization", None)
    quantization = getattr(quantization, "value", quantization)
    return f"{base}:{quantization}" if quantization else base


def resident_device_for(model_record: Any) -> Optional[str]:
    """The device(s) THIS model's resident copy is on, as a `device_spec`, or None.

    "cuda:1", "cpu", or "cuda:0+cuda:1" for a split copy. None when nothing is
    cached, or when the cached entry is another model. The readout task asks
    this before placing a job, so a copy it left loaded is reused rather than
    joined by a second copy on another card.
    """
    key = _CACHE.loaded_key
    if key is None:
        return None
    model_key, separator, device = key.rpartition("@")
    if not separator or model_key != _model_key(model_record):
        return None
    return device


def estimate_weights_mb(model_record: Any) -> Optional[float]:
    """The memory this model's weights take when a J-lens job loads it, in MB; None if unknown.

    WITHOUT A SIZE, AUTO CANNOT SPLIT. `gpu_placement.resolve_cards` splits a
    model across cards only when it knows the model fits no single card, and
    every J-lens job placed with no size: Auto took the card with the most free
    memory and a model larger than it died in `from_pretrained`.

    WEIGHTS ONLY, here. `jlens_progress.place_on_card` adds the activation
    headroom every other job places with (review round 2): sized at the weights
    alone, Auto put a model on the one card whose free memory just held its
    weights, and the first forward pass ran out of memory there. The fixed 2 GB
    of the loader's preflight, not a percentage: Qwen2.5-14B at bf16 (~28,170
    MiB + 2,048) still fits the node's two cards at ~31,952 MB of split budget,
    where a 20% allowance would take it past that.

    The PRECISION IS THE ONE THE LOAD USES. A quantized row loads through
    bitsandbytes at the loader preflight's per-parameter figure. Every other row
    loads with `dtype="auto"`, the checkpoint's own dtype, whatever the row says
    — so an FP32 row with a bf16 checkpoint takes 2 bytes a parameter, and a
    float32 checkpoint takes 4 whatever its row says.
    """
    from .base_model_budget import params_for_sizing

    quantization = getattr(model_record, "quantization", None)
    # A Q4 row's count is the packed count; its description is not.
    params = params_for_sizing(
        getattr(model_record, "params_count", None),
        quantization,
        getattr(model_record, "architecture_config", None),
    )
    if params is None:
        return None
    quantization = str(getattr(quantization, "value", quantization) or "").upper()

    from .resource_config import _BYTES_PER_PARAM

    if quantization in ("Q8", "Q4", "Q2"):
        per_param = _BYTES_PER_PARAM[quantization]
    else:
        per_param = _checkpoint_bytes_per_param(model_record)
    return params * per_param / (1024 * 1024)


def estimate_fit_working_mb(
    model_record: Any,
    prompts: Iterable[str],
    layers: Optional[Iterable[int]] = None,
    target_layer: str = "penultimate",
) -> Optional[float]:
    """MB a J-lens FIT allocates beyond the weights on its card; None when the row cannot say.

    ``jlens_fitter.fit_working_bytes`` over the row's ``architecture_config`` and the
    longest prompt in tokens, read with the model's own tokenizer (a few MB, no
    weights). A fit keeps its forward graph for a batched backward, which the fixed
    2 GiB every other J-lens task places with does not cover.

    The intermediate width falls back to ``4 x hidden``, as the loader's parameter
    estimate does. No hidden size, layer count, vocabulary or head count: None, and
    the caller keeps the fixed headroom — a dimension is never invented. A model
    whose tokenizer cannot be read counts a character a token, which is never fewer.
    """
    from ..ml.jlens_fitter import fit_working_bytes

    arch = getattr(model_record, "architecture_config", None)
    arch = arch if isinstance(arch, dict) else {}

    def dim(name: str) -> Optional[int]:
        value = arch.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

    hidden, n_layers, vocab, heads = (dim("hidden_size"), dim("num_hidden_layers"), dim("vocab_size"),
                                      dim("num_attention_heads"))

    # READ THE CHECKPOINT WHEN THE ROW CANNOT SAY. The row's
    # `architecture_config` is populated at download time and a row written by an
    # older path can be missing dimensions, but the dimensions themselves are on
    # disk in `config.json` — so falling back to it is reading the truth, not
    # inventing a dimension. This matters because the caller's alternative is the
    # FIXED 2 GiB headroom, while a fit needs that cap PLUS about 0.41 GB of fp32
    # accumulators per captured layer (measured: 3.63 GB of working set for four
    # layers of d_model 3840). Unsized, a fit is placed short by the accumulator
    # term and dies mid-backward on a card that looked like it had room.
    if not (hidden and n_layers and vocab and heads):
        snapshot_dims = _dims_from_checkpoint(model_record)
        hidden = hidden or snapshot_dims.get("hidden_size")
        n_layers = n_layers or snapshot_dims.get("num_hidden_layers")
        vocab = vocab or snapshot_dims.get("vocab_size")
        heads = heads or snapshot_dims.get("num_attention_heads")

    if not (hidden and n_layers and vocab and heads):
        missing = [
            name for name, value in (
                ("hidden_size", hidden), ("num_hidden_layers", n_layers),
                ("vocab_size", vocab), ("num_attention_heads", heads),
            ) if not value
        ]
        logger.warning(
            "%s records no %s, and its checkpoint config does not supply them, so its fit's "
            "working memory cannot be sized; it is placed with the fixed activation headroom, "
            "which is SHORT of what a fit needs by roughly 0.41 GB per captured layer",
            getattr(model_record, "repo_id", getattr(model_record, "id", "?")),
            ", ".join(missing),
        )
        return None
    prompts = [str(prompt) for prompt in prompts]
    try:
        tokenizer = tokenizer_for(model_record)
        longest = max((len(tokenizer(prompt)["input_ids"]) for prompt in prompts), default=1)
    except ModelNotAvailable as exc:
        logger.warning("Sizing the fit's prompts by characters: %s", exc)
        longest = max((len(prompt) for prompt in prompts), default=1)
    target = n_layers - 1 if target_layer == "final" else max(0, n_layers - 2)
    captured = len(list(layers)) if layers else target + 1
    quantization = str(getattr(getattr(model_record, "quantization", None), "value",
                               getattr(model_record, "quantization", None)) or "").upper()
    activation_bytes = 2 if quantization in ("Q8", "Q4", "Q2") else int(_checkpoint_bytes_per_param(model_record))
    return fit_working_bytes(
        d_model=hidden, intermediate=dim("intermediate_size") or 4 * hidden, layers_to_target=target + 1,
        n_captured=captured, max_seq_len=longest, vocab=vocab, heads=heads, activation_bytes=activation_bytes,
    ) / (1024 * 1024)


def _dims_from_checkpoint(model_record: Any) -> Dict[str, int]:
    """Architecture dimensions read from the snapshot's `config.json`.

    The fallback for a row whose `architecture_config` is missing fields. It reads
    the checkpoint, so it reports what the model IS — the alternative is placing a
    fit with the fixed headroom and discovering the shortfall as an OOM.

    NESTED CONFIGS COUNT. A multimodal checkpoint keeps the text stack's
    dimensions under `text_config` (gemma-4-12B is one), and reading only the top
    level finds nothing there — the same nesting that once made the J-lens final
    norm resolve to None and fall back to plain RMS silently.
    """
    import json

    from .analysis_service import resolve_snapshot_dir

    wanted = ("hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads")
    try:
        repo_id, resolved = locate_weights(model_record)
        snapshot = resolve_snapshot_dir(resolved, repo_id)
        config = json.loads((snapshot / "config.json").read_text()) if snapshot else {}
    except (ModelNotAvailable, OSError, ValueError, TypeError):
        return {}

    nested = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    found: Dict[str, int] = {}
    for name in wanted:
        for source in (config, nested):
            value = source.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                found[name] = value
                break
    return found


def _checkpoint_bytes_per_param(model_record: Any) -> float:
    """Bytes per parameter of the checkpoint on disk: 4 for float32, else 2.

    Read from the snapshot's config.json. Anything unreadable counts as 2 — an
    estimate that is too LOW only leaves Auto choosing as it did with no size,
    where one too HIGH refuses a model that fits.
    """
    import json

    from .analysis_service import resolve_snapshot_dir

    try:
        repo_id, resolved = locate_weights(model_record)
        snapshot = resolve_snapshot_dir(resolved, repo_id)
        config = json.loads((snapshot / "config.json").read_text()) if snapshot else {}
    except (ModelNotAvailable, OSError, ValueError, TypeError):
        return 2.0
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    dtype = config.get("dtype") or config.get("torch_dtype") or text.get("dtype") or text.get("torch_dtype")
    return 4.0 if str(dtype).lower() in ("float32", "float", "torch.float32") else 2.0


def release_idle_gpu_copy() -> bool:
    """Drop the resident model if it is on a GPU. Returns whether anything was dropped.

    Registered with `gpu_placement.register_idle_release`, so ANY job placed in
    this worker frees a copy a readout left loaded before its card is chosen —
    every card of a split copy included. A CPU copy holds no card and is kept,
    as the cache always has.
    """
    key = _CACHE.loaded_key
    if key is None:
        return False
    _model, separator, device = key.rpartition("@")
    if not separator or not any(_is_gpu(name) for name in parse_device_spec(device)):
        return False
    logger.info("Freeing the idle J-lens model %s before placing another job", key)
    clear_cache()
    return True


def _register_with_placement() -> None:
    from .gpu_placement import register_idle_release

    register_idle_release(release_idle_gpu_copy)


_register_with_placement()


def load_for_readout(
    model_record: Any, capture_device: Any = "cpu", placement: Any = None
) -> LoadedModel:
    """Resolve a Model row to everything `ReadoutService` needs.

    `capture_device` defaults to CPU rather than "auto". "auto" lets
    accelerate spread the model over whatever it finds, which is a choice of
    GPU nobody made. A task that wants a GPU passes the `placement` it was given
    (`gpu_placement.place_job`); `capture_device` is then ignored.

    A PLACEMENT, NOT ITS DEVICE. `placement.device` is only a split's FIRST
    card: loaded there, a model no single card holds OOMs, or the cache records
    one card for a copy that occupies two and frees only that one. A split
    loads with the placement's own `device_map` and GPU-only `max_memory`,
    passed through as the placement spells them — the map strategy is the
    core's decision, not this module's — and a load accelerate still maps off
    the GPUs is refused (operator decision 3: no CPU or disk spill).

    A `torch.device` and its string name are ONE device. The key and
    `device_map` both use the string form ("cuda:1"), so a placement's device
    and the same device spelled as text cannot become two cache entries holding
    two copies of one model on one card.

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

    model_key = _model_key(model_record)

    # None = "any resident copy of this model will do". Used by the readout,
    # which can capture on whatever device the model already occupies and does
    # its own maths on READOUT_DEVICE regardless — so evicting a GPU-resident
    # model just to reload it on CPU would cost a minute and free nothing.
    if placement is None and capture_device is None:
        resident = _CACHE.loaded_key
        if resident is not None and resident.rsplit("@", 1)[0] == model_key:
            entry = _CACHE.peek()
            if entry is not None:
                return entry
        capture_device = "cpu"

    if placement is not None:
        devices = tuple(placement.all_devices)
        device_map = placement.device_map
        split = bool(placement.is_shard)
    else:
        devices = (capture_device,)
        device_map = str(capture_device)
        split = False
    # THE KEY NAMES EVERY DEVICE THE COPY OCCUPIES. Keyed by the first card
    # alone, a split copy looked like a single-card copy on that card: a readout
    # naming that card would reuse a model spread over two, and the release
    # would free one card of the two.
    capture_device = device_spec(devices)
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

        # A SPLIT'S BUDGET, per card by torch index and with no "cpu" key.
        # Computed here, on a load, because a placement that REUSES a resident
        # split carries no budgets — it names the cards the copy already holds.
        max_memory = None
        if split:
            if placement.max_memory_mb is None:
                raise RuntimeError(
                    f"{repo_id} would be loaded split across {capture_device} with no "
                    "per-card budget; accelerate would then fill the cards unbounded. "
                    "A split load needs the placement that chose its cards."
                )
            max_memory = placement.max_memory

        # A SPLIT IS MAPPED BEFORE A WEIGHT IS READ, as the shared loader maps
        # its own (`ml/split_load.py`). transformers holds back the largest
        # layer's size on the lowest-index card for a CPU put-back this GPU-only
        # budget never makes, so a split the budgets held spilled to disk and
        # was refused below only after every weight had been read. The refusal
        # is raised here, outside the fallback, so it is never loaded again.
        load_max_memory = max_memory
        if split:
            planned = _plan_split(repo_id, resolved, quant_config, max_memory)
            if planned is not None:
                load_max_memory = planned.max_memory

        out_of_memory = None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                cache_dir=resolved,
                local_files_only=True,
                dtype="auto",
                device_map=device_map,
                max_memory=load_max_memory,
                quantization_config=quant_config,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                repo_id, cache_dir=resolved, local_files_only=True
            )
        except Exception as exc:  # noqa: BLE001 - fall back, reporting why
            if _is_out_of_memory(exc):
                # NEVER LOADED AGAIN. The fallback forces the ROW's format, so an
                # FP32 row on a bf16 checkpoint asks for twice what just ran out,
                # and the failed attempt still holds its memory here.
                out_of_memory = f"{type(exc).__name__}: {exc}"
            else:
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
                    device_map=device_map,
                    # And the split's budget, or the fallback would load it with
                    # none — free to spread past the placement's cards. The
                    # placement's own: the shared loader maps it for its format.
                    max_memory=max_memory,
                    local_files_only=True,
                )

        if out_of_memory is not None:
            # Released OUTSIDE the handler: its traceback holds the frames that
            # hold what the attempt allocated.
            _release_memory(devices)
            from ..ml.model_loader import OutOfMemoryError

            raise OutOfMemoryError(
                f"Out of memory loading {repo_id} on {capture_device}: {out_of_memory}. "
                "It is not loaded again at another precision. Free memory on the GPU(s), "
                "choose another GPU, or use a more aggressive quantization."
            )

        if split:
            # OUTSIDE the try above, so a refusal is not taken for a failed load
            # and retried through the fallback. `from_pretrained` keeps "disk"
            # as a last resort even with no "cpu" budget, so a model the cards
            # cannot hold still LOADS — and every forward pass then reads
            # layers from disk, hours slower, with nothing to say so.
            from ..ml.model_devices import detach_dispatch_hooks, off_gpu_modules

            offloaded = off_gpu_modules(model)
            if offloaded:
                detach_dispatch_hooks(model)
                del model
                _release_memory(devices)
                name, target = next(iter(offloaded.items()))
                raise ModelNotAvailable(
                    f"{repo_id} does not fit on {capture_device}: {len(offloaded)} "
                    f"module(s) would run from {'/'.join(sorted(set(offloaded.values())))} "
                    f"(for example {name} on {target}). J-lens jobs run models on GPUs "
                    "only; free memory on those cards or choose a smaller quantization."
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
