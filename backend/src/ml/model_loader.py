"""
Model loading and quantization utilities.

This module provides functions for loading language models from HuggingFace,
applying quantization, extracting architecture configuration, and estimating
memory requirements.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)

from ..models.model import QuantizationFormat

logger = logging.getLogger(__name__)


# Note: Architecture validation has been replaced with dynamic layer discovery.
# Any transformer model with standard attention + MLP structure is now supported.
# See layer_discovery.py for the dynamic introspection logic.


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class OutOfMemoryError(Exception):
    """Raised when loading fails due to insufficient memory."""
    pass


# validate_architecture() has been removed.
# Architecture validation now happens dynamically during hook registration.
# This allows any transformer model with standard structure to be used.


#: Fields that describe the shape of one transformer stack. Read off whichever
#: tower is being described, so the same list serves the text, vision and audio
#: towers of a composite model.
SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "intermediate_size",
    "max_position_embeddings",
    "vocab_size",
    "num_key_value_heads",       # GQA/MQA
    "hidden_act",
    "initializer_range",
    "layer_norm_eps",
    "use_cache",
    "tie_word_embeddings",
    "rope_theta",
    # Mixture-of-experts. A DIFFERENT axis from tower nesting: an MoE config is
    # flat, but each layer's FFN is a router over experts, so a reader that
    # knows only num_hidden_layers cannot tell a dense 8B from an 8x7B. The
    # names differ per implementation (Mixtral, Qwen-MoE, DeepSeek), and they
    # are recorded only when present.
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "n_routed_experts",
    "n_shared_experts",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    # Granite's shared MLP (granitemoeshared, granitemoehybrid) and ERNIE's shared
    # experts. Without them a Q4 granite-4.0-h-small was sized without its shared
    # MLP, 3 x 4,096 x 8,192 a layer over 40 layers: about 4B of its 32B parameters.
    # (ERNIE's routed count is already read as `num_experts`, which its config answers.)
    "shared_intermediate_size",
    "moe_num_shared_experts",
    "num_experts_shared",
)


def _read(sub: Any, field: str) -> Tuple[Any, bool]:
    """
    Read one field. Returns (value, refused).

    Three outcomes, and they are NOT the same thing:
      * a value            -> the stack has one global value for this field
      * (None, False)      -> the field is absent
      * (None, True)       -> the config REFUSED, because the value varies per
                              layer and no single global answer exists

    google/gemma-4-12B-it is heterogeneous: asking its text config for
    `num_key_value_heads` raises AmbiguousGlobalPerLayerAttributeError by
    design. `getattr(obj, field, None)` swallows AttributeError and nothing
    else, so that one refusal propagated and discarded the whole description --
    the backfill logged "repaired 0 model(s)" and the Training page stayed
    blank (2026-08-25).

    AttributeError means absent; anything else means the config declined to
    answer. That rule needs no import of a private exception class and so
    survives a transformers upgrade.
    """
    try:
        return getattr(sub, field), False
    except AttributeError:
        return None, False
    except Exception:
        return None, True


def _describe(sub: Any) -> Dict[str, Any]:
    """Shape fields present on one config object."""
    described: Dict[str, Any] = {}
    varies_per_layer = []

    for field in SHAPE_FIELDS:
        value, refused = _read(sub, field)
        if refused:
            varies_per_layer.append(field)
        elif value is not None:
            described[field] = value

    model_type, _ = _read(sub, "model_type")
    if model_type is not None:
        described["model_type"] = model_type

    # A stack whose layers do not share one shape. Recorded rather than
    # flattened away: anything reasoning per layer -- an SAE, a hook, a memory
    # estimate -- must not assume every layer looks like the first. Detected by
    # the config refusing, not by probing for a private attribute; in
    # transformers 5 every config exposes a per-layer VIEW, so its presence
    # says nothing.
    if varies_per_layer:
        described["heterogeneous_layers"] = True
        described["per_layer_fields"] = sorted(varies_per_layer)

    return described


def _text_tower(config: AutoConfig) -> Tuple[Any, Optional[str]]:
    """
    The sub-config describing the LANGUAGE model, and the attribute holding it.

    Asks transformers rather than matching names. `get_text_config()` returns
    the config itself on a flat model and the text section on a composite one,
    and the library extends it as new composite architectures land -- which a
    hand-maintained list of names does not.
    """
    getter = getattr(config, "get_text_config", None)
    try:
        text = getter() if callable(getter) else config
    except Exception:                                   # pragma: no cover
        text = config

    # Believe a separate text tower only if it presents an INTEGER depth.
    # gemma-4's sibling towers carry `num_hidden_layers: null`, and an object
    # that answers to every attribute would otherwise be accepted and then
    # supply every remaining field from nowhere.
    if text is None or text is config:
        return config, None
    if not isinstance(getattr(text, "num_hidden_layers", None), int):
        return config, None

    # Name it by the attribute it lives under, so callers can address the tower
    # rather than only read its numbers.
    for name in _tower_names(config):
        if getattr(config, name, None) is text:
            return text, name
    return text, "text_config"


def _tower_names(config: AutoConfig) -> Tuple[str, ...]:
    """
    Every sub-config this architecture declares.

    `sub_configs` is a class attribute maintained by transformers -- e.g.
    Gemma3Config declares {'text_config': Gemma3TextConfig, 'vision_config':
    SiglipVisionConfig}. Enumerating it means a new modality on a new
    architecture is described the day the library supports it, with no list
    here to update.
    """
    declared = getattr(type(config), "sub_configs", None) or {}
    return tuple(declared.keys())


def _describe_towers(config: AutoConfig) -> Dict[str, Dict[str, Any]]:
    """Each declared tower's own shape, keyed by its attribute name."""
    towers = {}
    for name in _tower_names(config):
        sub = getattr(config, name, None)
        if sub is None:
            continue
        described = _describe(sub)
        if described:
            towers[name] = described
    return towers


def extract_architecture_config(config: AutoConfig) -> Dict[str, Any]:
    """
    Describe a model's architecture, whatever shape its config has.

    Flat decoder-only configs are described directly. Composite configs -- any
    model with more than one tower, which today means vision-language, audio
    and "omni" models -- are described by their TEXT tower at the top level,
    with every declared tower recorded under `towers`.

    Top-level fields keep meaning "the stack an SAE is trained on", so existing
    readers (the Training page's layer picker, memory estimation, the data
    model docs) need no change. `towers` is additive, and is what interpreting
    a non-text modality will need.

    Why the text tower rather than the outer config: a composite config's outer
    level carries the fusion metadata, not a transformer stack.
    google/gemma-4-12B-it exposed exactly three usable keys there and no layer
    count, so the Training page offered no layers at all (2026-08-25).

    Returns:
        Dictionary containing architecture details. Always has `model_type`.
        Composite models also carry `towers` and `text_tower`.
    """
    text, text_tower_name = _text_tower(config)

    arch_config: Dict[str, Any] = {"model_type": config.model_type}

    # The text tower's own numbers win, then anything the outer config declares
    # that the tower does not (fusion-level settings such as tie_word_embeddings
    # often live only at the top).
    arch_config.update(_describe(config))
    arch_config.update(_describe(text))
    arch_config["model_type"] = config.model_type

    towers = _describe_towers(config)
    if towers:
        arch_config["towers"] = towers
    if text_tower_name is not None:
        # Which tower the top-level numbers came from. A reader comparing this
        # against a running model needs to know it describes one tower of a
        # composite, not the whole thing.
        arch_config["text_tower"] = text_tower_name

    return arch_config


def get_quantization_config(quant_format: QuantizationFormat) -> Optional[BitsAndBytesConfig]:
    """
    Get BitsAndBytes quantization configuration for the specified format.

    Args:
        quant_format: Quantization format enum value

    Returns:
        BitsAndBytesConfig for bitsandbytes quantization, or None for FP32/FP16
    """
    if quant_format == QuantizationFormat.FP32:
        return None  # Load in full precision

    elif quant_format == QuantizationFormat.FP16:
        return None  # Will use torch_dtype=torch.float16

    elif quant_format == QuantizationFormat.Q8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

    elif quant_format == QuantizationFormat.Q4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    elif quant_format == QuantizationFormat.Q2:
        # Q2 is experimental - use 4-bit with aggressive settings
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4",  # More aggressive than nf4
        )

    else:
        raise ValueError(f"Unknown quantization format: {quant_format}")


def estimate_model_memory(
    params_count: int,
    quant_format: QuantizationFormat,
) -> int:
    """
    Estimate memory requirements for a model in bytes.

    Args:
        params_count: Number of model parameters
        quant_format: Quantization format

    Returns:
        Estimated memory requirement in bytes
    """
    # Base memory per parameter based on quantization
    bytes_per_param = {
        QuantizationFormat.FP32: 4,
        QuantizationFormat.FP16: 2,
        QuantizationFormat.Q8: 1,
        QuantizationFormat.Q4: 0.5,
        QuantizationFormat.Q2: 0.25,
    }

    base_memory = params_count * bytes_per_param[quant_format]

    # Add overhead for activations, gradients, optimizer states (roughly 20%)
    overhead = base_memory * 0.2

    return int(base_memory + overhead)


def estimate_parameter_count(config) -> Optional[int]:
    """Parameter count from the config alone, before any weight is downloaded.

    Standard decoder arithmetic: embeddings + per-layer attention and MLP. Only
    an ESTIMATE — MoE, tied embeddings and multimodal towers all shift it — so
    the caller treats a shortfall as advisory sizing, not a measurement, and
    `preflight_gpu_capacity` skips entirely when this returns None.

    Reads through a sub-config when the top level has no `hidden_size`: unified
    and multimodal configs keep the text fields nested, which is the same shape
    that broke `vocab_size` (MIS-E2E-083's sibling, reported the same day).
    """
    def _get(name: str) -> Optional[int]:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
        for attr in dir(config):
            if attr.startswith("_"):
                continue
            try:
                child = getattr(config, attr)
            except Exception:
                continue
            nested = getattr(child, name, None)
            if isinstance(nested, int) and nested > 0:
                return nested
        return None

    hidden = _get("hidden_size")
    layers = _get("num_hidden_layers")
    vocab = _get("vocab_size")
    if not (hidden and layers and vocab):
        return None

    intermediate = _get("intermediate_size") or 4 * hidden
    # Attention: q,k,v,o ≈ 4·h². MLP: gate/up/down ≈ 3·h·i (2 for non-gated,
    # rounded up deliberately — under-estimating defeats the point).
    per_layer = 4 * hidden * hidden + 3 * hidden * intermediate
    embeddings = 2 * vocab * hidden          # input + output, untied worst case
    return int(layers * per_layer + embeddings)


def _release_failed_load(device_map: Any, max_memory: Optional[Dict[Any, str]]) -> None:
    """Give back to the driver what a failed load allocated, on every card it could have used.

    By torch index, as the rest of this loader addresses cards: a split's cards are
    its budget's keys, a named device ("cuda:1") is its own index, and anything
    else ("auto", "sequential", a dict) could have used every visible card.
    """
    import gc

    gc.collect()
    if not torch.cuda.is_available():
        return
    if max_memory is not None:
        indices = [int(index) for index in max_memory]
    else:
        try:
            target = device_map if isinstance(device_map, torch.device) else torch.device(device_map)
        except (TypeError, RuntimeError, ValueError):
            target = None
        if target is not None and target.index is not None:
            indices = [target.index]
        else:
            indices = list(range(torch.cuda.device_count()))
    for index in indices:
        with torch.cuda.device(index):
            torch.cuda.empty_cache()


def load_model_from_hf(
    repo_id: str,
    quant_format: QuantizationFormat = QuantizationFormat.FP16,
    cache_dir: Optional[Path] = None,
    device_map: str = "auto",
    trust_remote_code: bool = False,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    attn_implementation: Optional[str] = None,
    max_memory: Optional[Dict[Any, str]] = None,
    extra_mb_by_layer: Optional[Dict[int, float]] = None,
    working_mb_by_layer: Optional[Dict[int, float]] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, AutoConfig, Dict[str, Any]]:
    """
    Load a language model from HuggingFace Hub with specified quantization.

    Args:
        repo_id: HuggingFace model repository ID (e.g., "meta-llama/Llama-2-7b-hf")
        quant_format: Quantization format to apply
        cache_dir: Directory to cache downloaded models
        device_map: Device mapping strategy ("auto", "cpu", "cuda:0", etc.)
        max_memory: A SPLIT placement's per-card budget by torch index, with
            ``device_map="sequential"`` — ``Placement.max_memory``. It has no
            "cpu" key. The split is mapped before loading (``ml/split_load.py``)
            and refused there if transformers would put any of it on the CPU or
            disk; a load that still maps off the GPUs is refused after loading.
            None, the default, leaves every existing caller's call unchanged.
        extra_mb_by_layer: For a split, the MiB the caller will put beside each
            decoder layer on that layer's card, by layer index — its SAEs. The
            split is mapped so every card holds its layers' share of both
            (``plan_split_load``). Ignored for a single card.
        working_mb_by_layer: For a split, the MiB the caller allocates beside a
            layer only while it works on that layer (an SAE encode's codes); each
            card keeps room for the largest of its layers'. Ignored for a single card.
        trust_remote_code: Whether to trust remote code execution
        hf_token: HuggingFace API token for gated models
        local_files_only: If True, only use locally cached files (no network calls).
        attn_implementation: Force an attention backend, e.g. "eager". Left as
            None the model keeps transformers' own choice, which is SDPA
            wherever supported — so every existing caller's `from_pretrained`
            call is byte-identical to before. Pass "eager" only when the caller
            needs attention PROBABILITIES: SDPA and flash kernels never
            materialise them (`sdpa_attention_forward` returns
            `(attn_output, None)`), and eager is materially slower.
            Use this when the model is already downloaded to avoid HF API validation
            for gated models.

    Returns:
        Tuple of (model, tokenizer, config, metadata dict)

    Raises:
        ModelLoadError: If model loading fails
        OutOfMemoryError: The load ran out of GPU memory. It is never retried at
            another format: see the handler.
        ValueError: If architecture is unsupported
    """
    logger.info(f"Loading model {repo_id} with quantization {quant_format.value} (local_files_only={local_files_only})")

    try:
        # Load configuration first to validate architecture
        config = AutoConfig.from_pretrained(
            repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            local_files_only=local_files_only,
        )

        # Note: Architecture validation removed. Dynamic layer discovery handles this.
        # Any transformer model with standard attention + MLP blocks is now supported.
        logger.info(f"Model architecture: {config.model_type}")

        # Extract architecture configuration
        arch_config = extract_architecture_config(config)

        # WILL THIS FIT? ASK BEFORE SPENDING THE MINUTES (live 2026-08-23).
        #
        # An extraction on gemma-4-12B-it ran 2m47s and died with "CUDA out of
        # memory. Tried to allocate 120.00 MiB. GPU 0 has a total capacity of
        # 23.56 GiB of which 113.06 MiB is free" — the weights had taken the
        # card and the first forward pass had nowhere to go. A 12B model at FP16
        # is ~24 GB of weights on a 23.56 GB card; it was never going to fit,
        # and nothing said so.
        #
        # The check lives HERE rather than at the ten call sites, because a
        # guard added to one caller and not its siblings is this codebase's most
        # repeated defect. The config is already loaded above, so the parameter
        # count is available before a single weight is fetched.
        from ..services.resource_config import preflight_gpu_capacity

        # Against the memory the load will actually use: the one card
        # `device_map` names, every visible card together for "auto", or a
        # split's own cards (the keys of its `max_memory`) and no others.
        preflight_gpu_capacity(
            params_count=estimate_parameter_count(config),
            quantization=quant_format.value,
            device=device_map if max_memory is None else tuple(max_memory),
            model_name=repo_id,
        )

        # Get quantization configuration
        quantization_config = get_quantization_config(quant_format)

        # Determine torch dtype
        if quant_format == QuantizationFormat.FP16:
            torch_dtype = torch.float16
        elif quant_format in (QuantizationFormat.Q8, QuantizationFormat.Q4, QuantizationFormat.Q2):
            torch_dtype = torch.float16  # bitsandbytes uses fp16 for compute
        else:
            torch_dtype = torch.float32

        # A SPLIT IS MAPPED BEFORE A WEIGHT IS READ. transformers holds back the
        # largest layer's size on the lowest-index card for a CPU put-back this
        # GPU-only budget never makes, so a split the placement's budgets held
        # spilled to disk and was refused after loading. `plan_split_load` maps
        # it with transformers' own inference, returns that room within each
        # card's limit, and refuses a split that still spills — here, before the
        # weights are downloaded or loaded. See `ml/split_load.py`.
        load_max_memory = None if max_memory is None else dict(max_memory)
        split_plan = None
        if max_memory is not None and device_map == "sequential":
            from .split_load import SplitDoesNotFit, plan_split_load

            try:
                split_plan = plan_split_load(
                    config,
                    max_memory=max_memory,
                    dtype=torch_dtype,
                    quantization_config=quantization_config,
                    trust_remote_code=trust_remote_code,
                    model_name=f"{repo_id} at {quant_format.value}",
                    extra_mb_by_layer=extra_mb_by_layer,
                    working_mb_by_layer=working_mb_by_layer,
                )
            except SplitDoesNotFit as refusal:
                raise ModelLoadError(str(refusal)) from refusal
            if split_plan is not None:
                load_max_memory = dict(split_plan.max_memory)

        # Load model
        out_of_memory = None
        try:
            load_kwargs: Dict[str, Any] = dict(
                config=config,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                cache_dir=str(cache_dir) if cache_dir else None,
                trust_remote_code=trust_remote_code,
                token=hf_token,
                local_files_only=local_files_only,
            )
            # INSERTED ONLY WHEN ASKED FOR, so "a default caller's call is
            # unchanged" is a property a test can assert rather than a claim.
            if attn_implementation is not None:
                load_kwargs["attn_implementation"] = attn_implementation
            if load_max_memory is not None:
                load_kwargs["max_memory"] = load_max_memory
            model = AutoModelForCausalLM.from_pretrained(repo_id, **load_kwargs)
            if split_plan is not None and getattr(model, "hf_device_map", None) != split_plan.device_map:
                logger.warning(
                    "%s loaded with a different device map than the one mapped before loading "
                    "(planned %s MiB per card); the post-load check still applies.",
                    repo_id, split_plan.mapped_mb,
                )

            if max_memory is not None:
                # GPUs ONLY (operator decision 3, 2026-09-13). The budget has no
                # "cpu" key, but accelerate always keeps "disk" as a last resort,
                # so a model the cards cannot hold still loads — with layers read
                # from disk on every forward pass, hours slower, and nothing to
                # say so. Refuse it instead.
                from .model_devices import off_gpu_modules

                offloaded = off_gpu_modules(model)
                if offloaded:
                    del model
                    import gc

                    gc.collect()
                    for index in max_memory:
                        with torch.cuda.device(int(index)):
                            torch.cuda.empty_cache()
                    first = next(iter(offloaded.items()))
                    raise ModelLoadError(
                        f"{repo_id} does not fit on the GPUs it was split across at "
                        f"{quant_format.value}: {len(offloaded)} module(s) would run from "
                        f"{'/'.join(sorted(set(offloaded.values())))} (for example {first[0]} on "
                        f"{first[1]}). miStudio runs models on GPUs only; free memory on those "
                        f"cards or choose a smaller quantization."
                    )

            logger.info(f"Successfully loaded model with {quant_format.value} quantization")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            out_of_memory = str(e)

        if out_of_memory is not None:
            # NEVER RETRIED AT ANOTHER FORMAT. This used to fall back Q2 -> Q4 ->
            # Q8 -> FP16 -> FP32: every retry needs MORE memory than the load that
            # had just run out, so on a tight card or a split it could only fail
            # again minutes later (ending in a misleading "does not fit at FP32").
            # Had one succeeded, a training, an SAE extraction or a steering run
            # would have read the model at a precision nobody asked for, silently.
            #
            # RELEASED OUTSIDE THE HANDLER: the exception's traceback holds the
            # frames holding what the attempt allocated, so emptying the cache
            # inside it frees nothing.
            _release_failed_load(device_map, max_memory)
            raise OutOfMemoryError(
                f"Out of memory loading {repo_id} at {quant_format.value}: {out_of_memory}. "
                "Free memory on the GPU(s), choose another GPU, or use a more aggressive "
                "quantization."
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            local_files_only=local_files_only,
        )

        # Calculate metadata
        params_count = sum(p.numel() for p in model.parameters())
        memory_required = estimate_model_memory(params_count, quant_format)

        metadata = {
            "repo_id": repo_id,
            "quantization": quant_format.value,
            "params_count": params_count,
            "memory_required_bytes": memory_required,
            "architecture": config.model_type,
            "architecture_config": arch_config,
            # THE RESOLVED backend, not the requested one. A caller that needs
            # attention probabilities can assert on this and refuse in seconds,
            # rather than discovering minutes into a GPU job that its request
            # was ignored.
            "attn_implementation": getattr(
                model.config, "_attn_implementation", None),
        }

        return model, tokenizer, config, metadata

    except Exception as e:
        if isinstance(e, (OutOfMemoryError, ValueError, ModelLoadError)):
            raise
        logger.error(f"Failed to load model {repo_id}: {e}")
        raise ModelLoadError(f"Failed to load model {repo_id}: {e}")
