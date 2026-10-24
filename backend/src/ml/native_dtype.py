"""The dtype a model row loads at: the checkpoint's own 16-bit precision, never a hardcoded one.

⚠ WHY THIS MODULE EXISTS. Every load path here cast a 16-bit row to `torch.float16`, while the
checkpoints this estate runs (Llama-3.1, Qwen2.5, Gemma) are published in `bfloat16` and miLLM
serves them in `bfloat16`. So every probe, SAE and circuit was fitted on activations nothing
serves. It surfaced as a probe failing parity on import — combined Δ 0.251 against 0.10 —
and Phase 0 of the 2026-10-03 fix confirmed it: re-scoring at bf16 one request at a time
reproduced miLLM's per-vector differences to four decimals on 6 of 16 vectors.

THE RULE (shared with miLLM, pinned by `docs/schemas/native-dtype-cases.json`, which both repos
test their resolver against and which the cross-repo check keeps byte-identical):

    FP32 row                 -> float32. The row asked for 4 bytes a parameter and is priced at
                                4 — loading it at 16 bits was a mislabel in both repos.
    FP16 / Q8 / Q4 / Q2 row  -> the checkpoint's own 16-bit dtype: bfloat16 or float16. A
                                checkpoint that says float32, or names nothing, gets bfloat16 —
                                float16's 65,504 ceiling is the overflow risk, bfloat16's range
                                is float32's.

For a bnb row the resolved dtype is BOTH `torch_dtype` (the modules bitsandbytes leaves alone)
and `bnb_4bit_compute_dtype`, so the quantized and unquantized halves of the model agree.

⚠ WHY NOT `dtype="auto"`, which the J-lens readout uses: it hands `plan_split_load`, the GPU
preflight and the bnb compute dtype nothing to plan with, and it records nothing. A split
planned at one dtype and loaded at another is how a model spills. The resolver returns ONE
concrete object, and that object goes to the plan, the preflight and the load.

⚠ A CONFIG THAT NAMES NO DTYPE GIVES `None` HERE, never transformers' own default. Reading
`config.dtype` off an `AutoConfig` returns a default for a checkpoint that recorded nothing, and
presenting that as "the checkpoint's dtype" would be a fabricated fact wearing a fact's clothes.
`source` says where the answer came from, so a reader can tell a read value from a default.

Pure: no device code, no model, no file writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import torch

#: The dtypes this rule ever produces, by their canonical names.
LOAD_DTYPES: Tuple[str, ...] = ("float16", "bfloat16", "float32")

_TORCH = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

_ALIASES = {
    "float16": "float16", "half": "float16", "fp16": "float16", "f16": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float32": "float32", "float": "float32", "fp32": "float32", "f32": "float32",
}

#: What a 16-bit row falls back to when the checkpoint names no 16-bit dtype.
DEFAULT_16BIT = "bfloat16"


def normalise_dtype_name(value: Any) -> Optional[str]:
    """`torch.bfloat16`, `"torch.bfloat16"`, `"bf16"`, `"bfloat16"` -> `"bfloat16"`; unknown -> None."""
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        text = str(value)
    else:
        text = str(value)
    text = text.strip().lower()
    if text.startswith("torch."):
        text = text[len("torch."):]
    return _ALIASES.get(text)


def _field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def checkpoint_dtype_of(config: Any) -> Tuple[Optional[str], str]:
    """The dtype a checkpoint RECORDS, and where it was found: `(name, source)`.

    Read order — `dtype`, `torch_dtype`, then the same two under `text_config` (multimodal
    checkpoints keep the language model's config one level down). `source` is `"config"`,
    `"text_config"` or `"default"` (nothing recorded, `name` is None).

    Accepts a parsed `config.json` dict or a transformers config object. ⚠ For a config object,
    only values the checkpoint actually wrote are trusted: transformers fills `dtype` with its
    own default when the file had none, so an object without `to_dict()` evidence of the key is
    read as recording nothing.
    """
    if config is None:
        return None, "default"
    raw = config
    if not isinstance(config, Mapping) and hasattr(config, "to_diff_dict"):
        # `to_diff_dict` keeps only what differs from the class defaults — i.e. what the
        # checkpoint wrote — so a transformers-supplied default does not pass for a recorded one.
        try:
            raw = config.to_diff_dict()
        except Exception:  # noqa: BLE001 - fall back to attribute reads
            raw = config
    for name in ("dtype", "torch_dtype"):
        found = normalise_dtype_name(_field(raw, name))
        if found:
            return found, "config"
    text = _field(raw, "text_config")
    if text is not None:
        for name in ("dtype", "torch_dtype"):
            found = normalise_dtype_name(_field(text, name))
            if found:
                return found, "text_config"
    return None, "default"


@dataclass(frozen=True)
class ResolvedDtype:
    """What a row loads at, and why. Pass this ONE object to the plan, the preflight and the load."""

    torch_dtype: torch.dtype
    name: str
    #: What the checkpoint recorded (None when it recorded nothing).
    checkpoint_dtype: Optional[str]
    #: Where `checkpoint_dtype` came from: "config", "text_config" or "default".
    source: str
    quantization: str

    @property
    def storage_name(self) -> str:
        """The dtype activations are STORED at: float32 for a float32 load, else float16.

        A bf16 value converts to float16 exactly for 6.1e-5 <= |x| <= 65,504 — float16 carries
        more mantissa bits than bfloat16 (below that it rounds as a subnormal, error < 6e-8) — so
        16-bit loads keep today's disk cost and readers. A value above that range is refused at
        write time (`activation_storage.to_storage`), never written as inf.
        """
        return "float32" if self.name == "float32" else "float16"

    def as_record(self) -> dict:
        """The facts every artifact records about the load."""
        return {
            "model_dtype": self.name,
            "checkpoint_dtype": self.checkpoint_dtype,
            "dtype_source": self.source,
            "activation_storage_dtype": self.storage_name,
            # The row's quantization rides with the precision: a Q4 and an FP16 load of one
            # bfloat16 checkpoint are both "bfloat16" and still read different activations
            # (~0.93 cosine per token), so precision alone does not identify the distribution.
            "quantization": self.quantization,
        }


def _quant_name(quantization: Any) -> str:
    value = getattr(quantization, "value", quantization)
    return str(value).upper()


def resolve_load_dtype(
    quantization: Any,
    checkpoint_dtype: Optional[str],
    source: str = "config",
    pre_quantized: bool = False,
) -> ResolvedDtype:
    """THE rule. `quantization` is a QuantizationFormat or its value ("FP16", "Q4", ...).

    A PRE-QUANTIZED checkpoint (GPTQ/AWQ/FP8 — its config carries `quantization_config`) keeps
    the 16-bit rule whatever its row is labelled: its unquantized modules and the size plan are
    16-bit, and GPTQ/AWQ kernels at float32 are unsupported. In the shared table since v2.
    """
    quant = _quant_name(quantization)
    if pre_quantized and quant == "FP32":
        quant = "FP16"
    recorded = normalise_dtype_name(checkpoint_dtype)
    if recorded is None:
        source = "default"
    if quant == "FP32":
        name = "float32"
    elif quant in ("FP16", "Q8", "Q4", "Q2"):
        name = recorded if recorded in ("bfloat16", "float16") else DEFAULT_16BIT
    else:
        raise ValueError(f"unknown quantization {quantization!r}; known: FP32, FP16, Q8, Q4, Q2")
    return ResolvedDtype(
        torch_dtype=_TORCH[name],
        name=name,
        checkpoint_dtype=recorded,
        source=source,
        quantization=quant,
    )


def is_pre_quantized(config: Any) -> bool:
    """Whether a checkpoint ships already quantized (its config carries `quantization_config`)."""
    if config is None:
        return False
    return _field(config, "quantization_config") is not None


def resolve_for_config(quantization: Any, config: Any) -> ResolvedDtype:
    """Resolve from a parsed `config.json` dict or a transformers config object."""
    recorded, source = checkpoint_dtype_of(config)
    return resolve_load_dtype(quantization, recorded, source, pre_quantized=is_pre_quantized(config))


def read_config_json(snapshot: Union[str, Path]) -> Optional[dict]:
    """The snapshot's `config.json`, parsed, or None when it cannot be read."""
    path = Path(snapshot) / "config.json"
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def resolve_for_snapshot(quantization: Any, snapshot: Union[str, Path]) -> ResolvedDtype:
    """Resolve from a model snapshot directory (the one holding `config.json`)."""
    return resolve_for_config(quantization, read_config_json(snapshot))


#: The attribute a loaded model carries its resolved dtype record under.
LOAD_DTYPE_ATTR = "mistudio_load_dtype"


def attach_load_dtype(model: Any, resolved: ResolvedDtype) -> None:
    """Stamp a loaded model with what it was loaded at, so recorders read the FACT.

    Every artifact built from a model records its dtype from here — a probe run's environment,
    an extraction's metadata — rather than re-deriving it, because a second derivation is a
    second answer that can disagree with the load that actually happened.
    """
    setattr(model, LOAD_DTYPE_ATTR, resolved.as_record())


def load_dtype_record(model: Any) -> Optional[dict]:
    """The record `attach_load_dtype` stamped, or None when this model was not loaded through
    a resolver (a test double, or a path not yet migrated). None is recorded as "not recorded",
    never filled in with a guess."""
    record = getattr(model, LOAD_DTYPE_ATTR, None)
    return dict(record) if isinstance(record, dict) else None


def storage_dtype_of_load(model: Any) -> str:
    """The activation storage dtype for a loaded model: from its load record, else its parameters.

    The fallback reads the model's own parameter dtype — a fact, not a guess — for a model not
    loaded through a resolver (a test double).
    """
    record = load_dtype_record(model)
    if record and record.get("activation_storage_dtype"):
        return str(record["activation_storage_dtype"])
    try:
        first = next(model.parameters())
    except (AttributeError, StopIteration, TypeError):
        return "float16"
    return "float32" if first.dtype == torch.float32 else "float16"


def common_model_dtype(recorded: "list[Optional[str]]", what: str = "extractions") -> Optional[str]:
    """The ONE precision a set of artifacts was read at, or a refusal when they disagree.

    An SAE trained on a float16 extraction and a bfloat16 one learns a single dictionary over two
    distributions, and nothing downstream could tell. NULL (not recorded — every artifact before
    2026-10-03, which ran float16) counts as its own value, so a legacy extraction does not mix
    with a recorded bfloat16 one either. All NULL returns None: still not recorded.
    """
    distinct = sorted({str(value) if value is not None else "not recorded" for value in recorded})
    if len(distinct) > 1:
        raise ValueError(
            f"these {what} were read at different precisions ({', '.join(distinct)}); training on "
            "them together would fit one dictionary over different distributions. Use "
            f"{what} read at one precision — re-extract the older ones, which ran float16."
        )
    values = [value for value in recorded if value is not None]
    return values[0] if values else None
