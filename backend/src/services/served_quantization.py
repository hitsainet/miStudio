"""Which model quantizations miLLM serves — the one rule every miLLM-facing load refuses by.

⚠ Q2 IS NOT SERVED. miStudio loads a Q2 row as bitsandbytes fp4, while miLLM has no Q2 case and
serves the row unquantized. Anything measured on a Q2 load — a steered transcript, a usable band,
a circuit's causal effect size, its faithfulness — therefore describes a model nothing serves
(operator decision 2026-10-03). Before this module the rule existed as two copies with different
messages and exception types, and four circuit paths had no copy at all (review round 3, M-B).
"""

from __future__ import annotations

from typing import Any

#: Formats miLLM serves. A row in any other is refused before anything loads.
SERVED_QUANTIZATIONS = frozenset({"FP32", "FP16", "Q8", "Q4"})


class UnservedQuantization(RuntimeError):
    """A model row is quantized in a format miLLM does not serve."""


def normalise_quantization(quantization: Any) -> str:
    return str(getattr(quantization, "value", quantization)).upper()


def refuse_unserved_quantization(quantization: Any, model_id: str = "") -> str:
    """The row's quantization, normalised — or `UnservedQuantization` when miLLM does not serve it."""
    value = normalise_quantization(quantization)
    if value == "Q2":
        raise UnservedQuantization(
            f"Model {model_id or '?'} is quantized Q2, which miLLM does not serve, so anything "
            f"measured on it would describe a model nothing serves. Re-download it as Q4, Q8 or FP16."
        )
    if value not in SERVED_QUANTIZATIONS:
        raise ValueError(f"unknown model quantization {quantization!r}")
    return value
