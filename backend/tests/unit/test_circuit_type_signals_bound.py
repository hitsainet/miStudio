"""Per-edge `type_signals` is open-typed but bounded (016 R3-B4, suite tracker 4.22).

The field is deliberately open so the edge classifier's full disclosure always fits.
Open AND unbounded meant one edge could carry megabytes that no other check looked at.
The bound is a validator (not a JSON-Schema constraint), so the published
`circuit-definition-v1.json` that miLLM vendors byte-identically does not move —
`test_circuit_definition_schema_sync.py` pins that half.
"""

import json

import pytest
from pydantic import ValidationError

from src.schemas.circuit_definition import TYPE_SIGNALS_MAX_BYTES, CircuitEdge
from src.services.circuit_edge_type_service import classify_edge


def _edge(signals):
    return {"up": {"layer": 1, "feature_idx": 3}, "down": {"layer": 2, "feature_idx": 4},
            "type_signals": signals}


def test_the_classifiers_real_disclosure_fits_with_room():
    real = classify_edge(weight_prior=0.9, up_top_tokens=["a", "b"], down_top_tokens=["a", "c"],
                         up_label="dogs", down_label="dogs and cats")["signals"]
    size = len(json.dumps(real, separators=(",", ":")))
    assert size * 5 < TYPE_SIGNALS_MAX_BYTES, f"the cap leaves too little headroom ({size} B)"
    CircuitEdge.model_validate(_edge(real))


def test_an_oversized_disclosure_is_refused():
    with pytest.raises(ValidationError, match="type_signals is"):
        CircuitEdge.model_validate(_edge({"blob": "x" * TYPE_SIGNALS_MAX_BYTES}))


def test_exactly_at_the_cap_is_accepted():
    filler = TYPE_SIGNALS_MAX_BYTES - len(json.dumps({"b": ""}, separators=(",", ":")))
    CircuitEdge.model_validate(_edge({"b": "x" * filler}))


def test_the_published_schema_carries_no_new_constraint():
    """A JSON-Schema constraint would change the vendored file; the cap must not."""
    schema = CircuitEdge.model_json_schema()
    signals = schema["properties"]["type_signals"]
    assert "maxProperties" not in json.dumps(signals) and "maxLength" not in json.dumps(signals)
