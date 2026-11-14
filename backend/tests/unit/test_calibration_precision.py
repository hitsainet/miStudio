"""A calibration band records the precision it was measured at, and a reproduction compares it."""

import ast
import inspect
import textwrap

from src.services.circuit_calibration_service import CircuitCalibrationService as C


def test_a_request_cannot_supply_the_precision():
    """It is a fact about the loaded model, set after the load (review round 1, L9)."""
    assert C.create_config({"model_dtype": "float32"})["model_dtype"] is None


def test_build_band_keeps_the_precision_the_load_recorded():
    assert C.create_config({"model_dtype": "bfloat16"}, keep_model_dtype=True)["model_dtype"] == "bfloat16"
    src = textwrap.dedent(inspect.getsource(C.build_band))
    assert "create_config(config, keep_model_dtype=True)" in src


def test_the_reproduction_verdict_states_the_precision_change():
    tree = ast.parse(textwrap.dedent(inspect.getsource(C.reproduce)))
    assigned = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(ast.unparse(t) in ("verdict['precision']", 'verdict["precision"]') for t in n.targets)]
    assert assigned, "the reproduction verdict no longer reports the precision"
    keys = {k.value for k in assigned[0].value.keys}
    assert keys == {
        "original", "reproduced", "changed",
        # Quantization too: a Q4 and an FP16 load share a precision and read different activations.
        "original_quantization", "reproduced_quantization", "quantization_changed",
    }


def test_a_request_cannot_supply_the_quantization_either():
    """Like `model_dtype`, a fact about the loaded model — only the load may set it."""
    assert C.create_config({"model_quantization": "Q4"})["model_quantization"] is None
    assert C.create_config({"model_quantization": "Q4"}, keep_model_dtype=True)["model_quantization"] == "Q4"


def test_the_quantization_is_recorded_from_the_loaded_model():
    tree = ast.parse(textwrap.dedent(inspect.getsource(C._build_generation_fns)))
    assigned = [
        n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(ast.unparse(t) in ("cfg['model_quantization']", 'cfg["model_quantization"]') for t in n.targets)
    ]
    assert assigned and "load_dtype_record(model)" in ast.unparse(assigned[0].value)
