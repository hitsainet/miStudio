"""An SAE is applied only at the precision it was trained at (review round 1, M6).

Re-extracting features for an SAE fitted on float16 activations, with the model now loading at
bfloat16, encodes a different distribution through the same dictionary — plausible features with
shifted meanings, invisible in every metric.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

from src.services import extraction_service
from src.services.extraction_service import predicted_load_dtype, sae_training_dtype


def _db(training):
    class _Q:
        def filter(self, *a, **k): return self
        def first(self): return training

    return SimpleNamespace(query=lambda model: _Q())


def test_the_sae_precision_comes_from_its_training_row():
    sae = SimpleNamespace(training_id="t", sae_metadata={})
    assert sae_training_dtype(_db(SimpleNamespace(model_dtype="bfloat16")), sae) == "bfloat16"


def test_an_imported_sae_falls_back_to_its_cfg_metadata():
    sae = SimpleNamespace(training_id=None, sae_metadata={"model_dtype": "float16"})
    assert sae_training_dtype(_db(None), sae) == "float16"


def test_an_unrecorded_sae_is_none_not_float16():
    sae = SimpleNamespace(training_id="t", sae_metadata={})
    assert sae_training_dtype(_db(SimpleNamespace(model_dtype=None)), sae) is None


def test_the_predicted_load_dtype_reads_the_snapshot(tmp_path, monkeypatch):
    import src.services.activation_service as act

    (tmp_path / "config.json").write_text('{"torch_dtype": "bfloat16"}')
    monkeypatch.setattr(act, "resolve_model_snapshot", lambda p: p)
    assert predicted_load_dtype(SimpleNamespace(quantization="FP16"), tmp_path) == "bfloat16"
    assert predicted_load_dtype(SimpleNamespace(quantization="FP32"), tmp_path) == "float32"
    assert predicted_load_dtype(SimpleNamespace(quantization="FP16"), None) is None


def test_the_extraction_checks_before_it_loads_and_records_after():
    """Within the ONE function that loads the base model — not anywhere in the module."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(extraction_service)))
    owners = [fn for fn in ast.walk(tree) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
              and any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "load_model_from_hf"
                      for n in ast.walk(fn))]
    assert len(owners) == 1, [fn.name for fn in owners]
    fn = owners[0]

    def lines(name):
        return sorted(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                      and getattr(n.func, "id", None) == name)

    assert lines("predicted_load_dtype") and lines("predicted_load_dtype")[0] < lines("load_model_from_hf")[0]
    writes = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
              and any(ast.unparse(t) == "extraction_job.config" for t in n.targets)]
    assert writes and "sae_model_dtype" in ast.unparse(writes[0].value)


def test_an_imported_sae_records_what_its_cfg_says(tmp_path):
    """Round 3, MED-1: the gate's fallback read a key no import wrote."""
    import json

    from src.services.sae_manager_service import read_config_model_dtype

    (tmp_path / "cfg.json").write_text(json.dumps({"model_dtype": "bfloat16", "d_in": 4}))
    assert read_config_model_dtype(tmp_path) == "bfloat16"
    (tmp_path / "cfg.json").write_text(json.dumps({"d_in": 4}))
    assert read_config_model_dtype(tmp_path) is None


def test_importing_from_a_training_carries_its_precision():
    """The gate reads `sae_metadata["model_dtype"]` for an SAE whose training row is gone; the
    import from training must write it."""
    from src.services import sae_manager_service

    tree = ast.parse(textwrap.dedent(inspect.getsource(sae_manager_service.SAEManagerService._import_single_sae)))
    dicts = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)
             and any(isinstance(k, ast.Constant) and k.value == "training_hyperparameters" for k in n.keys)]
    assert dicts
    pairs = {k.value: ast.unparse(v) for k, v in zip(dicts[0].keys, dicts[0].values) if isinstance(k, ast.Constant)}
    assert pairs.get("model_dtype") == "training.model_dtype"
