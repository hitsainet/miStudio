"""A definition states the precision its probe was fitted at, and its vectors are scored at it.

Two failures this pins, both found 2026-10-03:
  * the contract carried no precision, so a consumer serving bfloat16 could not tell its parity
    failure (combined Δ 0.251) came from the producer having run float16;
  * test vectors scored in a padded batch carry batch noise into the reference — measured at up
    to 0.177 under bfloat16 — that a one-request-at-a-time consumer can never reproduce.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

from src.services import probe_definition_builder
from src.services.probe_definition_builder import ProbeExportRefused, _vector_dtype_or_refuse


def _model(dtype=None):
    from src.ml.native_dtype import LOAD_DTYPE_ATTR

    model = SimpleNamespace()
    if dtype:
        setattr(model, LOAD_DTYPE_ATTR, {"model_dtype": dtype})
    return model


class TestTheVectorDtype:
    def test_it_is_the_runs_recorded_dtype(self):
        run = SimpleNamespace(id="pmr_x", environment={"model_dtype": "bfloat16"})
        assert _vector_dtype_or_refuse(run, _model("bfloat16")) == "bfloat16"

    def test_a_run_that_predates_dtype_recording_is_refused_not_built_with_none(self):
        """⚠ Such a run trained at float16; today's loader runs bfloat16. Vectors built now would
        be bfloat16 scores beside a float16 threshold, and parity would pass against the wrong
        distribution."""
        run = SimpleNamespace(id="pmr_legacy", environment={"seed": 1337})
        with pytest.raises(ProbeExportRefused, match="predates dtype recording") as exc:
            _vector_dtype_or_refuse(run, _model("bfloat16"))
        assert exc.value.status == 409

    def test_a_run_with_no_environment_at_all_is_refused(self):
        with pytest.raises(ProbeExportRefused):
            _vector_dtype_or_refuse(SimpleNamespace(id="pmr_y", environment=None), _model("bfloat16"))

    def test_a_load_at_a_different_dtype_from_the_runs_is_refused(self):
        run = SimpleNamespace(id="pmr_z", environment={"model_dtype": "float16"})
        with pytest.raises(ProbeExportRefused, match="trained at float16") as exc:
            _vector_dtype_or_refuse(run, _model("bfloat16"))
        assert exc.value.status == 409


def _build_tree():
    return ast.parse(inspect.getsource(probe_definition_builder.build))


class TestBuildIsWired:
    def test_build_resolves_the_dtype_from_the_run_and_the_loaded_model(self):
        calls = [n for n in ast.walk(_build_tree()) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "_vector_dtype_or_refuse"]
        assert len(calls) == 1
        assert [ast.unparse(a) for a in calls[0].args] == ["run", "model"]

    def test_the_resolved_dtype_reaches_the_model_identity(self):
        identities = [n for n in ast.walk(_build_tree()) if isinstance(n, ast.Call)
                      and getattr(n.func, "id", None) == "ModelIdentity"]
        assert len(identities) == 1
        load = [kw for kw in identities[0].keywords if kw.arg == "load_dtype"]
        assert load and ast.unparse(load[0].value) == "load_dtype"
        assigned = [n for n in ast.walk(_build_tree()) if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == "load_dtype" for t in n.targets)]
        assert assigned and getattr(assigned[0].value.func, "id", None) == "_vector_dtype_or_refuse"

    def test_vectors_are_scored_one_input_at_a_time(self):
        """Every `forward_scores` call in build() receives a ONE-ELEMENT list."""
        calls = [n for n in ast.walk(_build_tree()) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "forward_scores"]
        assert calls, "build() no longer scores its vectors through forward_scores"
        for call in calls:
            examples = call.args[1]
            assert isinstance(examples, ast.List) and len(examples.elts) == 1, (
                f"forward_scores is given {ast.unparse(examples)}; a padded batch puts batch "
                "noise (up to 0.177 at bfloat16) into the parity reference"
            )


class TestTheRunRecordsTheDtypeTheBuilderReads:
    """The builder reads `environment["model_dtype"]`; the run must WRITE it, from the loaded model."""

    def test_execute_probe_run_spreads_the_load_record_into_its_environment(self):
        from src.services import probe_monitor_run

        tree = ast.parse(inspect.getsource(probe_monitor_run.execute_probe_run))
        updates = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and getattr(n.func, "attr", None) == "update"
                   and ast.unparse(n.func.value) == "environment"]
        spread = [
            v for call in updates for arg in call.args if isinstance(arg, ast.Dict)
            for k, v in zip(arg.keys, arg.values) if k is None
        ]
        assert any("load_dtype_record(model)" in ast.unparse(v) for v in spread), (
            "environment.update no longer records the model's load dtype"
        )


class TestATrainingNeverMixesPrecisions:
    """Review round 1, M5: an on-the-fly run resumed across the change would train its remaining
    steps at bfloat16 and record one precision for an SAE trained on two."""

    def _db(self, row):
        class _Q:
            def filter_by(self, **k): return self
            def first(self): return row

        return SimpleNamespace(query=lambda model: _Q(), commit=lambda: None)

    def test_a_fresh_run_records_its_precision(self):
        from src.workers.training_tasks import _record_training_dtype

        row = SimpleNamespace(model_dtype=None)
        _record_training_dtype(self._db(row), "t", "bfloat16")
        assert row.model_dtype == "bfloat16"

    def test_a_resume_at_the_same_precision_is_allowed(self):
        from src.workers.training_tasks import _record_training_dtype

        row = SimpleNamespace(model_dtype="bfloat16")
        _record_training_dtype(self._db(row), "t", "bfloat16", resuming=True)

    def test_a_resume_of_a_pre_change_run_is_refused(self):
        from src.workers.training_tasks import _record_training_dtype

        row = SimpleNamespace(model_dtype=None)
        with pytest.raises(ValueError, match="two precisions"):
            _record_training_dtype(self._db(row), "t", "bfloat16", resuming=True, legacy_dtype="float16")
        assert row.model_dtype is None

    def test_a_legacy_fp32_run_resumed_at_float32_is_allowed(self):
        """Round 3, LOW-6: the old loader ran an on-the-fly FP32 row at float32, so nothing
        changed."""
        from src.workers.training_tasks import _record_training_dtype

        row = SimpleNamespace(model_dtype=None)
        _record_training_dtype(self._db(row), "t", "float32", resuming=True, legacy_dtype="float32")
        assert row.model_dtype == "float32"

    def test_both_paths_pass_whether_they_are_resuming(self):
        from src.workers import training_tasks

        tree = ast.parse(inspect.getsource(training_tasks.train_sae_task))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "_record_training_dtype"]
        assert len(calls) == 2
        assert all(any(k.arg == "resuming" and "start_step" in ast.unparse(k.value) for k in c.keywords)
                   for c in calls)


def test_a_quantization_change_between_training_and_export_is_refused():
    """Round 3, MED-4: the post-load check compared precision only."""
    from src.ml.native_dtype import LOAD_DTYPE_ATTR

    run = SimpleNamespace(id="pmr_q", environment={"model_dtype": "bfloat16", "quantization": "Q4"})
    model = SimpleNamespace()
    setattr(model, LOAD_DTYPE_ATTR, {"model_dtype": "bfloat16", "quantization": "FP16"})
    with pytest.raises(ProbeExportRefused, match="same precision, different activations"):
        _vector_dtype_or_refuse(run, model)
