"""Review round 2 (2026-10-04) on the calibration and steering changes: three gaps, each pinned."""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from src.ml.probe_monitor_model import EmptyWindowError, ProbeHead
from src.services import probe_monitor_run
from src.services.probe_monitor_capture import ScoreSpec, forward_scores_by_scope
from src.services.probe_monitor_render import RenderedExample


class TestOneQ2Rule:
    """⚠ REVIEW ROUND 3 (M-B): two copies of the rule with different messages, and four circuit
    paths — capture, attribution, intervention, faithfulness — with no copy at all, so a circuit's
    effect size and faithfulness could be measured on a model miLLM does not serve."""

    def test_the_rule(self):
        from src.services.served_quantization import UnservedQuantization, refuse_unserved_quantization

        for value in ("FP32", "FP16", "Q8", "Q4"):
            assert refuse_unserved_quantization(value) == value
        with pytest.raises(UnservedQuantization, match="does not serve"):
            refuse_unserved_quantization("Q2", "m")
        with pytest.raises(ValueError):
            refuse_unserved_quantization("Q3")

    @pytest.mark.parametrize("module", [
        "circuit_capture_service", "circuit_attribution_service",
        "circuit_intervention_service", "circuit_faithfulness_service",
    ])
    def test_every_circuit_load_goes_through_it(self, module):
        import importlib

        tree = ast.parse(inspect.getsource(importlib.import_module(f"src.services.{module}")))
        loads = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_model_from_hf"
        ]
        assert loads, f"{module}: no load_model_from_hf call — the scan is looking at the wrong shape"
        for call in loads:
            quant = next(k.value for k in call.keywords if k.arg == "quant_format")
            assert "refuse_unserved_quantization" in {
                getattr(n.func, "id", None) for n in ast.walk(quant) if isinstance(n, ast.Call)
            }, f"{module}: a load passes the row's quantization without the served-format rule"


class TestQ2IsRefusedOnEverySteeringLoad:
    """M1: UI steering refused Q2, while the recorder and calibration — which load through
    `steering_core` — fp4-loaded it and recorded transcripts and bands for a model nothing serves."""

    def test_the_core_loader_refuses_a_q2_row_before_loading(self, monkeypatch):
        from src.ml import model_loader
        from src.models.model import QuantizationFormat
        from src.services.steering_core import SteeringCoreError, load_model_and_structure

        monkeypatch.setattr(
            model_loader, "load_model_from_hf",
            lambda **k: pytest.fail("a Q2 row was loaded before being refused"),
        )
        row = SimpleNamespace(
            id="m_q2", quantization=QuantizationFormat.Q2, file_path=None, repo_id="org/m",
        )
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = row
        with pytest.raises(SteeringCoreError, match="Q2"):
            load_model_and_structure("m_q2", db, torch.device("cpu"))


def _row(n_user: int, n_assistant: int) -> RenderedExample:
    roles = ["user"] * n_user + ["assistant"] * n_assistant
    messages = [0] * n_user + [1] * n_assistant
    return RenderedExample(
        input_ids=list(range(2, 2 + len(roles))), token_roles=roles, token_message=messages, text="",
    )


class TestOnlyAnEmptyWindowIsSkipped:
    """M3: any ValueError from `combine` read as "window absent" — a mask of the wrong shape or an
    unknown rule would have been downgraded to a warning."""

    @pytest.fixture(scope="class")
    def model(self):
        from transformers import AutoModelForCausalLM, LlamaConfig

        torch.manual_seed(0)
        return AutoModelForCausalLM.from_config(LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        )).eval()

    def _spec(self):
        return ScoreSpec(head=ProbeHead(weight=torch.ones(16), bias=0.0, layer=1), rule="mean")

    def test_the_empty_window_error_is_a_value_error(self):
        assert issubclass(EmptyWindowError, ValueError)

    def test_an_empty_window_is_skipped(self, model):
        out = forward_scores_by_scope(
            model, [_row(3, 0)], [self._spec()], scopes=["all", "last_assistant"],
            required_scopes=["all"],
        )
        assert set(out) == {"all"}

    def test_any_other_refusal_still_raises(self, model, monkeypatch):
        from src.ml import probe_monitor_model

        def broken(*args, **kwargs):
            raise ValueError("mask shape (1, 3) does not match scores (1, 4)")

        monkeypatch.setattr(probe_monitor_model, "combine", broken)
        with pytest.raises(ValueError, match="mask shape"):
            forward_scores_by_scope(
                model, [_row(3, 2)], [self._spec()], scopes=["all", "last_assistant"],
                required_scopes=[],
            )


class TestTheRecutArmKeepsABarItCouldNotRescore:
    """M2: one unscorable window used to be dropped from the table with only a log line."""

    def test_a_missing_window_keeps_its_previous_entry(self):
        previous = {"all": {"threshold": 1.0}, "prompt": {"threshold": 2.0}, "response": {"threshold": 3.0}}
        rescored = {"all": {"threshold": 1.5}, "prompt": {"threshold": 2.5}}
        merged = probe_monitor_run.merge_rescored_windows("pm_x", previous, rescored)
        assert merged == {"all": {"threshold": 1.5}, "prompt": {"threshold": 2.5},
                          "response": {"threshold": 3.0}}

    def test_nothing_rescored_is_none_so_the_arm_refuses(self):
        assert probe_monitor_run.merge_rescored_windows("pm_x", {"all": {}}, None) is None
        assert probe_monitor_run.merge_rescored_windows("pm_x", {"all": {}}, {}) is None

    def test_the_arm_uses_the_merge(self):
        """The RESULT must become the table — a bare call would pass a call-exists check."""
        tree = ast.parse(inspect.getsource(probe_monitor_run.recut_probe_windows_on_gpu))
        assigns = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "merge_rescored_windows"
        ]
        assert len(assigns) == 1
        assert [ast.unparse(t) for t in assigns[0].targets] == ["decisions"]
        assert ast.unparse(assigns[0].value.args[1]) == "probe.window_decisions"


class TestTheRecutArmRefusesBeforeWritingHalfARecut:
    """⚠ REVIEW ROUND 3 (M-A). Keeping a window the arm could not re-score is right only when that
    window can still be MOVED. Without stored negatives `propose` refuses — after the re-scored
    windows were committed at the old target — and its message sends the operator back to this
    same GPU job, which fails the same way."""

    def test_a_kept_window_without_stored_negatives_is_named(self):
        previous = {"all": {"scores_path": "a.npy"}, "prompt": {}, "response": {"scores_path": "r.npy"}}
        rescored = {"all": {"threshold": 1.0}}
        assert probe_monitor_run.unrecuttable_kept_windows(previous, rescored) == ["prompt"]

    def test_a_rescored_window_is_never_named(self):
        assert probe_monitor_run.unrecuttable_kept_windows({"prompt": {}}, {"prompt": {}}) == []

    def test_the_arm_refuses_and_writes_nothing(self, monkeypatch, tmp_path):
        """⚠ A SURVIVING MUTATION WROTE THIS. A source-order check stayed green with the refusal
        disabled. So the real arm runs, with only its GPU and database edges stubbed."""
        from src.models.probe_monitor import ProbeMonitor, ProbeMonitorRun
        from src.services.probe_recalibration import RecalibrationRefused

        previous = {
            "all": {"threshold": 1.0, "scores_path": "a.npy"},
            "prompt": {"threshold": 2.0},  # no stored negatives: cannot be moved
        }
        probe = SimpleNamespace(
            id="pm_x", run_id="pmr_x", calibration_dataset_id="pmd_cal", length_bands=None,
            calibration_lengths_path=None, threshold_source="calibration_set",
            window_decisions=dict(previous),
        )
        run = SimpleNamespace(id="pmr_x", calibration_dataset_id="pmd_cal")
        db = MagicMock()
        db.query.side_effect = lambda model: MagicMock(**{
            "filter.return_value.first.return_value": probe if model is ProbeMonitor else run,
        })
        context = SimpleNamespace(scope="all", target_fpr=0.01, artifact_dir=tmp_path)
        monkeypatch.setattr(probe_monitor_run, "context_from_row", lambda row: context)
        monkeypatch.setattr(probe_monitor_run, "probe_precision", lambda db, run: {"refusal": None})
        monkeypatch.setattr(probe_monitor_run, "_load_model_for_run", lambda run: (None, None, ""))
        monkeypatch.setattr(probe_monitor_run, "load_probe", lambda db, pid: (None, None))
        monkeypatch.setattr(probe_monitor_run, "_prepare_calibration_rows", lambda *a: ([], []))
        # Only `all` could be scored; `prompt` and `response` were empty windows.
        monkeypatch.setattr(
            probe_monitor_run, "_score_calibration_group",
            lambda *a, **k: [{"all": probe_monitor_run.CalibrationScores(
                negatives=[float(v) for v in range(200)], n_rows=200, lengths=[5] * 200)}],
        )
        with pytest.raises(RecalibrationRefused) as exc:
            probe_monitor_run.recut_probe_windows_on_gpu(db, "pm_x", target_fpr=0.05)
        assert exc.value.code == "window_not_rescorable"
        assert "prompt" in exc.value.detail
        assert probe.window_decisions == previous, "the table was rewritten before refusing"
        db.commit.assert_not_called()
