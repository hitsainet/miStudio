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

    def test_an_empty_window_scores_nothing_and_is_not_fatal(self, model):
        """Since 2026-10-04 a row with no tokens in a non-required window is left out of it
        (`n_scored == 0`), rather than the window being dropped for every row."""
        out = forward_scores_by_scope(
            model, [_row(3, 0)], [self._spec()], scopes=["all", "last_assistant"],
            required_scopes=["all"],
        )
        assert set(out) == {"all", "last_assistant"}
        assert out["last_assistant"][0][0].n_scored == 0

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


class TestTheRecutArmDropsABarItCouldNotPlace:
    """Review round 2 of the `last_user` work (M-B), which REVERSES this file's earlier M2 and M-A
    behaviour on purpose. A window the re-score could not place used to KEEP its old bar — and a
    pre-fix `last_user` bar was cut over a span including the template preamble, which `propose`
    then moved as though it were current. The re-score is the current code over the probe's own
    rows; a window it cannot place has no bar that code would produce, so it is dropped — loudly,
    in the history entry and the result."""

    def test_a_missing_window_is_dropped_and_named(self):
        previous = {"all": {"threshold": 1.0}, "prompt": {"threshold": 2.0}, "last_user": {"threshold": 3.0}}
        rescored = {"all": {"threshold": 1.5}, "prompt": {"threshold": 2.5}}
        table, dropped = probe_monitor_run.rescored_window_table("pm_x", previous, rescored)
        assert table == rescored and dropped == ["last_user"]

    def test_nothing_rescored_is_none_so_the_arm_refuses(self):
        assert probe_monitor_run.rescored_window_table("pm_x", {"all": {}}, None) == (None, [])
        assert probe_monitor_run.rescored_window_table("pm_x", {"all": {}}, {}) == (None, [])

    @staticmethod
    def _arm(monkeypatch, tmp_path, previous, *, propose, stored_negatives=None, loaded=None):
        from src.models.probe_monitor import ProbeMonitor
        from src.services import probe_recalibration

        probe = SimpleNamespace(
            id="pm_x", run_id="pmr_x", calibration_dataset_id="pmd_cal", length_bands=None,
            calibration_lengths_path=None, threshold_source="calibration_set",
            window_decisions=dict(previous), calibration_scores_path=stored_negatives,
            # The served render with its start-of-text record (2026-10-08); the gate is tested alone.
            render_form={"generation_prompt": True, "add_special_tokens": False, "bos_handling": {"template_wrote_bos": True, "tokenizer_added_bos": False, "bos_count": 1}},
        )
        run = SimpleNamespace(id="pmr_x", calibration_dataset_id="pmd_cal")
        db = MagicMock()
        db.query.side_effect = lambda model: MagicMock(**{
            "filter.return_value.first.return_value": probe if model is ProbeMonitor else run,
            "filter.return_value.all.return_value": [],
        })
        context = SimpleNamespace(scope="all", target_fpr=0.01, artifact_dir=tmp_path)
        monkeypatch.setattr(probe_monitor_run, "context_from_row", lambda row: context)
        monkeypatch.setattr(probe_monitor_run, "probe_precision", lambda db, run: {"refusal": None})
        def load(run):
            if loaded is not None:
                loaded.append(run)
            return (None, None, "")

        monkeypatch.setattr(probe_monitor_run, "_load_model_for_run", load)
        monkeypatch.setattr(probe_monitor_run, "load_probe", lambda db, pid: (None, None))
        monkeypatch.setattr(probe_monitor_run, "_prepare_calibration_rows", lambda *a: ([], []))
        # Only `all` could be placed; every other window came back empty.
        monkeypatch.setattr(
            probe_monitor_run, "_score_calibration_group",
            lambda *a, **k: [{"all": probe_monitor_run.CalibrationScores(
                negatives=[float(v) for v in range(200)], n_rows=200, lengths=[5] * 200)}],
        )
        seen = {}

        def fake_propose(p, evaluations, *, target_fpr, **kw):
            seen["table"] = dict(p.window_decisions)
            return propose(p)

        def fake_apply(db_, p, proposal, *, reason="", seed_windows=None):
            seen["reason"] = reason
            seen["seed_windows"] = seed_windows
            db_.commit()
            return {"committed": True}

        monkeypatch.setattr(probe_recalibration, "propose", fake_propose)
        monkeypatch.setattr(probe_recalibration, "apply", fake_apply)
        return probe, db, seen

    def test_the_real_arm_drops_the_stale_window_and_records_it(self, monkeypatch, tmp_path):
        previous = {
            "all": {"threshold": 1.0, "scores_path": "a.npy"},
            "last_user": {"threshold": 9.0, "scores_path": "stale.npy"},  # cut over the old span
        }
        probe, db, seen = self._arm(monkeypatch, tmp_path, previous, propose=lambda p: {})
        result = probe_monitor_run.recut_probe_windows_on_gpu(db, "pm_x", target_fpr=0.05)
        assert set(seen["table"]) == {"all"}, "propose saw the stale window"
        assert result["dropped_windows"] == ["last_user"]
        assert "last_user" in seen["reason"]
        # Review round 3 (L1): revision 1 is seeded from the table that was SERVED.
        assert set(seen["seed_windows"]) == {"all", "last_user"}

    def test_a_refused_move_writes_nothing(self, monkeypatch, tmp_path):
        """Review round 2 (L-D): the re-scored table was committed at the run's target before
        `propose` ran at the new one, so a refusal left half a re-cut on the row."""
        from src.services.probe_recalibration import RecalibrationRefused

        previous = {"all": {"threshold": 1.0, "scores_path": "a.npy"}}

        def refuse(p):
            raise RecalibrationRefused("target_fpr_unaffordable", "too tight")

        probe, db, _seen = self._arm(monkeypatch, tmp_path, previous, propose=refuse)
        with pytest.raises(RecalibrationRefused):
            probe_monitor_run.recut_probe_windows_on_gpu(db, "pm_x", target_fpr=0.0001)
        assert probe.window_decisions == previous, "the re-scored table survived the refusal"
        db.commit.assert_not_called()
        db.rollback.assert_called_once()


class TestARefusedRecutLeavesDiskAndRowAgreeing:
    """Review round 3 (M2). The re-score overwrites the arrays under fixed names before `propose`
    runs, so a rolled-back row pointed at bars cut from arrays that were no longer on disk."""

    def test_a_refusal_restores_the_arrays_it_overwrote(self, monkeypatch, tmp_path):
        import numpy as np

        from src.services.probe_recalibration import RecalibrationRefused

        old = tmp_path / "pm_x__calibration_negatives_window_all.npy"
        np.save(old, np.asarray([7.0, 8.0], dtype=np.float32))
        previous = {"all": {"threshold": 1.0, "scores_path": str(old)}}

        def refuse(p):
            raise RecalibrationRefused("stale_derived_bars", "refused after the re-score")

        probe, db, _ = TestTheRecutArmDropsABarItCouldNotPlace._arm(
            monkeypatch, tmp_path, previous, propose=refuse
        )
        with pytest.raises(RecalibrationRefused):
            probe_monitor_run.recut_probe_windows_on_gpu(db, "pm_x", target_fpr=0.05)
        assert np.load(old).tolist() == [7.0, 8.0], "the re-score's array survived the refusal"
        names = sorted(p.name for p in tmp_path.iterdir())
        assert names == [old.name], f"files the re-score created were left behind: {names}"

    def test_an_unaffordable_target_is_refused_before_the_model_loads(self, monkeypatch, tmp_path):
        import numpy as np

        from src.services.probe_recalibration import RecalibrationRefused

        stored = tmp_path / "pm_x__calibration_negatives.npy"
        np.save(stored, np.zeros(50, dtype=np.float32))
        loaded = []
        _probe, db, _ = TestTheRecutArmDropsABarItCouldNotPlace._arm(
            monkeypatch, tmp_path, {"all": {}}, propose=lambda p: {},
            stored_negatives=str(stored), loaded=loaded,
        )
        with pytest.raises(RecalibrationRefused) as exc:
            probe_monitor_run.recut_probe_windows_on_gpu(db, "pm_x", target_fpr=0.001)
        assert exc.value.code == "target_fpr_unaffordable"
        assert loaded == [], "the model was loaded for a target that could never be placed"
