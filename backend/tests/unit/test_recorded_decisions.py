"""`probe_breakdown.recorded_decisions` — the stored probe turned into the deployed decision (review R1, 040 S1-S2).

The breakdown, the success report and the recall selection each built `decide_many`'s arguments inline.
Six caller mutations survived every test (the window entry dropped, or the run's internal scope passed for
the contract scope), because every fixture was a run scoped `all` with probe bands, where neither argument
changes a bar. These fixtures are scoped `last_assistant` (window `response`, contract scope `response`),
where both do, and every bar differs: window bands 0.65/0.75, window bar 0.7, probe bands 0.4/0.6, global 0.5.
"""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from src.services import detector_success_report, probe_breakdown, probe_monitor_run
from src.services.probe_breakdown import recorded_decisions

PB = [{"min_tokens": 0, "max_tokens": 100, "threshold": 0.4}, {"min_tokens": 101, "max_tokens": None, "threshold": 0.6}]
WB = [{"min_tokens": 0, "max_tokens": 50, "threshold": 0.65}, {"min_tokens": 51, "max_tokens": None, "threshold": 0.75}]
N = np.array([10, 60, 500])


def probe(window_decisions):
    return SimpleNamespace(threshold=0.5, length_bands=PB, window_decisions=window_decisions)


def test_the_windows_own_bands_apply_over_its_window():
    d = recorded_decisions(probe({"response": {"threshold": 0.7, "length_bands": WB}}), "last_assistant", N)
    assert d.thresholds.tolist() == [0.65, 0.75, 0.75]
    assert d.rule_counts()["window_band"] == 3


def test_the_probe_bands_apply_over_the_contract_scope_window():
    """No window entry: the window IS the probe's exported scope (`response`), so its bands apply. Passing
    the internal scope (`last_assistant`) instead reads it as another window and judges at the global bar."""
    d = recorded_decisions(probe(None), "last_assistant", N)
    assert d.thresholds.tolist() == [0.4, 0.4, 0.6]
    assert d.rule_counts()["probe_band"] == 3


def test_a_run_scope_with_no_window_reads_the_probe_bands():
    d = recorded_decisions(probe({"response": {"threshold": 0.7, "length_bands": WB}}), "assistant", N)
    assert d.thresholds.tolist() == [0.4, 0.4, 0.6]


@pytest.mark.parametrize("function", [
    probe_breakdown.compute, detector_success_report._figures, probe_monitor_run.recall_at_decision,
])
def test_every_reader_of_stored_rows_decides_through_it(function):
    tree = ast.parse(inspect.getsource(function).lstrip())
    names = [getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert names.count("recorded_decisions") == 1
    assert "decide_many" not in names and "deployed_decision" not in names


class _Called(Exception):
    pass


@pytest.mark.parametrize("scope", ["last_assistant", "assistant", "all"])
def test_each_reader_passes_its_runs_scope_and_its_rows(monkeypatch, scope):
    """The PAYLOAD of each call, not only its presence: the run's own scope and the rows' `n_scored`. A caller
    passing a constant scope keeps the call and loses the window (the AST test above cannot see that)."""
    from src.services import probe_evaluation_scores

    n_scored = np.array([7, 8, 9])
    calls = []

    def spy(probe, run_scope, rows):
        calls.append((probe, run_scope, rows))
        raise _Called

    loaded = SimpleNamespace(arrays={"score": np.zeros(3), "position": np.arange(3), "label": np.ones(3),
                                     "n_scored": n_scored})
    monkeypatch.setattr(probe_evaluation_scores, "load_evaluation_arrays", lambda *a, **k: loaded)
    monkeypatch.setattr(probe_breakdown, "recorded_decisions", spy)
    monkeypatch.setattr(detector_success_report, "recorded_decisions", spy)
    p = probe(None)
    run = SimpleNamespace(config={"scope": scope})
    evaluation = SimpleNamespace(id="pme_x", positions_path="/x")
    with pytest.raises(_Called):
        detector_success_report._figures(p, run, evaluation, SimpleNamespace(), None, None)
    with pytest.raises(_Called):
        probe_monitor_run.recall_at_decision(p, run, evaluation)
    assert len(calls) == 2
    for got_probe, got_scope, rows in calls:
        assert got_probe is p and got_scope == scope and rows is n_scored


def test_the_contract_scope_is_the_exporters(monkeypatch):
    """`contract_scope_of` calls the definition builder's mapping rather than copying it."""
    from src.services import probe_definition_builder

    for scope in ("all", "user", "assistant", "last_assistant"):
        assert probe_breakdown.contract_scope_of(scope) == probe_definition_builder._contract_scope(scope)
    assert probe_breakdown.contract_scope_of("content") == "content"      # inexpressible: its own name
    assert probe_breakdown.contract_scope_of(None) == "all"
    monkeypatch.setattr(probe_definition_builder, "_contract_scope", lambda scope: "moved")
    assert probe_breakdown.contract_scope_of("user") == "moved"
