"""`deployed_decision` and `decide_many` (Feature 040 FR-15; FTID §7.1; FTASKS 7.2 – 7.3).

One fixture whose four bars ALL DIFFER — window bar 0.7, window bands 0.65/0.75, probe bands 0.4/0.6,
global 0.5 — so applying the wrong rule always gives a different number (the global fixture rule).
`score_verdict` is checked to be built on `deployed_decision` behaviourally (identical verdicts to the
pre-040 lookup on the existing fixtures) and by AST call.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from src.services import probe_monitor_run
from src.services.probe_monitor_metrics import (
    DECISION_RULES,
    band_for_length,
    decide_many,
    deployed_decision,
    threshold_for_length,
    usable_bands,
    window_is_calibrated,
)

PB = [{"min_tokens": 0, "max_tokens": 100, "threshold": 0.4}, {"min_tokens": 101, "max_tokens": None, "threshold": 0.6}]
WB = [{"min_tokens": 0, "max_tokens": 50, "threshold": 0.65}, {"min_tokens": 51, "max_tokens": None, "threshold": 0.75}]
ENTRY = {"threshold": 0.7, "length_bands": WB}


@pytest.mark.parametrize("window,entry,n,threshold,rule", [
    ("prompt", ENTRY, 10, 0.65, "window_band"),                                   # window bands
    ("prompt", {"threshold": 0.7, "length_bands": None}, 10, 0.7, "window_bar"),  # window bar
    (None, None, 10, 0.4, "probe_band"),                                          # probe bands (own scope)
    ("all", {"threshold": 0.7, "length_bands": None}, 500, 0.6, "probe_band"),     # all probe, own window
    ("prompt", None, 10, 0.5, "global"),                                          # another window: global
    ("prompt", {"threshold": None, "length_bands": WB}, 10, 0.5, "global"),        # bands need own bar
])
def test_each_rule(window, entry, n, threshold, rule):
    decision = deployed_decision(0.5, entry, PB, "all", window, n)
    assert (decision.threshold, decision.rule) == (threshold, rule)


def test_no_bar_is_none_not_an_error():
    decision = deployed_decision(None, None, None, "all", None, 10)
    assert (decision.threshold, decision.rule) == (None, "none")
    fires = decide_many(None, None, None, "all", None, np.array([5, 9])).fires(np.array([99.0, 99.0]))
    assert fires.tolist() == [False, False]


def test_a_non_all_probe_keeps_its_window_bar_over_its_bands():
    """The 2026-10-04 rule mirrored from miLLM: bands cut under another internal scope do not refine."""
    assert deployed_decision(0.5, {"threshold": 0.7}, PB, "prompt", "prompt", 10).rule == "window_bar"
    assert deployed_decision(0.5, None, PB, "prompt", "prompt", 10).rule == "probe_band"


def test_window_is_calibrated_only_over_its_own_scope():
    assert window_is_calibrated("all", None) and window_is_calibrated("all", "all")
    assert not window_is_calibrated("all", "prompt")


def test_decide_many_equals_the_row_rule_everywhere():
    rng = np.random.default_rng(7)
    lengths = rng.integers(1, 400, size=500)
    for window, entry in (("prompt", ENTRY), (None, None), ("all", {"threshold": 0.7}), ("prompt", None)):
        many = decide_many(0.5, entry, PB, "all", window, lengths)
        for n, t, code in zip(lengths.tolist(), many.thresholds.tolist(), many.rule_codes.tolist()):
            one = deployed_decision(0.5, entry, PB, "all", window, n)
            assert one.threshold == t and DECISION_RULES[code] == one.rule
        assert sum(many.rule_counts().values()) == len(lengths)


# ── score_verdict is re-expressed on the one decision function (7.3) ─────────────────────────


@pytest.mark.parametrize("bands", [None, [], PB, [{"min_tokens": 10, "max_tokens": 20, "threshold": 0.45}],
                                   [{"min_tokens": 0, "max_tokens": 100, "threshold": None,
                                     "threshold_source": "global"}]])
@pytest.mark.parametrize("n", [0, 5, 15, 100, 101, 1000])
@pytest.mark.parametrize("bar", [None, 0.5])
def test_score_verdict_is_unchanged(bands, n, bar):
    """The pre-040 lookup, recomputed here, gives the same verdict fields — over the table a server arms.

    Review R1 (040 S1-S2): the last two tables (a closed last band; a band with no threshold) are ones
    miLLM refuses to arm whole, so the deployed bar is the global one. Before the fix the offline verdict
    applied them as written, at bars miLLM never serves."""
    got = probe_monitor_run.score_verdict(bar, bands, 0.55, n)
    armed = usable_bands(bands)
    band = band_for_length(armed, n)
    threshold = threshold_for_length(armed, n, bar)
    assert got["threshold"] == threshold
    assert got["fires"] == (None if threshold is None else 0.55 >= threshold)
    assert (got["threshold_band"] is None) == (band is None)
    # Review R1: the rule that chose the bar is reported, and is the decision function's own.
    assert got["threshold_rule"] == deployed_decision(bar, None, bands, "own", None, n).rule
    assert got["threshold_rule"] == ("probe_band" if band is not None else ("global" if bar is not None else "none"))


@pytest.mark.parametrize("bands", [
    [{"min_tokens": 0, "max_tokens": 100, "threshold": None, "threshold_source": "global"},
     {"min_tokens": 101, "max_tokens": None, "threshold": 0.6}],                          # a null bar
    [{"min_tokens": 0, "max_tokens": 100, "threshold": 0.4}, {"min_tokens": 101, "max_tokens": 900, "threshold": 0.6}],
    [{"min_tokens": -1, "max_tokens": None, "threshold": 0.4}],                              # a negative bound
    [{"min_tokens": 0, "max_tokens": None, "threshold": True}],                              # a bool is not a bar
])
def test_a_table_miLLM_would_not_arm_decides_nothing(bands):
    """miLLM's `bands_from_block` drops such a table whole; the deployed bar is then the probe's own."""
    assert usable_bands(bands) == []
    for n in (10, 500):
        decision = deployed_decision(0.5, None, bands, "all", None, n)
        assert (decision.threshold, decision.rule, decision.band) == (0.5, "global", None)
    unusable_window = {"threshold": 0.7, "length_bands": bands}
    assert deployed_decision(0.5, unusable_window, PB, "all", "prompt", 10).rule == "window_bar"


def test_a_window_bar_that_is_not_a_number_is_no_bar():
    assert deployed_decision(0.5, {"threshold": True}, None, "all", "prompt", 10).rule == "global"


def test_score_verdict_calls_deployed_decision():
    tree = ast.parse(inspect.getsource(probe_monitor_run.score_verdict))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "deployed_decision"]
    assert len(calls) == 1
    names = {getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "threshold_for_length" not in names and "band_for_length" not in names
