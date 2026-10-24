"""Wilson's interval, moved unchanged from `jlens_causal` into `probe_monitor_metrics` (040-T6; FTASKS 7.1).

Hand-computed at 0, n and an interior value with z = 1.959963984540054 (two-sided 95%).
"""
from __future__ import annotations

import math

import pytest

from src.services import jlens_causal, probe_monitor_metrics
from src.services.probe_monitor_metrics import Z_95, wilson_interval


def test_no_trials_is_maximal_uncertainty():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_zero_of_twenty():
    # p = 0: centre = (z²/2n)/(1+z²/n); half = (z/(1+z²/n))·sqrt(z²/4n²)
    z, n = Z_95, 20
    denom = 1 + z * z / n
    hi = (z * z / (2 * n)) / denom + (z / denom) * math.sqrt(z * z / (4 * n * n))
    lo, got_hi = wilson_interval(0, 20)
    assert lo == 0.0 and got_hi == pytest.approx(hi, abs=1e-12) and got_hi == pytest.approx(0.161125, abs=1e-6)


def test_twenty_of_twenty():
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and lo == pytest.approx(1 - 0.161125, abs=1e-6)


def test_interior_value():
    # 31 of 40: p = 0.775 → [0.624969, 0.876839] (hand-computed with the formula above)
    lo, hi = wilson_interval(31, 40)
    assert lo == pytest.approx(0.624969, abs=1e-6) and hi == pytest.approx(0.876839, abs=1e-6)


def test_jlens_reads_the_same_function():
    assert jlens_causal.wilson_interval is probe_monitor_metrics.wilson_interval
    assert jlens_causal.Z_95 == Z_95
