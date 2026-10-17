"""The probe verdict fires AT the threshold: `score >= threshold` (Feature 034, P-03).

The boundary is suite-wide: miLLM decides live and offline verdicts with `value >= threshold`
(`millm/services/probe_runtime.py:693` at miLLM `6d7212b`, read 2026-10-16), and miStudio's
offline `score_verdict` must agree, or one input reads "fires" in one product and "silent" in the
other. `millm_score_probes` returns miLLM's verdict verbatim and never recomputes it (asserted in
`test_millm_inference_tools.py`).

Control (recorded 2026-10-16, task 5.3): `aggregate >= threshold` → `aggregate > threshold` in
`score_verdict` → `test_a_score_EQUAL_to_the_threshold_fires` red; restored, diff clean.
"""

from src.services.probe_monitor_run import score_verdict


def test_a_score_EQUAL_to_the_threshold_fires():
    assert score_verdict(0.5, None, aggregate=0.5, n_scored=10)["fires"] is True


def test_just_below_does_not_fire():
    assert score_verdict(0.5, None, aggregate=0.4999999, n_scored=10)["fires"] is False


def test_no_threshold_is_SILENCE_not_false():
    assert score_verdict(None, None, aggregate=99.0, n_scored=10)["fires"] is None
