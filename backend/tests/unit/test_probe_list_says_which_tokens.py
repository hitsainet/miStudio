"""The probe list says which tokens each probe reads (operator, 2026-10-04).

A run's nine probes are three LAYERS x three RULES; every one was trained on the run's scope and
carries a bar for EACH contract window. The list sent neither, so a tile could not show it, and the
nine read as "one probe per window per layer".
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.api.v1.endpoints import probe_monitors
from src.api.v1.endpoints.probe_monitors import summary_without_curve


def _probe(probe_id="pm_x", run_id="pmr_x", **extra):
    base = dict(
        id=probe_id, run_id=run_id, layer=11, rule="mean", rule_params={}, variant="dense",
        sae_id=None, sae_feature_indices=None, val_metrics={"val_auroc": 0.9}, selected=False,
        threshold=25.29, target_fpr=0.01, realised_fpr=0.01, threshold_source="calibration_set",
        streamable=True, rung=2, rung_reasons=[], definition_built_at=None,
        definition_sha256=None, definition_build=None, published=[],
        created_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
        window_decisions={
            "all": {"threshold": 25.29, "scope": "all"},
            "prompt": {"threshold": 14.69, "scope": "input"},
            "response": {"threshold": 25.04, "scope": "last_assistant"},
        },
        length_bands=[{}, {}, {}, {}],
    )
    base.update(extra)
    return SimpleNamespace(**base)


class TestTheSummary:
    def test_it_carries_each_windows_own_bar_and_the_band_count(self):
        summary = summary_without_curve(_probe(), "all")
        assert summary.scope == "all"
        assert summary.window_thresholds == {"all": 25.29, "prompt": 14.69, "response": 25.04}
        assert summary.length_band_count == 4

    def test_no_per_window_bar_is_an_empty_map_not_a_guess(self):
        summary = summary_without_curve(_probe(window_decisions=None, length_bands=None), "all")
        assert summary.window_thresholds == {} and summary.length_band_count == 0

    def test_a_gone_run_is_none_not_all(self):
        assert summary_without_curve(_probe()).scope is None


def _result(scalars=None, rows=None):
    result = MagicMock()
    result.scalars.return_value.all.return_value = scalars or []
    result.all.return_value = rows or []
    return result


def test_the_list_endpoint_sends_each_probes_run_scope():
    """Run for real against a stubbed session: two probes from two runs with different scopes."""
    probes = [_probe("pm_a", "pmr_a"), _probe("pm_b", "pmr_b")]
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[
        _result(scalars=probes),
        _result(rows=[("pmr_a", {"scope": "user"}), ("pmr_b", {})]),
    ])
    out = asyncio.run(probe_monitors.list_probes(run_id=None, selected_only=False, db=db))
    assert [s.scope for s in out] == ["user", "all"], "an unset scope must read as the run's default"
    assert out[0].window_thresholds["prompt"] == 14.69
    assert db.execute.await_count == 2
