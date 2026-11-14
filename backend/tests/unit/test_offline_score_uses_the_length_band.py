"""The offline score judges an input against the bar for ITS length, as the live monitor does.

⚠ FOUND FROM A PASTED SCREENSHOT, NOT A TEST. "Score this text" on `pm_f463a8a235ae` showed
`threshold 17.9802 · fires` for a ~50-token passage. That is the probe's GLOBAL bar. The probe has
a length table, and the band for 0-203 tokens is 12.806 — the bar miLLM's runtime would actually
apply. `score_one` returned `probe.threshold` for every input and never consulted `length_bands`.

That made the panel contradict the contract miStudio itself publishes, whose `LengthBand` docstring
opens "ONE CONSTANT THRESHOLD IS MISCALIBRATED AT EVERY LENGTH BUT THE ONE IT WAS CUT AT". And it
disagreed with the monitor in BOTH directions on that probe — too strict on short inputs (12.806 vs
17.980), too lenient on medium ones (26.602 vs 17.980) — so there was no consistent bias to reason
around.

⚠ AND `threshold_for_length`'s OWN DOCSTRING SAID IT WAS "the ONE place the lookup lives, so
miStudio's calibration, its evaluation and any consumer cannot disagree". The offline score simply
never called it. A single source of truth only helps the callers that use it.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from src.services.probe_monitor_metrics import band_for_length, threshold_for_length
from src.services.probe_monitor_run import score_verdict

#: `pm_f463a8a235ae`'s real table, read off the production row on 2026-10-03.
BANDS = [
    {"min_tokens": 0, "max_tokens": 203, "threshold": 12.806008338928223,
     "threshold_source": "band", "target_fpr": 0.01},
    {"min_tokens": 204, "max_tokens": 339, "threshold": 26.602256774902344,
     "threshold_source": "band", "target_fpr": 0.01},
    {"min_tokens": 340, "max_tokens": 518, "threshold": 16.48041534423828,
     "threshold_source": "band", "target_fpr": 0.01},
    {"min_tokens": 519, "max_tokens": None, "threshold": 20.311847686767578,
     "threshold_source": "band", "target_fpr": 0.01},
]
GLOBAL = 17.98017692565918


class TestTheBarIsTheBandsNotTheHeadline:
    def test_a_short_input_is_judged_against_the_FIRST_band(self):
        """The screenshot's case. 50 scored tokens falls in 0-203."""
        got = score_verdict(GLOBAL, BANDS, aggregate=28.9189, n_scored=50)
        assert got["threshold"] == pytest.approx(12.806008338928223)
        assert got["global_threshold"] == GLOBAL
        assert got["threshold_band"]["min_tokens"] == 0
        assert got["threshold_band"]["max_tokens"] == 203

    def test_the_panel_now_FIRES_where_it_used_to_stay_silent(self):
        """⚠ THE CASE THAT MATTERS. A short input scoring 15.0 is above its band (12.806) and
        below the headline (17.980). The old panel said "below threshold"; miLLM fires."""
        got = score_verdict(GLOBAL, BANDS, aggregate=15.0, n_scored=50)
        assert got["fires"] is True, (
            "a short input above its own band's bar was reported silent — judged against the "
            "global threshold the contract says is miscalibrated at this length"
        )

    def test_and_STAYS_SILENT_where_it_used_to_fire(self):
        """The other direction. 22.0 at 250 tokens clears the headline (17.980) and not its band
        (26.602). The old panel said "fires"; miLLM stays silent."""
        got = score_verdict(GLOBAL, BANDS, aggregate=22.0, n_scored=250)
        assert got["threshold"] == pytest.approx(26.602256774902344)
        assert got["fires"] is False

    def test_the_open_ended_last_band_catches_a_very_long_input(self):
        got = score_verdict(GLOBAL, BANDS, aggregate=1.0, n_scored=5000)
        assert got["threshold"] == pytest.approx(20.311847686767578)
        assert got["threshold_band"]["max_tokens"] is None

    @pytest.mark.parametrize("n, expected", [(203, 12.806008338928223), (204, 26.602256774902344),
                                              (339, 26.602256774902344), (340, 16.48041534423828)])
    def test_the_boundaries_are_inclusive_on_both_ends(self, n, expected):
        """A band's `max_tokens` is the last length IN it — the same rule the runtime uses, so a
        boundary input lands in the same band on both sides."""
        assert score_verdict(GLOBAL, BANDS, 1.0, n)["threshold"] == pytest.approx(expected)


class TestWithoutATableNothingChanges:
    def test_no_table_falls_back_to_the_global_bar(self):
        """Every probe without bands — most of what has ever been trained here — must behave
        exactly as before."""
        got = score_verdict(GLOBAL, None, aggregate=18.0, n_scored=50)
        assert got["threshold"] == GLOBAL
        assert got["threshold_band"] is None
        assert got["fires"] is True

    def test_an_empty_table_is_the_same_as_none(self):
        assert score_verdict(GLOBAL, [], 18.0, 50)["threshold_band"] is None

    def test_no_threshold_at_all_means_fires_is_NONE_not_false(self):
        """A probe with no calibration has said nothing, not "no"."""
        got = score_verdict(None, None, aggregate=99.0, n_scored=50)
        assert got["fires"] is None
        assert got["threshold"] is None


class TestAThinBandSaysItInheritedTheGlobal:
    def test_a_global_band_reports_its_source(self):
        """A band with too few negatives to cut carries `threshold_source: global` and the global
        bar. The panel must be able to say so rather than present it as a measured band bar."""
        bands = [{"min_tokens": 0, "max_tokens": None, "threshold": GLOBAL,
                  "threshold_source": "global"}]
        got = score_verdict(GLOBAL, bands, 1.0, 50)
        assert got["threshold_band"]["threshold_source"] == "global"


class TestBandForLengthIsTheOnePlaceTheDecisionLives:
    def test_threshold_for_length_is_BUILT_ON_band_for_length(self):
        """Naming the band and applying it must never disagree, so the threshold lookup must
        call the band lookup rather than repeat its loop."""
        tree = ast.parse(inspect.getsource(threshold_for_length))
        calls = {n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "band_for_length" in calls

    def test_band_for_length_agrees_with_threshold_for_length_everywhere(self):
        for n in range(0, 700, 7):
            band = band_for_length(BANDS, n)
            assert threshold_for_length(BANDS, n, GLOBAL) == band["threshold"]


class TestScoreOneUsesTheVerdict:
    """⚠ THE CALL, BY AST — `score_one` needs a model, so this is how its wiring is pinned."""

    @staticmethod
    def _calls():
        from src.services import probe_monitor_run

        tree = ast.parse(inspect.getsource(probe_monitor_run.score_one))
        return [n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "score_verdict"]

    def test_score_one_calls_score_verdict(self):
        assert self._calls(), "score_one does not call score_verdict, so the band is never applied"

    def test_it_passes_the_probes_OWN_length_bands_and_n_scored(self):
        """The payload, not just the call: passing `None` for the bands would satisfy a
        was-called assertion and restore the defect exactly."""
        call = self._calls()[0]
        src = ast.unparse(call)
        assert "probe.length_bands" in src
        assert "n_scored" in src
        assert "probe.threshold" in src

    def test_score_one_no_longer_reports_the_bare_global_bar(self):
        from src.services import probe_monitor_run

        src = inspect.getsource(probe_monitor_run.score_one)
        assert '"threshold": probe.threshold' not in src
