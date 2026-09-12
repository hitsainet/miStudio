"""The two histogram arrays Neuronpedia receives must be PARALLEL.

`np.histogram` returns `n_bins + 1` edges and `n_bins` counts, and the push sent
both straight through as `freq_hist_data_bar_values` / `freq_hist_data_bar_heights`.
Neuronpedia's hover template maps over the VALUES and indexes the HEIGHTS at the
same position, so the extra edge read one past the end:

    TypeError: Cannot read properties of undefined (reading 'toLocaleString')

That killed every feature page of every pushed model — 129,788 rows on
lfm2.5-1.2b-instruct — and the model page with them, because its Browse pane
renders a feature preview inline.
"""

import numpy as np
import pytest

from src.services.neuronpedia_local_service import bin_edges_to_bar_values


def test_the_off_by_one_is_removed():
    """The real shape: 51 edges, 50 counts."""
    edges = np.linspace(0.0, 5.0, 51).tolist()
    bars = bin_edges_to_bar_values(edges, 50)

    assert len(bars) == 50, (
        "values and heights are consumed as parallel arrays; a length mismatch "
        "is the crash this function exists to prevent"
    )


def test_the_bar_value_is_the_CENTRE_of_its_bin():
    """Not `edges[:-1]`.

    Neuronpedia renders the hover range as `value ± spacing/2`, so it treats the
    entry as the middle of the bar. Left edges are the right LENGTH and still
    mislabel every bar by half a bin — a fix that passes the length test and
    stays wrong.
    """
    edges = [0.0, 2.0, 4.0, 6.0]
    bars = bin_edges_to_bar_values(edges, 3)

    assert bars == [1.0, 3.0, 5.0]
    assert bars != edges[:-1], "left edges, not centres"


def test_reconstructing_the_range_recovers_the_original_bin():
    """The property the consumer actually relies on, computed the way it does.

    Neuronpedia derives `low = v - s/2`, `high = v + s/2` where `s` is the gap to
    the neighbouring value. On uniform bins that must return the true edges.
    """
    edges = np.linspace(0.0, 10.0, 11).tolist()   # 10 bins of width 1.0
    bars = bin_edges_to_bar_values(edges, 10)

    for i, v in enumerate(bars):
        spacing = bars[i + 1] - bars[i] if i < len(bars) - 1 else bars[i] - bars[i - 1]
        low, high = v - spacing / 2, v + spacing / 2
        assert low == pytest.approx(edges[i]), f"bar {i} low edge"
        assert high == pytest.approx(edges[i + 1]), f"bar {i} high edge"


def test_log_spaced_bins_are_handled_too():
    """The real data is log-spaced — the service defaults to `log_scale=True`."""
    edges = np.logspace(0.0, 1.0, 51).tolist()
    bars = bin_edges_to_bar_values(edges, 50)

    assert len(bars) == 50
    assert all(bars[i] < bars[i + 1] for i in range(len(bars) - 1)), "monotonic"
    assert edges[0] < bars[0] < edges[1], "first centre lies inside its bin"


def test_already_parallel_input_is_left_alone():
    """Idempotent — a re-push of corrected data must not shrink it again."""
    values = [1.0, 2.0, 3.0]
    assert bin_edges_to_bar_values(values, 3) == values


@pytest.mark.parametrize("edges,n", [([], 50), ([1.0, 2.0], 0), ([], 0)])
def test_empty_input_yields_empty_output(edges, n):
    """A feature with no activations must not raise inside a push."""
    assert bin_edges_to_bar_values(edges, n) == []


def test_an_unmodelled_mismatch_degrades_instead_of_raising(caplog):
    """A wrong histogram must not fail a push carrying good labels.

    It still truncates to the shorter array, because emitting the mismatch
    unchanged would reproduce the original crash.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        bars = bin_edges_to_bar_values([1.0, 2.0, 3.0, 4.0, 5.0], 2)

    assert len(bars) == 2
    assert any("disagree" in r.getMessage() for r in caplog.records), (
        "an unmodelled shape was silently reshaped; the next reader has no clue"
    )


def test_both_push_sites_route_through_the_helper():
    """Reachability: the computed path AND the DB-fallback path.

    The bug existed at TWO assignment sites. Fixing one and reading the other as
    'the same thing' is how half a fix ships — this repo's recorded
    'fixed one representative, never generalized' anti-pattern.
    """
    import inspect

    from src.services import neuronpedia_local_service as mod

    src = inspect.getsource(mod)
    assert src.count("bin_edges_to_bar_values(") >= 3, (
        "expected the definition plus BOTH call sites; one of the two push "
        "paths still sends raw bin edges"
    )
    assert "freq_hist_values = hist_result.bin_edges" not in src
    assert 'freq_hist_values = hist_data.get("bin_edges", [])' not in src
