"""OSD-23 — the fit cost estimate must not be an order of magnitude low.

The old model predicted **0.40 h for a fit that took 6.83 h** — 17x low, not the
~7x the tracker recorded — and its peak was `d_model^2 x 2 x n_layers`, i.e.
118 MB against 16.6 GB measured, because the resident weights were not modelled at
all. An estimate like that does not merely mislead: it made a 68-hour job look
like 9.

It now derives iterations from the FITTER'S OWN `narrowed_chunk`, so the estimate
cannot drift from the chunking it describes, and is calibrated on the one
end-to-end fit anyone has measured. Exact at that point by construction; an
extrapolation elsewhere, with exponents from the algorithm rather than guessed.

A NOTE ON THE RECORDED NUMBERS. The tracker says "~41 s/prompt → ~5.5 h for 600
prompts", but 41 x 600 = 6.83 h. The per-prompt rate is the directly measured
quantity, so it is the one used; the 5.5 h figure is not reproducible from it.
"""
import ast
import inspect
import math

import pytest

from src.services import jlens_watchlist as watchlist
from src.services.jlens_watchlist import FIT_REFERENCE, OperationClass, estimate_cost

REF = FIT_REFERENCE
MEASURED_SECONDS = REF["measured_seconds_per_prompt"] * REF["n_prompts"]


def _fit(**overrides):
    kwargs = dict(
        d_model=REF["d_model"],
        n_layers=REF["n_layers"],
        n_positions=64,
        n_prompts=REF["n_prompts"],
        weights_bytes=REF["measured_weights_bytes"],
    )
    kwargs.update(overrides)
    return estimate_cost(OperationClass.ARTIFACT_CONSTRUCTION, **kwargs)


class TestItReproducesTheOneMeasuredFit:

    def test_the_time_matches_what_was_measured(self):
        est = _fit().order_of_magnitude_seconds
        assert 0.8 * MEASURED_SECONDS <= est <= 1.25 * MEASURED_SECONDS, (
            f"{est/3600:.2f}h estimated against {MEASURED_SECONDS/3600:.2f}h measured"
        )

    def test_the_peak_matches_what_was_measured(self):
        est = _fit().order_of_magnitude_peak_bytes
        measured = REF["measured_peak_bytes"]
        assert 0.9 * measured <= est <= 1.1 * measured, (
            f"{est/1024**3:.2f} GiB estimated against {measured/1024**3:.2f} GiB measured"
        )

    def test_it_is_no_longer_an_order_of_magnitude_low(self):
        """The specific failure: the old formula, evaluated at the same point."""
        old_seconds = max(
            60.0,
            (REF["n_prompts"] * REF["n_layers"] * max(1, REF["d_model"] // 128)) / 50.0,
        )
        assert old_seconds < MEASURED_SECONDS / 10, (
            "sanity: the old model really was more than 10x low here"
        )
        assert _fit().order_of_magnitude_seconds > 5 * old_seconds


class TestTheShapeFollowsTheAlgorithm:

    def test_time_is_linear_in_prompts(self):
        one, ten = _fit(n_prompts=100), _fit(n_prompts=1000)
        ratio = ten.order_of_magnitude_seconds / one.order_of_magnitude_seconds
        assert 9.5 < ratio < 10.5

    def test_time_grows_with_layer_count(self):
        few, many = _fit(n_layers=4), _fit(n_layers=16)
        assert many.order_of_magnitude_seconds > 3.5 * few.order_of_magnitude_seconds

    def test_peak_does_not_grow_with_layers_the_naive_way(self):
        """Only the per-layer accumulators scale; the backward stays CAPPED.

        The naive model grew the whole peak with layer count. Here the delta from
        4 to 8 layers must be exactly four accumulator blocks — nothing else.
        """
        block = watchlist._ACCUMULATOR_COPIES_PER_LAYER * REF["d_model"] ** 2 * 4
        four = _fit(n_layers=4).order_of_magnitude_peak_bytes
        eight = _fit(n_layers=8).order_of_magnitude_peak_bytes
        assert eight - four == 4 * block

    def test_a_long_prompt_narrows_the_chunk_and_costs_more(self):
        """When the cap binds, iterations rise — the fitter's own behaviour."""
        short, long = _fit(n_positions=64), _fit(n_positions=4096)
        assert long.order_of_magnitude_seconds > short.order_of_magnitude_seconds

    def test_the_iteration_count_is_the_fitters_own(self):
        from src.ml.jlens_fitter import DEFAULT_CHUNK, narrowed_chunk

        chunk = narrowed_chunk(DEFAULT_CHUNK, 64, REF["d_model"], REF["n_layers"])
        expected = math.ceil(REF["d_model"] / chunk)
        assert f"x {expected} batched-backward iterations" in _fit().basis


class TestItSaysWhatItAssumed:

    def test_omitting_the_weights_says_so(self):
        est = _fit(weights_bytes=0)
        assert "EXCLUDES the model" in est.basis
        assert est.order_of_magnitude_peak_bytes < REF["measured_peak_bytes"] / 3, (
            "without the weights the peak must be visibly smaller, not quietly close"
        )

    def test_the_basis_names_the_calibration(self):
        assert "calibrated on the measured gemma-4-12B fit" in _fit().basis

    def test_the_reference_point_travels_with_the_constant(self):
        assert REF["recorded"].endswith("JLens_Fit_On_24GB_2026-09-05.md")
        assert REF["measured_seconds_per_prompt"] == 41.0


class TestItCannotDriftFromTheFitter:

    def test_the_estimate_CALLS_narrowed_chunk(self):
        """Not a restatement of the formula: walk the AST for the call."""
        tree = ast.parse(inspect.getsource(watchlist._fit_estimate))
        assert any(
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "narrowed_chunk"
            for node in ast.walk(tree)
        ), (
            "the estimate must call the fitter's narrowed_chunk, or the two can "
            "disagree about chunking and the estimate silently goes wrong again"
        )

    def test_it_uses_the_fitters_backward_cap(self):
        from src.ml.jlens_fitter import MAX_BACKWARD_BYTES

        bare = estimate_cost(
            OperationClass.ARTIFACT_CONSTRUCTION,
            d_model=8, n_layers=1, n_positions=1, n_prompts=1, weights_bytes=0,
        )
        block = watchlist._ACCUMULATOR_COPIES_PER_LAYER * 8 * 8 * 4
        assert bare.order_of_magnitude_peak_bytes == MAX_BACKWARD_BYTES + block


class TestTheOtherOperationsAreUntouched:

    @pytest.mark.parametrize("op", [
        OperationClass.READOUT,
        OperationClass.ANNOTATION_SWEEP,
        OperationClass.INTERVENTION_RUN,
        OperationClass.DECOMPOSITION,
    ])
    def test_they_still_estimate(self, op):
        est = estimate_cost(op, d_model=3840, n_layers=4, n_positions=64,
                            n_prompts=1, n_features=16384)
        assert est.order_of_magnitude_seconds > 0
        assert est.basis
