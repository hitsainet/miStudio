"""MIS-E2E-080 — "converged" measured sample count, not stabilisation.

The fit accumulated a running mean and declared convergence when
`relative_change(previous, accumulated) < delta`. The increment of a running
mean is O(sigma/n): it shrinks because the DENOMINATOR GROWS, not because
successive estimates of J agree. The stop point was therefore n ~ sigma/delta —
directly proportional to per-prompt variance, and reachable by any process with
bounded increments whether it has converged or not.

Reproduced here, and it matches the reviewer's simulation almost exactly
(518 / 1040 / 1988 against their 518 / 1050 / 2030) — numbers that BRACKET the
two real recorded fits the docs call "paper-aligned converged lenses" (gemma
634, LFM2 1097). Those fits are fully consistent with the criterion having
measured nothing but each model's per-prompt spread.

The word "converged" does evidential work in the artifact, the docs and the
gate decision. It now refers to split-half agreement: two independent
accumulators over alternating prompts, which asks "would a different half of
this corpus have produced the same lens?"
"""

import inspect

import pytest
import torch

from src.ml.jlens_fitter import (
    CONVERGENCE_CRITERION,
    DEFAULT_CONVERGENCE_DELTA,
    relative_change,
)

PATIENCE = 2


def _stop_point(noise: float, criterion: str, delta: float, cap: int = 6000):
    """Run both criteria against a stationary process plus noise."""
    g = torch.Generator().manual_seed(0)
    truth = torch.randn(16, 16, generator=g)
    acc, prev, a, b = {}, {}, {}, {}
    na = nb = 0
    stable = 0
    for n in range(1, cap + 1):
        x = truth + torch.randn(16, 16, generator=g) * noise
        acc[0] = x.clone() if 0 not in acc else acc[0] + (x - acc[0]) / n
        if n % 2:
            na += 1
            t, k = a, na
        else:
            nb += 1
            t, k = b, nb
        t[0] = x.clone() if 0 not in t else t[0] + (x - t[0]) / k

        if n >= 10:
            if criterion == "increment":
                d = relative_change(prev, acc)
                prev = {0: acc[0].clone()}
            else:
                d = relative_change(a, b) if na and nb else float("inf")
            stable = stable + 1 if d < delta else 0
            if stable >= PATIENCE:
                return n
    return None


def test_the_old_criterion_stops_in_proportion_to_noise():
    """The defect, demonstrated. This is why the old test could not be trusted.

    Stop points scale linearly with sigma, so the criterion is reporting the
    per-prompt spread and calling it convergence.
    """
    stops = [_stop_point(s, "increment", 1e-3) for s in (0.5, 1.0, 2.0)]
    assert all(s is not None for s in stops)
    # Linear in sigma: doubling the noise roughly doubles the stop point.
    assert 1.6 < stops[1] / stops[0] < 2.4, stops
    assert 1.6 < stops[2] / stops[1] < 2.4, stops


def test_the_new_criterion_is_not_the_running_means_own_step():
    """Split-half must be a genuinely different measurement.

    At the OLD threshold it does not converge at all within a realistic corpus
    — which is the point: the two criteria measure different quantities and the
    threshold cannot be carried across.
    """
    assert _stop_point(1.0, "splithalf", 1e-3, cap=3000) is None


def test_the_default_threshold_converges_on_a_realistic_corpus():
    """Calibration, pinned. 0.1 lands in the 100–2000 prompt range the real
    fits used (gemma 634, LFM2 1097); 0.01 never converges."""
    assert DEFAULT_CONVERGENCE_DELTA == 0.1
    n = _stop_point(1.0, "splithalf", DEFAULT_CONVERGENCE_DELTA)
    assert n is not None and 100 < n < 3000, n


def test_the_fit_loop_compares_two_independent_halves():
    from src.ml.jlens_fitter import JacobianFitter

    # Parse the CALLS, not the text. The replacement comment quotes the old
    # expression in order to explain why it was wrong, and a substring check
    # cannot tell the claim from the correction. (Fourth occurrence of this
    # trap in this remediation — the rule is now: parse, or strip the prose.)
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(JacobianFitter.fit)))
    calls = [
        tuple(getattr(a, "id", None) for a in n.args)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "relative_change"
    ]
    assert calls == [("half_a", "half_b")], (
        f"convergence must compare two independent estimates; found "
        f"relative_change{calls}"
    )


def test_the_published_lens_is_the_whole_corpus_not_one_half():
    """The split is the instrument, not the estimate.

    Publishing half the corpus would halve the data behind every artifact — a
    silent regression that no convergence test would notice.
    """
    from src.ml.jlens_fitter import JacobianFitter

    src = inspect.getsource(JacobianFitter.fit)
    assert "(a * count_a + b * count_b) / total" in src


def test_the_artifact_records_which_criterion_it_passed():
    """Absent must stay meaningful for lenses fitted before this change.

    An artifact with no criterion was fitted under the running-mean test. If
    reads defaulted the field to the new value, every old lens would claim a
    property it was never tested for.
    """
    from src.workers import jlens_fit_tasks

    src = inspect.getsource(jlens_fit_tasks)
    assert '"convergence_criterion": result.convergence_criterion' in src
    assert CONVERGENCE_CRITERION == "split_half_agreement"


def test_halves_are_interleaved_not_sequential():
    """A corpus ordered by topic would otherwise guarantee disagreement."""
    from src.ml.jlens_fitter import JacobianFitter

    src = inspect.getsource(JacobianFitter.fit)
    assert "if seen % 2:" in src, (
        "the halves must alternate; a sequential split makes the criterion a "
        "test of corpus ordering"
    )
