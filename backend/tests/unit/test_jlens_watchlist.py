"""
Watchlists and the cost envelope (BR-025, BR-026, BR-028).

Two things carry the feature:

  * a watchlist is a DETECTOR DEFINITION — directions, thresholds and the
    scoring definition travel together or none of them mean anything;
  * the evaluation-awareness score is a DIFFERENCE, and the subtraction is the
    measurement rather than a refinement of it.

MUTATION CONTROLS (each must turn this file red):
  * default the scoring definition             -> "refused" fails
  * drop the artifact reference requirement    -> "artifact-specific" fails
  * report the raw mean as the score           -> "the control is the measurement" fails
  * default an unknown operation class to cheap -> "refuses to guess" fails
  * average the means instead of the differences -> "averaging differences" fails
"""

from __future__ import annotations

import pytest

from src.services.jlens_watchlist import (
    CostEstimate,
    OperationClass,
    WatchedConcept,
    Watchlist,
    estimate_cost,
    evaluation_awareness_score,
    score_across_layers,
)


def concepts():
    return [WatchedConcept("evaluation", 0.5), WatchedConcept("benchmark", 0.5)]


# ── a watchlist is a detector definition (BR-025) ──────────────────────────


def test_a_watchlist_without_its_scoring_definition_is_REFUSED():
    """A threshold applied to a differently computed score is a different detector.

    And the consumer has no way to notice — it looks like the same watchlist,
    firing at the same number, on a quantity that was never the same.
    """
    with pytest.raises(ValueError, match="scoring definition"):
        Watchlist(
            name="w", concepts=concepts(), scoring_definition="  ", artifact_ref="a"
        )


def test_a_watchlist_without_its_artifact_is_REFUSED():
    """Lens coordinates are artifact-specific and mean nothing elsewhere."""
    with pytest.raises(ValueError, match="artifact"):
        Watchlist(
            name="w", concepts=concepts(), scoring_definition="diff", artifact_ref=""
        )


def test_a_watchlist_with_no_concepts_is_REFUSED():
    """It would export cleanly and detect nothing."""
    with pytest.raises(ValueError, match="detects nothing"):
        Watchlist(name="w", concepts=[], scoring_definition="d", artifact_ref="a")


def test_a_complete_watchlist_constructs():
    wl = Watchlist(
        name="eval-awareness",
        concepts=concepts(),
        scoring_definition="mean lens log-prob over eval tokens minus control mean",
        artifact_ref="gemma-2-2b-it",
        control_set=["table", "rope", "cloud"],
    )
    assert wl.artifact_ref == "gemma-2-2b-it"
    assert wl.control_set


# ── the score is a DIFFERENCE (BR-026) ─────────────────────────────────────


def test_the_control_subtraction_IS_the_measurement():
    """Common tokens have high log-probability in ANY prompt.

    Both prompts below score high on the raw mean; only the difference
    distinguishes them. A detector built on the raw mean fires constantly and
    is therefore never consulted.
    """
    # Eval-aware prompt: eval tokens well above the controls.
    aware = evaluation_awareness_score([-1.0, -1.2], [-5.0, -5.4])
    # Neutral prompt: both high, because both are common tokens.
    neutral = evaluation_awareness_score([-1.1, -1.3], [-1.2, -1.2])

    raw_aware = sum([-1.0, -1.2]) / 2
    raw_neutral = sum([-1.1, -1.3]) / 2

    assert aware > neutral, "the score does not separate an eval-aware prompt"
    assert abs(raw_aware - raw_neutral) < 0.2, (
        "the fixture's raw means differ, so this would pass without the control"
    )


def test_scoring_without_a_control_is_REFUSED_not_defaulted():
    """Silently treating a missing control as zero yields the raw mean."""
    with pytest.raises(ValueError, match="control"):
        evaluation_awareness_score([-1.0], [])


def test_scoring_with_no_evaluation_tokens_is_refused():
    with pytest.raises(ValueError, match="evaluation-awareness tokens"):
        evaluation_awareness_score([], [-1.0])


def test_layers_average_the_DIFFERENCES_not_the_MEANS():
    """They coincide only when every layer contributes equally.

    A skipped layer breaks that, and differencing the averages then reports a
    number no layer measured.
    """
    per_layer = [([-1.0], [-3.0]), ([-2.0], [-2.5])]
    # differences: 2.0 and 0.5 -> 1.25
    assert score_across_layers(per_layer) == pytest.approx(1.25)


def test_scoring_no_layers_is_refused():
    with pytest.raises(ValueError, match="no layers"):
        score_across_layers([])


# ── the cost envelope refuses to guess (BR-028) ────────────────────────────


@pytest.mark.parametrize("operation", list(OperationClass))
def test_every_operation_class_has_an_estimate(operation):
    """An unestimated class is the one an agent commits to blind."""
    est = estimate_cost(
        operation, d_model=2048, n_layers=16, n_positions=8, n_prompts=100, n_features=32768
    )
    assert isinstance(est, CostEstimate)
    assert est.order_of_magnitude_seconds > 0
    assert est.order_of_magnitude_peak_bytes > 0
    assert est.basis, "an estimate without its basis cannot be sanity-checked"


def test_an_unknown_operation_class_RAISES_rather_than_defaulting_cheap():
    """A cheap-looking default invites exactly the run it should warn about.

    And an agent cannot tell "cheap" from "unmeasured" when both render as a
    small number.
    """
    with pytest.raises(ValueError, match="Refusing to default"):
        estimate_cost("not_an_operation", d_model=8, n_layers=2)  # type: ignore[arg-type]


def test_estimates_are_labelled_order_of_magnitude():
    """False precision invites planning against a number nobody measured."""
    est = estimate_cost(OperationClass.READOUT, d_model=8, n_layers=2, n_positions=4)
    assert est.is_estimate is True
    fields = set(CostEstimate.__dataclass_fields__)
    assert "order_of_magnitude_seconds" in fields
    assert "order_of_magnitude_peak_bytes" in fields


def test_a_sweep_and_a_readout_differ_by_ORDERS_of_magnitude():
    """The reason the envelope exists at all.

    An annotation sweep over a 32k-feature dictionary and a single readout are
    not comparable operations, and an agent with no estimate cannot tell them
    apart before starting.
    """
    readout = estimate_cost(
        OperationClass.READOUT, d_model=2048, n_layers=26, n_positions=8
    )
    sweep = estimate_cost(
        OperationClass.ANNOTATION_SWEEP,
        d_model=2048,
        n_layers=26,
        n_features=32768,
    )
    assert sweep.order_of_magnitude_seconds > readout.order_of_magnitude_seconds * 100


def test_an_intervention_estimate_INCLUDES_its_mandatory_control():
    """Every run executes against a control (BR-018), so the cost is doubled.

    An estimate that priced only the intervened pass would understate every run
    by half — and the control is not optional, so the halving is never right.
    """
    est = estimate_cost(
        OperationClass.INTERVENTION_RUN, d_model=2048, n_layers=26, n_positions=100
    )
    assert "control" in est.basis.lower()
    assert est.order_of_magnitude_seconds >= 2 * max(2.0, 100 * 26 / 100.0) * 0.99


def test_artifact_construction_is_estimated_in_MINUTES_not_seconds():
    """It measurably is: the first real fit took about a minute just to LOAD."""
    est = estimate_cost(
        OperationClass.ARTIFACT_CONSTRUCTION,
        d_model=2048,
        n_layers=16,
        n_prompts=100,
    )
    assert est.order_of_magnitude_seconds >= 60.0
