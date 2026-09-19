"""
Dictionary annotation (BR-012..015).

The defect this feature is built around: MOTOR FEATURES SHARE HIGH LENS
KURTOSIS WITH WORKSPACE FEATURES. So a classifier using kurtosis alone labels
every motor feature a workspace feature, and the error is invisible because the
number it rests on is real.

Every test below is shaped to distinguish "measured" from "assumed" — the same
distinction the rest of this arc turns on.

MUTATION CONTROLS (each must turn this file red):
  * classify from kurtosis alone                   -> "two fields" fails
  * guess a band boundary when none exists         -> "no bands, no class" fails
  * fold UNKNOWN into a measured class             -> "unknown is reported" fails
  * score an unlabelled feature as disagreeing     -> "nothing to compare" fails
  * make the distribution check advisory           -> "implausible" fails
  * reimplement the projection instead of reusing  -> "same path" fails
"""

from __future__ import annotations

import pytest
import torch

from src.services.jlens_annotation import (
    IMPLAUSIBLE_ALIGNED_FRACTION,
    FeatureAnnotation,
    WorkspaceClass,
    annotate_direction,
    classify_behaviour,
    distribution_is_plausible,
    label_disagreement,
    summarise_distribution,
)
from src.services.jlens_readout_service import IdentityTransport


class Bands:
    def __init__(self, boundaries):
        self.boundaries = boundaries


BANDS = Bands({"workspace_start": 10, "motor_start": 20})


# ── two independent fields (BR-012) ────────────────────────────────────────


def test_high_kurtosis_alone_is_NOT_workspace_alignment():
    """The whole reason BR-012 requires two fields.

    A motor feature is sharp — it commits to a token — so it scores high on
    kurtosis exactly like a workspace feature. A classifier reading only the
    geometric field calls it workspace, and the number it used was real.
    """
    motor_like = FeatureAnnotation(
        feature_id="f1",
        layer=25,
        lens_kurtosis=40.0,  # very sharp
        workspace_class=WorkspaceClass.MOTOR,
    )
    assert motor_like.is_j_aligned is False, (
        "a motor feature with high kurtosis was reported as J-aligned"
    )


def test_alignment_requires_BOTH_fields():
    assert FeatureAnnotation("f", 12, 30.0, WorkspaceClass.WORKSPACE).is_j_aligned is True
    # Geometric present, behavioural says otherwise.
    assert FeatureAnnotation("f", 12, 30.0, WorkspaceClass.OUTSIDE).is_j_aligned is False
    # Behavioural says workspace but nothing was measured geometrically.
    assert FeatureAnnotation("f", 12, None, WorkspaceClass.WORKSPACE).is_j_aligned is False


# ── no band report, no behavioural claim (BR-002 one level down) ───────────


def test_without_a_band_report_the_class_is_UNKNOWN_not_guessed():
    """Separating motor from workspace is a question about position in the stack.

    Without boundaries measured for THIS model there is nothing to position
    against, and substituting a default would reintroduce the ported boundaries
    BR-002 forbids through a side door.
    """
    assert classify_behaviour(layer=15, band_report=None) is WorkspaceClass.UNKNOWN
    assert classify_behaviour(layer=15, band_report=Bands(None)) is WorkspaceClass.UNKNOWN


def test_with_a_band_report_the_class_follows_THIS_models_boundaries():
    assert classify_behaviour(5, BANDS) is WorkspaceClass.OUTSIDE
    assert classify_behaviour(15, BANDS) is WorkspaceClass.WORKSPACE
    assert classify_behaviour(25, BANDS) is WorkspaceClass.MOTOR


def test_the_boundaries_are_read_from_the_report_not_hardcoded():
    """Shift the report and every classification must move with it.

    A single-shape test passes against hardcoded numbers — the trap that let
    two mutations survive in the viewer.
    """
    shifted = Bands({"workspace_start": 40, "motor_start": 90})
    assert classify_behaviour(50, BANDS) is WorkspaceClass.MOTOR
    assert classify_behaviour(50, shifted) is WorkspaceClass.WORKSPACE


def test_the_depth_profile_overrides_the_nominal_layer_when_known():
    """Where a direction READS strongly is the question, not where it lives."""
    profile = {25: 0.9, 5: 0.1}
    assert classify_behaviour(5, BANDS, depth_profile=profile) is WorkspaceClass.MOTOR


# ── the projection is the readout's (BR-015) ───────────────────────────────


def test_annotation_uses_the_SAME_transport_as_the_readout():
    """A second projection path can disagree with what the user is shown.

    A dictionary annotated by a different path than the readout displays is
    worse than an unannotated one: the two disagree and nothing says so.
    """
    torch.manual_seed(0)
    d_model, n_vocab = 8, 40
    W_U = torch.randn(n_vocab, d_model)
    direction = torch.randn(d_model)
    transport = IdentityTransport()

    ann = annotate_direction(
        direction, transport, W_U, layer=3, feature_id="f",
        decode=lambda ids: [f"tok{i}" for i in ids], top_k=4,
    )

    # The readout of the same direction, computed directly through the same
    # transport — identical by construction only if the service reuses it.
    expected = torch.topk(W_U @ transport.apply(direction, 3), k=4).indices.tolist()
    assert ann.top_tokens == [f"tok{i}" for i in expected]


def test_annotation_records_the_geometric_field_as_a_number():
    torch.manual_seed(1)
    ann = annotate_direction(
        torch.randn(8), IdentityTransport(), torch.randn(40, 8), 3, "f",
        decode=lambda ids: [f"tok{i}" for i in ids],
    )
    assert isinstance(ann.lens_kurtosis, float)
    # No band report was supplied, so the behavioural field stays absent.
    assert ann.workspace_class is WorkspaceClass.UNKNOWN


# ── disagreement is a sortable queue (BR-013) ──────────────────────────────


def test_disagreement_is_a_SCORE_so_the_queue_can_be_sorted():
    """A flag alone gives a reviewer no way to start with the worst cases."""
    total = label_disagreement(["cat", "dog"], ["car", "truck"])
    partial = label_disagreement(["cat", "dog"], ["cat", "truck"])
    none = label_disagreement(["cat", "dog"], ["cat", "dog"])

    assert total == pytest.approx(1.0)
    assert 0.0 < partial < 1.0
    assert none == pytest.approx(0.0)
    assert total > partial > none, "scores do not order the queue"


def test_nothing_to_compare_is_NOT_disagreement():
    """Otherwise the queue fills with features that simply have no label yet."""
    assert label_disagreement([], ["cat"]) == 0.0
    assert label_disagreement(["cat"], []) == 0.0


def test_disagreement_is_case_and_whitespace_insensitive():
    """'Cat' vs 'cat ' is not a divergence, and reporting it as one buries the real ones."""
    assert label_disagreement([" Cat "], ["cat"]) == pytest.approx(0.0)


def test_the_annotation_carries_both_a_filterable_flag_and_a_sortable_score():
    ann = FeatureAnnotation("f", 1, 2.0, WorkspaceClass.WORKSPACE)
    assert hasattr(ann, "has_disagreement")
    assert hasattr(ann, "disagreement_score")


# ── distributional shape check (BR-014) ────────────────────────────────────


def _sweep(n_workspace: int, n_motor: int, n_outside: int = 0, n_unknown: int = 0):
    out = []
    out += [FeatureAnnotation(f"w{i}", 12, 20.0, WorkspaceClass.WORKSPACE) for i in range(n_workspace)]
    out += [FeatureAnnotation(f"m{i}", 25, 30.0, WorkspaceClass.MOTOR) for i in range(n_motor)]
    out += [FeatureAnnotation(f"o{i}", 2, 1.0, WorkspaceClass.OUTSIDE) for i in range(n_outside)]
    out += [FeatureAnnotation(f"u{i}", 2, 1.0, WorkspaceClass.UNKNOWN) for i in range(n_unknown)]
    return out


def test_a_modest_aligned_fraction_is_plausible():
    """The published finding, once motor features are excluded."""
    summary = summarise_distribution(_sweep(n_workspace=10, n_motor=20, n_outside=70))
    assert summary["fraction_aligned_excluding_motor"] < IMPLAUSIBLE_ALIGNED_FRACTION
    assert distribution_is_plausible(summary)


def test_a_sweep_calling_MOST_features_workspace_is_implausible():
    """The bug this check exists for: a mis-scaled threshold puts everything on one side."""
    summary = summarise_distribution(_sweep(n_workspace=90, n_motor=5, n_outside=5))
    assert not distribution_is_plausible(summary), (
        "a sweep labelling 90% of features workspace was reported as plausible"
    )


def test_motor_features_are_EXCLUDED_from_the_denominator():
    """The published finding is about non-motor features specifically.

    Including motor features dilutes the fraction and makes an implausible
    sweep look fine.
    """
    summary = summarise_distribution(_sweep(n_workspace=10, n_motor=90))
    assert summary["fraction_aligned_excluding_motor"] == pytest.approx(1.0)
    assert not distribution_is_plausible(summary)


def test_unknown_classifications_are_REPORTED_not_hidden():
    """A sweep run without a band report classifies nothing, and must say so.

    Folding UNKNOWN into a measured bucket would make "we could not ask" look
    like "we asked and the answer was no".
    """
    summary = summarise_distribution(_sweep(n_workspace=0, n_motor=0, n_unknown=50))
    assert summary["unknown"] == 50.0
    assert summary["j_aligned"] == 0.0


def test_an_empty_sweep_is_refused_rather_than_summarised_as_zero():
    """0.0 aligned out of nothing reads as a finding about the dictionary."""
    with pytest.raises(ValueError, match="empty"):
        summarise_distribution([])


def test_the_check_returns_a_verdict_rather_than_raising():
    """An implausible sweep is a RESULT the user needs to see.

    Refusing to produce it would hide the evidence that something is
    mis-scaled — the opposite of what the check is for.
    """
    summary = summarise_distribution(_sweep(n_workspace=90, n_motor=5, n_outside=5))
    assert distribution_is_plausible(summary) is False


# ── review round 2 ─────────────────────────────────────────────────────────


def test_top_tokens_are_DECODED_STRINGS_not_ids():
    """Ids here make EVERY feature disagree maximally.

    `top_tokens` feeds the disagreement queue, which compares them against an
    existing label's WORDS. Against ids the overlap is always empty, so the
    queue fills with the entire dictionary and tells a reviewer nothing — and
    what it shows them is unreadable anyway.
    """
    torch.manual_seed(3)
    ann = annotate_direction(
        torch.randn(8), IdentityTransport(), torch.randn(40, 8), 3, "f",
        decode=lambda ids: ["spider", "web", "legs", "eight"][: len(ids)],
        top_k=4,
    )
    assert ann.top_tokens[0] == "spider"
    assert not any(t.isdigit() for t in ann.top_tokens), "ids leaked into top_tokens"

    # And the consequence, stated: a real label overlaps a decoded readout.
    assert label_disagreement(["spider", "web"], ann.top_tokens) < 1.0


def test_a_decoder_is_REQUIRED_not_optional():
    """Optional means someone will omit it and get ids without noticing."""
    import inspect

    params = inspect.signature(annotate_direction).parameters
    assert params["decode"].default is inspect.Parameter.empty


def test_a_sweep_that_classified_NOTHING_is_NOT_ASSESSABLE():
    """"We measured nothing" must not read as "the distribution looks right".

    Without a band report every feature is UNKNOWN and the aligned fraction is
    trivially zero — which the two-valued version reported as plausible, a
    false reassurance about the one check meant to catch a mis-scaled
    threshold.
    """
    summary = summarise_distribution(_sweep(n_workspace=0, n_motor=0, n_unknown=40))
    assert distribution_is_plausible(summary) is None


def test_a_partially_classified_sweep_is_still_assessed():
    """Some UNKNOWN is normal; all UNKNOWN is the case that means nothing ran."""
    summary = summarise_distribution(
        _sweep(n_workspace=5, n_motor=10, n_outside=60, n_unknown=25)
    )
    assert distribution_is_plausible(summary) is True
