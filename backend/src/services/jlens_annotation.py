"""
Dictionary annotation: what an SAE feature looks like in J-space (BR-012..015).

TWO INDEPENDENT FIELDS, AND THE REASON IS NOT STYLISTIC. Motor features — the
ones committing to the next token — are SHARP, so they score high on lens
kurtosis exactly like workspace features do. Classifying on kurtosis alone
therefore labels every motor feature a workspace feature, and the error is
invisible because the number it rests on is real. The BRD locks this as a
decision (§ locked decision 6), not an implementation preference.

THE BEHAVIOURAL FIELD NEEDS BANDS, SO WITHOUT A BAND REPORT IT IS ABSENT.
Separating motor from workspace means asking WHERE in the stack a direction
reads strongly, and without a band report for this model there is no principled
"middle" to ask about. Guessing one would be the ported-boundary defect BR-002
forbids, arriving through a side door.

THE PROJECTION IS THE READOUT'S. Annotation applies `LensTransport` to a
WEIGHT-SPACE vector instead of a residual; it does not reimplement the
projection. A dictionary annotated by a different path than the readout shows is
worse than an unannotated one, because the two disagree and nothing says so.

NOTHING HERE RESOLVES A DISAGREEMENT. A lens readout is rung 0 (feature 026);
it raises a question about a label, it does not overrule one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence

import torch

from ..ml.jlens_metrics import excess_kurtosis

logger = logging.getLogger(__name__)


class WorkspaceClass(str, Enum):
    """Behavioural classification. `UNKNOWN` is a real answer, not a failure."""

    WORKSPACE = "workspace"
    MOTOR = "motor"
    OUTSIDE = "outside"
    #: No band report for this model, so the question cannot be asked. Distinct
    #: from OUTSIDE, which is a measurement.
    UNKNOWN = "unknown"


@dataclass
class FeatureAnnotation:
    """One feature's J-space description.

    Every optional field is ABSENT when not computable, never zero. A zero is
    averaged by a downstream consumer and silently understates; an absent value
    forces the consumer to decide.
    """

    feature_id: str
    layer: int

    #: GEOMETRIC field: excess kurtosis of the projected vocabulary distribution.
    lens_kurtosis: Optional[float] = None

    #: BEHAVIOURAL field: independent of the above, and absent without bands.
    workspace_class: WorkspaceClass = WorkspaceClass.UNKNOWN

    #: Top tokens the direction pushes toward — the readout, for review.
    top_tokens: List[str] = None  # type: ignore[assignment]

    #: BR-013. Sortable score and filterable flag, so the queue is a filter over
    #: the existing feature list rather than a new screen.
    disagreement_score: Optional[float] = None
    has_disagreement: bool = False

    def __post_init__(self) -> None:
        if self.top_tokens is None:
            self.top_tokens = []

    @property
    def is_j_aligned(self) -> bool:
        """Workspace-aligned, which REQUIRES both fields to agree.

        High kurtosis alone is not alignment: a motor feature has it too. This
        property is the one place that rule is expressed, so nothing downstream
        can reconstruct alignment from the geometric field by itself.
        """
        return (
            self.workspace_class is WorkspaceClass.WORKSPACE
            and self.lens_kurtosis is not None
        )


def annotate_direction(
    direction: torch.Tensor,
    transport,
    unembedding: torch.Tensor,
    layer: int,
    feature_id: str,
    decode,
    top_k: int = 8,
    band_report=None,
    depth_profile: Optional[Dict[int, float]] = None,
) -> FeatureAnnotation:
    """Project one weight-space direction through the lens and describe it.

    `transport` is the SAME `LensTransport` the readout uses. `band_report` and
    `depth_profile` are both required for the behavioural field; either being
    absent leaves the class UNKNOWN rather than guessed.

    `decode` IS REQUIRED, not optional. `top_tokens` must be DECODED STRINGS:
    the field feeds the disagreement queue, which compares them against an
    existing label's words. Token ids there make every feature disagree
    maximally — the queue fills with the whole dictionary and says nothing —
    and they are unreadable to the reviewer who opens it. Same rule the wire
    format enforces for readouts.
    """
    projected = transport.apply(direction.to(torch.float32), layer)
    logits = unembedding.to(torch.float32) @ projected

    kurtosis = excess_kurtosis(logits)
    top_ids = torch.topk(logits, k=min(top_k, logits.numel())).indices.tolist()
    top_tokens = [str(t) for t in decode(top_ids)]

    workspace_class = classify_behaviour(
        layer=layer, band_report=band_report, depth_profile=depth_profile
    )

    return FeatureAnnotation(
        feature_id=feature_id,
        layer=layer,
        lens_kurtosis=kurtosis,
        workspace_class=workspace_class,
        top_tokens=top_tokens,
    )


def classify_behaviour(
    layer: int, band_report=None, depth_profile: Optional[Dict[int, float]] = None
) -> WorkspaceClass:
    """Motor vs workspace, from WHERE in the stack the direction reads strongly.

    UNKNOWN without a band report. Separating motor from workspace is a question
    about position in the stack, and without boundaries measured for THIS model
    there is nothing to position against. Substituting a default here would
    reintroduce the ported boundaries BR-002 forbids, one level down from the
    place that forbids them.
    """
    if band_report is None or getattr(band_report, "boundaries", None) is None:
        return WorkspaceClass.UNKNOWN

    bounds = band_report.boundaries
    workspace_start = bounds["workspace_start"]
    motor_start = bounds["motor_start"]

    # Where the direction actually reads strongly, when we know; otherwise the
    # layer it lives at.
    peak_layer = layer
    if depth_profile:
        peak_layer = max(depth_profile, key=lambda k: depth_profile[k])

    if peak_layer >= motor_start:
        return WorkspaceClass.MOTOR
    if peak_layer >= workspace_start:
        return WorkspaceClass.WORKSPACE
    return WorkspaceClass.OUTSIDE


def label_disagreement(
    label_tokens: Sequence[str], readout_tokens: Sequence[str]
) -> float:
    """How far an existing label diverges from the lens readout, in [0, 1].

    The failure mode this exists for is documented: an example-driven label
    names what a feature fires ON, while the readout names what it pushes
    TOWARD. Those differ often enough that a silent divergence is a systematic
    labelling error nobody sees.

    Returns a SCORE rather than a boolean so the queue can be sorted — a flag
    alone gives a reviewer no way to start with the worst cases.
    """
    left = {t.strip().lower() for t in label_tokens if t.strip()}
    right = {t.strip().lower() for t in readout_tokens if t.strip()}
    if not left or not right:
        # Nothing to compare is NOT disagreement. Scoring it as maximal would
        # fill the queue with features that simply have no label yet.
        return 0.0
    overlap = len(left & right) / len(left | right)
    return 1.0 - overlap


def summarise_distribution(annotations: Sequence[FeatureAnnotation]) -> Dict[str, float]:
    """Shape of an annotation sweep (BR-014).

    A SHAPE OBSERVATION, not validation of the lens. The published finding is
    that only a modest fraction of features are J-aligned once motor features
    are excluded; a sweep that calls most features workspace has a mis-scaled
    threshold, and this is how that becomes visible.

    Reporting it as validation would be a rung claim it has not earned.
    """
    total = len(annotations)
    if total == 0:
        raise ValueError("cannot summarise an empty sweep")

    motor = sum(1 for a in annotations if a.workspace_class is WorkspaceClass.MOTOR)
    unknown = sum(1 for a in annotations if a.workspace_class is WorkspaceClass.UNKNOWN)
    non_motor = total - motor
    aligned = sum(1 for a in annotations if a.is_j_aligned)

    return {
        "total": float(total),
        "motor": float(motor),
        # Absent classifications are reported, not folded into a denominator
        # where they would look like measurements.
        "unknown": float(unknown),
        "j_aligned": float(aligned),
        "fraction_aligned_excluding_motor": (
            aligned / non_motor if non_motor else 0.0
        ),
    }


#: The published finding: a MODEST fraction, once motor features are excluded.
#: Above this, a sweep is reporting a threshold problem rather than a dictionary.
IMPLAUSIBLE_ALIGNED_FRACTION = 0.5


def distribution_is_plausible(summary: Dict[str, float]) -> Optional[bool]:
    """Whether a sweep's shape matches the published distribution.

    Three-valued, and the third value is the point. `None` means NOT
    ASSESSABLE: a sweep run without a band report classifies nothing, so every
    feature is UNKNOWN and the aligned fraction is trivially zero — which the
    two-valued version reported as PLAUSIBLE. "We measured nothing" reading as
    "the distribution looks right" is a false reassurance about the one check
    meant to catch a mis-scaled threshold.

    Otherwise a verdict rather than an exception: an implausible sweep is a
    RESULT the user needs to see, and refusing to produce it would hide the
    evidence that something is wrong.
    """
    classified = summary["total"] - summary["unknown"]
    if classified <= 0:
        return None
    return summary["fraction_aligned_excluding_motor"] < IMPLAUSIBLE_ALIGNED_FRACTION
