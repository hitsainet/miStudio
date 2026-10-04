"""
The workspace band report and the Phase-0 gate (BR-002, BR-003).

TWO PRODUCT RULES LIVE HERE AND BOTH ARE STRUCTURAL RATHER THAN POLICY.

1. BANDS ARE EARNED. `BandReport.boundaries` is `None` unless this model's own
   measured profile supports a split. There is no default, no fallback, and no
   constant anywhere in this module or in the frontend — the published ~L38-92
   figures were measured on one specific model, and BR-002 requires that porting
   them be impossible BY CONSTRUCTION. A `boundaries` field that could fall back
   to a default would be those figures under another name.

2. A NO-GO IS A RESULT. `GateDecision.NO_GO` is a first-class value that stores,
   renders and exports. A gate that can only conclude "yes" is not a gate, and
   the BRD is explicit that a NO-GO produces a publishable negative result
   rather than a blocked project.

NEXT-TOKEN AGREEMENT IS DESCRIBED, NEVER SCORED (BR-004). It appears in the
per-layer profile because the profile describes the model. It appears in no
gate condition, no ranking, and no threshold — the J-lens is deliberately worse
on it than the logit lens through most of the network, so a gate that rewarded
it would fail good models and pass bad ones. Enforced by an AST guard in
`test_jlens_band_report.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

from ..ml.jlens_metrics import LayerProfile, derive_boundaries

logger = logging.getLogger(__name__)


class GateDecision(str, Enum):
    GO = "go"
    NO_GO = "no_go"
    GO_AT_LARGER_SCALE = "go_at_larger_scale"


@dataclass
class BandReport:
    """One model's measured profile, and the boundaries it does or does not support."""

    model_id: str
    profiles: List[LayerProfile]
    cross_layer_cka: Dict[int, Dict[int, float]] = field(default_factory=dict)
    control_seed: Optional[int] = None
    # None means NO BAND REPORT BOUNDARIES for this model. Consumers draw
    # nothing; they never substitute a default.
    boundaries: Optional[Dict[str, int]] = None
    derivation: str = ""

    @property
    def has_bands(self) -> bool:
        return self.boundaries is not None


def build_band_report(
    model_id: str,
    profiles: Sequence[LayerProfile],
    control_seed: int,
    cross_layer_cka: Optional[Dict[int, Dict[int, float]]] = None,
) -> BandReport:
    """Assemble a report and derive this model's own boundaries.

    `control_seed` is required, not defaulted: the excess-FVE and autocorrelation
    figures in the profiles are defined against controls, and a report whose
    controls cannot be reconstructed cannot be checked by anyone else.
    """
    boundaries = derive_boundaries(profiles)
    if boundaries is None:
        derivation = (
            "No boundaries derived: this model's kurtosis profile does not "
            "support a three-way split. Bands are not shown. Boundaries "
            "measured on another model do not transfer."
        )
    else:
        derivation = (
            "Derived from this model's own excess-kurtosis profile: workspace "
            f"begins at the first sustained rise above the layer median "
            f"(L{boundaries['workspace_start']}), motor at the peak "
            f"(L{boundaries['motor_start']})."
        )

    return BandReport(
        model_id=model_id,
        profiles=list(profiles),
        cross_layer_cka=cross_layer_cka or {},
        control_seed=control_seed,
        boundaries=boundaries,
        derivation=derivation,
    )


@dataclass
class GateRecord:
    """The Phase-0 decision, with the evidence that produced it (BR-003)."""

    model_id: str
    decision: GateDecision
    rationale: str
    band_report: BandReport
    replication_report_id: Optional[str] = None

    def is_blocking(self) -> bool:
        """Whether product surface beyond the readout viewer stays closed.

        A NO_GO blocks further surface and is a complete, publishable outcome.
        GO_AT_LARGER_SCALE also blocks at THIS scale, which is the distinction
        that makes it worth having as a separate value rather than a softened
        GO.
        """
        return self.decision is not GateDecision.GO


def decide_gate(
    model_id: str,
    band_report: BandReport,
    claim_set_replicated: bool,
    larger_scale_indicated: bool,
    rationale: str,
    replication_report_id: Optional[str] = None,
) -> GateRecord:
    """Record the GO / NO-GO / GO-AT-LARGER-SCALE decision.

    THE INPUTS ARE FINDINGS, NOT SCORES. `claim_set_replicated` is the question
    BR-003 actually asks — whether the full workspace claim set replicates —
    and it is supplied by the analysis rather than thresholded here. There is
    deliberately no numeric criterion in this function: a threshold on any
    single metric would become the definition of the gate, and the one metric
    most likely to be reached for is next-token agreement, which BR-004
    forbids.

    A rationale is mandatory. A recorded decision without its reasoning is not
    a record.
    """
    if not rationale.strip():
        raise ValueError(
            "a gate decision requires a rationale; a decision without its "
            "reasoning cannot be reviewed, and BR-003 requires the decision be "
            "recorded with its evidence"
        )

    if claim_set_replicated:
        decision = GateDecision.GO
    elif larger_scale_indicated:
        decision = GateDecision.GO_AT_LARGER_SCALE
    else:
        decision = GateDecision.NO_GO

    logger.info(
        "Phase-0 gate for %s: %s (bands=%s)",
        model_id,
        decision.value,
        "derived" if band_report.has_bands else "none",
    )

    return GateRecord(
        model_id=model_id,
        decision=decision,
        rationale=rationale,
        band_report=band_report,
        replication_report_id=replication_report_id,
    )
