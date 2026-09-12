"""
J-space claims vocabulary (BR-019, BR-020, BR-024; PADR IDL-44).

ONE LADDER. J-space does NOT define rungs of its own — it maps its evidence
kinds onto `EvidenceRung`, the ladder circuits already use. A second enum would
be a second ladder, and two ladders is no ladder: a reviewer would have to hold
both and translate between them, which is exactly the confusion the ladder was
introduced to remove.

THE CAVEATS ARE DEFINED ONCE. Two copies of a sentence drift, and the drifted
copy is always the one on the surface nobody re-read. Every surface that reports
a negative imports the string from here.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict

from .evidence_ladder import EvidenceRung


class JSpaceEvidenceKind(str, Enum):
    """What kind of J-space observation is being reported.

    Deliberately NOT numbered. Numbering it would make it look like a ladder,
    and the ordering that matters is `EvidenceRung`'s.
    """

    READOUT = "readout"
    PROBE_CROSSING = "probe_crossing"
    ATTRIBUTION = "attribution"
    INTERVENTION_WITH_CONTROL = "intervention_with_control"


# The mapping IS the integration. A kind absent here has no rung, and
# `rung_for` refuses it rather than guessing — an unranked claim presented
# beside ranked ones reads as ranked.
_RUNG_BY_KIND: Dict[JSpaceEvidenceKind, EvidenceRung] = {
    # A concept appearing in a readout is an observation, not a cause.
    JSpaceEvidenceKind.READOUT: EvidenceRung.MINED,
    # A threshold crossing is a readout with a number attached, and a number is
    # not a mechanism.
    JSpaceEvidenceKind.PROBE_CROSSING: EvidenceRung.MINED,
    JSpaceEvidenceKind.ATTRIBUTION: EvidenceRung.ATTRIBUTION_SUPPORTED,
    # The FIRST rung that may be described in causal language, and only because
    # BR-018 requires a size-matched control alongside it.
    JSpaceEvidenceKind.INTERVENTION_WITH_CONTROL: EvidenceRung.CAUSALLY_VALIDATED,
}


def rung_for(kind: JSpaceEvidenceKind) -> EvidenceRung:
    if kind not in _RUNG_BY_KIND:
        raise KeyError(
            f"{kind!r} has no rung. Every J-space evidence kind must map onto "
            "the existing ladder — an unranked claim shown beside ranked ones "
            "reads as ranked."
        )
    return _RUNG_BY_KIND[kind]


def may_use_causal_language(kind: JSpaceEvidenceKind) -> bool:
    """Whether this evidence has earned intervention language.

    The single predicate every surface consults. Anything below
    CAUSALLY_VALIDATED describes what was OBSERVED, never what was CAUSED.
    """
    return rung_for(kind) >= EvidenceRung.CAUSALLY_VALIDATED


# ---------------------------------------------------------------- caveats

# BR-020. Both mechanisms are named because each alone reads as a hedge; together
# they say something specific about what the technique cannot see.
ABSENCE_CAVEAT = (
    "Absence of a signal is not evidence that the computation did not occur. "
    "Sufficiently automatic or well-practiced computation proceeds without "
    "engaging the workspace, and a concept with no single-token name may not "
    "surface even when it is represented."
)

# BR-020's second half, kept separate: a surface can be honest about one
# negative result while still implying the sweep saw everything.
NO_COVERAGE_CLAIM = (
    "This is not a comprehensive account of what the model is doing. Workspace "
    "evidence covers what the lens can name, not everything the model computes."
)

# BR-011. Stated wherever a readout is shown, because it bounds what any readout
# can contain at all.
READOUT_LIMITS = (
    "Readouts are limited to concepts with single-token names, and a readout "
    "that resists interpretation is not a null result."
)

# BR-019. What a rung-0 surface says instead of an intervention claim.
READOUT_NOT_CAUSAL = (
    "A concept appearing in a readout is not a causal claim: it was present, "
    "which is not the same as having been used."
)
