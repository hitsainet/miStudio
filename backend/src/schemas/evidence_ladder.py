"""
The evidence ladder — miStudio's single claims vocabulary (IDL-35, BR-026).

Mechanistic claims come in strictly increasing strength. This module is THE
one definition consumed by the contract (circuit_definition), the API/MCP
surfaces, and (via the mirrored TS type) the UI. No other rung/status enum
may exist anywhere in the codebase — a grep-guard test enforces it, and a
literal-value sync test pins the frontend mirror
(frontend/src/types/evidenceLadder.ts).

Language rules (enforced by the shared causal-language copy audit): the word
"causal" is FORBIDDEN below rung 2. All user-facing rung language must come
from RUNG_LANGUAGE / rung_language() — never hand-written per surface.
"""

from enum import IntEnum
from typing import List

from pydantic import BaseModel, Field


class EvidenceRung(IntEnum):
    """Rungs are strictly ordered; an artifact's rung is the highest PASSED."""

    MINED = 0                    # statistical association survived null + support (Tier-1)
    ATTRIBUTION_SUPPORTED = 1    # gradient attribution agrees (Tier-2)
    CAUSALLY_VALIDATED = 2       # real intervention satisfied the BR-018 criterion
    FAITHFULNESS_TESTED = 3      # circuit-level necessity (and, where run, sufficiency)


# The ONLY source of user-facing rung language (UI, MCP descriptions, exports).
RUNG_LANGUAGE: dict[EvidenceRung, str] = {
    EvidenceRung.MINED: "associated",
    EvidenceRung.ATTRIBUTION_SUPPORTED: "suggested (attribution-supported)",
    EvidenceRung.CAUSALLY_VALIDATED: "causally validated (edge)",
    EvidenceRung.FAITHFULNESS_TESTED: "faithfulness-tested (circuit)",
}

# What moves an artifact up one rung — surfaced as UI tooltips and MCP hints.
RUNG_NEXT_STEP: dict[EvidenceRung, str] = {
    EvidenceRung.MINED: "run the attribution pass (sign agreement + magnitude percentile)",
    EvidenceRung.ATTRIBUTION_SUPPORTED: "run intervention validation (effect size vs null + sign consistency)",
    EvidenceRung.CAUSALLY_VALIDATED: "run circuit-level faithfulness at promotion",
    EvidenceRung.FAITHFULNESS_TESTED: "top rung — nothing further",
}


def rung_language(rung: "EvidenceRung | int") -> str:
    """Server-rendered rung phrase — the single language source for all surfaces."""
    return RUNG_LANGUAGE[EvidenceRung(rung)]


class EdgeEvidence(BaseModel):
    """Rung state carried by every edge (and mirrored into the contract).

    The rung is the highest rung PASSED. Failing a test at rung N records N in
    tested_and_failed WITHOUT demoting the rungs below it — a failed rung-2
    intervention does not erase a real rung-0/1 association.
    """

    rung: EvidenceRung = EvidenceRung.MINED
    tested_and_failed: List[EvidenceRung] = Field(default_factory=list)

    def record_pass(self, rung: EvidenceRung) -> "EdgeEvidence":
        if rung > self.rung:
            self.rung = rung
        # A later pass supersedes an earlier failure at the same rung.
        self.tested_and_failed = [r for r in self.tested_and_failed if r != rung]
        return self

    def record_failure(self, rung: EvidenceRung) -> "EdgeEvidence":
        if rung not in self.tested_and_failed:
            self.tested_and_failed.append(rung)
        return self


def circuit_rung(edge_rungs: List[EvidenceRung]) -> EvidenceRung:
    """A circuit's displayed rung = MIN over member edges' rungs.

    An edge-less circuit (hand-assembled, no mined evidence) is rung 0 by
    definition — defined, not a crash (018 FTID pitfall).
    """
    if not edge_rungs:
        return EvidenceRung.MINED
    return EvidenceRung(min(int(r) for r in edge_rungs))
