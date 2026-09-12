"""
Pins for the evidence ladder (018 Task 1.1, IDL-35): single-source enum,
rung transition rules, TS mirror sync, and the no-parallel-enums grep guard.
"""

import ast
import re
from pathlib import Path

from src.schemas.evidence_ladder import (
    RUNG_LANGUAGE,
    RUNG_NEXT_STEP,
    EdgeEvidence,
    EvidenceRung,
    circuit_rung,
    rung_language,
)

REPO = Path(__file__).resolve().parents[3]


class TestRungModel:
    def test_rung_is_highest_passed(self):
        e = EdgeEvidence()
        e.record_pass(EvidenceRung.ATTRIBUTION_SUPPORTED)
        e.record_pass(EvidenceRung.MINED)  # lower pass never demotes
        assert e.rung == EvidenceRung.ATTRIBUTION_SUPPORTED

    def test_failure_records_without_demoting(self):
        e = EdgeEvidence(rung=EvidenceRung.ATTRIBUTION_SUPPORTED)
        e.record_failure(EvidenceRung.CAUSALLY_VALIDATED)
        assert e.rung == EvidenceRung.ATTRIBUTION_SUPPORTED
        assert e.tested_and_failed == [EvidenceRung.CAUSALLY_VALIDATED]

    def test_later_pass_supersedes_failure_at_same_rung(self):
        e = EdgeEvidence()
        e.record_failure(EvidenceRung.CAUSALLY_VALIDATED)
        e.record_pass(EvidenceRung.CAUSALLY_VALIDATED)
        assert e.rung == EvidenceRung.CAUSALLY_VALIDATED
        assert e.tested_and_failed == []

    def test_circuit_rung_is_min_over_edges(self):
        assert circuit_rung(
            [EvidenceRung.FAITHFULNESS_TESTED, EvidenceRung.MINED]
        ) == EvidenceRung.MINED

    def test_edgeless_circuit_is_mined_not_a_crash(self):
        assert circuit_rung([]) == EvidenceRung.MINED

    def test_language_map_total_and_causal_only_at_rung2plus(self):
        assert set(RUNG_LANGUAGE) == set(EvidenceRung) == set(RUNG_NEXT_STEP)
        for rung, phrase in RUNG_LANGUAGE.items():
            if rung < EvidenceRung.CAUSALLY_VALIDATED:
                assert "causal" not in phrase.lower()
        assert rung_language(2) == RUNG_LANGUAGE[EvidenceRung.CAUSALLY_VALIDATED]


class TestSingleSource:
    def test_ts_mirror_in_sync(self):
        ts = (REPO / "frontend" / "src" / "types" / "evidenceLadder.ts").read_text()
        for rung in EvidenceRung:
            assert re.search(rf"{rung.name}\s*=\s*{rung.value}\b", ts), (
                f"TS mirror missing/mismatched {rung.name}={rung.value}"
            )

    def test_no_parallel_rung_enums(self):
        """Grep-guard (IDL-35): the ladder must never be redefined locally."""
        offenders = []
        for py in (REPO / "backend" / "src").rglob("*.py"):
            if py.name == "evidence_ladder.py":
                continue
            text = py.read_text()
            if re.search(r"class\s+\w*(EvidenceRung|Rung)\w*\s*\((IntEnum|Enum|str,\s*Enum)", text):
                offenders.append(str(py))
        assert offenders == [], f"Parallel rung enums found: {offenders}"


class TestRungThreeIsCircuitLevelNotEdgeLevel:
    """Rung 3 is deliberately absent from `edge.rung`, and that is the DESIGN.

    A reviewer (2026-07-21) flagged that `FAITHFULNESS_TESTED` is never written
    anywhere outside this schema, and asked whether rung 3 was unreachable —
    the "declared but unwired" pattern that produced three shipped-unreachable
    capabilities in the circuits arc. It is not that. The normative rule is
    BRD-MIS-CIRCUITS-002:

        "A circuit's displayed rung = min over member edges' rungs, WITH
         FAITHFULNESS STATUS SHOWN SEPARATELY."

    Faithfulness is not an edge test — it ablates the whole member set at once
    (Appendix A.7) — so it cannot be an edge rung. `circuit.rung` carries the
    three edge-derivable tiers; `circuit.faithfulness` +
    `faithfulness_status` carry the circuit-level result, and the API surfaces
    both.

    This test exists so the absence reads as intentional. Writing rung 3 onto
    an edge would be the regression, not the fix.
    """

    def test_the_ladder_still_defines_rung_three(self):
        """It is a real tier with real language — just not an edge rung."""
        assert int(EvidenceRung.FAITHFULNESS_TESTED) == 3
        assert rung_language(EvidenceRung.FAITHFULNESS_TESTED) == (
            "faithfulness-tested (circuit)"
        )

    def test_circuit_rung_is_MIN_over_edges_and_never_invents_rung_three(self):
        """The aggregation cannot manufacture a tier no edge earned."""
        assert int(circuit_rung([2, 2, 2])) == 2
        assert int(circuit_rung([3, 2])) == 2, (
            "MIN must hold even if a rung-3 edge value somehow appears"
        )
        assert int(circuit_rung([2, 0])) == 0, "one unvalidated edge caps it"

    def test_no_production_code_writes_rung_three_onto_an_edge(self):
        """The guard. If this fails, someone has made faithfulness an EDGE
        property — which silently re-grades every circuit that has one, since
        `recompute_rung` takes MIN over edges."""
        src = Path(__file__).resolve().parents[2] / "src"
        offenders = []
        for py in src.rglob("*.py"):
            if py.name == "evidence_ladder.py":
                continue  # the ladder DEFINES the tier
            text = py.read_text()
            for node in ast.walk(ast.parse(text)):
                # The defect is EXECUTABLE code naming the tier — an import, a
                # comparison, an assignment. Prose that merely mentions it is
                # documentation, and forbidding that would mean the guidance
                # tool could not explain the rule this very test enforces.
                #
                # Found by exactly that collision: adding `mistudio_howto`,
                # whose evidence_ladder topic says "rung 3 is not an edge
                # rung", tripped a substring check that could not tell an
                # explanation from a write.
                if isinstance(node, ast.Attribute) and node.attr == "FAITHFULNESS_TESTED":
                    offenders.append(f"{py.relative_to(src)}:{node.lineno}")
                elif isinstance(node, ast.Name) and node.id == "FAITHFULNESS_TESTED":
                    offenders.append(f"{py.relative_to(src)}:{node.lineno}")
                elif isinstance(node, ast.ImportFrom) and any(
                    a.name == "FAITHFULNESS_TESTED" for a in node.names
                ):
                    offenders.append(f"{py.relative_to(src)}:{node.lineno}")
        assert offenders == [], (
            "FAITHFULNESS_TESTED is referenced in production code: "
            f"{offenders}. Rung 3 is a CIRCUIT-level result carried in "
            "`circuit.faithfulness`, not an edge rung. See this class's "
            "docstring and BRD-MIS-CIRCUITS-002 line 414."
        )
