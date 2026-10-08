"""
Feature 026: claims discipline (BR-019, BR-020, BR-024).

Three audits over three different corpora, because each catches what the others
cannot:

  RUNG MAPPING     J-space evidence sits on the EXISTING ladder, no second enum
  ABSENCE CAVEAT   a negative result says what it does not mean
  CONSCIOUSNESS    no shipped text implies subjective experience

The consciousness corpus is the widest deliberately: the likeliest home for that
language is a well-meaning paragraph in the manual, not a variable name.

MUTATION CONTROLS (each must turn this file red):
  * give J-space its own rung enum                  -> "one ladder" fails
  * let a rung-0 kind use causal language           -> "may_use_causal" fails
  * drop the manual from the consciousness corpus   -> "manual is audited" fails
  * duplicate a caveat string instead of importing  -> "one definition" fails
  * drop a mechanism from the absence caveat        -> "both mechanisms" fails
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.schemas.evidence_ladder import EvidenceRung
from src.schemas.jspace_claims import (
    ABSENCE_CAVEAT,
    NO_COVERAGE_CLAIM,
    READOUT_LIMITS,
    READOUT_NOT_CAUSAL,
    JSpaceEvidenceKind,
    may_use_causal_language,
    rung_for,
)

REPO = Path(__file__).resolve().parents[3]
BACKEND = REPO / "backend" / "src"
FRONTEND = REPO / "frontend" / "src"
MANUAL = REPO / "manual" / "docs"


# ── one ladder ─────────────────────────────────────────────────────────────


def test_every_jspace_evidence_kind_maps_onto_the_existing_ladder():
    """No kind may be unranked: an unranked claim beside ranked ones reads as ranked."""
    for kind in JSpaceEvidenceKind:
        assert isinstance(rung_for(kind), EvidenceRung)


def test_jspace_adds_no_rung_of_its_own():
    """A second enum would be a second ladder, and two ladders is no ladder.

    Asserted against `EvidenceRung`'s members, so renaming or removing a rung
    breaks this rather than silently diverging.
    """
    used = {rung_for(k) for k in JSpaceEvidenceKind}
    assert used <= set(EvidenceRung)
    assert EvidenceRung.MINED in used, "nothing maps to the observation rung"
    assert EvidenceRung.CAUSALLY_VALIDATED in used, "nothing can ever be causal"


def test_a_readout_is_rung_zero_and_may_not_use_causal_language():
    assert rung_for(JSpaceEvidenceKind.READOUT) is EvidenceRung.MINED
    assert may_use_causal_language(JSpaceEvidenceKind.READOUT) is False


def test_a_threshold_crossing_is_still_rung_zero():
    """Attaching a number to an observation does not promote it."""
    assert rung_for(JSpaceEvidenceKind.PROBE_CROSSING) is EvidenceRung.MINED
    assert may_use_causal_language(JSpaceEvidenceKind.PROBE_CROSSING) is False


def test_only_an_intervention_with_a_control_earns_causal_language():
    kind = JSpaceEvidenceKind.INTERVENTION_WITH_CONTROL
    assert rung_for(kind) is EvidenceRung.CAUSALLY_VALIDATED
    assert may_use_causal_language(kind) is True


def test_an_unmapped_kind_is_refused_rather_than_defaulted():
    class Rogue(str):
        pass

    with pytest.raises(KeyError, match="no rung"):
        rung_for(Rogue("invented"))


# ── the caveats are defined once ───────────────────────────────────────────


def test_the_absence_caveat_names_BOTH_mechanisms():
    """Either alone reads as a hedge; together they say what the lens cannot see."""
    assert "not evidence" in ABSENCE_CAVEAT
    assert "automatic" in ABSENCE_CAVEAT
    assert "single-token name" in ABSENCE_CAVEAT


def test_the_coverage_disclaimer_is_separate_from_the_absence_caveat():
    """A surface can be honest about one negative and still imply the sweep saw all."""
    assert ABSENCE_CAVEAT != NO_COVERAGE_CLAIM
    assert "comprehensive" in NO_COVERAGE_CLAIM


def test_the_readout_limits_bound_what_any_readout_can_contain():
    assert "single-token names" in READOUT_LIMITS
    assert "not a null result" in READOUT_LIMITS


def test_the_rung_zero_sentence_denies_rather_than_hedges():
    assert "not a causal claim" in READOUT_NOT_CAUSAL


def _shipped_text_files():
    """Everything a user or agent reads. Not code comments, not 0xcc/."""
    files = []
    files += sorted(FRONTEND.glob("components/jlens/*.tsx"))
    files += sorted(FRONTEND.glob("components/panels/JLensPanel.tsx"))
    files += sorted(BACKEND.glob("mcp_server/tools/jlens*.py"))
    files += sorted(MANUAL.glob("**/jlens.md"))
    return [f for f in files if f.exists()]


def test_the_shipped_text_corpus_is_not_empty_and_spans_every_surface():
    """Discovery, not a list — and the list this replaces shipped green over 16 modules."""
    files = _shipped_text_files()
    assert files, "no shipped text discovered at all"
    suffixes = {f.suffix for f in files}
    assert {".tsx", ".py", ".md"} <= suffixes, (
        f"a whole surface kind is unaudited: found {suffixes}"
    )


def test_the_manual_is_in_the_consciousness_corpus():
    """The likeliest home for the claim is a friendly paragraph, not a variable."""
    assert any(f.suffix == ".md" for f in _shipped_text_files())


# ── consciousness (BR-024) ─────────────────────────────────────────────────

#: Phrases that assert or invite the inference of subjective experience.
#: Anchored on the SUBJECT being the model, so "the user experiences a delay"
#: is not caught and "the model experiences" is.
CONSCIOUSNESS = re.compile(
    r"\b(?:model|it)\s+(?:is\s+)?"
    r"(?:conscious|aware\s+of\s+itself|self-aware|sentient|feels|experiences)\b"
    r"|\bsubjective\s+experience\b"
    r"|\bphenomenal\s+consciousness\b"
    r"|\bwhat\s+it\s+is\s+like\s+to\s+be\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize(
    "path", _shipped_text_files(), ids=lambda p: p.name
)
def test_no_shipped_text_implies_subjective_experience(path: Path):
    """BR-024. The source paper declines to take a position; the product inherits that."""
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = [m.group(0) for m in CONSCIOUSNESS.finditer(text)]
    assert not hits, f"{path.name} implies subjective experience: {hits}"


class TestTheConsciousnessAuditBites:
    """Negative controls. An audit nobody has seen fail is not evidence."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "The model experiences the prompt as unpleasant.",
            "This shows the model is conscious of the contradiction.",
            "a window into its subjective experience",
            "what it is like to be this model",
            "The model is self-aware at these layers.",
        ],
    )
    def test_a_planted_claim_is_caught(self, phrase):
        assert CONSCIOUSNESS.search(phrase), f"not caught: {phrase!r}"

    @pytest.mark.parametrize(
        "phrase",
        [
            "The user experiences a delay while the model loads.",
            "The model represents the concept at this layer.",
            "Readouts are limited to concepts with single-token names.",
            "This is not evidence that the computation did not occur.",
        ],
    )
    def test_legitimate_text_is_permitted(self, phrase):
        """A guard that flags the sentences the product must say is unusable."""
        assert not CONSCIOUSNESS.search(phrase), f"false positive: {phrase!r}"

    def test_the_required_caveats_survive_this_audit_too(self):
        for caveat in (ABSENCE_CAVEAT, NO_COVERAGE_CLAIM, READOUT_LIMITS, READOUT_NOT_CAUSAL):
            assert not CONSCIOUSNESS.search(caveat)


# ── one definition, mirrored not copied ────────────────────────────────────

TS_MIRROR = FRONTEND / "config" / "jspaceClaims.ts"


def _ts_string(name: str) -> str:
    """Extract a concatenated TS string constant as its rendered value."""
    text = TS_MIRROR.read_text(encoding="utf-8")
    m = re.search(rf"export const {name} =\s*(.*?);", text, re.S)
    assert m, f"{name} is absent from the TypeScript mirror"
    parts = re.findall(r"'([^']*)'", m.group(1))
    return "".join(parts)


def test_ts_mirror_in_sync():
    """The frontend copy must be the SAME SENTENCE, not a paraphrase.

    A caveat that drifts is worse than one that is missing: the surface still
    looks like it is warning the user, while saying something weaker than the
    requirement. Pinned the way the evidence ladder already is.
    """
    assert TS_MIRROR.exists(), "the TypeScript mirror is gone"
    for name, expected in (
        ("ABSENCE_CAVEAT", ABSENCE_CAVEAT),
        ("NO_COVERAGE_CLAIM", NO_COVERAGE_CLAIM),
        ("READOUT_LIMITS", READOUT_LIMITS),
        ("READOUT_NOT_CAUSAL", READOUT_NOT_CAUSAL),
    ):
        assert _ts_string(name) == expected, (
            f"{name} has drifted between Python and TypeScript"
        )


def test_the_panel_imports_the_caveat_rather_than_restating_it():
    """A hardcoded sentence in a component is a copy the sync test cannot see."""
    strip = FRONTEND / "components" / "jlens" / "ProvenanceStrip.tsx"
    source = strip.read_text(encoding="utf-8")
    assert "jspaceClaims" in source, (
        "ProvenanceStrip does not import the shared caveats, so its copy can "
        "drift from the requirement without any test noticing"
    )
