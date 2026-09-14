"""Copy-audit: Feature 016 surfaces produce rungs 0–1 ONLY, so no
user-facing 016 string may use unqualified causal language (IDL-35 /
evidence-ladder discipline). Extends the 018 grep-guard to 016.

Allowed occurrences: the word inside a NEGATION/qualifier ("NOT causal",
"not a causal", "causally validated" is a rung-2 LABEL, disclosed as such),
or in meta-text that names the discipline itself. This test pins the
discovery/attribution reports + MCP docstrings + service copy against a
regression that would let a mined association be described as a mechanism.
"""

import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "src"

# 016 + 017 user-facing surfaces (report strings, MCP docstrings, service
# copy). 017's validation IS the rung-2 tier, so 'causal' is EARNED there —
# but only inside validation/faithfulness contexts, which ALLOWED_CONTEXT
# covers via 'causally validated'/'rung 2'/'017'.
# F20 R3-16: this WAS the hand-maintained list of five files whose failure mode
# the comment below already describes — and it left sixteen circuit modules
# unaudited, including all three REST endpoint modules whose responses reach the
# UI directly. Planting
#
#     """OVERCLAIM CONTROL: This circuit is causally validated by observation
#        alone; the mechanism is confirmed.
#
# at the top of `api/v1/endpoints/circuit_discovery.py` left the suite 26/26
# green. The file argued for globbing, globbed the MCP tools for exactly that
# reason, and then did not apply the argument to itself.
#
# Now discovered. A circuit module added later is audited the moment it exists.
SURFACES = sorted(
    set(
        list((BACKEND / "services").glob("circuit_*.py"))
        + list((BACKEND / "api" / "v1" / "endpoints").glob("circuit*.py"))
        + list((BACKEND / "schemas").glob("circuit_*.py"))
        + list((BACKEND / "schemas").glob("evidence_*.py"))
        + [BACKEND / "mcp_server" / "tools" / "circuits.py"]
    )
)

# F20 task 3.4: EVERY `millm_*` MCP tool module, discovered rather than listed.
#
# `SURFACES` above is hand-maintained, and a hand-maintained list is only as
# good as the list — a new module simply is not audited, silently. The MCP
# tools are the surface an AGENT reads and relays verbatim, so they are
# globbed: adding a module opts it in automatically.
MILLM_TOOL_MODULES = sorted(
    (BACKEND / "mcp_server" / "tools").glob("millm_*.py")
)

# Feature 026 (BR-019): J-space surfaces, DISCOVERED for the same reason.
#
# J-space evidence is overwhelmingly rung 0 — a readout is an observation, not
# a cause — so these are the modules most likely to reach for intervention
# language and least entitled to it. Globbed, never listed: `jlens_watchlist.py`
# added next week is audited the moment it exists, which is precisely what the
# hand-maintained version of this list failed to do for 16 circuit modules.
JSPACE_SURFACES = sorted(
    set(
        list((BACKEND / "services").glob("jlens_*.py"))
        + list((BACKEND / "ml").glob("jlens_*.py"))
        + list((BACKEND / "api" / "v1" / "endpoints").glob("jlens*.py"))
        + list((BACKEND / "schemas").glob("jlens*.py"))
        + list((BACKEND / "schemas").glob("jspace_*.py"))
        + list((BACKEND / "workers").glob("jlens_*.py"))
        + list((BACKEND / "mcp_server" / "tools").glob("jlens*.py"))
    )
)

# Clusters: DISCOVERED, and under BOTH names.
#
# A cluster is a correlational object. Its members are grouped by co-occurrence
# and TF-IDF context similarity, and the strength budget allocates a DIAL, not
# evidence — so these surfaces are entitled to no causal claim whatsoever. The
# only route by which a cluster earns one is BR-016: being mined into a circuit
# as `member_kind: cluster_ref` and passing rung 2 on the ordinary ladder, which
# happens in the circuit modules already audited above.
#
# The audit reached none of them. Twelve modules across six layers — including
# both REST endpoints and the `groups` MCP tool surface, which an agent reads and
# relays verbatim — were unaudited while the circuit and J-space corpora were
# globbed. The argument for globbing had simply never been applied here.
#
# BOTH NAMING CONVENTIONS, and this is the trap. The Feature Groups → Clusters
# rename was UI-ONLY (IDL-28): the backend still calls half of these
# `feature_group*`. A `cluster_*` glob alone reads as thorough and silently
# misses `feature_grouping_service.py`, `feature_groups.py`, `feature_group.py`,
# `feature_grouping_tasks.py`, `feature_grouping.py` and `groups.py` — six of
# twelve, including the service that does the grouping and the tool that reports
# it. A glob is only as good as its pattern, which is the same failure as a
# hand-maintained list wearing better clothes.
CLUSTER_SURFACES = sorted(
    set(
        list((BACKEND / "services").glob("cluster_*.py"))
        + list((BACKEND / "services").glob("feature_group*.py"))
        + list((BACKEND / "api" / "v1" / "endpoints").glob("cluster*.py"))
        + list((BACKEND / "api" / "v1" / "endpoints").glob("feature_group*.py"))
        + list((BACKEND / "schemas").glob("cluster_*.py"))
        + list((BACKEND / "schemas").glob("feature_group*.py"))
        + list((BACKEND / "workers").glob("cluster_*.py"))
        + list((BACKEND / "workers").glob("feature_group*.py"))
        + list((BACKEND / "models").glob("cluster_*.py"))
        + list((BACKEND / "models").glob("feature_group*.py"))
        + list((BACKEND / "mcp_server" / "tools").glob("group*.py"))
        + list((BACKEND / "mcp_server" / "tools").glob("cluster*.py"))
    )
)

# F20 R1-09/10: NEGATION-anchored, not topic-anchored.
#
# The previous `ALLOWED_CONTEXT` whitelisted TOPIC WORDS — `causal\s+evidence`,
# `causal\s+effect`, `causal\s+validation`, `real\s+causal`, `rung\s*2`. So
# naming the topic legitimised any claim about it, and six real overclaims
# passed, verified by execution:
#
#   "These edges are backed by causal evidence from live traffic."
#   "The observation constitutes a causal effect on the output."
#   "Sensing performs a causal intervention on each edge."   <- the worst one
#   "This is a real causal mechanism."
#   "rung 2 reached by observation alone; causal proof implied."
#   "Edge firing gives causal validation automatically."
#
# An audit that passes "sensing performs a causal intervention" is worse than
# no audit: it certifies the single most dangerous false claim this surface can
# make, and it does so with a green tick.
#
# The rule now: an occurrence is permitted only when the SAME SENTENCE denies
# the claim, gates it on a rung, or names the ladder vocabulary as vocabulary.
CAUSAL = re.compile(r"\bcausal(?:ly)?\b", re.IGNORECASE)

#: DENIALS — the claim is being refused.
_DENIAL = (
    r"not\s+(?:a\s+|yet\s+)?causal",
    r"never\s+causal",
    # `n[o']?t?\s+…` degenerated to matching mere WHITESPACE — every char
    # after `n` was optional — so it licensed "has BEEN causally validated",
    # the exact claim it was written to deny. Caught by this file's own
    # wrapped-overclaim control, which is why that control exists.
    r"\b(?:not|n't|never)\s+causally\s+validated",
    r"never\s+describ\w*\s+.{0,40}causal",
    r"forbid\w*\s+.{0,40}causal",
    r"must\s+not\s+.{0,40}causal",
    r"no\s+causal",
    r"without\s+.{0,20}causal",
    # "raising a rung REQUIRES a causal intervention" — states what this
    # surface cannot do, so it denies rather than claims.
    r"requires?\s+a\s+causal",
    r"only\s+a\s+causal",
)

#: RUNG GATES — the claim is explicitly conditioned on the ladder.
_GATED = (
    r"rung\s*(?:>=|≥)\s*2",
    r"rung\s*<\s*2",
    r"below\s+rung\s+2",
    r"rung-2\s+label",
)

#: VOCABULARY — the phrase is NAMED as a ladder label rather than ASSERTED of
#: the thing at hand.
#:
#: The distinction that matters: "rung 2 is called 'causally validated'" is
#: vocabulary; "this circuit is causally validated" is a claim. The tell is an
#: adjacent rung NUMBER or tier name — the ladder enumerating itself.
_VOCABULARY = (
    r"causal\s+(?:words|language|vocabulary|copy)",
    r"rung_language",
    # The ladder enumerated: "2 causally validated, 3 faithfulness-tested".
    r"\b[0-3]\s+causally\s+validated",
    # NOT a bare "rung 2 … causal": that let
    # "rung 2 reached by observation alone; causal proof implied" through —
    # the ladder-label rule licensing the exact overclaim it should catch.
    # The label is `causally validated`; only that pairing is vocabulary.
    r"rung\s*2[^.]{0,20}causally\s+validated",
    r"causal\s+validation[^.]{0,30}rung\s*2",
    # "Rung 3 sits ABOVE rung 2 causal validation" — comparing TIERS, which
    # names the ladder rather than claiming a circuit's place on it.
    r"rung\s*[0-3][^.]{0,20}(?:above|below)[^.]{0,20}causal",
    r"rung\s*2\)?:",                    # "…(rung 2): suppress the upstream…"
    # A TIER is a level of the ladder, not a property of a circuit.
    r"causal\s+tier",
    # The IMPERATIVE action only ("Causally validate the top-K edges"), NOT the
    # past participle. `causally validate\b` also matched "has been causally
    # VALIDATED against a held-out set" — the vocabulary rule licensing the
    # exact claim it exists to catch. Caught by this file's own wrapped-
    # overclaim control.
    r"^causally\s+validate\b",
    # Naming a LATER step denies the claim about the current one: "re-ranks
    # the shortlist BEFORE 017's causal validation" says this pass is not it.
    r"before\s+[^.]{0,30}causal",
    r"causal\s+validation\s+(?:happens|runs|occurs)",
)

#: F20 R3-16. Files that ARE the rung-2 machinery, or the vocabulary itself.
#:
#: Widening SURFACES from 5 hand-listed files to 17 discovered ones surfaced
#: three hits. All three are legitimate and are exempted BY FILE rather than by
#: loosening ALLOWED_CONTEXT, because loosening the pattern would weaken the
#: audit everywhere to accommodate three places:
#:
#:   * `evidence_ladder.py`         — DEFINES "causally validated (edge)". The
#:                                    vocabulary source cannot be forbidden from
#:                                    containing the vocabulary.
#:   * `circuit_intervention_*.py`  — IS the intervention that EARNS rung 2.
#:   * `circuit_validation_math.py` — the statistics behind that promotion.
#:   * `jlens_intervention.py`      — F26/F25: the J-space equivalent, and the
#:                                    ONLY J-space module that produces a causal
#:                                    claim. Its docstring says so, which is a
#:                                    statement about the module rather than a
#:                                    claim about a finding — the same
#:                                    distinction the circuit entries rest on.
#:
#: Deliberately a short, named list with a reason each: an exemption without a
#: reason is how an audit stops auditing. These files are still covered by the
#: PROOF/synonym checks; only the "causal" token is permitted here.
RUNG_TWO_MACHINERY = {
    "evidence_ladder.py",
    "circuit_intervention_hooks.py",
    "circuit_intervention_service.py",
    "circuit_validation_math.py",
    "circuit_faithfulness_service.py",
    # F25/F26: the J-space equivalent — the only J-space module that produces a
    # causal claim, and whose docstring says so.
    "jlens_intervention.py",
    # The worker that RUNS one: it perturbs the residual stream, continues the
    # forward pass and scores the model's own output against a matched control.
    # That is the rung-2 experiment itself, not a readout describing one.
    #
    # DELIBERATELY NARROW. `jlens_artifact_service.py` STORES these results and
    # is NOT exempt: it is mostly rung-0 serving and validation, and exempting a
    # whole file to quiet a few lines would blind the audit exactly where an
    # overclaim would do most damage. Its wording says what was run and leaves
    # the verdict to each record's own `evidence_rung`.
    "jlens_intervention_tasks.py",
}

ALLOWED_CONTEXT = re.compile(
    "|".join(_DENIAL + _GATED + _VOCABULARY), re.IGNORECASE
)


def _prose_units(text: str):
    """Yield `(line_no, sentence)` over PROSE ONLY — docstrings and comments.

    F20 R1-11/12: the audit was LINE-based, so a qualifier on the previous line
    was invisible and an overclaim could be hidden by wrapping at the right
    column (Black does this routinely at line-length 100). Rejoining lines
    fixed that and introduced a new problem: multi-line function SIGNATURES
    were glued into "sentences", producing false positives on parameter lists.

    Patching the tokeniser twice was the signal to change the unit. The audit
    is about COPY — what a human or an agent reads — and copy lives in
    docstrings. Code is not copy, so code is not scanned. Comments ARE included
    because a reviewer reads them, but they are stripped by `_code_only` where
    that is the caller's intent.
    """
    import ast
    import re as _re

    chunks: list[tuple[int, str]] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Not parseable (a tool DESCRIPTION rather than a module) — treat the
        # whole thing as prose.
        tree = None

    if tree is None:
        chunks.append((1, text))
    else:
        docstring_nodes = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    line = getattr(node, "lineno", 1)
                    chunks.append((line, doc))
                    # Remember the node so its literal is not scanned twice.
                    body = getattr(node, "body", None)
                    if body and isinstance(body[0], ast.Expr):
                        docstring_nodes.add(id(body[0].value))

        # F20 R2-01: EVERY string literal, not just docstrings.
        #
        # The AST rewrite (R1-11) fixed the line-wrapping bypass and opened a
        # new one: an overclaim in a RETURNED ERROR MESSAGE — copy an agent
        # relays verbatim — became invisible, because only docstrings were
        # scanned. Verified:
        #
        #     return {"error": "This edge is causally validated by observation."}
        #
        # passed the audit entirely. An R1 fix reintroducing the defect class
        # it removed is the pattern this increment has hit in every round.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in docstring_nodes:
                    continue
                chunks.append((getattr(node, "lineno", 1), node.value))
        # Standalone comments: prose a reviewer reads.
        #
        # F26: CONSECUTIVE comment lines are joined into one unit, exactly as
        # wrapped docstring lines already are. Scanned line-by-line, a comment
        # whose negation sits on the previous line reads as a bare claim —
        #
        #     # A threshold crossing is a readout. The number does
        #     # not make it causal.
        #
        # was flagged, while the same sentence in a docstring passed. The
        # asymmetry cuts both ways and the dangerous direction is the other
        # one: an overclaim SPLIT across two comment lines evaded the sentence
        # matcher entirely, which is the wrapped-line bypass R1-11 fixed for
        # docstrings and left open here.
        run_start = None
        run_lines = []
        for i, raw in enumerate(text.splitlines() + [""], 1):
            stripped = raw.strip()
            if stripped.startswith("#"):
                if run_start is None:
                    run_start = i
                run_lines.append(stripped.lstrip("# "))
                continue
            if run_lines:
                chunks.append((run_start, " ".join(run_lines)))
                run_start, run_lines = None, []

    for line_no, chunk in chunks:
        # Split into sentences, with wrapped lines rejoined first.
        flat = " ".join(l.strip() for l in chunk.splitlines() if l.strip())
        for sentence in _re.split(r"(?<=[.!?])\s+", flat):
            if sentence.strip():
                yield line_no, sentence.strip()


def _offending_lines(path: Path):
    bad = []
    for line_no, sentence in _prose_units(path.read_text()):
        if CAUSAL.search(sentence) and not ALLOWED_CONTEXT.search(sentence):
            bad.append((line_no, sentence[:120]))
    return bad


@pytest.mark.parametrize("path", SURFACES, ids=lambda p: p.name)
def test_no_unqualified_causal_language(path):
    if path.name in RUNG_TWO_MACHINERY:
        pytest.skip(
            f"{path.name} IS the rung-2 machinery or the vocabulary source — "
            "see RUNG_TWO_MACHINERY for why each is exempt. Skipped rather "
            "than silently passing, so the exemption stays visible in the "
            "test output."
        )
    offending = _offending_lines(path)
    assert not offending, (
        f"{path.name} has unqualified causal language on a rung-0/1 surface "
        f"(IDL-35): {offending}")


def test_discovery_report_lag0_disclosure_present():
    src = (BACKEND / "services" / "circuit_discovery_service.py").read_text()
    assert "lag0_disclosure" in src
    assert "lag-0" in src.lower()


# ── F20: the MCP tool surface ──────────────────────────────────────────────


def _code_only(text: str) -> str:
    """Strip comment lines: prose explaining the prohibition is not copy."""
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )


@pytest.mark.parametrize(
    "path", MILLM_TOOL_MODULES, ids=lambda p: p.name
)
def test_no_unqualified_causal_language_in_millm_tools(path):
    """F20 task 3.4. A mutation planting "causally validated" in a tool
    description SURVIVED the entire reachability suite until this existed."""
    bad = []
    for line_no, sentence in _prose_units(_code_only(path.read_text())):
        if CAUSAL.search(sentence) and not ALLOWED_CONTEXT.search(sentence):
            bad.append((line_no, sentence[:120]))
    assert not bad, (
        f"{path.name} makes an unqualified causal claim on a surface an AGENT "
        f"reads and relays: {bad}"
    )


def test_the_millm_module_list_is_not_empty():
    """An audit over an empty set passes forever. The glob is the point — but
    a glob that matches nothing is worse than a list."""
    names = {p.name for p in MILLM_TOOL_MODULES}
    assert "millm_circuits.py" in names, (
        "the circuit tools are not being audited"
    )
    assert len(names) >= 4


def test_registered_DESCRIPTIONS_are_audited_too():
    """F20 task 3.5. A source scan cannot see a description composed at
    runtime, inherited, or rewritten by a decorator — and that is exactly what
    `list_tools()` hands an agent."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from mcp.server.fastmcp import FastMCP

    from src.mcp_server.tools import MILLM_CATEGORY_MODULES

    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, None))
    mcp = FastMCP("audit")
    for modules in MILLM_CATEGORY_MODULES.values():
        for module in modules:
            module.register(mcp, MagicMock(), gate)

    tools = asyncio.run(mcp.list_tools())
    assert any(t.name == "millm_circuit_status" for t in tools), (
        "the scan is not seeing the circuit tools, so it would pass however "
        "they were worded"
    )

    bad = []
    for tool in tools:
        for _no, sentence in _prose_units(tool.description or ""):
            if CAUSAL.search(sentence) and not ALLOWED_CONTEXT.search(sentence):
                bad.append(f"{tool.name}: {sentence[:120]}")
    assert not bad, (
        "a REGISTERED tool description asserts causal evidence:\n  "
        + "\n  ".join(bad)
    )


class TestTheAuditCanActuallyFail:
    """F20 R1-09..12. The audit's own negative controls, in the file rather
    than in a reviewer's memory.

    An audit nobody has watched fail is an audit nobody has tested. The
    previous `ALLOWED_CONTEXT` whitelisted TOPIC WORDS, so naming the topic
    legitimised any claim about it — all six of these passed, verified by
    execution. The worst was "sensing performs a causal intervention": an audit
    that certifies that is worse than no audit at all.
    """

    OVERCLAIMS = [
        "These edges are backed by causal evidence from live traffic.",
        "The observation constitutes a causal effect on the output.",
        "Sensing performs a causal intervention on each edge.",
        "This is a real causal mechanism.",
        "rung 2 reached by observation alone; causal proof implied.",
        "Edge firing gives causal validation automatically.",
    ]

    #: Legitimate uses that MUST keep passing. An audit that cannot tell these
    #: apart gets disabled by whoever it blocks next.
    LEGITIMATE = [
        "Raising one requires a causal intervention, which happens in miStudio.",
        "This circuit is NOT causally validated — its rung is below 2.",
        "min_rung filters by rung (0 mined, 1 attribution-supported, "
        "2 causally validated, 3 faithfulness-tested).",
        "Observation NEVER raises a rung.",
        "Rung 3 sits above rung 2 causal validation.",
        "re-ranks the shortlist before 017's causal validation",
    ]

    @pytest.mark.parametrize("sentence", OVERCLAIMS)
    def test_an_overclaim_is_caught(self, sentence):
        assert CAUSAL.search(sentence) and not ALLOWED_CONTEXT.search(sentence), (
            f"the audit PASSES an overclaim: {sentence!r}"
        )

    @pytest.mark.parametrize("sentence", LEGITIMATE)
    def test_a_legitimate_use_is_permitted(self, sentence):
        assert not (CAUSAL.search(sentence) and not ALLOWED_CONTEXT.search(sentence)), (
            f"the audit FLAGS legitimate copy: {sentence!r} — false positives "
            "are how an audit gets loosened until it stops auditing"
        )

    def test_an_overclaim_in_a_STRING_LITERAL_is_caught(self):
        """F20 R2-01. The AST rewrite (R1-11) scanned docstrings only, so an
        overclaim in a RETURNED ERROR MESSAGE — copy an agent relays verbatim —
        became invisible. Verified: this exact snippet passed the audit.

        An R1 fix reintroducing the defect class it removed is the pattern this
        increment has hit in every round, which is why the control is here."""
        src = (
            'def tool():\n'
            '    """A clean docstring."""\n'
            '    return {"error": "This edge is causally validated by '
            'observation."}\n'
        )
        flagged = [
            u for _n, u in _prose_units(src)
            if CAUSAL.search(u) and not ALLOWED_CONTEXT.search(u)
        ]
        assert flagged, (
            "an overclaim in a returned message is invisible to the audit — "
            "the agent relays it and the docstrings stay clean"
        )

    def test_a_WRAPPED_overclaim_is_still_caught(self):
        """R1-11: the audit was LINE-based, so wrapping at the right column
        hid an overclaim — and Black wraps docstrings routinely."""
        wrapped = (
            '"""This circuit has been causally\n'
            '        validated against a held-out set."""'
        )
        units = list(_prose_units(wrapped))
        flagged = [
            s for _n, s in units
            if CAUSAL.search(s) and not ALLOWED_CONTEXT.search(s)
        ]
        assert flagged, (
            "a claim split across two lines is invisible to the audit — "
            "wrapping is now a bypass"
        )


# ── Feature 026: J-space claims discipline (BR-019, BR-020, BR-024) ─────────


class TestJSpaceSurfacesAreDiscovered:
    """The corpus must come from the FILESYSTEM, not from a list.

    This is the finding that produced the whole approach: `SURFACES` was
    hand-maintained at 5 files while 16 circuit modules shipped unaudited, and
    the suite was green throughout. A list is only as good as the last person
    who remembered it.
    """

    def test_the_jspace_corpus_is_not_empty(self):
        assert JSPACE_SURFACES, "J-space discovery matched nothing at all"

    def test_it_contains_the_modules_that_exist_today(self):
        names = {p.name for p in JSPACE_SURFACES}
        for expected in (
            "jlens_readout_service.py",
            "jlens_validation.py",
            "jlens_band_report.py",
            "jlens_fitter.py",
            "jlens.py",
            "jspace_claims.py",
        ):
            assert expected in names, f"{expected} is not audited"

    def test_discovery_spans_every_layer_a_claim_can_reach(self):
        """A claim can ship from a service, an endpoint, a worker or a tool."""
        parents = {p.parent.name for p in JSPACE_SURFACES}
        for layer in ("services", "ml", "endpoints", "schemas", "workers", "tools"):
            assert layer in parents, f"no J-space {layer} module is audited"


@pytest.mark.parametrize("path", JSPACE_SURFACES, ids=lambda p: p.name)
def test_jspace_surface_makes_no_unearned_causal_claim(path):
    """J-space evidence is rung 0 unless an intervention with a control says otherwise."""
    if path.name in RUNG_TWO_MACHINERY:
        pytest.skip(
            f"{path.name} IS the rung-2 machinery — see RUNG_TWO_MACHINERY for "
            "why. Skipped rather than silently passing, so the exemption stays "
            "visible in the test output."
        )
    violations = _offending_lines(path)
    assert not violations, (
        f"{path.name} states a causal claim in user-facing text. J-space "
        "readouts are rung 0 — present is not used (BR-019).\n"
        + "\n".join(f"  line {ln}: {text}" for ln, text in violations)
    )


class TestTheAuditWouldCatchAPlantedClaim:
    """Negative control. An audit nobody has seen fail is not evidence."""

    def test_a_causal_string_is_detected(self, tmp_path):
        planted = tmp_path / "jlens_planted.py"
        planted.write_text(
            "MESSAGE = 'This readout shows the feature causally drives the output.'\n"
        )
        assert _offending_lines(planted), "the audit does not see a planted claim"

    def test_a_NEGATED_statement_is_permitted(self, tmp_path):
        """The audit is negation-anchored, and that is what J-space needs most.

        Nearly every honest J-space sentence is a DENIAL of a causal claim —
        "a readout is not a causal claim", "absence is not evidence of
        absence". A topic-anchored audit would flag exactly the sentences the
        product is required to say.
        """
        planted = tmp_path / "jlens_negated.py"
        planted.write_text(
            "MESSAGE = 'A concept in a readout is not a causal claim.'\n"
        )
        assert not _offending_lines(planted)

    def test_a_DENIAL_SPLIT_ACROSS_COMMENT_LINES_is_still_a_denial(self, tmp_path):
        """Consecutive comment lines are one prose unit, as wrapped docstrings are.

        Scanned line-by-line, the second line of

            # A concept in a readout is not
            # a causal claim.

        reads as the bare phrase "a causal claim" and is flagged — while the
        identical sentence in a docstring passes, because docstrings already
        rejoin wrapped lines. The asymmetry cost real precision: it pushes
        authors to reword the very caveats BR-019 requires until the audit
        stops objecting.
        """
        planted = tmp_path / "jlens_split_denial.py"
        planted.write_text(
            "# A concept appearing in a readout is not\n"
            "# a causal claim.\n"
            "VALUE = 1\n"
        )
        assert not _offending_lines(planted), (
            "a denial split across two comment lines was read as a claim"
        )

    def test_an_OVERCLAIM_SPLIT_ACROSS_COMMENT_LINES_is_still_caught(self, tmp_path):
        """And the joining must not open the bypass it closes.

        The dangerous direction: an overclaim split so neither line contains a
        whole claim. Joining is what makes the sentence visible at all.
        """
        planted = tmp_path / "jlens_split_claim.py"
        planted.write_text(
            "# This readout shows the feature\n"
            "# causally drives the output.\n"
            "VALUE = 1\n"
        )
        assert _offending_lines(planted), (
            "an overclaim split across two comment lines evaded the audit"
        )

    def test_the_shipped_caveats_survive_their_own_audit(self):
        """The strings BR-019/BR-020 REQUIRE must not be what the audit rejects.

        A rule that forbids its own remedy is unimplementable, and the way that
        surfaces is a caveat quietly reworded until it passes — losing the
        precision that made it worth saying.
        """
        from src.schemas import jspace_claims

        path = BACKEND / "schemas" / "jspace_claims.py"
        assert not _offending_lines(path)
        assert "not a causal claim" in jspace_claims.READOUT_NOT_CAUSAL
        assert "not evidence" in jspace_claims.ABSENCE_CAVEAT


# ── Clusters: a correlational object may make no causal claim ──────────────


class TestClusterSurfacesAreDiscovered:
    """The corpus comes from the filesystem, under both names.

    `cluster_allocation_service.py` and `cluster_profile_service.py` were the
    two modules I could name from a directory listing. Globbing found ten more,
    six of which do not contain the word "cluster" at all — the rename never
    reached the backend.
    """

    def test_the_cluster_corpus_is_not_empty(self):
        assert CLUSTER_SURFACES, "cluster discovery matched nothing at all"

    def test_it_contains_the_modules_that_exist_today(self):
        names = {p.name for p in CLUSTER_SURFACES}
        for expected in (
            "cluster_allocation_service.py",
            "cluster_profile_service.py",
            "cluster_profiles.py",
            "cluster_profile.py",
        ):
            assert expected in names, f"{expected} is not audited"

    def test_the_PRE_RENAME_names_are_audited_too(self):
        """The half a `cluster_*` glob misses.

        Fails if someone 'tidies' the globs down to the current vocabulary.
        """
        names = {p.name for p in CLUSTER_SURFACES}
        for expected in (
            "feature_grouping_service.py",   # the grouping itself
            "feature_groups.py",             # REST, reaches the UI directly
            "feature_group.py",
            "feature_grouping_tasks.py",
            "feature_grouping.py",
            "groups.py",                     # MCP: an agent relays this verbatim
        ):
            assert expected in names, (
                f"{expected} is not audited — the Feature Groups → Clusters "
                "rename was UI-only, so a cluster_* glob alone misses it"
            )

    def test_discovery_spans_every_layer_a_claim_can_reach(self):
        parents = {p.parent.name for p in CLUSTER_SURFACES}
        for layer in ("services", "endpoints", "schemas", "workers", "models", "tools"):
            assert layer in parents, f"no cluster {layer} module is audited"


@pytest.mark.parametrize("path", CLUSTER_SURFACES, ids=lambda p: p.name)
def test_cluster_surface_makes_no_causal_claim(path):
    """Co-occurrence is not mechanism, and a dial is not evidence.

    Nothing in the CLUSTERS chain earns a causal claim: across all thirteen of
    its documents the words causal, rung and intervention appear zero times.
    The code must hold the same line.
    """
    if path.name in RUNG_TWO_MACHINERY:
        pytest.skip(
            f"{path.name} IS the rung-2 machinery — see RUNG_TWO_MACHINERY for "
            "why. Skipped rather than silently passing, so the exemption stays "
            "visible in the test output."
        )
    violations = _offending_lines(path)
    assert not violations, (
        f"{path.name} states a causal claim in user-facing text. A cluster is "
        "features grouped by co-occurrence and context similarity; it earns a "
        "causal claim only by being mined into a circuit (BR-016) and passing "
        "rung 2.\n"
        + "\n".join(f"  line {ln}: {text}" for ln, text in violations)
    )
