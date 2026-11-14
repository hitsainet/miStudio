"""Every workflow gate accepted this repository's old AND new name until P1 closed; from the export whose
header says `closed: "P1"` it names the new one only (042 FR-042.12; FTDD §11.1; CF-13, CF-17, CF-21;
FTASKS 7.1, 10.2).

ENH-009 P1 renamed the private and the public repository (each `…/miStudio` → `…/miWorkbench`). A gate naming only the old name would
have stopped the mirror sync, the image build or the gitops sync the moment GitHub renamed the repository;
a gate still naming the old one after P1 blocks S001's P5 gate (no old private name outside history). The
pairs come from the export's `repo` records, the mode from its header's `closed` (never `phase`).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from tests.support import legacy_export as le
from tests.support import workflow_gates as wg


def _committed() -> le.Export:
    export = le.load()
    if export.is_public and any(n.startswith("repo:sha256:") for n in export.withheld):
        pytest.skip("on the public mirror the private repository's repo records are withheld: "
                    + ", ".join(n for n in export.withheld if n.startswith("repo:")))
    return export


def _fixture(tmp_path: Path, **header) -> le.Export:
    """The committed export with header keys changed (a variant that loads through the real loader)."""
    src = _committed().path.read_text(encoding="utf-8")
    head, _, rest = src.partition("```jsonl\n")
    first, _, others = rest.partition("\n")
    h = json.loads(first)
    if "closed" in header and "phase" not in header:
        # A header names the phase it was exported for: `closed` or the one after it (CF-21). Staging an earlier
        # `closed` keeps that rule, so the variant is the export as committed DURING the next phase (P2 close,
        # 2026-11-10: the committed export moved past P1 and a kept `phase` made every variant unloadable).
        header = {**header, "phase": le.PHASES[le.PHASES.index(header["closed"]) + 1]}
    h.update(header)
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / le.FULL_NAME if h["variant"] == "full" else tmp_path / le.PUBLIC_NAME
    out.write_text(head + "```jsonl\n" + json.dumps(h, separators=(",", ":")) + "\n" + others, encoding="utf-8")
    return le.load(path=out)


def _with_old_halves(workflows: dict) -> dict:
    """The real workflows with each gate's old half put back (what the tree held until P1 closed)."""
    text = json.dumps(workflows)
    for old, new in sorted(wg.this_repo_pairs(_committed()).values()):
        text = re.sub(r"github\.repository == '" + re.escape(new) + "'",
                      f"github.repository == '{old}' || github.repository == '{new}'", text)
    return json.loads(text)


def test_discovery_finds_the_known_gates():
    found = wg.gates(wg.load_workflows())
    files = {g.workflow for g in found}
    # Fail closed: the check reads nothing if discovery breaks.
    assert len(found) >= 8, found
    assert {"docker-images.yml", "release.yml", "sync-to-clean.yml", "sync-main-to-gitops-pilot.yml",
            "build-neuronpedia.yml", "backend-tests.yml", "frontend-ci.yml"} <= files, files


def test_from_closed_p1_gates_name_new_only():
    """The real tree against the COMMITTED export (no fixture): P1 has closed, so every gate names the new
    name and no gate keeps an old half (FTASKS 10.2)."""
    export = _committed()
    assert export.closed_at_least("P1"), (
        "the committed export must say closed: P1 (FTASKS 10.1) for this test to run in new-only mode")
    real = wg.load_workflows()
    assert len(wg.gates(real)) >= 8 and wg.this_repo_pairs(export), "nothing to check: fail closed"
    problems = wg.findings(real, export)
    assert problems == [], "\n".join(problems)
    # The rule bites: with the old halves back, every gate is an old-half finding.
    old_halves = [p for p in wg.findings(_with_old_halves(real), export) if "the old half" in p]
    assert len(old_halves) == len(wg.gates(real)), old_halves


def test_every_gate_accepts_both_names_before_p1_closes(tmp_path):
    """The before-P1 rule, kept on a fixture export (`closed: "P0"`) now that the committed one says P1:
    the tree as it stood until P1 closed passes it, the new-only tree does not."""
    before_p1 = _fixture(tmp_path, closed="P0")
    real = wg.load_workflows()
    assert wg.findings(_with_old_halves(real), before_p1) == []
    both = [p for p in wg.findings(real, before_p1) if "must name both" in p]
    assert len(both) == len(wg.gates(real)), both


def test_switch_keys_to_closed_not_phase(tmp_path):
    # A --phase P1 export committed DURING P1 says closed: P0 and keeps the both-names mode; the same export
    # differing only in `closed` (P1) switches to new-only. The committed export (closed P1 or later) is new-only.
    committed = _committed()
    during_p1 = _fixture(tmp_path / "during", phase="P1", closed="P0")
    at_p1_close = _fixture(tmp_path / "closed", phase="P1", closed="P1")
    assert during_p1.header["phase"] == at_p1_close.header["phase"]
    real = wg.load_workflows()
    assert wg.findings(real, during_p1)
    assert wg.findings(_with_old_halves(real), during_p1) == []
    assert wg.findings(real, at_p1_close) == []
    assert wg.findings(real, committed) == []


def test_a_gate_naming_another_repository_or_form_is_a_finding(tmp_path):
    export = _committed()
    pairs = wg.this_repo_pairs(export)
    owner, (old, new) = sorted(pairs.items())[0]
    docs = {"x.yml": {"jobs": {
        "a": {"if": f"github.repository == '{owner}/elsewhere' || github.repository == '{old}' "
                    f"|| github.repository == '{new}'"},
        "b": {"if": f"github.repository_owner == '{owner}'"},
        "c": {"if": f"github.repository == '{old}'"},
        "d": {"steps": [{"if": f"github.repository == 'nobody/{new.split('/')[1]}'"}]},
    }}}
    for mode in (export, _fixture(tmp_path, closed="P0")):
        problems = wg.findings(docs, mode)
        assert any("jobs.a" in p and "neither" in p for p in problems), problems
        assert any("jobs.b" in p and "cannot read" in p for p in problems), problems
        assert any("steps[0]" in p and "no repo record" in p for p in problems), problems
    closed = wg.findings(docs, export)
    assert any("jobs.c" in p and "the old half" in p for p in closed), closed
    assert any("jobs.c" in p and "must name" in p for p in closed), closed
    assert any("jobs.a" in p and "the old half" in p for p in closed), closed
    before = wg.findings(docs, _fixture(tmp_path, closed="P0"))
    assert any("jobs.c" in p and "both" in p for p in before), before


def _staged_close(tmp_path: Path, closed: str, phase: str) -> le.Export:
    """The committed export exactly as S001 will commit it at a LATER phase close: the same records, the
    header's `phase` and `closed` moved (a P-close export changes nothing else that these tests read)."""
    src = le.load().path.read_text(encoding="utf-8")
    head, _, rest = src.partition("```jsonl\n")
    first, _, others = rest.partition("\n")
    h = {**json.loads(first), "closed": closed, "phase": phase}
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / (le.FULL_NAME if h["variant"] == "full" else le.PUBLIC_NAME)
    out.write_text(head + "```jsonl\n" + json.dumps(h, separators=(",", ":")) + "\n" + others, encoding="utf-8")
    return le.load(path=out)


def _close_headers() -> list[tuple[str, str]]:
    """Every `(closed, phase)` a committed export can carry from P1's close to the end: the close export
    itself (`phase == closed`) and the export regenerated during the next phase."""
    out = []
    for i, closed in enumerate(le.PHASES):
        if i == 0:
            continue
        out.append((closed, closed))
        if i + 1 < len(le.PHASES):
            out.append((closed, le.PHASES[i + 1]))
    return out


@pytest.mark.parametrize("closed,phase", _close_headers())
def test_fixtures_survive_every_later_phase_close(tmp_path, monkeypatch, closed, phase):
    """A P-close export must never again break this file's fixtures silently (review of 2026-11-10: the
    P2-close export moved the committed header past P1 and every `_fixture` variant became unloadable,
    found only when CI ran the P2 tree).

    Every fixture here is DERIVED from the committed export, so its validity depends on a header S001
    rewrites at each close. This test stages the committed export as it will read at every later close
    and runs each fixture-building test of this file against it. A fixture that only loads for today's
    `closed` is red here, phases before the close that would break it.
    """
    _committed()  # the mirror skip, by name, before `_committed` is replaced below
    staged = _staged_close(tmp_path / "staged", closed, phase)
    assert (staged.closed, staged.phase) == (closed, phase)
    monkeypatch.setattr(sys.modules[__name__], "_committed", lambda: staged)
    assert _committed() is staged
    # The fixture builder itself, for every `closed` it is asked to stage below the committed one.
    for earlier in le.PHASES[: le.PHASES.index(closed)]:
        built = _fixture(tmp_path / f"f-{earlier}", closed=earlier)
        assert built.closed == earlier and built.records == staged.records
    # Every test of this file that builds a fixture or reads the committed header.
    test_from_closed_p1_gates_name_new_only()
    test_every_gate_accepts_both_names_before_p1_closes(tmp_path / "t1")
    test_switch_keys_to_closed_not_phase(tmp_path / "t2")
    test_a_gate_naming_another_repository_or_form_is_a_finding(tmp_path / "t3")
    test_mirror_name_is_derived()


def test_the_close_staging_covers_the_committed_header():
    """Fail closed: the committed header is one of the staged ones (so today's tree is a case of the test
    above, not an exception to it), and later closes are staged too."""
    export = _committed()
    headers = _close_headers()
    assert (export.closed, export.phase) in headers, (export.closed, export.phase)
    later = [h for h in headers if le.PHASES.index(h[0]) > le.PHASES.index(export.closed)]
    assert later or export.closed == le.PHASES[-1], "no later close is staged: the guard checks nothing ahead"


def test_mirror_name_is_derived():
    workflows = wg.load_workflows()
    export = _committed()
    private, public = wg.private_and_public_owner(workflows, export)
    sync = workflows["sync-to-clean.yml"]["jobs"]["sync"]
    assert sync["env"]["MIRROR"] == public + "/${{ github.event.repository.name }}", (
        "MIRROR must be derived from this repository's own name (PD-16), never spelled, and never "
        "the private repository")
    assert public != private
