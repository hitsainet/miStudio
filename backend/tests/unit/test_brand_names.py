"""The brand guard, backend half (042 FR-042.13, FR-042.34; FTDD §6.7; FTID §3.14; FTASKS 7.8, 7.9; T-16).

Report mode until S001 closes P4 (`BRAND_GUARD_MODE`, tied to the export by `test_brand_guard_mode.py`):
`test_real_tree_report` prints every finding and passes; in enforce mode it fails on any. Everything else
here is enforced in both modes: the scanner finds one plant per surface class, discovery is never empty,
the MCP surface is the live registry, old repository names pass only through `transition` repo records,
the guard modules carry no repository name, the suite-phrase rules come from the export's guard records by
phase, and the export itself is never a surface (CF-20).

Nothing here re-types a register fact: allowances come from `docs/legacy-identifiers*.md` through
`tests/support/legacy_export.py`; repository names are taken from its `repo` records.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from tests.support import brand_surfaces as bs
from tests.support import legacy_export as le

ROOT = le.REPO_ROOT


@pytest.fixture(scope="module")
def export() -> le.Export:
    return le.load()


def _with(export: le.Export, *, header: dict | None = None, records=None) -> le.Export:
    """A copy of the committed export with header keys or records replaced (never re-typed)."""
    return le.Export(export.path, {**export.header, **(header or {})},
                     tuple(export.records if records is None else records))


def _git_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "tree"
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return root


# One plant per surface class. Each text carries the plant once, beside text the guard must ignore.
FIXTURE_FILES = {
    "README.md": "# Plant miStudio here\n\nBody text naming miLLM is not a heading.\n",
    "manual/docs/intro.md": "---\ntitle: Planted millm page\n---\n\nBody miStudio text.\n",
    "manual/docusaurus.config.ts": "export default {\n  title: 'Planted MechInterp Studio',\n  url: 'x',\n};\n",
    "frontend/public/icon.svg": "<svg><title>miLLM plant</title><desc>miStudio not a surface</desc></svg>\n",
    "frontend/index.html": "<html><head><title>miStudio plant</title></head><body>miLLM</body></html>\n",
    "backend/src/planted.py": '"""Docstring naming miStudio is not scanned."""\nLABEL = "Open miLLM now"\n# miStudio comment\n',
}


def _tiny_app():
    from fastapi import FastAPI

    app = FastAPI(title="Planted app")

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "miStudio planted service"}

    return app


def _tiny_mcp():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("planted", instructions="Instructions with nothing old.")

    @server.tool()
    async def plain_tool() -> str:
        """Description naming miLLM once."""
        return "x"

    return server


def test_fixture_tree_one_finding_per_surface_class(tmp_path, export):
    root = _git_tree(tmp_path, FIXTURE_FILES)
    rules = bs.rules_for(export)
    per_class: dict[str, list[bs.Finding]] = {}
    for cls in bs.FILE_CLASSES:
        _files, items = bs.file_items(cls, root)
        per_class[cls] = bs.scan(items, rules)
    per_class["api"] = bs.scan(bs.api_items(_tiny_app()), rules)
    per_class["mcp"] = bs.scan(bs.mcp_items(_tiny_mcp()), rules)
    assert set(per_class) == set(bs.ENFORCED_CLASSES) | set(bs.REPORT_ONLY_CLASSES)
    counts = {cls: len(f) for cls, f in per_class.items()}
    assert counts == {cls: 1 for cls in per_class}, {c: [f.line() for f in fs] for c, fs in per_class.items()}


@pytest.mark.parametrize("cls", bs.FILE_CLASSES)
def test_discovery_not_empty_per_class(cls):
    files, items = bs.file_items(cls)
    assert files, f"surface class {cls!r} discovered no file: a guard that discovers nothing asserts nothing"
    assert items, f"surface class {cls!r} extracted no text from {len(files)} file(s)"
    assert not [f for f in files if f.startswith(("0xcc/", ".claude/"))]


@pytest.mark.parametrize("cls", bs.LIVE_CLASSES)
def test_discovery_not_empty_per_live_class(cls):
    items = bs.live_items(cls)
    assert items and all(i.text for i in items if i.where != "server instructions"), cls


def test_real_tree_report(export):
    """Prints every finding per class; fails only where the guard is enforced: the constant says
    `enforce` AND this tree holds the full export (R23: the private repository enforces, the public
    mirror reports). The report-only class (TPD-16) is printed and never enforced."""
    rules = bs.rules_for(export)
    failures: list[str] = []
    for cls in (*bs.FILE_CLASSES, *bs.LIVE_CLASSES):
        items = bs.file_items(cls)[1] if cls in bs.FILE_CLASSES else bs.live_items(cls)
        findings = bs.scan(items, rules)
        mode = "report" if cls in bs.REPORT_ONLY_CLASSES else bs.effective_mode(export)
        failures += bs.report(cls, findings, mode)
    assert not failures, "\n".join(failures)


def test_mcp_surface_read_from_build_server(monkeypatch):
    """The MCP surface is the BUILT server's registry: the same tool set the live-registry recipe sees."""
    from src.mcp_server.tools import howto
    from tests.support.next_calls import live_input_schemas

    items = bs.live_items("mcp")
    names = {i.text for i in items if i.where.startswith("tool name ")}
    assert names == set(live_input_schemas()), "brand guard's tool list differs from the live registry"
    assert len(names) > 300
    described = {i.where.removeprefix("tool description ") for i in items if i.where.startswith("tool description ")}
    assert described == names
    server = [i for i in items if i.where == "server name"]
    assert len(server) == 1 and server[0].text  # the server's own name, read from the object
    topics = [i for i in items if i.where.startswith("howto topic ")]
    assert len(topics) >= 10 and all(len(t.text) > 100 for t in topics), "howto topics were not read"
    # A PLANT through the real howto path. Before 2026-11-11 this line asked only that SOME topic held a
    # finding, which was true while the real guidance still named an old brand and would have gone red,
    # proving nothing, the day the guidance was clean. Now one old brand is put into the text the real
    # tool serves for one topic, the server is built again, and exactly that topic gains exactly one finding.
    rules = bs.rules_for(le.load())
    topic = sorted(howto.TOPICS)[-1]
    where = f"howto topic {topic}"
    before = [f.line() for f in bs.scan([i for i in topics if i.where == where], rules)]
    marker = "Planted: open mi" + "Studio now."
    monkeypatch.setitem(howto.TOPICS, topic, howto.TOPICS[topic] + "\n" + marker + "\n")
    planted = [i for i in bs.mcp_items(bs.live_mcp_server()) if i.where == where]
    assert len(planted) == 1 and marker in planted[0].text, "the planted topic text was not read from the server"
    after = [f.line() for f in bs.scan(planted, rules)]
    gained = list(after)
    for line in before:
        gained.remove(line)
    assert len(gained) == 1 and gained[0].startswith(f"[mcp] {where}: mi" + "Studio "), (before, after)


def _howto_mcp(text: str):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("planted", instructions="Nothing old here.")

    @server.tool()
    async def planted_howto(topic: str | None = None) -> dict:
        """A guide with one topic."""
        return {"topics": ["shapes"]} if topic is None else {"topic": topic, "text": text}

    return server


def test_howto_text_is_scanned_as_the_agent_reads_it(export):
    """A tool answers with a JSON document. Scanned raw, a line starting with an old brand was reported
    as the token `n<brand>` (the `\\n` escape), and a quoted document kind the register allows
    (DR-01, with its quotes) was a finding because its closing quote arrived escaped. The parsed strings
    are what an agent reads, so they are what is scanned."""
    kind = '"mi' + 'studio.circuit-definition"'
    old = "mi" + "Studio"
    text = f"The kind is {kind} exactly.\n{old} starts this line."
    items = [i for i in bs.mcp_items(_howto_mcp(text)) if i.where == "howto topic shapes"]
    assert len(items) == 1 and kind in items[0].text and "\\n" not in items[0].text, items
    findings = bs.scan(items, bs.rules_for(export))
    assert [f.token for f in findings] == [old], [f.line() for f in findings]
    # The raw form is what the two defects came from (so the decode is doing the work here).
    raw = bs.scan([bs.Item("mcp", "raw", json.dumps({"text": text}))], bs.rules_for(export))
    assert sorted(f.token for f in raw) != [old], [f.line() for f in raw]


def _repo_names(export: le.Export) -> list[str]:
    names = {r[k] for r in export.repos() for k in ("old", "new")}
    assert names, "the export lists no repo record"
    return sorted(names)


def test_old_repo_name_allowed_only_by_transition_record(export):
    # Since ENH-009 P7 every repo record is `retired`; the allowance rule is exercised on the committed
    # records put back into `transition` (a copy, never re-typed), and the retired export itself is checked
    # below and by `test_retired_repo_records_allow_nothing`.
    if not export.repos():
        pytest.fail("the export has no repo record to test with")
    if not export.repos("transition"):
        export = _with(export, records=[{**r, "status": "transition"} if r["section"] == "repo" else r
                                        for r in export.records])
    olds = [r["old"] for r in export.repos("transition")]
    rules = bs.rules_for(export)
    for old in olds:
        url = f"Source: https://github.com/{old}.git and {old}/issues"
        assert bs.scan([bs.Item("readme", "x", url)], rules) == [], old
        # Only the WHOLE <owner>/<name> token: a longer name is not covered (TPD-33, miForge N31).
        longer = bs.scan([bs.Item("readme", "x", f"see {old}-backup")], rules)
        assert len(longer) == 1, (old, [f.line() for f in longer])
        # Not the name part alone, under another owner.
        other = bs.scan([bs.Item("readme", "x", "see someone/" + old.split("/", 1)[1])], rules)
        assert len(other) == 1, old
    # P7: the record turns `retired`; every remaining spelling is then a finding.
    retired = _with(export, records=[{**r, "status": "retired"} if r["section"] == "repo" else r
                                     for r in export.records])
    for old in olds:
        found = bs.scan([bs.Item("readme", "x", f"https://github.com/{old}")], bs.rules_for(retired))
        assert found, f"a retired repo record still allows {old}"


def test_retired_repo_records_allow_nothing(export):
    """ENH-009 P7 (FTASKS 18.12): the committed export retires every repo record, so no old repository name
    passes the guard any more."""
    retired = export.repos("retired")
    if export.header.get("phase") in ("P7", "P8") and not export.header.get("withheld"):
        assert retired and not export.repos("transition"), "the P7 export still holds a transition repo record"
    rules = bs.rules_for(export)
    for rec in retired:
        found = bs.scan([bs.Item("readme", "x", f"https://github.com/{rec['old']}")], rules)
        assert found, f"the retired repo record still allows {rec['old']}"


def test_guard_modules_hold_no_repo_literal(export):
    """TPD-33: no hand-kept allow list. The guard modules name no repository (old or new); every allowance
    comes from the export's `repo` records."""
    names = _repo_names(export)
    modules = [ROOT / "backend/tests/support/brand_surfaces.py", ROOT / "frontend/src/test/brandNames.test.ts",
               ROOT / "frontend/src/test/brandMode.ts", ROOT / "frontend/src/test/legacyExport.ts"]
    offenders = []
    for path in modules:
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            strings = [n.value for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            text = text + "\n" + "\n".join(strings)
        for name in names:
            if re.search(re.escape(name), text, re.IGNORECASE):
                offenders.append(f"{path.relative_to(ROOT)} names {name}")
    assert not offenders, offenders


def _at(export: le.Export, phase: str) -> bs.Rules:
    closed = {"P4": "P3", "P5": "P4"}[phase]
    return bs.rules_for(_with(export, header={"phase": phase, "closed": closed}))


def _found(rules: bs.Rules, text: str) -> list[bs.Finding]:
    return bs.scan([bs.Item("readme", "plant", text)], rules)


def test_suite_phrase_rules_by_phase(export):
    """S001 FR-S001.69 / CF-15 / CF-22 / CF-23 (selectable with `-k suite_phrase`). Phase rules come only
    from the export's guard records: neither guard module holds a phrase or a phase."""
    p4, p5 = _at(export, "P4"), _at(export, "P5")
    suite_phrase, open_app = "part of the miStudio suite", "Open miStudio"
    assert _found(p4, suite_phrase) and not _found(p5, suite_phrase)
    assert _found(p4, open_app) and _found(p5, open_app)
    # miSuite: the suite's name until P5 (allowed), then a G-MISUITE finding.
    assert _found(p4, "miSuite") == []
    p5_misuite = _found(p5, "miSuite")
    assert [f.rules for f in p5_misuite] == [["G-MISUITE"]], [f.line() for f in p5_misuite]
    # CF-22: the suite's paths through G-STUDIO-PATH from P5. Withheld from the public export (it names a
    # workstation path), so the mirror skips this part BY NAME.
    le.require(export, "guard", "G-STUDIO-PATH")
    for path in ("~/app/miStudio/scripts", "-home-x-sean-app-miStudio"):
        assert _found(p4, path), path
        assert _found(p5, path) == [], path


def test_export_files_are_never_surfaces(tmp_path):
    """CF-20: the export names every old name by design, so it is never scanned — root-level and nested
    copies, under a `docs/` positive pathspec and a repository-wide one. Both layers are checked: git's
    glob exclusions (both forms; the `**/` form alone leaks the root-level file on git 2.43) and the
    module's own filter. The finding count is unchanged by the export's lines."""
    export_text = (le.paths()[0] if le.paths()[0].exists() else le.paths()[1]).read_text(encoding="utf-8")
    plant = "# Heading naming miStudio\n"
    base = {"README.md": plant, "docs/notes.md": plant}
    copies = {"docs/legacy-identifiers.md": export_text + plant,
              "docs/legacy-identifiers.public.md": export_text + plant,
              "sub/docs/legacy-identifiers.md": export_text + plant}
    with_export = _git_tree(tmp_path / "a", {**base, **copies})
    without = _git_tree(tmp_path / "b", base)
    for globs in (("docs/*.md",), ("**/*.md",), ("docs/**",), ("*.md", "**/*.md")):
        specs = [f":(glob){g}" for g in globs]
        raw = subprocess.run(["git", "ls-files", "--", *specs, *bs.EXPORT_EXCLUDES], cwd=with_export,
                             capture_output=True, text=True, check=True).stdout.split()
        assert not [p for p in raw if "legacy-identifiers" in p], (globs, raw)  # git layer
        listed = bs.ls_files(with_export, globs)
        assert listed == bs.ls_files(without, globs), globs                      # both layers
    for p in copies:
        assert bs._EXPORT_NAME.search(p), p
    # The module's own filter alone (git's exclusions switched off) still drops every copy.
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(bs, "EXPORT_EXCLUDES", ())
        for globs in (("docs/*.md",), ("**/*.md",)):
            assert bs.ls_files(with_export, globs) == bs.ls_files(without, globs), globs
    finally:
        mp.undo()
    count = lambda root: len(bs.scan([i for f in bs.ls_files(root, ("**/*.md",))
                                      for i in bs.readme_items(f, (root / f).read_text())],
                                     bs.rules_for(le.load())))
    assert count(with_export) == count(without) == 2


def _allowed_elsewhere(export: le.Export, entry: dict, text: str) -> bool:
    """An INDEPENDENT reference (never the scanner's rules): is every brand stem of `text` covered by an
    export record other than `entry`? A scanner allowance that the export does not hold cannot hide here."""
    spans: list[tuple[int, int]] = []
    for other in export.entries():
        if other is entry or other["status"] == "retired":
            continue
        spans += [m.span() for m in re.finditer(other["pattern"], text)]
        for lit in other["literals"]:
            spans += [(m.start(), m.end()) for m in re.finditer(re.escape(lit), text)]
    for r in [*export.env(), *export.hosts(), *export.history_tools()]:
        if r.get("status", "transition") == "transition":
            spans += [(m.start(), m.end()) for m in re.finditer(re.escape(r["old"]), text, re.IGNORECASE)]
    for r in export.repos("transition"):
        spans += [m.span() for m in re.finditer(r"(?<![\w-])" + re.escape(r["old"]) + r"(?![\w-])", text, re.I)]
    for g in export.active_guards():
        if g["effect"] == "allow":
            spans += [m.span() for m in re.finditer(g["pattern"], text)]
    stems = [m.span() for m in re.finditer(r"mi(?:studio|llm)|mechinterp[ -]?studio", text, re.IGNORECASE)]
    return all(any(s <= a and b <= e for s, e in spans) for a, b in stems)


def test_deleted_entry_is_reported_in_enforce_mode(export):
    """FTASKS 6.22 (FR-042.7): remove one entry from the export while the code keeps its value, and the brand
    guard reports the value — a register entry cannot be dropped silently, and the scanner holds no allowance
    of its own. For every entry literal naming an old brand that NO OTHER export record covers (decided by an
    independent reference, not the scanner), the scanner must report it once the entry is gone."""
    checked = []
    rules_full = bs.rules_for(export)
    for entry in export.entries():
        if entry["status"] == "retired":
            continue
        rules_without = bs.rules_for(_with(export, records=[r for r in export.records if r is not entry]))
        for lit in entry["literals"]:
            text = f'"{lit}"'
            if not bs.STEM.search(lit):
                continue
            item = [bs.Item("readme", f"{entry['id']} literal", text)]
            assert bs.scan(item, rules_full) == [], (entry["id"], lit)  # allowed while the entry exists
            if _allowed_elsewhere(export, entry, text):
                continue
            failures = bs.report(f"without {entry['id']}", bs.scan(item, rules_without), "enforce")
            assert failures, f"{entry['id']} removed, yet {lit!r} is still allowed: an allowance outside the export"
            checked.append(entry["id"])
    assert len(set(checked)) >= 5, f"too few entries checked: {sorted(set(checked))}"
