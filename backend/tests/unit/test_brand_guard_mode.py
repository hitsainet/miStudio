"""The brand guard's mode follows the export's `closed`, and both halves agree (042 T-48; TPD-31; CF-21;
FTASKS 7.11).

`BRAND_GUARD_MODE` is a constant in this repository (the export has no `brand_guard` field, CF-1). It must be
`enforce` once the export's header `closed` is P4 or later — keyed to `closed`, never to `phase`, so an export
regenerated `--phase P4` during P4 never forces the flip early, and a forgotten flip after S001 closes P4 is red.
The frontend mirror (`frontend/src/test/brandMode.ts`) must hold the same value; its own Vitest
(`brandMode.test.ts`) checks the same pair from the other side.
"""
from __future__ import annotations

import re
import shutil

import pytest

from tests.support import brand_surfaces as bs
from tests.support import legacy_export as le

FRONTEND = le.REPO_ROOT / "frontend" / "src" / "test" / "brandMode.ts"
#: The one declaration, at column 0 (a comment line starts with `//` or ` *`, so it cannot match).
_DECL = re.compile(r"^export const BRAND_GUARD_MODE(?:: \w+)? = '(report|enforce)';$", re.MULTILINE)


def frontend_mode() -> str:
    hits = _DECL.findall(FRONTEND.read_text(encoding="utf-8"))
    assert len(hits) == 1, f"expected one BRAND_GUARD_MODE declaration in {FRONTEND}, found {hits}"
    return hits[0]


def _with(export: le.Export, **header) -> le.Export:
    return le.Export(export.path, {**export.header, **header}, export.records)


def test_mode_constants_agree():
    assert bs.BRAND_GUARD_MODE in ("report", "enforce")
    assert frontend_mode() == bs.BRAND_GUARD_MODE


def test_enforce_from_closed_p4():
    export = le.load()
    assert bs.BRAND_GUARD_MODE == bs.mode_required_by(export), (
        f"export closed={export.closed!r} requires {bs.mode_required_by(export)!r}; "
        f"brand_surfaces.BRAND_GUARD_MODE is {bs.BRAND_GUARD_MODE!r} (flip both halves after P4 closes)")
    for closed, mode in ((None, "report"), ("P3", "report"), ("P4", "enforce"), ("P7", "enforce")):
        phase = "P0" if closed is None else closed
        assert bs.mode_required_by(_with(export, phase=phase, closed=closed)) == mode, closed


def test_mode_keys_to_closed_not_phase():
    export = le.load()
    assert bs.mode_required_by(_with(export, phase="P4", closed="P3")) == "report"
    assert bs.mode_required_by(_with(export, phase="P5", closed="P4")) == "enforce"
    assert bs.enforced("enforce") and not bs.enforced("report")


# ── R23 (operator, 2026-11-11): enforced in the private repository, reported on the public mirror ──────────
def _tree(tmp_path, *names: str):
    """A tree holding only the named export variants, copied from the committed ones."""
    root = tmp_path / "-".join(n.replace(".", "_") for n in names)
    (root / "docs").mkdir(parents=True)
    for name in names:
        src = le.REPO_ROOT / "docs" / name
        if not src.exists():
            pytest.skip(f"{name} is not in this tree (the public mirror keeps only the public export)")
        shutil.copy2(src, root / "docs" / name)
    return root


PLANT = [bs.Item("readme", "planted.md:1", "Open mi" + "Studio now")]


def test_public_export_always_reports(tmp_path):
    """A mirror-shaped tree (the public export alone): the real loader picks the public variant and the
    run reports, whatever the constant says. Runs in both trees: the public export is in each."""
    mirror = le.load(_tree(tmp_path, le.PUBLIC_NAME))
    assert mirror.is_public and mirror.withheld, "the public export withholds nothing: nothing to protect"
    assert bs.effective_mode(mirror, "enforce") == "report"
    assert bs.effective_mode(mirror, "report") == "report"
    findings = bs.scan(PLANT, bs.rules_for(mirror))
    assert len(findings) == 1
    assert bs.report("mirror", findings, bs.effective_mode(mirror, "enforce")) == []
    with pytest.raises(ValueError):
        bs.effective_mode(mirror, "off")


def test_full_export_enforces_when_the_constant_says_so(tmp_path):
    """A private-shaped tree (both variants): the loader picks the full one and `enforce` fails a finding.
    Skips BY NAME on the mirror, where the full export does not exist."""
    private = le.load(_tree(tmp_path, le.FULL_NAME, le.PUBLIC_NAME))
    assert not private.is_public and private.withheld == ()
    assert bs.effective_mode(private, "enforce") == "enforce"
    assert bs.effective_mode(private, "report") == "report"
    findings = bs.scan(PLANT, bs.rules_for(private))
    assert len(findings) == 1
    assert bs.report("private", findings, bs.effective_mode(private, "enforce")) == [findings[0].line()]
    assert bs.report("private", findings, bs.effective_mode(private, "report")) == []


def test_effective_mode_reads_the_constant_at_call_time(monkeypatch):
    export = le.load()
    here = "report" if export.is_public else "enforce"
    monkeypatch.setattr(bs, "BRAND_GUARD_MODE", "enforce")
    assert bs.effective_mode(export) == here
    monkeypatch.setattr(bs, "BRAND_GUARD_MODE", "report")
    assert bs.effective_mode(export) == "report"


@pytest.mark.parametrize("variant", [le.FULL_NAME, le.PUBLIC_NAME])
def test_the_real_tree_test_follows_the_effective_mode(tmp_path, monkeypatch, variant):
    """The WIRING: `test_brand_names.test_real_tree_report`, with the constant flipped to `enforce` and one
    planted finding on an enforced surface, FAILS against the full export and PASSES (reporting it)
    against the public one. The scan is stubbed to the plant; the mode path is the real one."""
    from tests.unit import test_brand_names as real

    names = (le.FULL_NAME, le.PUBLIC_NAME) if variant == le.FULL_NAME else (le.PUBLIC_NAME,)
    export = le.load(_tree(tmp_path, *names))
    monkeypatch.setattr(bs, "BRAND_GUARD_MODE", "enforce")
    monkeypatch.setattr(bs, "file_items", lambda cls, root=None: (["planted.md"], PLANT if cls == "readme" else []))
    monkeypatch.setattr(bs, "live_items", lambda cls: [])
    if export.is_public:
        real.test_real_tree_report(export)  # reported, not failed
    else:
        with pytest.raises(AssertionError, match="planted.md:1"):
            real.test_real_tree_report(export)
    # A report-only class never fails, in either tree.
    monkeypatch.setattr(bs, "file_items",
                        lambda cls, root=None: (["x.py"], [bs.Item("backend_strings", "x.py:1", PLANT[0].text)]
                                                if cls == "backend_strings" else []))
    real.test_real_tree_report(export)
