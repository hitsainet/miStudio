"""Every image a manual page references exists, and none is named for the old brand (042 FR-042.32; PD-17;
FTASKS 15.1, 15.10; control K-15.14).

The manual's captures were named `<old brand>_<Panel>…`. P4 renames them `miWorkbench_…`; a reference left
behind renders as a broken image (Docusaurus' `onBrokenLinks` does not check Markdown image targets under
`/img/`), so this test reads every reference and resolves it against `manual/static/`.

Discovered, never listed: every tracked `.md`/`.mdx` under `manual/docs/` and `manual/src/`, every Markdown
image (`![alt](target)`) and every `<img src=…>`. The count of captures is a dated reading and is not used.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from tests.support import legacy_export as le

REPO = le.REPO_ROOT
MANUAL = REPO / "manual"
STATIC = MANUAL / "static"
OLD_NAME = re.compile(r"mi(?:studio|llm)", re.IGNORECASE)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_HTML_IMAGE = re.compile(r"<img\b[^>]*?\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
_REMOTE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:)?//|^data:", re.IGNORECASE)


def _tracked(root: Path, *globs: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", *[f":(glob){g}" for g in globs]], cwd=root,
                         capture_output=True, check=True).stdout.decode()
    return sorted(p for p in out.split("\0") if p)


def references(root: Path = REPO) -> list[tuple[str, int, str]]:
    """`(doc, line, target)` for every local image reference in the manual's pages."""
    found = []
    for rel in _tracked(root, "manual/docs/**/*.md", "manual/docs/**/*.mdx", "manual/src/**/*.md",
                        "manual/src/**/*.mdx"):
        for n, line in enumerate((root / rel).read_text(encoding="utf-8").splitlines(), 1):
            for m in (*_MD_IMAGE.finditer(line), *_HTML_IMAGE.finditer(line)):
                target = m.group(1)
                if not _REMOTE.match(target):
                    found.append((rel, n, target))
    return found


def resolve(root: Path, doc: str, target: str) -> Path:
    """Docusaurus serves `manual/static/` at the site root; a relative target is beside the page."""
    clean = target.split("#", 1)[0].split("?", 1)[0]
    if clean.startswith("/"):
        return root / "manual" / "static" / clean.lstrip("/")
    return (root / doc).parent / clean


def problems(root: Path = REPO) -> list[str]:
    out = []
    for doc, line, target in references(root):
        path = resolve(root, doc, target)
        if not path.is_file():
            out.append(f"{doc}:{line}: {target} does not exist ({path.relative_to(root)})")
        if OLD_NAME.search(Path(target).name):
            out.append(f"{doc}:{line}: {target} is named for the old brand")
    return out


def test_every_referenced_image_exists_none_named_mistudio():
    refs = references()
    # Fail closed: the manual references images, most of them captures under /img/.
    assert len(refs) >= 20, len(refs)
    assert sum(1 for _d, _l, t in refs if t.startswith("/img/")) >= 20
    assert problems() == []
    # No capture is still STORED under an old name (a file nothing references would otherwise survive).
    stored = [p for p in _tracked(REPO, "manual/static/**") if OLD_NAME.search(Path(p).name)]
    assert stored == [], stored


def test_the_check_finds_a_missing_and_an_old_named_image(tmp_path):
    root = tmp_path / "tree"
    old = "mi" + "Studio_Panel.jpg"
    files = {
        "manual/docs/a.md": f"![ok](/img/miWorkbench_Panel.jpg)\n![gone](/img/missing.png)\n![old](/img/{old})\n",
        "manual/docs/sub/b.mdx": '<img src="/img/miWorkbench_Panel.jpg" alt="x" />\n![rel](./beside.png)\n'
                                 "![remote](https://example.org/x.png)\n",
        "manual/static/img/miWorkbench_Panel.jpg": "x",
        f"manual/static/img/{old}": "x",
        "manual/docs/sub/beside.png": "x",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    assert len(references(root)) == 5  # the remote one is not a local reference
    found = problems(root)
    assert len(found) == 2, found
    assert any("missing.png does not exist" in p for p in found), found
    assert any(old in p and "old brand" in p for p in found), found
