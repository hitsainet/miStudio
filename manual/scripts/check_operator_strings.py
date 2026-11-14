#!/usr/bin/env python3
"""Refuse operator-specific hosts, addresses and credentials in the public manual.

The manual publishes to GitHub Pages from the public mirror, so anything in
`manual/docs` (and the site built from it) is world-readable. The reference
deployment's private addresses and unapproved host names, the private repository's owner
name, and anything shaped like a credential must not appear there. Use neutral
placeholders instead: `mistudio.example.local`, `<your-host>`, `<ingress-ip>`.

Usage: check_operator_strings.py DIR [DIR ...]   (exit 1 and list each hit)

One authority for the patterns: `manual/publish-rules.json`, read by this file.
The backend unit test `backend/tests/unit/test_manual_has_no_operator_strings.py`
imports this file to scan the sources, `.github/workflows/deploy-manual.yml`
runs it over `manual/build` before the site is uploaded, and the suite census
reads the same JSON for this repository's public legacy-identifiers export.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

#: The rules live in `manual/publish-rules.json` (`forbidden`: `name`, `pattern`, `flags`), so that this checker and
#: the suite census, which writes the public legacy-identifiers export against the same list, read ONE file and can
#: never disagree (miSuite ENH-009 0.53, operator decision 2026-11-08). Every rule today carries `i`.
RULES_FILE = Path(__file__).resolve().parent.parent / "publish-rules.json"

#: The only rule flags read; any other is refused, never dropped (a dropped `i` would pass `Onegaishimas`).
_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def load_rules(path: Path = RULES_FILE) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """(name, compiled pattern) for every rule of ``path``'s ``forbidden`` list. A missing or unreadable file, an
    empty list, a rule without a name or pattern, an unknown flag or a pattern that does not compile raises: the
    check never runs with a partial list."""
    rules = json.loads(path.read_text(encoding="utf-8"))["forbidden"]
    if not rules:
        raise ValueError(f"{path}: the forbidden list is empty")
    out = []
    for rule in rules:
        name, pattern, flags = rule["name"], rule["pattern"], rule.get("flags") or ""
        unknown = sorted(set(flags) - set(_FLAGS))
        if unknown:
            raise ValueError(f"{path}: rule {name!r} has unknown flag(s) {''.join(unknown)!r}")
        bits = 0
        for c in flags:
            bits |= _FLAGS[c]
        out.append((name, re.compile(pattern, bits)))
    return tuple(out)


FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = load_rules()

#: Text files only; images and fonts are skipped.
TEXT_SUFFIXES = {
    ".md", ".mdx", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json",
    ".css", ".html", ".xml", ".txt", ".yml", ".yaml", ".svg",
}
SKIP_DIRS = {"node_modules", ".docusaurus"}


def scan_text(text: str) -> list[tuple[int, str, str]]:
    """Every (line number, pattern name, matched text) in ``text``."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for name, rx in FORBIDDEN:
            m = rx.search(line)
            if m:
                hits.append((lineno, name, m.group(0)))
    return hits


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and not (
            SKIP_DIRS & set(path.relative_to(root).parts)
        ):
            yield path


def scan_tree(root: Path) -> tuple[int, list[str]]:
    """(files scanned, findings as ``path:line: name: match``)."""
    count, findings = 0, []
    for path in iter_files(root):
        count += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, name, match in scan_text(text):
            findings.append(f"{path}:{lineno}: {name}: {match}")
    return count, findings


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    failed = False
    for arg in argv:
        root = Path(arg)
        if not root.is_dir():
            print(f"not a directory: {root}", file=sys.stderr)
            return 2
        count, findings = scan_tree(root)
        if count == 0:
            # An empty scan proves nothing; refuse rather than pass.
            print(f"{root}: no text files found — refusing to report clean", file=sys.stderr)
            return 2
        for f in findings:
            print(f)
        print(f"{root}: {count} files scanned, {len(findings)} finding(s)", file=sys.stderr)
        failed |= bool(findings)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
