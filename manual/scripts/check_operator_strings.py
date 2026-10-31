#!/usr/bin/env python3
"""Refuse operator-specific hosts, addresses and credentials in the public manual.

The manual publishes to GitHub Pages from the public mirror, so anything in
`manual/docs` (and the site built from it) is world-readable. The reference
deployment's private addresses and unapproved host names, the private repository's owner
name, and anything shaped like a credential must not appear there. Use neutral
placeholders instead: `mistudio.example.local`, `<your-host>`, `<ingress-ip>`.

Usage: check_operator_strings.py DIR [DIR ...]   (exit 1 and list each hit)

One authority for the patterns: the backend unit test
`backend/tests/unit/test_manual_has_no_operator_strings.py` imports this file
to scan the sources, and `.github/workflows/deploy-manual.yml` runs it over
`manual/build` before the site is uploaded.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: (name, pattern). Case-insensitive.
FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(rx, re.IGNORECASE))
    for name, rx in (
        ("private address 192.168.x", r"\b192\.168\."),
        ("private address 10.152.x", r"\b10\.152\."),
        ("operator host *.hitsai.local", r"\bhitsai\.local\b"),
        # The three public MCP endpoints are intentionally published in the setup guide.
        # All other deployment hosts remain forbidden, including lookalike subdomains.
        ("operator host *.hitsai.net",
         r"(?<![a-z0-9.-])(?!(?:mcp-studio|mcp-dataworks|mcp-forge)\.hitsai\.net\b)"
         r"[a-z0-9.-]+\.hitsai\.net\b"),
        ("operator machine name", r"\bmcs-lnx[a-z0-9-]*"),
        ("private repository owner", r"\bonegaishimas\b"),
        ("Hugging Face token", r"\bhf_[A-Za-z0-9]{30,}"),
        ("OpenAI-style key", r"\bsk-[A-Za-z0-9_-]{20,}"),
        ("GitHub token", r"\b(?:ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{20,}"),
        ("AWS access key", r"\bAKIA[0-9A-Z]{16}\b"),
        ("JSON Web Token", r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}"),
        # A literal bearer value, not a placeholder ($VAR, <token>, ${...}).
        ("literal bearer token", r"Bearer\s+(?![$<{])[A-Za-z0-9._~+/-]{24,}"),
    )
)

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
