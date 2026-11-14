"""The forbidden-token set of the generality guard, DERIVED from behaviour-bearing data (Feature 040 FR-8).

⚠ THE LIST IS NEVER WRITTEN BY HAND. A hand list in the guard's own file is a list the guard's author
remembered; a new behaviour added as data would be missing from it and the guard would pass over code
that names it (suite memory `a-guard-can-work-by-accident`). Tokens are read from the files matched by
`TOKEN_GLOBS` — the seeded judge prompts and the behaviour packs — and the derivation refuses to return a
set smaller than `MIN_TOKENS` from fewer than `MIN_FILES` files, so a wrong glob is a failure rather than
an empty set that matches nothing.

What is a token (FTID TI-15): a behaviour key (`behaviour` / `key`), its label (`behaviour_label` /
`name` of a pack), every sub-question key and every rule id in a pack. A judge prompt's own `name` is NOT
a token: it names an instrument the library loads, not a behaviour.

Matching (FTDD §6.8): case-insensitive, on the boundaries of the slug alphabet `[a-z0-9-]`, so a key does
not match inside a longer slug (`<k>` does not match `<k>-presence`) and a label matches as a phrase with
any run of whitespace between its words.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

BACKEND = Path(__file__).resolve().parents[2]

#: Globs (relative to `backend/`) whose files carry behaviour names. A new pack file is picked up here.
TOKEN_GLOBS: Tuple[str, ...] = (
    "src/data/judge_prompt_seeds/*.json",
    "tests/fixtures/behaviour_packs/*.json",
)
#: Fixed when S1 landed (2026-10-23): two seed files, four tokens (two keys, two labels). The pack
#: fixtures added later only raise the count.
MIN_FILES = 2
MIN_TOKENS = 4

_BOUNDARY = "a-z0-9-"


def _strings(value) -> Iterable[str]:
    if isinstance(value, str) and value.strip():
        yield value.strip()


def tokens_from_document(doc: dict) -> List[str]:
    """The tokens one data file contributes (a judge-prompt seed or a behaviour pack)."""
    out: List[str] = []
    for field in ("behaviour", "behaviour_label"):
        out += list(_strings(doc.get(field)))
    if str(doc.get("format", "")).startswith("mistudio.behaviour-pack/"):
        # A pack holds one behaviour at its top level, or several under `behaviours`.
        for entry in doc.get("behaviours") or [doc]:
            out += list(_strings(entry.get("key")))
            out += list(_strings(entry.get("name")))
            for sub in entry.get("sub_questions") or []:
                out += list(_strings((sub or {}).get("key")))
            for rule_id in entry.get("rule_ids") or []:
                out += list(_strings(rule_id))
    return out


def token_files(globs: Sequence[str] = TOKEN_GLOBS, root: Path = BACKEND) -> List[Path]:
    files: List[Path] = []
    for pattern in globs:
        files += sorted(root.glob(pattern))
    return files


def derive_tokens(globs: Sequence[str] = TOKEN_GLOBS, root: Path = BACKEND) -> List[str]:
    """Every token, lower-cased, de-duplicated, sorted. Refuses a set too small to be the real one."""
    files = token_files(globs, root)
    if len(files) < MIN_FILES:
        raise AssertionError(
            f"the generality guard's globs {list(globs)} matched {len(files)} file(s) under {root}; "
            f"at least {MIN_FILES} are expected — a wrong glob would make the guard match nothing"
        )
    found = set()
    for path in files:
        for token in tokens_from_document(json.loads(path.read_text(encoding="utf-8"))):
            found.add(" ".join(token.lower().split()))
    if len(found) < MIN_TOKENS:
        raise AssertionError(
            f"the generality guard derived {len(found)} token(s) from {len(files)} file(s); at least "
            f"{MIN_TOKENS} are expected"
        )
    return sorted(found)


def pattern_for(token: str) -> "re.Pattern[str]":
    words = [re.escape(word) for word in token.split()]
    body = r"\s+".join(words)
    return re.compile(rf"(?<![{_BOUNDARY}]){body}(?![{_BOUNDARY}])", re.IGNORECASE)


@dataclass(frozen=True)
class Hit:
    where: str
    line: int
    token: str
    text: str

    def __str__(self) -> str:
        return f"{self.where}:{self.line}: {self.token!r} in {self.text.strip()[:120]!r}"


class Matcher:
    def __init__(self, tokens: Sequence[str]):
        self.tokens = list(tokens)
        self._patterns = [(token, pattern_for(token)) for token in self.tokens]

    def find(self, text: str) -> List[str]:
        return [token for token, pattern in self._patterns if pattern.search(text)]

    def find_lines(self, text: str) -> List[Tuple[str, int, str]]:
        """`(token, 0-based line offset inside text, that line)` for every match."""
        out: List[Tuple[str, int, str]] = []
        lines = text.splitlines() or [text]
        for token, pattern in self._patterns:
            for match in pattern.finditer(text):
                offset = text.count("\n", 0, match.start())
                out.append((token, offset, lines[min(offset, len(lines) - 1)]))
        return out
