"""`lexical_weights` signals (Feature 040 FR-27; 016 FR-016.10; FTDD §6.1).

One match pass over all terms: the terms compiled to ONE RE2 alternation, longest term first at a position
(ties in declared order), wrapped in ASCII ``\\b`` when ``word_boundary``, under the signal's own
``ignore_case`` and ``fold_fullwidth_ascii``. The score is the sum of the weights of matched terms in text
order — each term once under ``presence``, once per match under ``occurrences``. The signal fires when the
score is at least ``cut``; its spans are the FIRST match of each distinct matched term, so ``count`` is the
number of distinct terms. Pinned by the vendored ``rule_kinds.lexical_weights`` cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from . import dialect


@dataclass(frozen=True)
class CompiledLexical:
    regex: Any
    #: (term, weight) in alternation order (longest first); the matching group's index selects one.
    terms: Tuple[Tuple[str, float], ...]
    counting: str
    cut: float
    fold: bool


def compile_lexical(signal: Dict[str, Any], max_mem: int) -> CompiledLexical:
    ordered = sorted(enumerate(signal["terms"]), key=lambda it: (-len(it[1]["term"]), it[0]))
    alternation = "|".join(f"({dialect.escape_literal(t['term'])})" for _, t in ordered)
    body = f"(?:{alternation})"
    pattern = rf"\b{body}\b" if signal["word_boundary"] else body
    regex = dialect.compile_pattern(pattern, bool(signal["ignore_case"]), max_mem)
    return CompiledLexical(regex, tuple((t["term"], float(t["weight"])) for _, t in ordered),
                           signal["counting"], float(signal["cut"]), bool(signal["fold_fullwidth_ascii"]))


def spans(compiled: CompiledLexical, view: Callable[[bool], "dialect.Text"]) -> List[Tuple[int, int, Dict[str, Any]]]:
    """`[(start, end, {term, weight})]` when the signal fires, else ``[]``."""
    score = 0.0
    first: Dict[int, Tuple[int, int, Dict[str, Any]]] = {}
    for m in dialect.finditer(compiled.regex, view(compiled.fold)):
        index = m.first_group()
        term, weight = compiled.terms[index]
        if compiled.counting == "occurrences" or index not in first:
            score += weight
        if index not in first:
            first[index] = (m.start, m.end, {"term": term, "weight": weight})
    if score >= compiled.cut:
        return sorted(first.values(), key=lambda s: (s[0], s[1]))
    return []
