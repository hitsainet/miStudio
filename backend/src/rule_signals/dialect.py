"""The rule-signal pattern dialect: RE2 on miDataworks' subset (Feature 040 FR-27, FR-30; FTDD §6.1; 016 FR-016.11).

The semantics are the producer's (the `$comment` of the vendored `midataworks-rule-set-v1.json`), pinned by
the vendored `rule-signal-cases-v1.json`. The scanner below is a port of miDataworks'
`backend/src/rules/dialect.py`, as miLLM 039 ported it (TD39-24): the vectors pin all three apps alike, and
no app imports another's code.

**Syntax:** RE2's, on the subset the scanner admits, with RE2's default options: case-sensitive unless a
signal declares ``ignore_case`` (RE2's Unicode simple case folding: ``k`` matches U+212A KELVIN SIGN); ``.``
does not match ``\\n``; ``^`` and ``$`` match only at the start and end of the text; ``\\d \\w \\s \\b`` are
ASCII and ``\\s`` is ``[\\t\\n\\f\\r ]``. Spans are Unicode code points.

**Refused, each naming the construct and the code-point position of its first character:** backreference,
lookaround, conditional, atomic, possessive, inline_flag, unicode_property, any_byte, repeat_too_large (a
count over 1,000, or nested counted repeats multiplying past 1,000), too_long (over 2,000 code points) and
syntax (see `scan`).

**The budget is deterministic (040-T3).** RE2 runs in time linear in the text, so no step budget exists.
A compiled program over ``max_mem`` (the runtime setting ``rule_pattern_max_mem_bytes``, passed in) is
refused at load as ``pattern_too_large``, naming the signal; it never runs. Nothing here reads a clock.

⚠ BYTES IN, CODE POINTS OUT. The ``re2`` binding's ``str`` path re-encodes the WHOLE text on every
``search`` call, which makes "every non-overlapping match" quadratic in the text. `Text` encodes once,
matches on the UTF-8 bytes, and converts byte offsets to code points through a prefix table built once per
text (FTID §3.7). The binding's own ``finditer`` is never used: it repeats an empty match after multi-byte
text (miLLM, measured 2026-10-23).

This module is the ONLY importer of ``re2`` and imports it unconditionally (D31): an image without the
wheel fails at start, never silently falls back to Python ``re``.
"""
from __future__ import annotations

import functools
import re
from array import array
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

import re2  # type: ignore[import-untyped]

MAX_PATTERN_CHARS = 2000
MAX_REPEAT = 1000

REFUSAL_RULES = (
    "backreference",
    "lookaround",
    "conditional",
    "atomic",
    "possessive",
    "inline_flag",
    "unicode_property",
    "any_byte",
    "repeat_too_large",
    "too_long",
    "syntax",
)

_ESCAPE_ATOMS = set("dDsSwWaftnrv0")
_ESCAPE_ASSERTIONS = set("bBAz")
_CLASS_ESCAPES = set("dDsSwW")
_CLASS_LITERAL_ESCAPES = {"a": "\a", "f": "\f", "t": "\t", "n": "\n", "r": "\r", "v": "\v", "0": "\0"}
_HEX = set("0123456789abcdefABCDEF")
_QUANT = re.compile(r"\{(\d*)(?:(,)(\d*))?\}")
_GROUP_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class PatternRefused(ValueError):
    """A pattern outside the dialect: the rule and the code-point position (None: RE2 refused what the
    scanner passed — a scanner gap, reported as `syntax`)."""

    def __init__(self, rule: str, position: Optional[int], message: str):
        super().__init__(f"{rule}{'' if position is None else f' at {position}'}: {message}")
        self.rule = rule
        self.position = position
        self.message = message


class PatternTooLarge(ValueError):
    """The compiled program exceeds the memory cap (`rule_pattern_max_mem_bytes`)."""

    def __init__(self, max_mem: int, error: str):
        super().__init__(f"the compiled pattern exceeds {max_mem} bytes ({error})")
        self.max_mem = max_mem


def _punct(ch: str) -> bool:
    return ch.isascii() and not ch.isalnum() and ch.isprintable()


@dataclass
class _Frame:
    open_at: int
    inner_max: int = 1


def _nothing(prev: str) -> str:
    if prev == "assertion":
        return "an anchor or boundary cannot be repeated"
    if prev in ("quant", "lazy"):
        return "a quantifier cannot follow another quantifier"
    return "nothing to repeat"


def scan(pattern: str) -> Optional[PatternRefused]:  # noqa: C901 - one state machine, kept whole
    """The first construct outside the dialect, or None. Positions are code points.

    `syntax` also covers what one engine of the suite reads differently, so the dialect leaves it out: an
    escape outside ``\\d \\D \\s \\S \\w \\W \\b \\B \\A \\z \\a \\f \\t \\n \\r \\v \\0 \\xHH`` and escaped
    ASCII punctuation; a quantifier on an anchor or boundary; ``{,n}``; ``(?<name>``; a repeated or
    malformed group name; ``[`` inside a class; a class escape or reversed range as a range endpoint;
    unbalanced brackets; a quantifier with nothing to repeat or after another one (a lazy ``?`` is allowed).
    """
    if len(pattern) > MAX_PATTERN_CHARS:
        return PatternRefused("too_long", MAX_PATTERN_CHARS, f"a pattern is at most {MAX_PATTERN_CHARS} characters")
    n = len(pattern)
    stack: List[_Frame] = [_Frame(-1)]
    names: set = set()
    prev = "none"  # none | atom | assertion | quant | lazy
    atom_product = 1
    quant_at = -1
    i = 0
    while i < n:
        c = pattern[i]
        if c == "\\":
            if i + 1 >= n:
                return PatternRefused("syntax", i, "a trailing backslash escapes nothing")
            e = pattern[i + 1]
            if e in "123456789" or e == "k":
                return PatternRefused("backreference", i, "backreferences are refused")
            if e in "pP":
                return PatternRefused("unicode_property", i, "Unicode property classes are refused")
            if e == "C":
                return PatternRefused("any_byte", i, "\\C matches one byte and is refused")
            if e == "x":
                if not (i + 3 < n and set(pattern[i + 2:i + 4]) <= _HEX):
                    return PatternRefused("syntax", i, "write a hex escape as \\xHH")
                prev, atom_product = "atom", 1
                i += 4
                continue
            if e in _ESCAPE_ATOMS or _punct(e):
                prev, atom_product = "atom", 1
            elif e in _ESCAPE_ASSERTIONS:
                prev = "assertion"
            else:
                return PatternRefused("syntax", i, f"the escape \\{e} is not in the dialect")
            i += 2
            continue
        if c == "[":
            end, refused = _scan_class(pattern, i)
            if refused is not None:
                return refused
            prev, atom_product = "atom", 1
            i = end
            continue
        if c == "(":
            opened = i
            if pattern.startswith("(?", i):
                nxt = pattern[i + 2:i + 3]
                if nxt == ":":
                    i += 3
                elif nxt == "P" and pattern[i + 3:i + 4] == "<":
                    m = _GROUP_NAME.match(pattern, i + 4)
                    if m is None or pattern[m.end():m.end() + 1] != ">":
                        return PatternRefused("syntax", i, "a group name is letters, digits and _")
                    if m.group() in names:
                        return PatternRefused("syntax", i, f"the group name {m.group()!r} is repeated")
                    names.add(m.group())
                    i = m.end() + 1
                elif nxt == "P" and pattern[i + 3:i + 4] == "=":
                    return PatternRefused("backreference", i, "backreferences are refused")
                elif nxt in ("=", "!") or pattern[i + 2:i + 4] in ("<=", "<!"):
                    return PatternRefused("lookaround", i, "lookahead and lookbehind are refused")
                elif nxt == "(":
                    return PatternRefused("conditional", i, "conditionals are refused")
                elif nxt == ">":
                    return PatternRefused("atomic", i, "atomic groups are refused")
                elif nxt and (nxt.isalpha() or nxt == "-"):
                    return PatternRefused("inline_flag", i, "inline flags are refused; declare options")
                else:
                    return PatternRefused("syntax", i, "this group form is not in the dialect")
            else:
                i += 1
            stack.append(_Frame(opened))
            prev = "none"
            continue
        if c == ")":
            if len(stack) == 1:
                return PatternRefused("syntax", i, "unbalanced parenthesis")
            frame = stack.pop()
            stack[-1].inner_max = max(stack[-1].inner_max, frame.inner_max)
            prev, atom_product = "atom", frame.inner_max
            i += 1
            continue
        if c == "|":
            prev = "none"
            i += 1
            continue
        if c in "*+?":
            if prev == "quant" and c == "?":
                prev = "lazy"
                i += 1
                continue
            if prev in ("quant", "lazy") and c == "+":
                return PatternRefused("possessive", quant_at, "possessive quantifiers are refused")
            if prev != "atom":
                return PatternRefused("syntax", i, _nothing(prev))
            prev, quant_at = "quant", i
            stack[-1].inner_max = max(stack[-1].inner_max, atom_product)
            i += 1
            continue
        if c == "{":
            m = _QUANT.match(pattern, i)
            if m is not None and (m.group(1) or m.group(2)):
                if not m.group(1):
                    return PatternRefused("syntax", i, "{,n} is not in the dialect; write {0,n}")
                if prev != "atom":
                    return PatternRefused("syntax", i, _nothing(prev))
                low = int(m.group(1))
                high = None if m.group(2) and not m.group(3) else int(m.group(3) or m.group(1))
                if low > MAX_REPEAT or (high is not None and high > MAX_REPEAT):
                    return PatternRefused("repeat_too_large", i, f"a repeat count is at most {MAX_REPEAT}")
                if high is not None and high < low:
                    return PatternRefused("syntax", i, "the repeat range is reversed")
                product = (high if high is not None else low) * atom_product
                if product > MAX_REPEAT:
                    return PatternRefused("repeat_too_large", i,
                                          f"nested repeat counts multiply to {product}; at most {MAX_REPEAT}")
                stack[-1].inner_max = max(stack[-1].inner_max, product)
                prev, quant_at = "quant", i
                i = m.end()
                continue
            prev, atom_product = "atom", 1
            i += 1
            continue
        if c in "^$":
            prev = "assertion"
            i += 1
            continue
        prev, atom_product = "atom", 1
        i += 1
    if len(stack) > 1:
        return PatternRefused("syntax", stack[-1].open_at, "unclosed parenthesis")
    return None


def _scan_class(pattern: str, start: int) -> Tuple[int, Optional[PatternRefused]]:
    """Scan ``[...]`` from ``start``; the index after ``]`` or a refusal."""
    n = len(pattern)
    j = start + 1
    if j < n and pattern[j] == "^":
        j += 1
    first = True
    last_literal: Optional[Tuple[int, str]] = None
    pending_range = False
    while j < n:
        c = pattern[j]
        if c == "]" and not first:
            return j + 1, None
        first = False
        if c == "[":
            return j, PatternRefused("syntax", j, "'[' inside a character class is refused")
        if c == "\\":
            if j + 1 >= n:
                return j, PatternRefused("syntax", j, "a trailing backslash escapes nothing")
            e = pattern[j + 1]
            if e in "pP":
                return j, PatternRefused("unicode_property", j, "Unicode property classes are refused")
            if e == "C":
                return j, PatternRefused("any_byte", j, "\\C matches one byte and is refused")
            if e == "x":
                if not (j + 3 < n and set(pattern[j + 2:j + 4]) <= _HEX):
                    return j, PatternRefused("syntax", j, "write a hex escape as \\xHH")
                char, width = chr(int(pattern[j + 2:j + 4], 16)), 4
            elif e in _CLASS_ESCAPES:
                if pending_range:
                    return j, PatternRefused("syntax", j, "a class escape cannot end a range")
                if pattern[j + 2:j + 3] == "-" and pattern[j + 3:j + 4] not in ("]", ""):
                    return j, PatternRefused("syntax", j, "a class escape cannot start a range")
                last_literal = None
                j += 2
                continue
            elif e in _CLASS_LITERAL_ESCAPES:
                char, width = _CLASS_LITERAL_ESCAPES[e], 2
            elif _punct(e):
                char, width = e, 2
            else:
                return j, PatternRefused("syntax", j, f"the escape \\{e} is not in a class")
        else:
            char, width = c, 1
        if c == "-" and last_literal is not None and not pending_range and j + 1 < n and pattern[j + 1] != "]":
            pending_range = True
            j += 1
            continue
        if pending_range and last_literal is not None:
            if ord(char) < ord(last_literal[1]):
                return j, PatternRefused("syntax", last_literal[0], "the range is reversed")
            pending_range = False
            last_literal = None
        else:
            last_literal = (j, char)
        j += width
    return start, PatternRefused("syntax", start, "unterminated character class")


def check(pattern: str) -> None:
    refused = scan(pattern)
    if refused is not None:
        raise refused


@functools.lru_cache(maxsize=4096)
def compile_pattern(pattern: str, ignore_case: bool, max_mem: int) -> Any:
    """A scanner-passed pattern compiled by RE2 under the memory cap. Raises `PatternRefused` or
    `PatternTooLarge`. Cached per (pattern, ignore_case, max_mem): a cap change compiles again."""
    check(pattern)
    options = re2.Options()
    options.log_errors = False
    options.case_sensitive = not ignore_case
    options.max_mem = int(max_mem)
    try:
        return re2.compile(pattern, options)
    except re2.error as exc:
        text = str(exc)
        if "too large" in text or "out of memory" in text:
            raise PatternTooLarge(int(max_mem), text) from None
        raise PatternRefused("syntax", None, f"RE2 refused it: {text}") from None


# ── matching on bytes, reporting code points ──────────────────────────────────────────────────────


def _char_len(lead: int) -> int:
    if lead < 0x80:
        return 1
    if lead < 0xE0:
        return 2
    if lead < 0xF0:
        return 3
    return 4


class Text:
    """A string matched once as UTF-8 bytes; byte offsets map to code points through one table."""

    __slots__ = ("text", "data", "_table")

    def __init__(self, text: str):
        self.text = text
        self.data = text.encode("utf-8", "surrogatepass")
        self._table: Optional[array] = None

    @property
    def ascii(self) -> bool:
        return len(self.data) == len(self.text)

    def cp(self, byte_offset: int) -> int:
        if self.ascii:
            return byte_offset
        if self._table is None:
            # ⚠ A BYTE OFFSET INSIDE A CHARACTER maps to the NEXT code point (the number of characters that
            # START before it), as the binding's ``str`` path does (``BytesToCharLen``) and so as miDataworks
            # and miLLM do. RE2's unanchored search tries every byte, so an empty match (``\B``, ``a?``) can
            # land inside a multi-byte character; leaving those entries 0 reported it at code point 0.
            table = array("I", bytes(4 * (len(self.data) + 1)))
            b = 0
            for i, ch in enumerate(self.text):
                table[b] = i
                o = ord(ch)
                width = 1 if o < 0x80 else 2 if o < 0x800 else 3 if o < 0x10000 else 4
                for k in range(1, width):
                    table[b + k] = i + 1
                b += width
            table[b] = len(self.text)
            self._table = table
        return self._table[byte_offset]


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    #: Byte spans of the groups (index 0 is the whole match); (-1, -1) for a group that did not take part.
    groups: Tuple[Tuple[int, int], ...]

    def first_group(self) -> int:
        """The index (0-based, over groups 1..n) of the first participating group."""
        for index, span in enumerate(self.groups[1:]):
            if span[0] != -1:
                return index
        raise LookupError("no group took part in the match")


def finditer(compiled: Any, text: Text) -> Iterator[Match]:
    """Every non-overlapping match, left to right, with the producer's rule for empty matches: after a
    match ending at ``e`` the search resumes at ``e``, one code point further when the match was empty."""
    data = text.data
    pos, n = 0, len(data)
    groups = compiled.groups
    while pos <= n:
        m = compiled.search(data, pos)
        if m is None:
            return
        s, e = m.span()
        yield Match(text.cp(s), text.cp(e), tuple(m.span(g) for g in range(groups + 1)))
        if e == s:
            # One code point further, counted from the code point the empty match reports (the producer
            # resumes its ``str`` search at ``cp(e) + 1``): an empty match inside a character first moves
            # to that character's end.
            while e < n and (data[e] & 0xC0) == 0x80:
                e += 1
            if e >= n:
                return
            pos = e + _char_len(data[e])
        else:
            pos = e


#: U+FF01 – U+FF5E → U+0021 – U+007E and U+3000 → U+0020: one code point to one, so spans need no mapping.
FULLWIDTH_TABLE = {**{cp: cp - 0xFEE0 for cp in range(0xFF01, 0xFF5F)}, 0x3000: 0x20}


def fold_fullwidth_ascii(text: str) -> str:
    return text.translate(FULLWIDTH_TABLE)


def escape_literal(text: str) -> str:
    """``text`` as a dialect pattern that matches it literally (ASCII punctuation escaped)."""
    return "".join("\\" + c if _punct(c) else c for c in text)
