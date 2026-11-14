"""The one reader of the legacy-identifier export (042 FR-042.6, FR-042.7; FTID §3.13; S001 FTDD §4.3).

`docs/legacy-identifiers.md` (full, this private repository only) and `docs/legacy-identifiers.public.md`
(the only variant the public mirror keeps, CF-2) are written by the suite's `census.py --export miStudio`
and are never edited here. Every test that needs a frozen value, an old environment name, an old host,
an old repository name or a brand-guard rule reads it through this module, so no test re-types a
register fact.

Every rule REFUSES loudly (`ExportError`, naming the file, the line and the value). A reader that skipped
an unknown record would let the frozen-identifier test and the brand guard pass over a rule they never
saw (memory `source-scrape-guards-fail-open`).

Nothing under `backend/src/` imports this module: the runtime keeps its own hand-written table
(`src/core/legacy_names.py`, CF-3), and `test_legacy_names_table.py` compares the two.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

#: `backend/tests/support/legacy_export.py` → parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_NAME = "legacy-identifiers.md"
PUBLIC_NAME = "legacy-identifiers.public.md"

FORMAT = 1
#: The header's `repo` is always the NEW name (S001 TID-5), whatever key census was run with.
REPO_NAME = "miWorkbench"
SECTIONS = frozenset({"entry", "env", "host", "history_tool", "repo", "guard"})
ENTRY_STATUSES = frozenset({"frozen", "scheduled", "retired"})
TRANSITION_STATUSES = frozenset({"transition", "retired"})
RETIRE_PHASES = frozenset({"P8", None})
FLAGS = frozenset({"history_only", "text_only", "public_withheld", "pending"})
EFFECTS = frozenset({"deny", "allow"})
PHASES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")
JS_FLAGS = frozenset({"", "i"})

HEADER_FIELDS = frozenset({"format", "repo", "variant", "register_hash", "phase", "closed", "withheld"})
#: Optional header keys, the only value each may take, and the variant that may carry it.
#: `forbidden_list: "unread"` is written by `census.py --public-unchecked` while S001 FTASKS 0.53 has not
#: read this repository's forbidden-content list; only the PUBLIC variant may carry it.
OPTIONAL_HEADER: dict[str, frozenset[str]] = {"forbidden_list": frozenset({"unread"})}
FIELDS: dict[str, frozenset[str]] = {
    "entry": frozenset({"section", "id", "kind", "group", "status", "retire_phase", "pattern", "pattern_js",
                        "js_flags", "literals", "exercise", "flags"}),
    "env": frozenset({"section", "old", "new", "status"}),
    "host": frozenset({"section", "old", "new", "status", "observed"}),
    "history_tool": frozenset({"section", "old", "new"}),
    "repo": frozenset({"section", "old", "new", "status"}),
    "guard": frozenset({"section", "id", "pattern", "pattern_js", "js_flags", "effect", "from_phase",
                        "until_phase"}),
}

#: CF-25 m4: an inline flag GROUP (`(?i)`, `(?i:…)`) has no JavaScript meaning; a lookaround or `(?:…)`
#: does, so a plain "contains `(?`" refusal would be wrong (control K-6.15).
INLINE_FLAG = re.compile(r"\(\?[a-zA-Z]+[:)]")
#: CF-25 m5: constructs with no identical JavaScript meaning, refused in either pattern.
NO_JS = ("\\A", "\\Z", "(?P", "(?#")
#: TI-5: exercise paths are repository-root relative and name one of this repository's two suites.
EXERCISE = re.compile(
    r"^(?:pytest:(?P<py>backend/tests/[\w./-]+\.py)::(?P<pytest>[\w\[\]./:-]+)"
    r"|vitest:(?P<vt>frontend/src/[\w./-]+\.test\.tsx?)::(?P<vitest>\S.*))$")
STAND_IN = re.compile(r"^(?P<section>[a-z_]+):sha256:[0-9a-f]{16}$")


class ExportError(ValueError):
    """The export breaks a rule of S001 FTDD §4.3 / 042 FTID §3.13."""


class ExportMissing(ExportError):
    """Neither variant exists: nothing that reads the export may pass on nothing."""


def stand_in(section: str, old: str) -> str:
    """The name a public variant gives a withheld record whose old name it must not print (CF-2):
    `<section>:sha256:` + the first 16 hex digits of the SHA-256 of the part after the colon."""
    return f"{section}:sha256:" + hashlib.sha256(old.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Export:
    path: Path
    header: dict[str, Any]
    records: tuple[dict[str, Any], ...]

    # ── header ─────────────────────────────────────────────────────────────────────────────────
    @property
    def variant(self) -> str:
        return self.header["variant"]

    @property
    def is_public(self) -> bool:
        return self.header["variant"] == "public"

    @property
    def phase(self) -> str:
        return self.header["phase"]

    @property
    def closed(self) -> str | None:
        return self.header["closed"]

    @property
    def withheld(self) -> tuple[str, ...]:
        return tuple(self.header["withheld"])

    def phase_at_least(self, phase: str) -> bool:
        return PHASES.index(self.header["phase"]) >= PHASES.index(phase)

    def closed_at_least(self, phase: str) -> bool:
        """`closed` is `phase` or later (CF-21: behaviour that changes at a phase CLOSE keys here, never
        to `phase`). `None` (nothing closed yet) is before every phase."""
        closed = self.header["closed"]
        return closed is not None and PHASES.index(closed) >= PHASES.index(phase)

    # ── records ────────────────────────────────────────────────────────────────────────────────
    def section(self, name: str) -> list[dict[str, Any]]:
        if name not in SECTIONS:
            raise ExportError(f"unknown section asked for: {name!r}")
        return [r for r in self.records if r["section"] == name]

    def entries(self, status: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.section("entry") if status is None or r["status"] == status]

    def entry(self, entry_id: str) -> dict[str, Any]:
        for r in self.section("entry"):
            if r["id"] == entry_id:
                return r
        if entry_id in self.header["withheld"]:
            raise ExportError(f"{entry_id} is withheld from {self.path.name}")
        raise ExportError(f"{self.path.name} has no entry {entry_id}")

    def env(self) -> list[dict[str, Any]]:
        return self.section("env")

    def hosts(self) -> list[dict[str, Any]]:
        return self.section("host")

    def history_tools(self) -> list[dict[str, Any]]:
        return self.section("history_tool")

    def repos(self, status: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.section("repo") if status is None or r["status"] == status]

    def guards(self) -> list[dict[str, Any]]:
        return self.section("guard")

    def guard(self, guard_id: str) -> dict[str, Any]:
        for g in self.guards():
            if g["id"] == guard_id:
                return g
        raise ExportError(f"{self.path.name} has no guard {guard_id}")

    def active_guards(self) -> list[dict[str, Any]]:
        """Guards in force at the header's `phase`: `from_phase` ≤ phase < `until_phase` (null = unbounded).
        Guard ACTIVITY keys to `phase`; enforcement keys to `closed` (CF-21)."""
        now = PHASES.index(self.header["phase"])
        out = []
        for g in self.guards():
            start = -1 if g["from_phase"] is None else PHASES.index(g["from_phase"])
            end = len(PHASES) if g["until_phase"] is None else PHASES.index(g["until_phase"])
            if start <= now < end:
                out.append(g)
        return out

    def pending_ids(self) -> list[str]:
        return [r["id"] for r in self.section("entry") if "pending" in r["flags"]]

    def is_withheld(self, section: str, old_or_id: str) -> bool:
        """True when the public variant leaves this record out, named `<id>` (entries, guards),
        `<section>:<old>`, or by its stand-in `<section>:sha256:<16 hex>` (CF-2)."""
        held = set(self.header["withheld"])
        return (old_or_id in held or f"{section}:{old_or_id}" in held
                or stand_in(section, old_or_id) in held)


def require(export: Export, section: str, old_or_id: str) -> None:
    """The ONE skip in this module (CF-2): a record the public variant withholds is skipped BY NAME on
    the mirror; on the full variant nothing is withheld, so this never skips in the private repo."""
    if export.is_public and export.is_withheld(section, old_or_id):
        pytest.skip(f"withheld from the public export (mirror): {section} {old_or_id}")


def paths(root: Path = REPO_ROOT) -> tuple[Path, Path]:
    return root / "docs" / FULL_NAME, root / "docs" / PUBLIC_NAME


def load(root: Path = REPO_ROOT, *, path: Path | None = None) -> Export:
    """The full variant when present (this private repository), else the public one (the mirror)."""
    if path is None:
        full, public = paths(root)
        path = full if full.exists() else public
        if not path.exists():
            raise ExportMissing(f"neither {full} nor {public} exists: the export is required")
    path = Path(path)
    if not path.exists():
        raise ExportMissing(f"{path} does not exist")
    return parse(path.read_text(encoding="utf-8"), path)


def _fail(path: Path, line: int, why: str, value: Any) -> ExportError:
    return ExportError(f"{path}:{line}: {why}: {value!r}")


def _block(text: str, path: Path) -> Iterator[tuple[int, str]]:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "```jsonl"]
    if len(starts) != 1:
        raise _fail(path, 0, "expected exactly one ```jsonl block", len(starts))
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == "```"), None)
    if end is None:
        raise _fail(path, start + 1, "the ```jsonl block is not closed", "```")
    for i in range(start + 1, end):
        if lines[i].strip():
            yield i + 1, lines[i]


def _phase_or_null(path: Path, line: int, field: str, value: Any) -> None:
    if value is not None and value not in PHASES:
        raise _fail(path, line, f"{field} must be one of P0…P8 or null", value)


def _check_pattern(path: Path, line: int, rec: dict[str, Any]) -> None:
    pattern, pattern_js, js_flags = rec["pattern"], rec["pattern_js"], rec["js_flags"]
    for name, value in (("pattern", pattern), ("pattern_js", pattern_js)):
        if not isinstance(value, str) or not value:
            raise _fail(path, line, f"{name} must be a non-empty string", value)
    for bad in NO_JS:
        if bad in pattern or bad in pattern_js:
            raise _fail(path, line, f"a pattern uses {bad} (no identical JavaScript meaning, CF-25 m5)",
                        pattern if bad in pattern else pattern_js)
    try:
        re.compile(pattern)
    except re.error as exc:
        raise _fail(path, line, f"pattern does not compile ({exc})", pattern) from exc
    if INLINE_FLAG.search(pattern_js):
        raise _fail(path, line, "pattern_js holds an inline flag group (CF-16, CF-25 m4)", pattern_js)
    if js_flags not in JS_FLAGS:
        raise _fail(path, line, "js_flags must be '' or 'i' (CF-16; never 'g')", js_flags)


def parse(text: str, path: Path) -> Export:
    items = list(_block(text, path))
    if not items:
        raise _fail(path, 0, "the jsonl block is empty", "")
    parsed: list[tuple[int, dict[str, Any]]] = []
    for line, raw in items:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _fail(path, line, f"not JSON ({exc.msg})", raw[:80]) from exc
        if not isinstance(value, dict):
            raise _fail(path, line, "a record must be an object", value)
        parsed.append((line, value))

    hline, header = parsed[0]
    expected_variant = "public" if path.name.endswith(".public.md") else "full"
    if "section" in header:
        raise _fail(path, hline, "the first record must be the header", header.get("section"))
    if "format" in header and header["format"] != FORMAT:
        raise _fail(path, hline, f"format must be {FORMAT}", header["format"])
    if "closed" not in header:
        raise _fail(path, hline, "the header has no `closed` key (CF-21)", sorted(header))
    extra = set(header) - HEADER_FIELDS
    missing = HEADER_FIELDS - set(header)
    if missing or not extra <= set(OPTIONAL_HEADER):
        raise _fail(path, hline, f"header keys wrong (missing {sorted(missing)}, unknown "
                    f"{sorted(extra - set(OPTIONAL_HEADER))})", sorted(header))
    for key in extra:
        if expected_variant != "public":
            raise _fail(path, hline, f"{key} may appear only in the public variant", key)
        if header[key] not in OPTIONAL_HEADER[key]:
            raise _fail(path, hline, f"{key} must be one of {sorted(OPTIONAL_HEADER[key])}", header[key])
    if header["repo"] != REPO_NAME:
        raise _fail(path, hline, f"repo must be {REPO_NAME!r} (S001 TID-5)", header["repo"])
    if header["variant"] != expected_variant:
        raise _fail(path, hline, f"variant must be {expected_variant!r} for this file name", header["variant"])
    if not isinstance(header["register_hash"], str) or not header["register_hash"].startswith("sha256:"):
        raise _fail(path, hline, "register_hash must be sha256:<hex>", header["register_hash"])
    if header["phase"] not in PHASES:
        raise _fail(path, hline, "phase must be one of P0…P8", header["phase"])
    _phase_or_null(path, hline, "closed", header["closed"])
    closed = header["closed"]
    allowed_phase = {"P0"} if closed is None else {closed} | (
        {PHASES[PHASES.index(closed) + 1]} if closed != PHASES[-1] else set())
    if header["phase"] not in allowed_phase:
        raise _fail(path, hline, "phase must equal closed or the phase after it (CF-21)",
                    (header["phase"], closed))
    withheld = header["withheld"]
    if not isinstance(withheld, list) or not all(isinstance(w, str) and w for w in withheld):
        raise _fail(path, hline, "withheld must be a list of non-empty strings", withheld)
    if expected_variant == "full" and withheld:
        raise _fail(path, hline, "the full variant withholds nothing", withheld)

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_old: set[tuple[str, str]] = set()
    for line, rec in parsed[1:]:
        section = rec.get("section")
        if section not in SECTIONS:
            raise _fail(path, line, "unknown section", section)
        if "locations" in rec:
            raise _fail(path, line, "a record carries `locations` (CF-1: locations stay in the suite)",
                        rec.get("id", rec.get("old")))
        if section == "host" and not isinstance(rec.get("observed"), bool):
            raise _fail(path, line, "a host record needs a boolean `observed` (CF-24)", rec.get("observed"))
        if set(rec) != FIELDS[section]:
            raise _fail(path, line, f"{section} fields wrong (extra {sorted(set(rec) - FIELDS[section])}, "
                        f"missing {sorted(FIELDS[section] - set(rec))})", sorted(rec))
        if section in ("entry", "guard"):
            if not isinstance(rec["id"], str) or not rec["id"]:
                raise _fail(path, line, "id must be a non-empty string", rec["id"])
            if rec["id"] in seen_ids:
                raise _fail(path, line, "duplicate id", rec["id"])
            seen_ids.add(rec["id"])
            _check_pattern(path, line, rec)
        else:
            for f in ("old", "new"):
                if not isinstance(rec[f], str) or not rec[f]:
                    raise _fail(path, line, f"{f} must be a non-empty string", rec[f])
            if (section, rec["old"]) in seen_old:
                raise _fail(path, line, "duplicate (section, old)", (section, rec["old"]))
            seen_old.add((section, rec["old"]))
        if section == "entry":
            _check_entry(path, line, rec, expected_variant)
        elif section in ("env", "host", "repo"):
            if rec["status"] not in TRANSITION_STATUSES:
                raise _fail(path, line, f"{section} status must be transition or retired", rec["status"])
        elif section == "guard":
            if rec["effect"] not in EFFECTS:
                raise _fail(path, line, "guard effect must be deny or allow", rec["effect"])
            _phase_or_null(path, line, "from_phase", rec["from_phase"])
            _phase_or_null(path, line, "until_phase", rec["until_phase"])
        records.append(rec)
    if not any(r["section"] == "entry" for r in records):
        raise _fail(path, hline, "zero entry records", 0)
    for name in withheld:
        m = STAND_IN.match(name)
        if m and m["section"] not in SECTIONS:
            raise _fail(path, hline, "a stand-in names an unknown section", name)
    return Export(path, header, tuple(records))


def _check_entry(path: Path, line: int, rec: dict[str, Any], variant: str) -> None:
    if rec["status"] not in ENTRY_STATUSES:
        raise _fail(path, line, "entry status must be frozen, scheduled or retired", rec["status"])
    if rec["retire_phase"] not in RETIRE_PHASES:
        raise _fail(path, line, "retire_phase must be 'P8' or null", rec["retire_phase"])
    flags = rec["flags"]
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        raise _fail(path, line, "flags must be a list of strings", flags)
    unknown = sorted(set(flags) - FLAGS)
    if unknown:
        raise _fail(path, line, f"unknown flag(s) {unknown}; allowed {sorted(FLAGS)}", flags)
    if "public_withheld" in flags and variant == "public":
        raise _fail(path, line, "a public_withheld record is in the public variant", rec["id"])
    literals = rec["literals"]
    if not isinstance(literals, list) or not all(isinstance(x, str) and x for x in literals):
        raise _fail(path, line, "literals must be a list of non-empty strings", literals)
    for literal in literals:
        if not re.search(rec["pattern"], literal):
            raise _fail(path, line, "a literal its own pattern misses (CF-25 m5)", literal)
    exercise = rec["exercise"]
    if exercise is None:
        if not set(flags) & {"history_only", "pending"}:
            raise _fail(path, line, "exercise null needs history_only or pending", rec["id"])
    elif not isinstance(exercise, str) or not EXERCISE.match(exercise):
        raise _fail(path, line, "exercise must be pytest:backend/tests/<…>.py::<test> or "
                                "vitest:frontend/src/<…>.test.ts(x)::<name> (TI-5)", exercise)
    if "pending" in flags and exercise is not None:
        raise _fail(path, line, "a pending record has an exercise", rec["id"])


def parse_exercise(exercise: str) -> tuple[str, str, str]:
    """`(runner, path, name)` of an exercise id; `runner` is `pytest` or `vitest`."""
    m = EXERCISE.match(exercise)
    if not m:
        raise ExportError(f"not an exercise id: {exercise!r}")
    if m["py"]:
        return "pytest", m["py"], m["pytest"]
    return "vitest", m["vt"], m["vitest"]
