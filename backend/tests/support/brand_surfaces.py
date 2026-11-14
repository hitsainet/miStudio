"""The brand guard, backend half: surface discovery and the scanner (042 FR-042.13, FR-042.34; FTDD §6.7;
FTID §3.14; TPD-15, TPD-16, TPD-31, TPD-33; CF-13, CF-15, CF-20 – CF-23).

WHAT IS A SURFACE (PD-3). Discovered, never listed by file: README-family H1–H3 headings
(`README*.md`, `CONTRIBUTING.md`, `SECURITY.md` anywhere tracked, outside the record trees `0xcc/` and
`.claude/`), every manual doc's front-matter `title`/`sidebar_label` and first heading, the Docusaurus config's
`title`/`tagline`/navbar/footer and `sidebars.ts` labels, SVG `<title>`/`<text>` under `frontend/public` and
`frontend/src/assets`, `frontend/index.html`'s `<title>`, the FastAPI title and health `service` from the REAL
app object, and the MCP server name, instructions, every tool's name and description and the howto topics'
text from the REAL built server (`build_server()`, every category on). A report-only class (TPD-16) lists
backend string constants under `src/` naming an old brand; it is printed, never enforced.

WHAT IS ALLOWED. Only what the census export says (FTID §3.14), read through `tests/support/legacy_export.py`:
(a) a match of a non-retired `entry` record's `pattern`, or one of its `literals`; (b) the `old` of a
`transition` `env`/`host`/`history_tool` record; (c) the WHOLE `<owner>/<name>` token of a `transition` `repo`
record (so `<owner>/<name>-backup` is not covered; TPD-33 — this module holds no repository name); (d) a
match of an ACTIVE `allow` guard record. An active `deny` guard's match is a finding unless (a)–(d) cover it.
Guard activity keys to the header's `phase`; enforcement keys to its `closed` (CF-21).

The export is never a surface (CF-20): discovery always passes BOTH glob-correct exclusions, because on
git 2.43 `:(exclude,glob)**/docs/legacy-identifiers*.md` does not exclude the root-level file under a `docs/`
positive pathspec (miForge P0 review, finding 13).
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from tests.support import legacy_export as le

#: P0 – P4: findings are printed and the test passes. Flipped to "enforce" in this repository's first commit
#: after S001 closes P4 (CF-26). `test_brand_guard_mode.py` ties it to the export's `closed` and to the
#: frontend mirror `frontend/src/test/brandMode.ts`. What a run USES is `effective_mode(export)`: this
#: constant with the full export (the private repository), always "report" with the public one (R23).
BRAND_GUARD_MODE = "enforce"

REPO_ROOT = le.REPO_ROOT

#: The hand-kept old-brand stems (the brand list is not a register fact; FTID §3.14). `miSuite` is NOT here:
#: it is the suite's name until P5, when the export's `G-MISUITE` deny record makes it a finding (CF-15).
STEM = re.compile(r"mi(?:studio|llm)|mechinterp[ -]?studio", re.IGNORECASE)
#: `_` and `.` are word characters for a token, so `millm_probe_id` is reported (and allowed) whole.
TOKEN_CHAR = re.compile(r"[\w.]")

#: CF-20, both forms (see the module docstring).
EXPORT_EXCLUDES = (":(exclude,glob)**/docs/legacy-identifiers*.md", ":(exclude,glob)docs/legacy-identifiers*.md")
#: Record trees (dated history and agent context, stripped from the mirror) are never a brand surface. They
#: are dropped in Python, NOT by pathspec: on git 2.43 an extra `:(exclude,glob)0xcc/**` (or `…/**/*`) beside
#: the export exclusions made unrelated positive pathspecs (`frontend/index.html`, `manual/docs/**/*.md`) match
#: NOTHING (measured 2026-11-08), which would have emptied whole classes.
_EXPORT_NAME = re.compile(r"(^|/)docs/legacy-identifiers[^/]*\.md$")
_RECORD_PREFIXES = ("0xcc/", ".claude/")

#: Surface classes enforced at P4 (PD-3), and the one report-only class (TPD-16).
ENFORCED_CLASSES = ("readme", "manual_docs", "manual_config", "svg", "index_html", "api", "mcp")
REPORT_ONLY_CLASSES = ("backend_strings",)
FILE_CLASSES = ("readme", "manual_docs", "manual_config", "svg", "index_html", "backend_strings")
LIVE_CLASSES = ("api", "mcp")


@dataclass(frozen=True)
class Item:
    cls: str
    where: str
    text: str


@dataclass
class Finding:
    cls: str
    where: str
    token: str
    rules: list[str] = field(default_factory=list)

    def line(self) -> str:
        return f"[{self.cls}] {self.where}: {self.token} ({', '.join(self.rules)})"


# ── discovery ──────────────────────────────────────────────────────────────────────────────────────────
def ls_files(root: Path, globs: Iterable[str]) -> list[str]:
    """Tracked files matching `globs`, never the export and never a record tree."""
    specs = [f":(glob){g}" for g in globs]
    out = subprocess.run(["git", "ls-files", "-z", "--", *specs, *EXPORT_EXCLUDES],
                         cwd=root, capture_output=True, check=True).stdout.decode("utf-8")
    paths = sorted(p for p in out.split("\0") if p)
    # Defence in depth: git pathspec magic has failed in both directions on 2.43, so the exclusions are
    # re-applied here. `test_export_files_are_never_surfaces` plants both and checks each layer.
    return [p for p in paths if not _EXPORT_NAME.search(p) and not p.startswith(_RECORD_PREFIXES)]


GLOBS: dict[str, tuple[str, ...]] = {
    "readme": ("README*.md", "**/README*.md", "CONTRIBUTING.md", "**/CONTRIBUTING.md", "SECURITY.md",
               "**/SECURITY.md"),
    "manual_docs": ("manual/docs/**/*.md", "manual/docs/**/*.mdx"),
    "manual_config": ("manual/docusaurus.config.ts", "manual/sidebars.ts"),
    "svg": ("frontend/public/**/*.svg", "frontend/src/assets/**/*.svg"),
    "index_html": ("frontend/index.html",),
    "backend_strings": ("backend/src/**/*.py",),
}

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_FRONT_KEY = re.compile(r"^(title|sidebar_label)\s*:\s*(.+?)\s*$", re.MULTILINE)
_CONFIG_KEY = re.compile(r"\b(title|tagline|label|copyright)\s*:\s*(['\"`])((?:\\.|(?!\2).)*)\2", re.DOTALL)
_SVG_TEXT = re.compile(r"<(title|text)\b[^>]*>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_HTML_TITLE = re.compile(r"<title\b[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(text: str) -> list[tuple[int, str]]:
    out, fenced = [], False
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if not fenced:
            out.append((n, line))
    return out


def readme_items(path: str, text: str) -> list[Item]:
    items = []
    for n, line in _strip_code_fences(text):
        m = _HEADING.match(line)
        if m:
            items.append(Item("readme", f"{path}:{n}", m.group(2)))
    return items


def manual_doc_items(path: str, text: str) -> list[Item]:
    items = []
    body_start = 0
    fm = _FRONT.match(text)
    if fm:
        body_start = fm.end()
        for m in _FRONT_KEY.finditer(fm.group(1)):
            items.append(Item("manual_docs", f"{path} front matter {m.group(1)}", m.group(2).strip("'\"")))
    offset = text[:body_start].count("\n")
    for n, line in _strip_code_fences(text[body_start:]):
        m = _HEADING.match(line)
        if m and len(m.group(1)) == 1:
            items.append(Item("manual_docs", f"{path}:{n + offset} first heading", m.group(2)))
            break
    return items


def manual_config_items(path: str, text: str) -> list[Item]:
    """Known config shape: `title`, `tagline`, `label`, `copyright` string values (navbar, footer, sidebars)."""
    items = []
    for m in _CONFIG_KEY.finditer(text):
        line = text[: m.start()].count("\n") + 1
        items.append(Item("manual_config", f"{path}:{line} {m.group(1)}", m.group(3)))
    return items


def svg_items(path: str, text: str) -> list[Item]:
    return [Item("svg", f"{path} <{m.group(1)}>", re.sub(r"\s+", " ", m.group(2)).strip())
            for m in _SVG_TEXT.finditer(text)]


def index_html_items(path: str, text: str) -> list[Item]:
    return [Item("index_html", f"{path} <title>", m.group(1).strip()) for m in _HTML_TITLE.finditer(text)]


def backend_string_items(path: str, text: str) -> list[Item]:
    """String constants of a module (AST), docstrings excluded (TPD-16; never comments)."""
    tree = ast.parse(text, filename=path)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    items = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            # Wrapped in quotes, the literal's source form: register patterns anchor on them (DR-01's
            # quoted kinds), and a quote is never a token character.
            items.append(Item("backend_strings", f"{path}:{node.lineno}", f'"{node.value}"'))
    return items


EXTRACTORS: dict[str, Callable[[str, str], list[Item]]] = {
    "readme": readme_items,
    "manual_docs": manual_doc_items,
    "manual_config": manual_config_items,
    "svg": svg_items,
    "index_html": index_html_items,
    "backend_strings": backend_string_items,
}


def file_items(cls: str, root: Path = REPO_ROOT) -> tuple[list[str], list[Item]]:
    """`(files discovered, items extracted)` for one file-backed surface class."""
    files = ls_files(root, GLOBS[cls])
    items: list[Item] = []
    for rel in files:
        items.extend(EXTRACTORS[cls](rel, (root / rel).read_text(encoding="utf-8", errors="replace")))
    return files, items


def api_items(app: Any) -> list[Item]:
    """The FastAPI title and the health `service`, read from the app object and its served route."""
    from fastapi.testclient import TestClient

    items = [Item("api", "FastAPI title", app.title)]
    body = TestClient(app).get("/api/health").json()
    items.append(Item("api", "GET /api/health service", body["service"]))
    return items


def _run(coro: Any) -> Any:
    # `asyncio.run` cannot nest inside a running loop (async tests): run in a worker thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _tool_text(result: Any) -> str:
    """The text a tool call answered with, whatever the mcp version's return shape."""
    parts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
        elif hasattr(value, "text") and isinstance(value.text, str):
            parts.append(value.text)
        elif hasattr(value, "content"):
            walk(value.content)
        elif isinstance(value, dict):
            parts.append(json.dumps(value))
        elif isinstance(value, str):
            parts.append(value)

    walk(result)
    return "\n".join(parts)


def _decoded(text: str) -> str:
    """A tool answer as the agent READS it. A tool returns a JSON document, so the raw text holds
    `\\n` and `\\"` escapes: scanning it reported the token `nmiStudio` for a line that starts with the
    word, and could never match a quoted document kind (`"mistudio.circuit-definition"`, which the
    register's DR-01 record allows only with its quotes) because the closing quote arrived as `\\"`.
    The strings of the parsed document are scanned instead, each re-quoted the way JSON shows a string
    value; text that is not JSON is scanned as it is."""
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return text
    parts: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            for k, item in v.items():
                parts.append(str(k))
                walk(item)
        elif isinstance(v, (list, tuple)):
            for item in v:
                walk(item)

    walk(value)
    return "\n".join(parts) if parts else text


def mcp_items(server: Any) -> list[Item]:
    """Server name, instructions, every tool's name and description, and the howto topics' text — all from
    the built server object, never a tool module."""
    items = [Item("mcp", "server name", server.name), Item("mcp", "server instructions", server.instructions or "")]
    tools = _run(server.list_tools())
    for tool in tools:
        items.append(Item("mcp", f"tool name {tool.name}", tool.name))
        items.append(Item("mcp", f"tool description {tool.name}", tool.description or ""))
    howto = [t.name for t in tools if t.name.endswith("_howto")]
    for name in howto:
        index = json.loads(_tool_text(_run(server.call_tool(name, {}))))
        for topic in index.get("topics", []):
            if topic == "tools":  # derived from the registry: the tool names/descriptions above
                continue
            text = _decoded(_tool_text(_run(server.call_tool(name, {"topic": topic}))))
            items.append(Item("mcp", f"howto topic {topic}", text))
    return items


def live_mcp_server() -> Any:
    """The real server, every category on and every product URL set (the `next_calls` recipe)."""
    import pytest

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    mp = pytest.MonkeyPatch()
    try:
        for spec in PRODUCTS:
            mp.setenv(spec.url_env, f"http://{spec.name}.test")
        settings = MCPSettings(tool_categories=",".join(sorted(all_categories())), allow_anonymous=True)
        server, _client = build_server(settings, stdio=True)
    finally:
        mp.undo()
    return server


_LIVE: dict[str, list[Item]] = {}


def live_items(cls: str) -> list[Item]:
    """Built once per test process (the server takes ~20 s to build and list)."""
    if cls not in _LIVE:
        _LIVE[cls] = _live_items(cls)
    return list(_LIVE[cls])


def _live_items(cls: str) -> list[Item]:
    if cls == "api":
        from src.main import app

        return api_items(app)
    if cls == "mcp":
        return mcp_items(live_mcp_server())
    raise KeyError(cls)


# ── the scanner ────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Rules:
    patterns: tuple[re.Pattern[str], ...]          # (a) entry patterns, (d) active allow guards
    literals: tuple[str, ...]                       # (a) entry literals (exact)
    olds_exact: tuple[str, ...]                     # (b) env, history_tool olds
    olds_ci: tuple[str, ...]                        # (b) host olds (DNS is case-insensitive)
    repos: tuple[re.Pattern[str], ...]              # (c) whole <owner>/<name> tokens
    deny: tuple[tuple[str, re.Pattern[str]], ...]   # active deny guards


def rules_for(export: le.Export) -> Rules:
    entries = [e for e in export.entries() if e["status"] != "retired"]
    active = export.active_guards()
    return Rules(
        patterns=tuple([re.compile(e["pattern"]) for e in entries]
                       + [re.compile(g["pattern"]) for g in active if g["effect"] == "allow"]),
        literals=tuple(lit for e in entries for lit in e["literals"]),
        olds_exact=tuple([r["old"] for r in export.env() if r["status"] == "transition"]
                         + [r["old"] for r in export.history_tools()]),
        olds_ci=tuple(r["old"] for r in export.hosts() if r["status"] == "transition"),
        repos=tuple(re.compile(r"(?<![\w-])" + re.escape(r["old"]) + r"(?![\w-])", re.IGNORECASE)
                    for r in export.repos("transition")),
        deny=tuple((g["id"], re.compile(g["pattern"])) for g in active if g["effect"] == "deny"),
    )


def _allowed_spans(text: str, rules: Rules) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in (*rules.patterns, *rules.repos):
        spans.extend(m.span() for m in pattern.finditer(text) if m.end() > m.start())
    low = text.lower()
    for needle, hay in ([(x, text) for x in (*rules.literals, *rules.olds_exact)]
                        + [(x.lower(), low) for x in rules.olds_ci]):
        start = hay.find(needle)
        while start != -1:
            spans.append((start, start + len(needle)))
            start = hay.find(needle, start + 1)
    return spans


def _covered(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(s <= start and end <= e for s, e in spans)


def _token(text: str, start: int, end: int) -> str:
    while start > 0 and TOKEN_CHAR.match(text[start - 1]):
        start -= 1
    while end < len(text) and TOKEN_CHAR.match(text[end]):
        end += 1
    return text[start:end].strip(".")


def scan(items: Iterable[Item], rules: Rules) -> list[Finding]:
    findings: list[Finding] = []
    for item in items:
        text = item.text
        spans = _allowed_spans(text, rules)
        brand: list[tuple[int, int, Finding]] = []
        for m in STEM.finditer(text):
            if _covered(spans, *m.span()):
                continue
            f = Finding(item.cls, item.where, _token(text, *m.span()), [f"brand:{m.group(0).lower()}"])
            brand.append((m.start(), m.end(), f))
            findings.append(f)
        for guard_id, pattern in rules.deny:
            for m in pattern.finditer(text):
                if m.end() == m.start() or _covered(spans, *m.span()):
                    continue
                hit = [f for s, e, f in brand if s < m.end() and m.start() < e]
                if hit:
                    for f in hit:
                        if guard_id not in f.rules:
                            f.rules.append(guard_id)
                else:
                    findings.append(Finding(item.cls, item.where, m.group(0), [guard_id]))
    return findings


def enforced(mode: str = BRAND_GUARD_MODE) -> bool:
    if mode not in ("report", "enforce"):
        raise ValueError(f"BRAND_GUARD_MODE must be 'report' or 'enforce', not {mode!r}")
    return mode == "enforce"


def mode_required_by(export: le.Export) -> str:
    """The mode the export's `closed` requires: `enforce` once P4 is closed (CF-21), else `report`."""
    return "enforce" if export.closed_at_least("P4") else "report"


def effective_mode(export: le.Export, mode: str | None = None) -> str:
    """The mode a run against `export` actually uses (R23, operator 2026-11-11).

    ENFORCED where the full export exists: the private repository, whose CI a commit must pass before it
    is mirrored. REPORTED on the public mirror, whatever the constant says: the mirror carries the public
    export alone, which withholds the old host names (DR-29) and the private repository names, so a
    string those records allow would be a finding there and an enforced mirror run would fail on text
    the private run has already passed. The private CI is the enforcement point; nothing withheld is
    published to make the mirror able to enforce.

    `mode` defaults to the module constant READ AT CALL TIME (so the P4-close flip needs no other edit).
    """
    mode = BRAND_GUARD_MODE if mode is None else mode
    enforced(mode)  # refuses anything but the two modes
    return "report" if export.is_public else mode


def report(label: str, findings: list[Finding], mode: str = BRAND_GUARD_MODE) -> list[str]:
    """Print every finding; in enforce mode return them as failures (the caller asserts the list empty)."""
    lines = sorted(f.line() for f in findings)
    print(f"\n[brand guard {mode}] {label}: {len(lines)} finding(s)")
    for line in lines:
        print("  " + line)
    return lines if enforced(mode) else []
