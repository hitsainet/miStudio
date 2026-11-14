"""Generality guard, backend half (Feature 040 FR-8 – FR-10; FTDD §6.8; FTASKS 1.1 – 1.2, 1.7).

No source names a behaviour. A behaviour is DATA — a catalogue row, a seeded judge prompt, a pack — and
code that spells one key works for that key and silently not for the next. This guard reads the tokens
from the data files (`tests/support/behaviour_tokens.py`, never a hand list) and fails on any match in:

* every string and comment token of `src/**/*.py` (Python's own `tokenize`, so a token inside an
  identifier is not a match and a comment is);
* the LIVE MCP registry — tool names, descriptions and every string in each tool's input schema — built
  by the real `build_server()` with every category on (what an agent reads);
* every `next` constructor's output in `services/detector_next.py`, called with its registered example;
* every file under `k8s/`.

Data files (seeds, fixtures) may name behaviours; that is where they belong.
"""
from __future__ import annotations

import asyncio
import io
import tokenize
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List

import pytest

from tests.support.behaviour_tokens import BACKEND, Hit, Matcher, derive_tokens

SRC = BACKEND / "src"
K8S = BACKEND.parent / "k8s"
_TEXT_TOKENS = {tokenize.STRING, tokenize.COMMENT}
for _name in ("FSTRING_MIDDLE", "TSTRING_MIDDLE"):
    if hasattr(tokenize, _name):
        _TEXT_TOKENS.add(getattr(tokenize, _name))


def source_files(root: Path = SRC) -> List[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def scan_python(files: Iterable[Path], matcher: Matcher, base: Path = BACKEND) -> List[Hit]:
    hits: List[Hit] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        try:
            relative = str(path.relative_to(base))
        except ValueError:
            relative = str(path)
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in _TEXT_TOKENS:
                continue
            for token, offset, line in matcher.find_lines(tok.string):
                hits.append(Hit(relative, tok.start[0] + offset, token, line))
    return hits


def live_tools():
    """Every tool the REAL server registers with every category and product on."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    mp = pytest.MonkeyPatch()
    try:
        for spec in PRODUCTS:
            mp.setenv(spec.url_env, f"http://{spec.name}.test")
        settings = MCPSettings(tool_categories=",".join(sorted(all_categories())), allow_anonymous=True)
        mcp, _client = build_server(settings, stdio=True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, mcp.list_tools()).result()
    finally:
        mp.undo()


def _strings_in(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings_in(k)
            yield from _strings_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings_in(v)


def scan_registry(tools, matcher: Matcher) -> List[Hit]:
    hits: List[Hit] = []
    for tool in tools:
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
        texts = [tool.name, tool.description or "", *list(_strings_in(schema))]
        for text in texts:
            for token in matcher.find(text):
                hits.append(Hit(f"mcp:{tool.name}", 0, token, text))
    return hits


def scan_next_constructors(matcher: Matcher) -> List[Hit]:
    from src.services import detector_next

    hits: List[Hit] = []
    for name, constructor in sorted(detector_next.CONSTRUCTORS.items()):
        produced = constructor.example()
        for text in _strings_in(produced):
            for token in matcher.find(text):
                hits.append(Hit(f"detector_next:{name}", 0, token, text))
    return hits


def scan_k8s(matcher: Matcher) -> List[Hit]:
    hits: List[Hit] = []
    for path in sorted(p for p in K8S.rglob("*") if p.is_file()):
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for token in matcher.find(line):
                hits.append(Hit(str(path.relative_to(BACKEND.parent)), number, token, line))
    return hits


@pytest.fixture(scope="module")
def matcher() -> Matcher:
    return Matcher(derive_tokens())


def _report(hits: List[Hit]) -> str:
    files = sorted({h.where for h in hits})
    return f"{len(hits)} hit(s) in {len(files)} place(s):\n" + "\n".join(str(h) for h in hits)


def test_source_names_no_behaviour(matcher):
    files = source_files()
    assert len(files) > 100, f"found only {len(files)} source files under {SRC}"
    hits = scan_python(files, matcher)
    assert not hits, _report(hits)


def test_live_registry_names_no_behaviour(matcher):
    tools = live_tools()
    assert len(tools) > 100, f"the live registry built only {len(tools)} tools"
    hits = scan_registry(tools, matcher)
    assert not hits, _report(hits)


def test_next_constructors_name_no_behaviour(matcher):
    from src.services import detector_next

    assert detector_next.CONSTRUCTORS, "detector_next registers no constructor"
    hits = scan_next_constructors(matcher)
    assert not hits, _report(hits)


def test_k8s_names_no_behaviour(matcher):
    assert K8S.is_dir() and any(K8S.rglob("*.yaml")), f"no manifests under {K8S}"
    hits = scan_k8s(matcher)
    assert not hits, _report(hits)


# ── the guard's own controls, permanent ─────────────────────────────────────────────────────────


def test_canary_in_a_source_file_is_found(matcher, tmp_path):
    """A file holding one derived token, scanned by the same function over the same discovery, is red."""
    canary = tmp_path / "canary_module.py"
    token = matcher.tokens[0]
    canary.write_text(f"# handles {token} specially\nVALUE = 'x'\n", encoding="utf-8")
    hits = scan_python([*source_files()[:5], canary], matcher, base=tmp_path)
    assert [h.where for h in hits] == ["canary_module.py"], hits


def test_a_code_branch_on_a_key_is_found(matcher, tmp_path):
    key = next(t for t in matcher.tokens if " " not in t)
    module = tmp_path / "branchy.py"
    module.write_text(f'def f(behaviour):\n    if behaviour.key == "{key}":\n        return 1\n', encoding="utf-8")
    assert [h.line for h in scan_python([module], matcher, base=tmp_path)] == [2]


def test_identifiers_are_not_matched(matcher, tmp_path):
    key = next(t for t in matcher.tokens if " " not in t and "-" not in t)
    module = tmp_path / "ident.py"
    module.write_text(f"{key}_count = 1\n", encoding="utf-8")
    assert scan_python([module], matcher, base=tmp_path) == []


def test_a_wrong_glob_fails_rather_than_matching_nothing(tmp_path):
    with pytest.raises(AssertionError, match="matched 0 file"):
        derive_tokens(("no/such/dir/*.json",), root=tmp_path)


def test_the_token_set_is_derived_not_listed():
    """The derivation reads every glob; it covers each seed's key and label."""
    import json

    tokens = derive_tokens()
    for path in sorted((SRC / "data" / "judge_prompt_seeds").glob("*.json")):
        doc = json.loads(path.read_text())
        assert doc["behaviour"].lower() in tokens
        assert " ".join(doc["behaviour_label"].lower().split()) in tokens
