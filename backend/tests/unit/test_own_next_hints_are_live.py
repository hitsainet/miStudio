"""This app's OWN `next` hints name tools the live registry has (042 TPD-7, FR-042.18's own half; FTDD §5.5;
FTASKS 12.11, control K-12.36).

The cut renamed every `millm_<x>` tool, and the hints this repo writes itself (`miserver_send.refusal`,
`miserver_approvals.moved_refusal`, the probe-limit refusal, `services/next_calls.py`'s `NextCall`s) were renamed
with them as strings. A hint left at an old name points an agent at a tool that does not exist, and nothing else
fails: so every string constant shaped like a tool name (`^[a-z][a-z0-9_]*$`, one word) that is

* the value of a `"next"` key in a dict literal, or of a `"tool"` key (a `next` list entry), or
* the first argument of a `_next(...)` call,

under `src/mcp_server/` and in `src/services/next_calls.py`, must be a tool of the LIVE registry (every category
on, every product URL set). Prose hints ("retry later …") are not tool-shaped and are skipped. The walk must
find hints at all (it fails on an empty discovery).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.support.next_calls import live_input_schemas

SRC = Path(__file__).resolve().parents[2] / "src"
FILES = sorted((SRC / "mcp_server").rglob("*.py")) + [SRC / "services" / "next_calls.py"]
TOOL_SHAPED = re.compile(r"^[a-z][a-z0-9_]*$")


def _hints() -> list[tuple[str, int, str]]:
    out = []
    for path in FILES:
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = str(path.relative_to(SRC.parent))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value in ("next", "tool") and \
                            isinstance(value, ast.Constant) and isinstance(value.value, str) and \
                            TOOL_SHAPED.match(value.value):
                        out.append((rel, node.lineno, value.value))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_next" and \
                    node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                out.append((rel, node.lineno, node.args[0].value))
    return out


def test_every_tool_shaped_next_is_live():
    hints = _hints()
    assert len(hints) >= 5, f"the walk found {len(hints)} hints: it is not reading the modules"
    assert any(h[2].startswith("miserver_") for h in hints), "no miserver_* hint found: the walk is blind"
    live = set(live_input_schemas())
    dead = [f"{rel}:{line}: next names {name!r}, not a live tool" for rel, line, name in hints if name not in live]
    assert dead == [], "\n".join(dead)
