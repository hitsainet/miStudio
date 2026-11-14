"""What stays of the P3 MCP cut after its script was deleted at ENH-009 P7 (042 FTASKS 18.12; FTID §3.9).

`scripts/rebrand/mcp_cut.py` (plan / apply / verify) did its work at P3 and was deleted at P7. Its OUTPUT stays as
history and as the authority the cut's lasting guarantees are checked against: `mcp_rename_map.json` (the map the
cut applied) and `assertion_counts.json` (the per-file counts it planned against). The two helpers the
registry checks need moved here unchanged, so `test_mcp_cut.py` keeps holding FR-042.15 (a) - (e) on the live
registry of every later tree.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SUPPORT = Path(__file__).resolve().parent
MAP_PATH = SUPPORT / "mcp_rename_map.json"


def load_map() -> dict[str, Any]:
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


def digest(value: Any, renames: Mapping[str, str] | None = None) -> str:
    """SHA-256 of `value`'s canonical JSON; with `renames` ({old: new}), each `new` is read back as `old`."""
    text = json.dumps(value, sort_keys=True, default=repr)
    for old, new in (renames or {}).items():
        text = text.replace(new, old)
    return hashlib.sha256(text.encode()).hexdigest()


def registered_names_with_duplicates() -> list[str]:
    """Every tool registration of the real build, counting a second registration of one name.

    MCPServer's tool manager keeps one entry per name (a second `add_tool` warns and keeps the first), so a
    module registered twice is invisible in `list_tools()`. Count `add_tool` calls instead, on the real build.
    """
    import pytest
    from mcp.server.mcpserver.tools.tool_manager import ToolManager

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    seen: list[str] = []
    original = ToolManager.add_tool

    def counting(self, fn, name=None, *args, **kwargs):  # noqa: ANN001
        tool = original(self, fn, name, *args, **kwargs)
        seen.append(name or getattr(tool, "name", None) or fn.__name__)
        return tool

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(ToolManager, "add_tool", counting)
        for spec in PRODUCTS:
            mp.setenv(spec.url_env, f"http://{spec.name}.test")
        build_server(MCPSettings(tool_categories=",".join(sorted(all_categories())), allow_anonymous=True),
                     stdio=True)
    finally:
        mp.undo()
    return seen


def duplicate_problems(names: list[str], data: Mapping[str, Any]) -> list[str]:
    """FR-042.15 (d): no name registered twice, and no old name registered beside its new one."""
    problems = []
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        problems.append(f"(d) registered twice: {dups}")
    beside = sorted(o for o, n in data["tools"].items() if o in names and n in names)
    if beside:
        problems.append(f"(d) old name registered beside its new one: {beside}")
    return problems
