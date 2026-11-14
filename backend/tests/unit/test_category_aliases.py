"""MCP category keys after the alias removal (042 TPD-5, FR-042.17, FR-042.35; FTDD §6.4; FTID §3.6; FTASKS 3.19,
12.12, 18.5).

From the P3 cut the canonical keys are `miserver_*`; until ENH-009 P7 the OLD `millm_*` keys mapped to them and were
recorded. P7 removed the aliases (group G2): the new keys build every miServer-product tool and are never recorded,
and an old key is an unknown category (`test_legacy_retired.py::test_old_category_keys_unknown`, control K-18.G2).
Read from the LIVE registry `build_server()` assembles.
"""
from __future__ import annotations

import pytest

from src.core import legacy_names
from src.mcp_server.config import VALID_CATEGORIES, MCPSettings
from src.mcp_server.server import build_server

NEW_KEYS = sorted(c for c in VALID_CATEGORIES if c.startswith("miserver_"))


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setenv("MISERVER_API_URL", "http://miserver.test")
    monkeypatch.delenv("MILLM_API_URL", raising=False)
    legacy_names.reset()
    yield
    legacy_names.reset()


def _tools(categories: str) -> tuple[set[str], MCPSettings]:
    settings = MCPSettings(tool_categories=categories, allow_anonymous=True, _env_file=None)
    mcp, _ = build_server(settings, stdio=True)
    return {t.name for t in mcp._tool_manager.list_tools()}, settings  # noqa: SLF001


def test_the_tables_agree():
    assert len(NEW_KEYS) == 7
    assert legacy_names.CATEGORY_ALIASES == {}, "P7 removed the category aliases (FTASKS 18.5)"
    assert legacy_names.watching()["category"] == 0


def test_new_keys_enable_every_miserver_tool_and_record_nothing():
    with_new, settings = _tools(",".join(NEW_KEYS))
    assert {n for n in with_new if n.startswith("miserver_")}, "the miserver_* categories registered nothing"
    assert settings.unknown_categories() == set()
    assert settings.enabled_categories() == set(NEW_KEYS)
    assert legacy_names.snapshot()["in_use"] == [], "the NEW keys must not be recorded"
