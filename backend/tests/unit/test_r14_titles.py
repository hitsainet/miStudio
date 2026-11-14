"""R14 (operator, 2026-11-08): the long title is "miWorkbench" and the tagline "Mechanistic interpretability
workbench" (042 FR-042.31, FR-042.32; FTASKS 15.2, 15.3, 15.9).

The brand guard reports old names on these surfaces but, until S001 closes P4, does not FAIL on them, and
it never says what a title SHOULD be. These tests pin the decided values where a person reads them: the
browser tab, the API's own title and health answers, the manual's site config and the wordmark. Each value
is read from the real surface (the app object, the served routes, the files as parsed by the guard's own
extractors), not from a copy.
"""
from __future__ import annotations

import re

from tests.support import brand_surfaces as bs

TITLE = "miWorkbench"
TAGLINE = "Interpretability Workbench"  # R30 (operator 2026-11-11), replacing R14's


def _texts(cls: str, path: str) -> dict[str, str]:
    files, items = bs.file_items(cls)
    assert path in files, (path, files)
    return {i.where: i.text for i in items if i.where.startswith(path)}


def test_browser_title():
    assert list(_texts("index_html", "frontend/index.html").values()) == [TITLE]


def test_api_title_and_health_service():
    from fastapi.testclient import TestClient

    from src.main import app

    items = {i.where: i.text for i in bs.api_items(app)}
    assert items == {"FastAPI title": f"{TITLE} API", "GET /api/health service": f"{TITLE} Backend API"}
    client = TestClient(app)
    assert client.get("/").json()["message"] == f"{TITLE} API"
    assert client.get("/api/v1/system/health").json() == {"status": "healthy", "service": f"{TITLE} API"}


def test_manual_site_title_and_tagline():
    config = _texts("manual_config", "manual/docusaurus.config.ts")
    titles = [text for where, text in config.items() if where.endswith(" title")]
    taglines = [text for where, text in config.items() if where.endswith(" tagline")]
    assert taglines == [TAGLINE]
    # The site title and the navbar title; the footer's column titles name no product.
    assert titles.count(f"{TITLE} Manual") == 2, titles
    assert not [t for t in titles if bs.STEM.search(t)]


def test_wordmark_text():
    marks = _texts("svg", "frontend/src/assets/logo-wordmark.svg")
    assert list(marks.values()) == [TITLE], marks


def test_readme_first_heading():
    readme = _texts("readme", "README.md")
    first = next(text for where, text in sorted(readme.items(), key=lambda kv: int(kv[0].rsplit(":", 1)[1])))
    assert re.fullmatch(rf"{TITLE} — {TAGLINE}", first), first
