"""`manual/scripts/check_published_site.py` reports every page and asset that is not served (042 FR-042.5).

A local `http.server` serves a tiny built site whose sitemap lists three pages; one page references a
stylesheet the server does not have and one page is missing entirely. The script must list exactly those,
fetch every same-origin `/assets/` and `/img/` reference (and nothing off-origin), and exit 1. Control
K-1.16 makes the script exit 0 regardless and requires `test_exit_code_is_one_on_any_failure` red.
"""
from __future__ import annotations

import functools
import importlib.util
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "manual" / "scripts" / "check_published_site.py"


def _module():
    spec = importlib.util.spec_from_file_location("check_published_site", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CHECK = _module()

SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>https://docs-miworkbench.hitsai.net/</loc></url>
<url><loc>https://docs-miworkbench.hitsai.net/advanced/approvals</loc></url>
<url><loc>https://docs-miworkbench.hitsai.net/advanced/gone</loc></url>
</urlset>
"""


def _page(css: str) -> str:
    return (f'<html><head><link rel="stylesheet" href="/assets/css/{css}">'
            '<link rel="icon" href="/img/favicon.svg"></head>'
            '<body><a href="https://neuronpedia.org/x">x</a><img src="/img/logo.svg"></body></html>')


@pytest.fixture
def served(tmp_path):
    """(base url, build dir): the served tree lacks one page and one stylesheet the build has."""
    build = tmp_path / "build"
    (build / "advanced").mkdir(parents=True)
    (build / "sitemap.xml").write_text(SITEMAP)
    (build / "index.html").write_text(_page("styles.ok.css"))
    (build / "advanced" / "approvals.html").write_text(_page("styles.missing.css"))
    (build / "advanced" / "gone.html").write_text(_page("styles.ok.css"))
    site = tmp_path / "site"
    (site / "advanced").mkdir(parents=True)
    (site / "assets" / "css").mkdir(parents=True)
    (site / "img").mkdir()
    (site / "index.html").write_text(_page("styles.ok.css"))
    (site / "advanced" / "approvals").write_text(_page("styles.missing.css"))  # served without suffix
    (site / "assets" / "css" / "styles.ok.css").write_text("body{}")
    (site / "img" / "favicon.svg").write_text("<svg/>")
    (site / "img" / "logo.svg").write_text("<svg/>")
    handler = functools.partial(_Quiet, directory=str(site))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/", build
    finally:
        server.shutdown()
        server.server_close()


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep the test output readable
        pass


def test_reports_every_non_200_page_and_asset(served):
    base, build = served
    pages, assets, failures = CHECK.check(base, build)
    assert pages == 3
    # Same-origin assets only: the two stylesheets, the favicon and the logo; neuronpedia is off-origin.
    assert assets == 4, assets
    assert sorted(failures) == sorted([
        f"404 {base}advanced/gone",
        f"404 {base}assets/css/styles.missing.css",
    ]), failures


def test_exit_code_is_one_on_any_failure(served, capsys):
    base, build = served
    assert CHECK.main([base, str(build)]) == 1
    out = capsys.readouterr()
    assert f"404 {base}advanced/gone" in out.out
    assert "3 pages (sitemap 3)" in out.err and "2 failure(s)" in out.err


def test_exit_code_is_zero_when_everything_is_served(served):
    base, build = served
    (build / "sitemap.xml").write_text(SITEMAP.replace(
        "<url><loc>https://docs-miworkbench.hitsai.net/advanced/gone</loc></url>\n", "").replace(
        "<url><loc>https://docs-miworkbench.hitsai.net/advanced/approvals</loc></url>\n", ""))
    assert CHECK.main([base, str(build)]) == 0


def test_the_old_base_path_fails_every_page(served):
    """The live negative control's shape (FTASKS 2.5): the right host under the wrong base path."""
    base, build = served
    pages, _assets, failures = CHECK.check(base + "miStudio/", build)
    assert pages == 3 and len(failures) == 3, failures


def test_an_empty_sitemap_is_never_clean(tmp_path):
    (tmp_path / "sitemap.xml").write_text('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>')
    assert CHECK.main(["http://127.0.0.1:9/", str(tmp_path)]) == 2
