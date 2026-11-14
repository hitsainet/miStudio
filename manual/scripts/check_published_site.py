#!/usr/bin/env python3
"""Every page of a built manual is served at a published address (042 FR-042.5; the P0 exit gate).

Usage: check_published_site.py BASE_URL BUILD_DIR

Reads BUILD_DIR/sitemap.xml (the pages Docusaurus built), maps each page onto BASE_URL, requests it, and for
each page requests every same-origin `/assets/` and `/img/` reference it carries. Prints every non-200 with
its status, then a summary. Exit 1 on any failure, 0 when every page and asset answered 200, 2 on bad usage
or an empty sitemap (no pages is never "all served").

Why: the manual moves to https://docs-miworkbench.hitsai.net/ at its root (baseUrl '/'). Until the custom
domain, its DNS and its certificate are all in place, the pages or their stylesheets 404; P1 (the repository
renames) must not start before this run is clean. A run against the custom domain with the old base path
`/miStudio/` fails on every page (the live negative control, FTASKS 2.5).

Standard library only (urllib, 10 s timeout per request, 4 workers), so it runs anywhere Python does.
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

TIMEOUT_S = 10
WORKERS = 4
ASSET_PREFIXES = ("assets/", "img/")
SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


class _Refs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.refs.append(value)


def sitemap_paths(build_dir: Path) -> list[str]:
    """The page paths of the sitemap, relative to the site's base ('' for the root page)."""
    root = ET.parse(build_dir / "sitemap.xml").getroot()
    locs = [el.text.strip() for el in root.iter(f"{SITEMAP_NS}loc") if el.text]
    out = []
    for loc in locs:
        path = urlsplit(loc).path
        out.append(path)
    # The sitemap's URLs carry the build's own baseUrl; strip their common base so they map onto BASE_URL.
    base = _common_dir(out)
    return sorted({p[len(base):].lstrip("/") for p in out})


def _common_dir(paths: list[str]) -> str:
    if not paths:
        return "/"
    first = paths[0]
    prefix = first[: first.rfind("/") + 1] if "/" in first else "/"
    while prefix and not all(p.startswith(prefix) for p in paths):
        prefix = prefix[: prefix[:-1].rfind("/") + 1]
    return prefix or "/"


def fetch(url: str) -> tuple[int, bytes]:
    try:
        with urlopen(Request(url, headers={"User-Agent": "check_published_site/1"}), timeout=TIMEOUT_S) as r:
            return r.status, r.read()
    except HTTPError as exc:
        return exc.code, b""
    except (URLError, TimeoutError, OSError) as exc:
        return -1, str(exc).encode()


def page_assets(base_url: str, page_url: str, html: bytes) -> set[str]:
    parser = _Refs()
    parser.feed(html.decode("utf-8", errors="replace"))
    base = urlsplit(base_url)
    out = set()
    for ref in parser.refs:
        absolute = urljoin(page_url, ref)
        parts = urlsplit(absolute)
        if (parts.scheme, parts.netloc) != (base.scheme, base.netloc):
            continue
        rel = parts.path[len(base.path):] if parts.path.startswith(base.path) else None
        if rel is not None and rel.startswith(ASSET_PREFIXES):
            out.add(f"{base.scheme}://{base.netloc}{parts.path}")
    return out


def check(base_url: str, build_dir: Path) -> tuple[int, int, list[str]]:
    """(pages checked, assets checked, failures as 'STATUS URL')."""
    base_url = base_url if base_url.endswith("/") else base_url + "/"
    pages = [urljoin(base_url, p) for p in sitemap_paths(build_dir)]
    failures: list[str] = []
    assets: set[str] = set()
    with ThreadPoolExecutor(WORKERS) as pool:
        for url, (status, body) in zip(pages, pool.map(fetch, pages)):
            if status != 200:
                failures.append(f"{status} {url}")
            else:
                assets |= page_assets(base_url, url, body)
        ordered = sorted(assets)
        for url, (status, _body) in zip(ordered, pool.map(fetch, ordered)):
            if status != 200:
                failures.append(f"{status} {url}")
    return len(pages), len(assets), failures


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    base_url, build_dir = argv[0], Path(argv[1])
    if not re.match(r"^https?://", base_url) or not (build_dir / "sitemap.xml").is_file():
        print("usage: BASE_URL must be http(s); BUILD_DIR must hold sitemap.xml", file=sys.stderr)
        return 2
    pages, assets, failures = check(base_url, build_dir)
    if pages == 0:
        print(f"{build_dir}/sitemap.xml lists no page — refusing to report clean", file=sys.stderr)
        return 2
    for f in failures:
        print(f)
    print(f"{base_url}: {pages} pages (sitemap {pages}), {assets} assets, {len(failures)} failure(s)",
          file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
