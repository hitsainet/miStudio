"""The GCP edge nginx bounds request bodies (016 R3-B6, suite tracker 4.22).

`docker/nginx.gcp.conf` carried `client_max_body_size 0` at the http level — no limit on
any server of a public edge. The file IS the artifact (there is no registry to read), so
this parses it, and fails CLOSED: a directive that cannot be found is a failure, never a
pass, because a scrape that matches nothing asserts nothing (memory:
source-scrape-guards-fail-open).
"""

import re
from pathlib import Path

import pytest

from src.core.body_limit import MAX_BODY_BYTES

CONF = Path(__file__).resolve().parents[3] / "docker" / "nginx.gcp.conf"

_UNITS = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}


def _bytes(value: str) -> int:
    m = re.fullmatch(r"(\d+)([kKmMgG]?)", value)
    assert m, f"unparseable size {value!r}"
    return int(m.group(1)) * _UNITS[m.group(2).lower()]


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _blocks(text: str):
    """Yield (server_name, body) for every `server { ... }` block, brace-matched."""
    for m in re.finditer(r"\bserver\s*\{", text):
        depth, i = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[m.end():i - 1]
        name = re.search(r"server_name\s+([^;]+);", body)
        yield (name.group(1).split()[0] if name else None), body


@pytest.fixture(scope="module")
def conf():
    assert CONF.is_file(), f"{CONF} is gone — the guard must not pass on a missing file"
    return _strip_comments(CONF.read_text())


def _http_level(conf: str) -> str:
    """The http block with every server block removed."""
    out, i = [], 0
    for m in re.finditer(r"\bserver\s*\{", conf):
        if m.start() < i:
            continue
        depth, j = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(conf[j], 0)
            j += 1
        out.append(conf[i:m.start()])
        i = j
    out.append(conf[i:])
    return "".join(out)


def test_no_unbounded_body_anywhere(conf):
    sizes = re.findall(r"client_max_body_size\s+([^;\s]+)\s*;", conf)
    assert sizes, "no client_max_body_size directive at all — nginx would default to 1m silently"
    assert "0" not in sizes, "client_max_body_size 0 disables the limit"


def test_the_http_default_sits_just_above_the_backends_own_cap(conf):
    found = re.findall(r"client_max_body_size\s+([^;\s]+)\s*;", _http_level(conf))
    assert len(found) == 1, f"expected one http-level bound, found {found}"
    size = _bytes(found[0])
    assert size >= MAX_BODY_BYTES, (
        "below the backend's 1 MiB cap, nginx would refuse bodies the API accepts"
    )
    assert size <= 4 * MAX_BODY_BYTES, (
        f"{found[0]} is far above the backend's cap: the edge reads bodies the API "
        "will refuse anyway"
    )


def test_the_mistudio_server_does_not_widen_the_default(conf):
    blocks = dict(_blocks(conf))
    # A dict lookup by exact server_name, not a substring test (CodeQL #65/#66 read it as one).
    assert blocks.get("mistudio.hitsai.net") is not None, "the miStudio server block was not found"
    assert "client_max_body_size" not in blocks["mistudio.hitsai.net"]


def test_neuronpedia_keeps_its_own_bound(conf):
    blocks = dict(_blocks(conf))
    # A dict lookup by exact server_name, not a substring test (CodeQL #65/#66 read it as one).
    assert blocks.get("neuron.hitsai.net") is not None, "the Neuronpedia server block was not found"
    found = re.findall(r"client_max_body_size\s+([^;\s]+)\s*;", blocks["neuron.hitsai.net"])
    assert found == ["100m"]
