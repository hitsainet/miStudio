"""The new Hugging Face discovery tag beside the frozen one (042 FR-042.37; BR-W12, BR-042; TPD-24; FTASKS 17.x).

From P6 a published probe carries `mistudio-probe-definition` (frozen, DR-06: every existing search filters on
it) AND `miworkbench-probe-definition`. Both card writers (one probe, a probe family) go through
`render_repo_readme`, which builds its tags from `render_readme`, so both are read here.

The second test asks miServer itself, in miServer's own interpreter (no import of a sibling's code into this
process), whether its hub search accepts every discovery tag these cards write. miServer's search learned the
twin pair in its P4 (040 12.14); a tag it did not know would publish a probe no miServer search could find.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest
import yaml

from src.services import probe_definition_publisher as publisher
from tests.support.cross_repo import require_cross_repo
from tests.support.sibling_repos import miserver_import, miserver_path_or_none, miserver_repo


def _definition(name: str = "probe-a") -> dict[str, Any]:
    from tests.unit.test_probe_definition_publisher import _definition as base

    d = base()
    d["name"] = name
    return d


def _cards() -> dict[str, str]:
    one = _definition("probe-a")
    two = _definition("probe-b")
    e1 = publisher.manifest_entry(one, "probe-a.probe.json")
    e2 = publisher.manifest_entry(two, "probe-b.probe.json")
    return {
        "render_readme": publisher.render_readme(one, {"probes": [e1]}),
        # The family writer's shape: several probes in one manifest, each card's tags merged.
        "render_repo_readme": publisher.render_repo_readme(
            {"probes": [e1, e2]}, {"probe-a.probe.json": one, "probe-b.probe.json": two}),
    }


def _tags(card: str) -> list[str]:
    assert card.startswith("---\n"), card[:80]
    return list(yaml.safe_load(card.split("---\n", 2)[1])["tags"])


def test_publisher_writes_each_tag_once():
    assert publisher.DISCOVERY_TAGS == ("mistudio-probe-definition", "miworkbench-probe-definition")
    for writer, card in _cards().items():
        tags = _tags(card)
        for tag in publisher.DISCOVERY_TAGS:
            assert tags.count(tag) == 1, (writer, tag, tags)


def test_every_written_tag_accepted_by_miserver_search():
    repo = miserver_repo()
    python = repo / "venv" / "bin" / "python"
    hub = miserver_path_or_none(repo, "services", "cluster_hub_service.py")
    if hub is None or not hub.is_file() or not python.is_file():
        message = f"no miServer checkout with a venv at {repo}: the written tags were NOT checked against its search"
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip("CROSS-REPO CHECK SKIPPED LOUDLY: " + message)
    written = sorted({t for card in _cards().values() for t in _tags(card) if t.endswith("-definition")})
    assert written, "the cards wrote no discovery tag"
    out = subprocess.run(
        [str(python), "-c",
         f"import json, sys; from {miserver_import('services.cluster_hub_service', repo)} import hub_tag_twin; "
         "print(json.dumps({t: hub_tag_twin(t) for t in sys.argv[1:]}))", *written],
        cwd=repo, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    twins = json.loads(out.stdout.strip().splitlines()[-1])
    unknown = [t for t, twin in twins.items() if twin is None]
    assert not unknown, f"miServer's hub search does not know {unknown}"
    # The pair the cards write is the pair miServer pairs: each tag's twin is the other written tag.
    for tag, twin in twins.items():
        assert twin in written, (tag, twin, written)
