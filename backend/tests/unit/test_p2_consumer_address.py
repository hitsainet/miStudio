"""P2 consumer address (042 FTASKS 11.3; FTDD §13 P2; S001 §11.3 row 62; CF-27).

In the S001 P2 window miServer's Service moves to `miserver-backend.miserver`. This repo's two consumers of
it, the API container of the backend Deployment and the MCP server, then set the NEW variable name
(`MISERVER_API_URL`, read first since P0) to the new in-cluster address, and the old `MILLM_API_URL` entries
leave the manifests. The MCP pod skips every miServer category at registration when the address is empty,
so a commit that deletes the old entry and forgets the new one would switch ~94 tools off silently; the
backend reads it for the behaviour checklist and spot-checks.

The push is config-only (no image path changes), so the same commit bumps the pod-template annotation
`rebrand.hitsai/config-rev` on both Deployments to the short SHA the change is made on (FTID §9.4 step 3a):
P-WATCH-CONFIG reads that annotation back from the live objects. In the private repository the value must
name an ancestor commit whose manifests do NOT yet carry the new address (so the bump moved with the change).

The names come from `src/core/legacy_names.ENV` (the registry), not from this file. When miServer's checkout
is reachable (`MISERVER_REPO`), its rendered manifests must define the Service the address
names; a checkout without miServer's P2 manifests skips that one test loudly (fails under the cross-repo
switch), because miServer pushes first in the window.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from src.core import legacy_names
from tests.support.cross_repo import require_cross_repo
from tests.support.legacy_export import REPO_ROOT
from tests.support.sibling_repos import miserver_repo

PAIR = next(r for r in legacy_names.ENV if r.new == "MISERVER_API_URL")
ADDRESS = "http://miserver-backend.miserver.svc.cluster.local:8000"
ANNOTATION = "rebrand.hitsai/config-rev"
#: `(Deployment, container)` of every consumer of miServer's address in this repo's manifests.
CONSUMERS = {("miworkbench-backend", "backend"), ("miworkbench-mcp", "mcp-server")}
PRIVATE = (REPO_ROOT / "0xcc").is_dir()


def _docs(root: Path) -> list[tuple[str, dict]]:
    out = []
    for path in sorted((root / "k8s").rglob("*.yaml")):
        try:
            loaded = list(yaml.safe_load_all(path.read_text()))
        except yaml.YAMLError:
            if root == REPO_ROOT:
                raise  # this repo's manifests must parse; a sibling's template file may not
            continue
        for doc in loaded:
            if isinstance(doc, dict):
                out.append((str(path.relative_to(root)), doc))
    return out


def _containers(root: Path = REPO_ROOT):
    for rel, doc in _docs(root):
        if doc.get("kind") not in ("Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"):
            continue
        spec = doc.get("spec") or {}
        if doc["kind"] == "CronJob":
            spec = ((spec.get("jobTemplate") or {}).get("spec")) or {}
        template = spec.get("template") or {}
        pod = template.get("spec") or {}
        for c in [*(pod.get("containers") or []), *(pod.get("initContainers") or [])]:
            yield rel, doc, template, c


def _env(container: dict) -> dict[str, object]:
    return {e["name"]: e.get("value", e.get("valueFrom")) for e in container.get("env") or []}


def test_consumers_set_the_new_name_to_the_new_address():
    setting = {(doc["metadata"]["name"], c["name"]): _env(c)[PAIR.new]
               for _, doc, _, c in _containers() if PAIR.new in _env(c)}
    assert setting == {k: ADDRESS for k in CONSUMERS}, setting


def test_old_name_left_every_manifest():
    still = sorted(f"{rel}: {doc['metadata']['name']}/{c['name']}"
                   for rel, doc, _, c in _containers() if PAIR.old in _env(c))
    assert not still, f"{PAIR.old} is still set (the new name {PAIR.new} replaces it from P2): {still}"


def _revs() -> dict[str, str | None]:
    names = {d for d, _ in CONSUMERS}
    return {doc["metadata"]["name"]: ((t.get("metadata") or {}).get("annotations") or {}).get(ANNOTATION)
            for _, doc, t, _ in _containers() if doc.get("kind") == "Deployment" and doc["metadata"]["name"] in names}


def test_consumer_pod_templates_carry_the_config_rev():
    revs = _revs()
    assert set(revs) == {d for d, _ in CONSUMERS}, revs
    for name, rev in revs.items():
        assert rev and re.fullmatch(r"[0-9a-f]{7,40}", str(rev)), f"{name}: {ANNOTATION}={rev!r}"
    assert len(set(revs.values())) == 1, f"one config-only commit, one value: {revs}"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def test_config_rev_names_the_commit_the_change_was_made_on():
    if not PRIVATE:
        pytest.skip("CONFIG-REV ANCESTRY SKIPPED LOUDLY: mirror-shaped tree (orphan snapshot, no history)")
    rev = next(iter(_revs().values()))
    assert _git("cat-file", "-e", f"{rev}^{{commit}}").returncode == 0, f"{ANNOTATION}={rev} is not a commit here"
    assert _git("merge-base", "--is-ancestor", rev, "HEAD").returncode == 0, (
        f"{ANNOTATION}={rev} is not an ancestor of HEAD (rebased? set it to the commit the change sits on)")
    for path in ("k8s/base/backend.yaml", "k8s/base/mcp.yaml"):
        shown = _git("show", f"{rev}:{path}")
        assert shown.returncode == 0, shown.stderr
        assert ADDRESS not in shown.stdout, (
            f"{path} already held {ADDRESS} at {rev}: the annotation was not bumped with the change")


def test_miserver_manifests_define_the_addressed_service():
    host = ADDRESS.split("//", 1)[1].rsplit(":", 1)
    service, namespace = host[0].split(".")[:2]
    port = int(host[1])
    repo = miserver_repo()
    if not (repo / "k8s").is_dir():
        if require_cross_repo():
            pytest.fail(f"miServer's checkout is required and absent: {repo}")
        pytest.skip(f"CROSS-REPO SKIPPED LOUDLY: miServer's checkout is absent ({repo})")
    found = []
    for _, doc in _docs(repo):
        meta = doc.get("metadata") or {}
        if doc.get("kind") == "Service" and meta.get("name") == service and meta.get("namespace") == namespace:
            found.append([p.get("port") for p in (doc.get("spec") or {}).get("ports") or []])
    if not found:
        msg = (f"{repo} defines no Service {service} in namespace {namespace}: miServer's P2 manifests are not "
               "in this checkout (set MISERVER_REPO to a tree that has them)")
        if require_cross_repo():
            pytest.fail(msg)
        pytest.skip("CROSS-REPO SKIPPED LOUDLY: " + msg)
    assert any(port in ports for ports in found), found
