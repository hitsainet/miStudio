"""Argo CD reads this repository at its new name from ENH-009 P1 on (042 FR-042.30; FTDD T-24; FTASKS 9.2, 9.4).

GitHub redirects the old URL after the rename, but Argo CD's repository credential is matched by URL and a redirect
is not a contract, so every Application's `repoURL` names the new repository. From P1 to P3 every AppProject's
`sourceRepos` admitted BOTH URLs (S001 runbook 1.21 patched `sourceRepos` in the cluster before `repoURL`, so the
in-git file could not narrow it). From P4 (FR-042.39; FTASKS 15.8) it admits the NEW URL only: every Application
has read the new URL since P1 closed, and S001's P5 gate allows no tracked line naming the old private repository
(`test_p5_repo_name_gate.py`). The Neuronpedia app is edited in place: same name, project and destination.

The old/new pair comes from the export's `repo` records for the private owner (CF-13): no repository name is
spelled here. On the public mirror those records are withheld and the test skips by name, as
`test_workflow_repo_gates.py` does.

Control (FTASKS 9.6, K-9.6): revert one `repoURL` → `::test_repo_urls_and_source_repos` red.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.support import legacy_export as le
from tests.support import workflow_gates as wg

REPO = Path(__file__).resolve().parents[3]
ARGOCD = REPO / "k8s" / "argocd"


def _urls() -> tuple[str, str]:
    export = le.load()
    if export.is_public and any(n.startswith("repo:sha256:") for n in export.withheld):
        pytest.skip("on the public mirror the private repository's repo records are withheld: "
                    + ", ".join(n for n in export.withheld if n.startswith("repo:")))
    private, _public = wg.private_and_public_owner(wg.load_workflows(), export)
    old, new = wg.this_repo_pairs(export)[private]
    return f"https://github.com/{old}.git", f"https://github.com/{new}.git"


def _docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def _all() -> dict[str, list[dict]]:
    found = {p.name: _docs(p) for p in sorted(ARGOCD.glob("*.yaml"))}
    assert found, f"no Argo CD manifests under {ARGOCD}"
    return found


def test_repo_urls_and_source_repos():
    old, new = _urls()
    apps = projects = 0
    for name, docs in _all().items():
        for d in docs:
            if d["kind"] == "Application":
                apps += 1
                assert d["spec"]["source"]["repoURL"] == new, (name, d["metadata"]["name"])
            elif d["kind"] == "AppProject":
                projects += 1
                repos = d["spec"]["sourceRepos"]
                assert repos == [new], (name, repos)
                assert old not in repos and old != new
    # Fail closed: the two apps this repository deploys (miStudio and Neuronpedia) were both read.
    assert apps >= 2 and projects >= 2, (apps, projects)


def test_neuronpedia_app_edited_in_place():
    _old, new = _urls()
    docs = _docs(ARGOCD / "neuronpedia-app.yaml")
    kinds = {d["kind"]: d for d in docs}
    assert sorted(d["kind"] for d in docs) == ["AppProject", "Application", "ImageUpdater"], docs
    project, app, updater = kinds["AppProject"], kinds["Application"], kinds["ImageUpdater"]
    assert project["metadata"]["name"] == "neuronpedia"
    assert project["spec"]["destinations"] == [{"namespace": "neuronpedia", "server": "https://kubernetes.default.svc"}]
    assert app["metadata"]["name"] == "neuronpedia" and app["spec"]["project"] == "neuronpedia"
    assert app["spec"]["destination"] == {"server": "https://kubernetes.default.svc", "namespace": "neuronpedia"}
    assert app["spec"]["source"] == {"repoURL": new, "targetRevision": "gitops/pilot", "path": "k8s/neuronpedia/base"}
    assert [r["namePattern"] for r in updater["spec"]["applicationRefs"]] == ["neuronpedia"]
    # One Neuronpedia app across the folder: edited in place, not added beside a new one.
    named = [d["metadata"]["name"] for docs_ in _all().values() for d in docs_ if d["kind"] == "Application"]
    assert named.count("neuronpedia") == 1, named


def test_new_app_project_destinations():
    """FTASKS 13.10 (FTDD §11.3): the miWorkbench Application replaces the old one in git. AppProject
    `miworkbench` admits its own namespace only (the old one, admitted for the ExternalName, left at P7), the
    Application and its
    Image Updater are named `miworkbench`, the updater tracks the `hitsai/miworkbench-*` images under the names the
    kustomization rewrites, and no file still defines the old Application (S001 deletes the live one
    `--cascade=false` in the window)."""
    docs = _docs(ARGOCD / "miworkbench-app.yaml")
    kinds = {d["kind"]: d for d in docs}
    assert sorted(d["kind"] for d in docs) == ["AppProject", "Application", "ImageUpdater"], docs
    project, app, updater = kinds["AppProject"], kinds["Application"], kinds["ImageUpdater"]
    assert project["metadata"]["name"] == "miworkbench"
    server = "https://kubernetes.default.svc"
    assert project["spec"]["destinations"] == [{"namespace": "miworkbench", "server": server}]
    assert app["metadata"]["name"] == "miworkbench" and app["spec"]["project"] == "miworkbench"
    assert app["spec"]["destination"] == {"server": server, "namespace": "miworkbench"}
    assert app["spec"]["source"]["path"] == "k8s/base" and app["spec"]["source"]["targetRevision"] == "gitops/pilot"
    assert updater["metadata"]["name"] == "miworkbench-images"
    assert [r["namePattern"] for r in updater["spec"]["applicationRefs"]] == ["miworkbench"]
    notes = app["metadata"]["annotations"]
    listed = dict(item.strip().split("=", 1) for item in notes["argocd-image-updater.argoproj.io/image-list"].split(","))
    assert set(listed.values()) == {f"docker.io/hitsai/miworkbench-{x}:latest" for x in
                                    ("backend", "frontend", "rule-runner")}, listed
    kust = yaml.safe_load((REPO / "k8s" / "base" / "kustomization.yaml").read_text())
    rewritten = {i["name"] for i in kust["images"]}
    assert {notes[f"argocd-image-updater.argoproj.io/{alias}.kustomize.image-name"] for alias in listed} == rewritten
    assert not (ARGOCD / "mistudio-app.yaml").exists()
    named = [d["metadata"]["name"] for docs_ in _all().values() for d in docs_ if d.get("kind") == "Application"]
    assert "mistudio" not in named and named.count("miworkbench") == 1, named
