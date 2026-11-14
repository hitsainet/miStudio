"""T-31 (042 FTASKS 13.2, 13.3; FTDD §11.3; FR-042.21): the P3 rename set, read from the rendered manifests.

No object, namespace or object REFERENCE (labels, selectors, Secret, claim and volume names, Ingress backends,
the websocket-services annotation, Argo-visible names) carries the old brand, except:
* (until P7: the ExternalName Service `mistudio-backend` in namespace `mistudio`, PD-12, recognised by its type
  and its `rebrand.hitsai/transition` label, not by a name list; removed on evidence, FTASKS 18.9);
* DR-19: the database and role stay `mistudio` (`POSTGRES_DB`, `POSTGRES_USER`, `PGUSER`, `PGDATABASE` values),
  and DR-42: the dump prefix stays `mistudio-` — both are values, which this test does not read.

Every image this repository builds is `hitsai/miworkbench-*` (P1 pushes both names; DR-27 keeps the old ones
pushed, `test_frozen_identifiers.py::test_dr27_old_image_repos_still_pushed`).
"""
from __future__ import annotations

import re

import yaml

from tests.support import k8s_render

OLD = re.compile(r"mistudio", re.I)
#: Keys whose string values name an object or a reference to one.
NAME_KEYS = {"name", "namespace", "claimName", "volumeName", "secretName", "serviceName", "app",
             "app.kubernetes.io/part-of", "nginx.ingress.kubernetes.io/websocket-services"}
OWN_IMAGES = re.compile(r"^(docker\.io/)?hitsai/")


def _is_transition_alias(doc: dict) -> bool:
    meta = doc.get("metadata") or {}
    return (doc.get("kind") == "Service" and (doc.get("spec") or {}).get("type") == "ExternalName"
            and (meta.get("labels") or {}).get("rebrand.hitsai/transition") == "until-p7")


def _walk(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "env":  # env entries: `name` is a variable name, read below under its own rule
                for e in v or []:
                    yield (*path, "env", e.get("name")), "env-name", e.get("name")
                    yield from _walk({kk: vv for kk, vv in e.items() if kk not in ("name", "value")},
                                     (*path, "env", e.get("name")))
                continue
            if isinstance(v, str) and k in NAME_KEYS:
                yield (*path, k), k, v
            else:
                yield from _walk(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, (*path, i))


def findings(docs) -> list[str]:
    out = []
    for d in docs:
        if _is_transition_alias(d):
            continue
        where = f"{d.get('kind')} {d.get('metadata', {}).get('namespace', '')}/{d.get('metadata', {}).get('name')}"
        for path, key, value in _walk(d):
            if value and OLD.search(value):
                out.append(f"{where}: {'.'.join(map(str, path))} = {value!r}")
    return out


def test_no_mistudio_object_name_or_namespace():
    docs = k8s_render.render()
    assert len(docs) >= 15, len(docs)
    problems = findings(docs)
    assert not problems, "\n".join(problems)
    namespaces = {d["metadata"].get("namespace") for d in docs if d.get("kind") not in ("Namespace", "PersistentVolume")
                  and not _is_transition_alias(d)}
    assert namespaces == {"miworkbench"}, namespaces
    aliases = [d for d in docs if _is_transition_alias(d)]
    assert aliases == [], "the ExternalName left at ENH-009 P7 (FTASKS 18.9)"
    # Self-test: an object left under its old name is a finding.
    assert findings([{"kind": "Deployment", "metadata": {"name": "mistudio-frontend", "namespace": "miworkbench"}}])


def test_images_are_miworkbench():
    docs = k8s_render.render()
    images = {c["image"] for d in docs for c in k8s_render.containers(d)}
    own = {i for i in images if OWN_IMAGES.match(i)}
    assert own, images
    assert all(re.match(r"^(docker\.io/)?hitsai/miworkbench-(backend|frontend|rule-runner)[:@]", i) for i in own), own
    kust = yaml.safe_load((k8s_render.BASE / "kustomization.yaml").read_text())
    names = {i["name"] for i in kust.get("images") or []}
    assert names == {"hitsai/miworkbench-backend", "hitsai/miworkbench-frontend", "hitsai/miworkbench-rule-runner"}
