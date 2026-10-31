"""The production ingress bounds request bodies PER PATH (operator decision 2026-10-16).

`k8s/base/ingress.yaml` read `proxy-body-size: "0"` — no limit at the edge — on both
ingresses. The decision: the general API is capped at 2m (just above the backend's 1 MiB
middleware cap, `core/body_limit.py`, matching `docker/nginx.gcp.conf`), and only routes
that genuinely take large bodies get a large limit, through their OWN Ingress object
scoped to their paths and labelled `mistudio.io/body-size: large`.

THE LARGE-BODY ROUTE SET IS DERIVED, NOT LISTED: every operation in the app's SERVED
OpenAPI schema whose request body is not JSON (multipart, octet-stream — an UploadFile or
a raw stream), plus every path the body-limit middleware exempts. On 2026-10-16 that set
is EMPTY — every body is JSON — so no large-limit Ingress exists, and the moment an upload
route is added this test fails until it is given one.

The manifests are RENDERED: `kubectl kustomize k8s/base` when kubectl is installed, else
the kustomization's own `resources` list loaded in order (the base has no patches that
touch an Ingress). Either way the render failing to yield an Ingress is a failure.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "k8s" / "base"
ANNOTATION = "nginx.ingress.kubernetes.io/proxy-body-size"
LARGE_LABEL = ("mistudio.io/body-size", "large")
GENERAL_LIMIT = "2m"

_UNITS = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}


def _bytes(value) -> int:
    m = re.fullmatch(r"(\d+)([kKmMgG]?)", str(value).strip())
    assert m, f"unparseable proxy-body-size {value!r}"
    return int(m.group(1)) * _UNITS[m.group(2).lower()]


def render_base():
    """Every document `kustomize build k8s/base` produces."""
    kubectl = shutil.which("kubectl")
    if kubectl:
        out = subprocess.run([kubectl, "kustomize", str(BASE)], capture_output=True,
                             text=True, check=True).stdout
        return [d for d in yaml.safe_load_all(out) if d]
    kustomization = yaml.safe_load((BASE / "kustomization.yaml").read_text())
    docs = []
    for resource in kustomization["resources"]:
        docs.extend(d for d in yaml.safe_load_all((BASE / resource).read_text()) if d)
    return docs


@pytest.fixture(scope="module")
def ingresses():
    found = [d for d in render_base() if d.get("kind") == "Ingress"]
    assert found, "the kustomize render produced no Ingress — the guard would assert nothing"
    return found


def _paths(ingress):
    for rule in ingress["spec"].get("rules", []):
        for p in rule.get("http", {}).get("paths", []):
            yield p["path"], p.get("pathType", "Prefix")


def _covers(prefix: str, path_type: str, route: str) -> bool:
    if path_type == "Exact":
        return route == prefix
    return route == prefix or route.startswith(prefix.rstrip("/") + "/")


def large_body_routes(app) -> set:
    """Served paths that take a non-JSON body, or that the body limit exempts."""
    from src.core.body_limit import EXEMPT_PREFIXES

    routes = set()
    for path, ops in app.openapi()["paths"].items():
        for op in ops.values():
            if not isinstance(op, dict):
                continue
            content = (op.get("requestBody") or {}).get("content", {})
            if any(ct != "application/json" for ct in content):
                routes.add(path)
            if any(path.startswith(prefix) for prefix in EXEMPT_PREFIXES):
                routes.add(path)
    return routes


def uncovered(routes, ingresses) -> set:
    """Large-body routes no LARGE-labelled Ingress with a large limit covers."""
    large = [
        i for i in ingresses
        if (i["metadata"].get("labels") or {}).get(LARGE_LABEL[0]) == LARGE_LABEL[1]
        and _bytes(i["metadata"]["annotations"][ANNOTATION]) > _bytes(GENERAL_LIMIT)
    ]
    return {
        r for r in routes
        if not any(_covers(p, t, r) for i in large for p, t in _paths(i))
    }


def test_no_ingress_is_unbounded(ingresses):
    for ingress in ingresses:
        value = (ingress["metadata"].get("annotations") or {}).get(ANNOTATION)
        assert value is not None, f"{ingress['metadata']['name']} sets no body limit"
        assert str(value) != "0", f"{ingress['metadata']['name']}: proxy-body-size \"0\" is no limit"


def test_the_general_ingresses_are_at_2m(ingresses):
    general = [i for i in ingresses
               if (i["metadata"].get("labels") or {}).get(LARGE_LABEL[0]) != LARGE_LABEL[1]
               and i["metadata"]["name"].startswith("mistudio")
               and i["metadata"]["name"] != "mistudio-mcp-ingress"]
    names = {i["metadata"]["name"] for i in general}
    assert {"mistudio-ingress", "mistudio-websocket-ingress"} <= names, names
    for ingress in general:
        assert _bytes(ingress["metadata"]["annotations"][ANNOTATION]) == _bytes(GENERAL_LIMIT), (
            f"{ingress['metadata']['name']} is not bounded at {GENERAL_LIMIT}"
        )


def test_every_large_body_route_has_a_large_limit(ingresses):
    from src.main import app

    missing = uncovered(large_body_routes(app), ingresses)
    assert not missing, (
        f"{sorted(missing)} take large or non-JSON bodies, and no Ingress labelled "
        f"{LARGE_LABEL[0]}: {LARGE_LABEL[1]} with a limit above {GENERAL_LIMIT} covers them — "
        "they would be refused at the edge with nginx's bare 413"
    )


def test_the_derivation_finds_an_upload_route_and_reports_it_uncovered(ingresses):
    """Negative control in test form: the set is empty today, so prove the derivation
    and the coverage check both bite on a real UploadFile route rather than passing
    vacuously."""
    from fastapi import FastAPI, File, UploadFile

    toy = FastAPI()

    @toy.post("/api/v1/datasets/upload")
    async def upload(file: UploadFile = File(...)):  # pragma: no cover - schema only
        return {}

    routes = large_body_routes(toy)
    assert routes == {"/api/v1/datasets/upload"}
    assert uncovered(routes, ingresses) == routes
    covering = {
        "kind": "Ingress",
        "metadata": {"name": "mistudio-uploads", "labels": {LARGE_LABEL[0]: LARGE_LABEL[1]},
                     "annotations": {ANNOTATION: "2g"}},
        "spec": {"rules": [{"http": {"paths": [
            {"path": "/api/v1/datasets/upload", "pathType": "Prefix"}]}}]},
    }
    assert uncovered(routes, ingresses + [covering]) == set()
