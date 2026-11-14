"""The restart labels name the Deployments that really run the code (042 FR-042.25; FTDD §6.12, TPD-17; FTASKS
12.14 – 12.15, control K-12.39).

The Settings page tells the operator which pod to restart for a read-only setting, and warns beside every row whose
pod is the backend (restarting it stops every running job). Those labels were the literal `"mistudio-backend pod"`
and `"mistudio-mcp pod"`, and the warning was decided by a prefix match on the old name. P3 renamed the
Deployments, so the labels are now constants tied to the RENDERED manifests (not re-typed here): `BACKEND_POD` is
`"<name> pod"` for the Deployment whose container runs the API (`SERVICE_TYPE=api`), `MCP_POD` the one running
`python -m src.mcp_server`. The warning is decided by EQUALITY with `BACKEND_POD`, checked through the real route.
"""
from __future__ import annotations

from src.core import config
from tests.support import k8s_render

API = "/api/v1/environment"


def _deployment_running(predicate) -> str:
    names = []
    for doc in k8s_render.render():
        if doc.get("kind") != "Deployment":
            continue
        for container in k8s_render.containers(doc):
            if predicate(container):
                names.append(doc["metadata"]["name"])
    assert len(names) == 1, f"expected exactly one Deployment, found {names}"
    return names[0]


def _is_api(container: dict) -> bool:
    return any(e.get("name") == "SERVICE_TYPE" and e.get("value") == "api" for e in container.get("env") or [])


def _is_mcp(container: dict) -> bool:
    return [*(container.get("command") or []), *(container.get("args") or [])][-2:] == ["-m", "src.mcp_server"]


def test_backend_pod_label_equals_manifest_deployment():
    assert config.BACKEND_POD == f"{_deployment_running(_is_api)} pod"
    assert config.MCP_POD == f"{_deployment_running(_is_mcp)} pod"
    # Every label the page can show is one of the two, never a hand-typed copy.
    labels = set(config.RESTART_KEYS.values()) | {pod for _, pod, _ in config.EXTERNAL_RESTART_KEYS}
    assert labels == {config.BACKEND_POD, config.MCP_POD}
    assert config.BACKEND_POD.split()[0] in config.BACKEND_RESTART_WARNING


async def test_rows_naming_backend_pod_get_warning_through_route(client):
    rows = (await client.get(API)).json()["settings"]
    restart_rows = {k: r for k, r in rows.items() if r.get("restart")}
    backend = {k for k, r in restart_rows.items() if r["restart"] == config.BACKEND_POD}
    mcp = {k for k, r in restart_rows.items() if r["restart"] == config.MCP_POD}
    assert backend and mcp, "the route served no row for one of the pods: the check would read nothing"
    assert backend | mcp == set(restart_rows)
    for key in backend:
        assert restart_rows[key]["restart_warning"] == config.BACKEND_RESTART_WARNING, key
    for key in mcp:
        assert restart_rows[key]["restart_warning"] is None, key
