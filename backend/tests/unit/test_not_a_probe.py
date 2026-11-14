"""A blend report id is never taken for a probe id (Feature 041 FR-18; FTASKS 9.3).

Every route with a `{probe_id}` on the served application — read from `app.openapi()`, never a list — is
called with a `pbr_` id and must answer 404 `not_a_probe`, naming the blend-report route. The guard is a
router-level dependency on BOTH routers that carry `{probe_id}`, so a route added later is covered.
"""
from __future__ import annotations

import pytest

API_ROOT = "/api/v1"


def probe_id_routes():
    from src.main import app

    out = []
    for path, operations in app.openapi()["paths"].items():
        if "{probe_id}" not in path:
            continue
        for method, operation in operations.items():
            required = tuple(sorted(p["name"] for p in operation.get("parameters", [])
                                    if p.get("in") == "query" and p.get("required")))
            out.append((method.upper(), path, required))
    return sorted(out)


def test_the_route_list_is_not_empty():
    routes = probe_id_routes()
    assert len(routes) >= 12, routes
    assert ("PUT", "/api/v1/probe-monitors/behaviours/{key}/probes/{probe_id}/role", ()) in routes


@pytest.mark.parametrize("method, path, required", probe_id_routes())
async def test_every_probe_id_route_refuses_a_report_id(client, method, path, required):
    url = path.replace("{probe_id}", "pbr_0123456789ab")
    for name in ("key", "evaluation_id", "task_id"):
        url = url.replace("{" + name + "}", "x-y" if name == "key" else "pme_x")
    response = await client.request(method, url, params={name: "x" for name in required},
                                    json={} if method in ("POST", "PUT", "PATCH") else None)
    assert response.status_code == 404, (method, path, response.text)
    detail = response.json()["detail"]
    assert detail["code"] == "not_a_probe" and detail["route"] == "/api/v1/probe-monitors/blend-reports/{id}"
    assert detail["next"][0]["tool"] == "get_blend_report"


async def test_a_probe_id_still_passes(client):
    response = await client.get(f"{API_ROOT}/probe-monitors/probes/pm_does_not_exist")
    assert response.status_code == 404 and "not_a_probe" not in response.text
