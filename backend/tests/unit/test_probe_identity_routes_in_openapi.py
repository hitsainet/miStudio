"""Feature 039 wiring, read from the SERVED app (memory: app-routes-is-not-a-route-list).

`GET /probes/{probe_id}/decision` is served, and the list route takes the two digest filters as
query parameters — read from `app.openapi()["paths"]`, so deleting the `@router.get` or a `Query`
turns this red, whatever the source says.
"""
from __future__ import annotations

from src.main import app

PREFIX = "/api/v1/probe-monitors"


def _paths():
    return app.openapi()["paths"]


def test_the_decision_route_is_served():
    route = _paths().get(f"{PREFIX}/probes/{{probe_id}}/decision")
    assert route is not None and "get" in route
    schema = route["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["$ref"].endswith("/ProbeDecisionResponse")


def test_the_list_takes_both_digest_filters_as_64_hex():
    params = {p["name"]: p for p in _paths()[f"{PREFIX}/probes"]["get"]["parameters"]}
    for name in ("detector_digest", "decision_digest"):
        assert name in params, name
        schema = params[name]["schema"]
        pattern = schema.get("pattern") or next(
            (alt.get("pattern") for alt in schema.get("anyOf", []) if alt.get("pattern")), None)
        assert pattern == "^[0-9a-f]{64}$"


def test_the_recalibrate_request_takes_expect_decision_digest():
    schemas = app.openapi()["components"]["schemas"]
    assert "expect_decision_digest" in schemas["RecalibrateRequest"]["properties"]
