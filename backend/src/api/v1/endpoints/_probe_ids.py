"""A blend report id is never a probe id (Feature 041 FR-18; TI18).

A router-level dependency on the probe-monitors and behaviours routers: any route whose path carries
`{probe_id}` refuses a `pbr_…` id with 404 `not_a_probe`, naming the blend-report route. On the ROUTER, not
on each route, so a `{probe_id}` route added later is covered without anyone remembering a list.
"""
from __future__ import annotations

from fastapi import HTTPException, Request

REPORT_PREFIX = "pbr_"
REPORT_ROUTE = "/api/v1/probe-monitors/blend-reports/{id}"


def refuse_blend_report_ids(request: Request) -> None:
    probe_id = request.path_params.get("probe_id")
    if isinstance(probe_id, str) and probe_id.startswith(REPORT_PREFIX):
        raise HTTPException(status_code=404, detail={
            "code": "not_a_probe",
            "message": f"{probe_id} is a blend report, not a probe: a blend has no rung, threshold or definition of "
                       "its own. Read it at the blend-report route.",
            "route": REPORT_ROUTE,
            "next": [{"tool": "get_blend_report", "arguments": {"report_id": probe_id},
                      "why": "Read the blend report this id names."}],
        })
