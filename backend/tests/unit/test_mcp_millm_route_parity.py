"""Route → tool parity for miLLM: every route miLLM's LIVE app serves is called by a REGISTERED
`millm_*` tool, or carries a written reason it is not (MCP gap closure, 2026-10-18).

CROSS-REPO. The served half is miLLM's own `create_app().openapi()["paths"]`, built by miLLM's
interpreter in miLLM's checkout (`MILLM_REPO`, else the `miLLM` sibling of this repo; interpreter
`MILLM_PYTHON`, else `<miLLM>/venv/bin/python`). Never a scrape of miLLM's router files: a scrape
reads what the source says, the app serves what it registered. When the sibling or its
interpreter is absent this SKIPS LOUDLY, and FAILS under `MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1` —
an unverified parity is not a verified one.

The covered half is the same as miStudio's parity guard: the routes REGISTERED tools call
(`tests/support/mcp_route_map`), membership from the built server's `list_tools()`.

Controls (recorded in the 2026-10-19 hand-back; restored from saved bytes):
  C3 drop `millm_probes` from MILLM_CATEGORY_MODULES → test 1 red (the probe routes)
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.support.mcp_route_map import called_routes, normalise, registered_tools
from tests.support.sibling_repos import millm_repo

# ── reasons ──────────────────────────────────────────────────────────────────────────────────

M_WEIGHTS = ("miLLM Feature 31 D26: /api/weights/* is miForge's weight-attachment and step-barrier "
             "protocol, not an MCP surface; millm_status reports the `weights` block")
M_CHECKPOINTS = ("miLLM Feature 32 non-goal (§9): no MCP tools for checkpoint save/load/lineage — "
                 "they replace the served weights with a hash-verified checkpoint")
M_ADAPTERS = "miLLM Feature 33 D16: adapter serving has neither an admin UI control nor MCP tools"
M_HF_TOKEN = ("credentials never via MCP: the HuggingFace token is saved and tested on miLLM's "
              "Settings page and downloads use it server-side (millm_download_model says so)")
M_SAES = ("miLLM SAE download/attach/detach, the attachment set and per-feature steering change "
          "what EVERY client is served; operator-managed in miLLM's admin UI. Agents steer per "
          "request (millm_generate `steering`) or through profiles/clusters/circuits. Tracked "
          "debt: a millm_saes category")
M_MONITORING = ("miLLM feature-activation monitoring (configure/enable/history/statistics) is an "
                "admin-UI dashboard; the sensing tools are the agent-facing detection surface")
M_HEALTH = ("liveness/readiness, Prometheus metrics, version, GPU and breaker telemetry for "
            "orchestrators and dashboards; millm_status (/api/health/detailed) is the agent's "
            "one-call status and the gate probes liveness itself")
M_BREAKER_RESET = "inference circuit-breaker reset is operator break-glass after a fault"
M_LOCK = ("model lock/unlock pins a model for an admin-UI steering session; leases "
          "(millm_acquire_lease) are the agent's residency mechanism (Feature 29)")
M_PROFILES_CRUD = ("manual steering-profile authoring (create/save-current/edit/delete/import/"
                   "export, the active-profile read) is admin-UI work; agents list, activate and "
                   "deactivate profiles and import clusters, and millm_status names the active one")
M_CLUSTER_ADMIN = ("cluster deletion is irreversible and stays in the admin UI; the per-cluster "
                   "dial is the admin UI's, agents use the active dial (millm_set_intensity)")
M_OPENAI_MODELS = ("the OpenAI-compatible model listing; millm_list_models reads the management "
                   "view with residency, quantization, served names and the actual context")
M_FILES_ADMIN = ("batch-file and batch listing housekeeping (list, read metadata, delete); the "
                 "batch tools track the ids they create and page results with millm_batch_results")


M_SESSIONS = ("ENH-005 / miLLM 036 recorded probe sessions hold users' prompt and reply text; they "
              "are read by miStudio's backend for session spot-checks (038, services/millm_read.py) "
              "and deleted by the operator in miLLM's admin UI, not exposed to agents. Agents start "
              "and read spot-checks with the probe spot-check tools. Open for the operator: a "
              "metadata-only millm_probe_sessions read tool")
# ENH-006 D54 (2026-10-24): the M_PROBE_CAPACITY exemptions are GONE — `millm_probe_capacity` reads,
# `millm_request_probe_limits` and `millm_measure_probe_capacity` proxy the two writes behind miStudio's operator
# approval (`mcp_server/tools/millm_behaviours.py`).


M_APPROVAL_DECISIONS = ("ENH-008 (D1, OSD-56 applied in miLLM): approving or rejecting a request is the "
                        "OPERATOR's, in miLLM's Approvals page; miLLM refuses either route to any caller "
                        "sending X-miLLM-Agent (403 approval_operator_only), and every request miStudio's "
                        "client sends carries that header. Agents read the queue with millm_list_approvals "
                        "and millm_get_approval_status")


M_APPROVAL_ACTIONS = ("ENH-008: not a tool of its own — every relayed tool reads it FIRST through "
                      "tools/millm_approvals.ApprovalQueueCheck (fail closed: a miLLM without the queue is "
                      "refused and sent nothing). The AST walk follows helpers in the tool's own module only; "
                      "the call, its order and its count are asserted in test_millm_approvals_relay.py")


def _routes(reason: str, *pairs: str) -> dict[tuple[str, str], str]:
    out = {}
    for pair in pairs:
        method, path = pair.split(" ", 1)
        out[(method, normalise(path))] = reason
    return out


M_ENV_WRITES = ("ENH-003 D5 / ENH-006 D75-D76 (miLLM contract v1.23): the per-key writes are the "
                "operator's, on miLLM's Settings page. An agent's write of a telemetry key is gated "
                "in miLLM (telemetry_write) and reached through millm_set_telemetry, which calls "
                "PUT /api/environment with TELEMETRY_* keys only; agents read every setting with "
                "millm_get_environment / millm_get_environment_history")

MILLM_ROUTE_EXEMPT: dict[tuple[str, str], str] = {
    **_routes(M_ENV_WRITES, "PUT /api/environment/{key}", "DELETE /api/environment/{key}"),
    **_routes(M_APPROVAL_DECISIONS, "POST /api/approvals/{id}/approve", "POST /api/approvals/{id}/reject",
             "POST /api/approvals/approve-batch"),  # ENH-006 D79/D81: operator-only, like a single approve
    **_routes(M_APPROVAL_ACTIONS, "GET /api/approvals/actions"),
    **_routes(M_WEIGHTS, "POST /api/weights/attach", "POST /api/weights/detach",
              "POST /api/weights/heartbeat", "POST /api/weights/quiesce",
              "POST /api/weights/resume", "POST /api/weights/commit-begin",
              "POST /api/weights/commit-end", "POST /api/weights/clear-stale",
              "GET /api/weights/status"),
    **_routes(M_CHECKPOINTS, "POST /api/weights/save", "POST /api/weights/load",
              "GET /api/weights/lineage"),
    **_routes(M_ADAPTERS, "POST /api/weights/adapter/load", "POST /api/weights/adapter/unload",
              "GET /api/weights/adapter"),
    **_routes(M_HF_TOKEN, "GET /api/settings/hf-token", "PUT /api/settings/hf-token",
              "DELETE /api/settings/hf-token", "POST /api/settings/hf-token/test"),
    **_routes(M_SAES, "GET /api/saes", "POST /api/saes/download", "POST /api/saes/{id}/cancel",
              "POST /api/saes/preview", "GET /api/saes/attachment", "GET /api/saes/attachments",
              "POST /api/saes/attach-set", "GET /api/saes/steering", "POST /api/saes/steering",
              "POST /api/saes/steering/batch", "POST /api/saes/steering/enable",
              "POST /api/saes/steering/disable", "DELETE /api/saes/steering/{idx}",
              "DELETE /api/saes/steering", "POST /api/saes/monitoring", "GET /api/saes/{id}",
              "DELETE /api/saes/{id}", "GET /api/saes/{id}/compatibility",
              "POST /api/saes/{id}/attach", "POST /api/saes/{id}/detach"),
    **_routes(M_MONITORING, "GET /api/monitoring", "POST /api/monitoring/configure",
              "POST /api/monitoring/enable", "GET /api/monitoring/history",
              "DELETE /api/monitoring/history", "GET /api/monitoring/statistics",
              "DELETE /api/monitoring/statistics", "POST /api/monitoring/statistics/top"),
    # `GET /api/health/gpus` is no longer exempt: gpu_preflight reads it (MCP gap G9, 2026-10-21).
    **_routes(M_HEALTH, "GET /api/health", "GET /api/health/ready",
              "GET /api/health/circuits", "GET /api/health/metrics",
              "GET /api/health/metrics/prometheus", "GET /api/health/version"),
    **_routes(M_BREAKER_RESET, "POST /api/health/circuits/{name}/reset"),
    **_routes(M_LOCK, "POST /api/models/{id}/lock", "POST /api/models/{id}/unlock"),
    **_routes(M_PROFILES_CRUD, "POST /api/profiles", "POST /api/profiles/save-current",
              "GET /api/profiles/active", "GET /api/profiles/{id}", "PATCH /api/profiles/{id}",
              "DELETE /api/profiles/{id}", "GET /api/profiles/{id}/export",
              "POST /api/profiles/import"),
    **_routes(M_CLUSTER_ADMIN, "DELETE /api/clusters/{id}", "PUT /api/clusters/{id}/intensity"),
    **_routes(M_OPENAI_MODELS, "GET /v1/models", "GET /v1/models/{id}"),
    **_routes(M_FILES_ADMIN, "GET /v1/files", "GET /v1/files/{id}", "DELETE /v1/files/{id}",
              "GET /v1/batches"),
    **_routes(M_SESSIONS, "GET /api/probes/sessions", "GET /api/probes/sessions/{id}",
              "DELETE /api/probes/sessions/{id}", "POST /api/probes/sessions/sample",
              "GET /api/probes/sessions/{id}/transcript"),
}

#: Feature 039 (X3, TI9): routes a registered tool calls that FastAPI serves OUTSIDE `paths` — so
#: they can never appear in `create_app().openapi()["paths"]`. Consulted ONLY by
#: `test_every_millm_route_a_tool_calls_is_served`; `test_the_outside_paths_allowance_is_exact`
#: holds it to exactly this path, actually called by a registered tool.
MILLM_CALLED_OUTSIDE_PATHS: dict[tuple[str, str], str] = {
    ("GET", "/openapi.json"): (
        "FastAPI serves its own OpenAPI document outside `paths`; millm_recalibrate_probe reads it "
        "to detect which identity fields miLLM's ProbeRecalibrationRequest takes (039 D3 (a), "
        "the millm_read.OPENAPI pattern), cached per process"
    ),
}

#: Read by miLLM's interpreter in miLLM's checkout. `-B`: never write bytecode into a sibling.
_DUMP = (
    "import json\n"
    "from millm.main import create_app\n"
    "paths = create_app().openapi()['paths']\n"
    "print('@@ROUTES@@' + json.dumps(sorted([m.upper(), p] for p, ops in paths.items() "
    "for m in ops if m in ('get', 'post', 'put', 'patch', 'delete'))))\n"
)


def _absent(reason: str) -> None:
    if os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
        pytest.fail("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1 but the miLLM parity guard cannot run: "
                    + reason)
    pytest.skip("CROSS-REPO, SKIPPED LOUDLY: " + reason)


@pytest.fixture(scope="module")
def millm_served() -> set[tuple[str, str]]:
    repo = millm_repo()
    python = Path(os.environ.get("MILLM_PYTHON") or repo / "venv" / "bin" / "python")
    if not (repo / "millm" / "main.py").is_file():
        _absent(f"no miLLM checkout at {repo} (set MILLM_REPO)")
    if not python.is_file():
        _absent(f"no miLLM interpreter at {python} (set MILLM_PYTHON)")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_URL", "PYTHONPATH"))}
    proc = subprocess.run(
        [str(python), "-B", "-c", _DUMP], cwd=repo, env=env, capture_output=True, text=True,
        timeout=240,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@ROUTES@@")]
    if proc.returncode != 0 or not lines:
        # The checkout and interpreter exist, so a failure to build the app is a real failure,
        # not an absence: report it rather than skip.
        pytest.fail(f"miLLM's app did not build (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    return {(m, normalise(p)) for m, p in json.loads(lines[0][len("@@ROUTES@@"):])}


@pytest.fixture(scope="module")
def covered() -> dict[tuple[str, str], set[str]]:
    mp = pytest.MonkeyPatch()
    try:
        registered = registered_tools(mp)
    finally:
        mp.undo()
    return called_routes(registered, "millm")


def uncovered(served, covered) -> set[tuple[str, str]]:
    return {key for key in served if key not in covered and key not in MILLM_ROUTE_EXEMPT}


def test_the_probe_can_see(millm_served, covered):
    assert len(millm_served) > 100, len(millm_served)
    assert ("POST", "/v1/completions") in millm_served
    assert "millm_score_completion" in covered[("POST", "/v1/completions")]
    assert "millm_embeddings" in covered[("POST", "/v1/embeddings")]


def test_every_served_millm_route_has_a_tool_or_a_reasoned_exemption(millm_served, covered):
    missing = sorted(uncovered(millm_served, covered))
    assert not missing, (
        f"{len(missing)} routes miLLM SERVES have no registered millm_* tool and no exemption. "
        "Add a tool, or a reason in MILLM_ROUTE_EXEMPT:\n  "
        + "\n  ".join(f"{m} {p}" for m, p in missing)
    )


def test_every_millm_route_a_tool_calls_is_served(millm_served, covered):
    dangling = sorted((k, sorted(v)) for k, v in covered.items()
                      if k not in millm_served and k not in MILLM_CALLED_OUTSIDE_PATHS)
    assert not dangling, f"millm_* tools call routes miLLM does not serve: {dangling}"


def test_no_exemption_is_stale_or_contradicted(millm_served, covered):
    stale = sorted(k for k in MILLM_ROUTE_EXEMPT if k not in millm_served)
    assert not stale, f"exemptions for routes miLLM does not serve: {stale}"
    both = sorted(set(MILLM_ROUTE_EXEMPT) & set(covered))
    assert not both, f"exempted routes a tool now calls — remove the exemption: {both}"


def test_every_exemption_has_a_reason():
    thin = [key for key, reason in MILLM_ROUTE_EXEMPT.items() if len(reason.strip()) < 40]
    assert not thin, thin


def test_a_route_with_no_tool_IS_reported():
    assert uncovered({("GET", "/api/__parity_dummy")}, {}) == {("GET", "/api/__parity_dummy")}


def test_the_outside_paths_allowance_is_exact(covered):
    """X3/TI9: the allowance names exactly `GET /openapi.json`, with a reason, and a REGISTERED tool
    actually calls it — a stale allowance would excuse a route nothing reads."""
    assert set(MILLM_CALLED_OUTSIDE_PATHS) == {("GET", "/openapi.json")}
    assert all(len(reason.strip()) >= 40 for reason in MILLM_CALLED_OUTSIDE_PATHS.values())
    for key in MILLM_CALLED_OUTSIDE_PATHS:
        assert key in covered, f"no registered tool calls {key}: the allowance is stale"
    assert "millm_recalibrate_probe" in covered[("GET", "/openapi.json")]
