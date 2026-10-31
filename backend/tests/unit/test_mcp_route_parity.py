"""Route → tool parity for miStudio: every route the LIVE app serves is called by a REGISTERED MCP
tool, or carries a written reason it is not (MCP gap closure, operator decision 2026-10-18).

Modelled on miDataworks' `backend/tests/unit/test_mcp_parity.py` (FR-010.44). Both halves come
from an authority:

* **Served routes** — `app.openapi()["paths"]` of the real FastAPI app. Never `app.routes`:
  included routers are wrapped with no `.path` (memory: app-routes-is-not-a-route-list).
* **Covered routes** — the routes REGISTERED tools call. Membership is the built server's
  `list_tools()` with every category on (`tests/support/mcp_route_map.registered_tools`); the AST
  walk only says which route each registered tool issues. Delete a category's registration line
  and its routes go uncovered here — that is the negative control this file was written for.

Three failures are kept apart:

1. a served route with no tool and no exemption — add the tool, or a reason below;
2. an exemption for a route nobody serves — stale, remove it;
3. an exemption for a route a tool now calls — a contradiction, remove the exemption.

Controls (recorded 2026-10-19 in the hand-back; each red, each restored from saved bytes):
  C1 delete `"probe_monitors": [probe_monitors],` from CATEGORY_MODULES → test 1 red (22 routes)
  C2 add a dummy `GET /probe-monitors/__parity_dummy` route with no tool → test 1 red
"""
from __future__ import annotations

import pytest

from tests.support.mcp_route_map import (
    called_routes,
    normalise,
    registered_tools,
    tool_routes,
    unreadable_calls,
)

API = "/api/v1"

# ── reasons ──────────────────────────────────────────────────────────────────────────────────

R_INFRA = ("framework or infrastructure route (root page, liveness, backend<->worker internal "
           "callbacks), not an API capability")
R_SYSTEM = ("host telemetry for the dashboard (CPU/disk/network/GPU metrics, worker internals); "
            "list_gpus exposes the one fact tools need (the GPU ids), and an agent has no action "
            "for the rest")
R_RESTART = ("restarts the backend process: operator-only, because an agent-triggered restart "
             "kills every in-flight GPU job (memory: deploys-kill-inflight-gpu-runs)")
R_SETTINGS = ("application settings hold credentials (HuggingFace token, API keys) and the UI PIN; "
              "credentials never travel via MCP — tools resolve them server-side")
R_APPROVE = ("approving or denying is the OPERATOR's act; no MCP code path can approve (Feature 034 "
             "FR-20) — tools only file and consume their own requests")
R_APPROVAL_LIST = ("the approval queue listing is the operator banner's surface; an agent reads "
                   "its own request by re-calling the tool with its approval_request_id")
R_TEMPLATES = ("UI preset management (extraction/prompt/training/labeling-prompt templates): "
               "agents pass explicit configs to the tools that run jobs, and a preset store "
               "would be a second, mutable source of truth")
R_DATASET_ACQ = ("dataset acquisition and tokenization (multi-GB downloads, CPU-hours of "
                 "tokenizing, deletes of shared corpora) are operator-run; agents read datasets "
                 "with list_datasets/get_dataset, and miDataworks' detector-set send downloads "
                 "into miStudio through its own approval")
R_SAE_ADMIN = ("SAE acquisition, HuggingFace upload, import, deletion and feature extraction are "
               "operator-run (Hub pushes, TB-scale extraction, irreversible deletes); agents read "
               "SAEs with list_saes/get_sae and features with search_features/get_feature_by_index")
R_TRAINING = ("SAE training (multi-hour GPU runs, checkpoint retention and pruning, finalize) is "
              "operator-run; agents read runs with list_trainings/get_training. Tracked debt: "
              "read-only metrics/evaluation/checkpoint readers")
R_EXTRACTION_JOBS = ("activation extraction (multi-hour GPU, TB-scale disk; memory: "
                     "sae-16k-unique-run-constraints) is operator-run from the Models panel")
R_MODEL_ADMIN = ("model row edits, re-download, the HF cache listing and the download task poll "
                 "are operator maintenance; the six `models` tools cover acquisition, inspection, "
                 "progress and deletion")
R_BULK_LABEL = ("bulk labeling jobs (paid LLM calls across every feature of an extraction) are "
                "operator-run; agents label per feature (run_enhanced_labeling, "
                "update_feature_label) and through the trial/resume tools")
R_NLP = ("NLP-analysis batch jobs and their reset are operator maintenance; "
         "get_feature_nlp_analysis reads the stored result")
R_UI_CURATION = ("UI curation state (favorite/star toggles, profile edits/deletes, bundle "
                 "import/export) with no agent workflow; agents save and export profiles with "
                 "save_cluster_profile/export_cluster_definition")
R_NEURONPEDIA = ("Neuronpedia export/push publishes to an external service (decision 2026-07-24: "
                 "not an agent surface); operator-run")
R_TASK_QUEUE_ADMIN = ("failure-queue triage (dismiss/retry/delete) is the operator's; "
                      "get_task_status reads task state")
R_EVIDENCE_DELETE = ("irreversible deletion of GPU-produced evidence (captures, discovery runs) "
                     "stays in the UI, like the `admin` category's deletes")
R_SYNC_STEERING = ("synchronous steering variants; agents use /steering/async/*, which carries the "
                   "concurrency budget and approval mode and cannot hit the 60 s ingress timeout")
R_STEERING_RECOVERY = ("steering-worker recovery (reset/cleanup) is operator break-glass; "
                       "exit_steering_mode stops an idle worker; generation workers exit after each task")
R_BULK_EXP_DELETE = ("bulk experiment deletion; single deletes are delete_experiment in the "
                     "`admin` category (off by default)")
R_CAPTURE_READ = ("list_circuit_captures returns every capture run's state; the per-run read adds "
                  "nothing an agent acts on")
R_DISCOVERY_LIST = ("get_discovery_results reads a run by id and run ids come back from "
                    "run_circuit_discovery. Tracked debt: a listing tool")
R_REFINE = ("Appendix A.4 member-pair refinement is a UI drill-down on one cluster edge. Tracked "
            "debt: a tool when an agent workflow needs member-level edges")
R_PROBE_EXPORT = ("the same document export_probe_definition returns, served as a file download "
                  "(Content-Disposition) for the browser")
R_ANALYSIS_CLEANUP = "feature-analysis cache housekeeping, operator maintenance"
R_VERSION = "build metadata for the UI footer and support requests"
R_ENV_WRITE = ("runtime environment writes are the operator's in v1 (Feature 036, ENH-003 D5): "
               "miStudio's approvals are caller-executed, so a write tool would need a backend "
               "executor; agents read with get_environment_settings")
R_PURGE = ("operator-only permanent delete of an ARCHIVED probe dataset view (decision 2026-10-21, "
           "the miForge/miDataworks pattern): agents archive with archive_probe_dataset, and every "
           "purge route also refuses the MCP server's X-MiStudio-Agent header with 403")
R_BLEND_PRUNE = ("Feature 041 FR-19: pruning a blend report's per-row values is the operator's alone (the route "
                 "refuses the MCP server's X-MiStudio-Agent header with 403); agents read the report")
R_BLEND_BYTES = ("Feature 041: the stored report.json bytes behind get_blend_report (which returns the parsed "
                 "report and its sha256) — for byte comparison and download")
R_FAMILY_ZIP = ("Feature 041 T13: a binary ZIP archive; agents use export_probe_family, whose index names each "
                "member's single-export route and sha256")
R_BEHAVIOUR_DELETE = ("deleting a catalogued behaviour is the operator's (Feature 037 TI6); it is refused "
                      "while any record uses the key, and agents archive with archive_behaviour")
R_CONFIRM_POSITIVE = ("D48: the OPERATOR confirms that a view's positive labels are a behaviour — the check "
                      "that lets a probe be made primary — so the route refuses the MCP server's "
                      "X-MiStudio-Agent header with 403 (Feature 037 TI5)")
R_SPOT_TEXT = ("carries recorded user text from miLLM sessions (judged excerpts and judge replies, D35): "
               "the item list and the JSONL export are the operator's, from the Behaviours tab; agents read "
               "the counts and the report through get_probe_spot_check")
R_SPOT_ADMIN = ("keeping (pinning) a spot-check's excerpts past seven days, or deleting a check and its text, "
                "is the operator's decision about recorded user text (D39)")
R_CODE_RULE_DECIDE = ("Feature 040 S4 (FTDD Appendix C.6): approving and revoking operator code is the OPERATOR's — "
                      "revoke and re-approve refuse the MCP server's X-MiStudio-Agent header with 403 and have no "
                      "tool by design; agents file a request with request_code_rule_approval")
R_CODE_RULE_READ = ("Feature 040 S4: the rule-set screen's read of one code rule (document, state, Revoke); an agent "
                    "learns a hash's state from validate_rule_set / evaluate_rule_signals (code_rules) and the "
                    "approval it filed (get_approval_status)")
R_ENV_HISTORY = ("the change history is the operator page's audit view; get_environment_settings "
                 "carries each setting's last change")


def _routes(reason: str, *pairs: str) -> dict[tuple[str, str], str]:
    out = {}
    for pair in pairs:
        method, path = pair.split(" ", 1)
        full = path if path.startswith("/api/") or path in ("/", "/ws") else API + path
        out[(method, normalise(full))] = reason
    return out


#: Every served route NO registered tool calls, with the reason. Adding a route to the app adds a
#: red here until it gets a tool or a sentence; that is the whole design.
MISTUDIO_ROUTE_EXEMPT: dict[tuple[str, str], str] = {
    **_routes(R_INFRA, "GET /", "GET /api/health", "POST /api/internal/steering/reconcile-worker",
              "POST /api/internal/ws/emit"),
    **_routes(R_SYSTEM, "GET /system/all", "GET /system/disk-rates", "GET /system/disk-usage",
              "GET /system/gpu-info", "GET /system/gpu-metrics", "GET /system/gpu-metrics/all",
              "GET /system/gpu-processes", "GET /system/health", "GET /system/metrics",
              "GET /system/network-rates", "GET /system/resource-estimate",
              "GET /workers/active", "GET /workers/health", "GET /workers/queues",
              "GET /workers/stats"),
    **_routes(R_RESTART, "POST /system/restart"),
    **_routes(R_SETTINGS, "GET /settings", "PUT /settings", "PUT /settings/bulk",
              "GET /settings/{key}", "DELETE /settings/{key}", "POST /settings/pin/set",
              "GET /settings/pin/status", "POST /settings/pin/verify",
              "POST /settings/pin/remove"),
    **_routes(R_APPROVE, "POST /mcp/approvals/{id}/approve", "POST /mcp/approvals/{id}/deny",
              "POST /mcp/approvals/approve-batch"),
    **_routes(R_APPROVAL_LIST, "GET /mcp/approvals"),
    **_routes(R_TEMPLATES,
              *(f"{m} /{kind}{p}" for kind in ("extraction-templates", "prompt-templates",
                                               "training-templates")
                for m, p in (("POST", ""), ("GET", ""), ("POST", "/export"),
                             ("GET", "/favorites"), ("POST", "/import"), ("GET", "/{id}"),
                             ("PATCH", "/{id}"), ("DELETE", "/{id}"),
                             ("POST", "/{id}/favorite"))),
              "POST /prompt-templates/{id}/duplicate",
              "POST /labeling-prompt-templates", "GET /labeling-prompt-templates/default",
              "POST /labeling-prompt-templates/export", "POST /labeling-prompt-templates/import",
              "GET /labeling-prompt-templates/{id}", "PATCH /labeling-prompt-templates/{id}",
              "DELETE /labeling-prompt-templates/{id}",
              "POST /labeling-prompt-templates/{id}/clone",
              "POST /labeling-prompt-templates/{id}/set-default",
              "GET /labeling-prompt-templates/{id}/usage-count"),
    **_routes(R_DATASET_ACQ, "POST /datasets", "POST /datasets/download",
              "POST /datasets/tokenize-preview", "PATCH /datasets/{id}", "DELETE /datasets/{id}",
              "DELETE /datasets/{id}/cancel", "GET /datasets/{id}/samples",
              "GET /datasets/{id}/task-status", "DELETE /datasets/{id}/tokenization",
              "GET /datasets/{id}/tokenizations", "GET /datasets/{id}/tokenizations/{m}",
              "DELETE /datasets/{id}/tokenizations/{t}",
              "POST /datasets/{id}/tokenizations/{t}/cancel", "POST /datasets/{id}/tokenize"),
    **_routes(R_SAE_ADMIN, "POST /saes/batch-extract-features", "POST /saes/delete",
              "POST /saes/download", "POST /saes/hf/preview", "POST /saes/import/file",
              "POST /saes/import/training", "GET /saes/training/{id}/available",
              "POST /saes/upload", "DELETE /saes/{id}", "POST /saes/{id}/cancel-extraction",
              "POST /saes/{id}/extract-features", "GET /saes/{id}/extraction-status",
              "GET /saes/{id}/features"),
    **_routes(R_TRAINING, "POST /trainings", "POST /trainings/checkpoints/prune-all",
              "GET /trainings/checkpoints/prune-preview-all", "DELETE /trainings/{id}",
              "GET /trainings/{id}/checkpoints", "GET /trainings/{id}/checkpoints/best",
              "POST /trainings/{id}/checkpoints/prune",
              "GET /trainings/{id}/checkpoints/prune-preview",
              "DELETE /trainings/{id}/checkpoints/{c}", "POST /trainings/{id}/control",
              "POST /trainings/{id}/evaluate", "GET /trainings/{id}/evaluation",
              "GET /trainings/{id}/features", "POST /trainings/{id}/finalize",
              "GET /trainings/{id}/metrics"),
    **_routes(R_EXTRACTION_JOBS, "POST /models/{id}/estimate-extraction",
              "POST /models/{id}/extract-activations", "GET /models/{id}/extractions",
              "DELETE /models/{id}/extractions", "GET /models/{id}/extractions/active",
              "POST /models/{id}/extractions/{e}/cancel",
              "POST /models/{id}/extractions/{e}/retry"),
    **_routes(R_MODEL_ADMIN, "GET /models/local-cache/list", "GET /models/tasks/{id}",
              "PATCH /models/{id}", "POST /models/{id}/redownload"),
    **_routes(R_BULK_LABEL, "POST /labeling", "GET /labeling", "GET /labeling/models/available",
              "POST /labeling/models/openai", "GET /labeling/{id}", "DELETE /labeling/{id}",
              "GET /labeling/{id}/available-judges", "POST /labeling/{id}/cancel",
              "POST /extractions/{id}/label", "POST /extractions/{id}/label/enhanced/batch",
              "POST /enhanced-labeling/{id}/cancel"),
    **_routes(R_NLP, "POST /extractions/{id}/analyze-nlp", "POST /extractions/{id}/cancel-nlp",
              "POST /extractions/{id}/reset-nlp", "POST /features/{id}/analyze-nlp"),
    **_routes(R_UI_CURATION, "POST /features/{id}/favorite", "POST /features/{id}/star",
              "POST /cluster-profiles/export-bundle", "POST /cluster-profiles/import",
              "PATCH /cluster-profiles/{id}", "DELETE /cluster-profiles/{id}"),
    **_routes(R_NEURONPEDIA, "POST /neuronpedia/compute-dashboard-data", "POST /neuronpedia/export",
              "GET /neuronpedia/export/{id}", "DELETE /neuronpedia/export/{id}",
              "POST /neuronpedia/export/{id}/cancel", "GET /neuronpedia/export/{id}/download",
              "GET /neuronpedia/exports", "GET /neuronpedia/local-status",
              "POST /neuronpedia/push-local", "GET /neuronpedia/push-local",
              "GET /neuronpedia/push-local/{id}"),
    **_routes(R_TASK_QUEUE_ADMIN, "GET /task-queue/failed", "POST /task-queue/failed/dismiss-all",
              "POST /task-queue/failed/{t}/{s}/dismiss",
              "DELETE /task-queue/failed/{t}/{s}/dismiss", "DELETE /task-queue/{id}",
              "POST /task-queue/{id}/retry"),
    **_routes(R_EVIDENCE_DELETE, "DELETE /circuit-capture/{id}", "DELETE /circuit-discovery/{id}"),
    **_routes(R_SYNC_STEERING, "POST /steering/compare", "POST /steering/sweep"),
    **_routes(R_STEERING_RECOVERY, "POST /steering/reset", "POST /steering/cleanup"),
    **_routes(R_BULK_EXP_DELETE, "POST /steering/experiments/delete"),
    **_routes(R_CAPTURE_READ, "GET /circuit-capture/{id}"),
    **_routes(R_DISCOVERY_LIST, "GET /circuit-discovery"),
    **_routes(R_REFINE, "POST /circuit-discovery/refine-cluster-edge"),
    **_routes(R_PROBE_EXPORT, "GET /probe-monitors/probes/{id}/export"),
    **_routes(R_PURGE, "DELETE /probe-monitors/datasets/{id}", "POST /probe-monitors/datasets/purge",
              "POST /probe-monitors/datasets/purge/plan", "GET /probe-monitors/dataset-purges"),
    **_routes(R_ANALYSIS_CLEANUP, "POST /analysis/cleanup"),
    **_routes(R_VERSION, "GET /version"),
    **_routes(R_ENV_WRITE, "PUT /environment/{key}", "DELETE /environment/{key}"),
    **_routes(R_ENV_HISTORY, "GET /environment/history"),
    **_routes(R_BEHAVIOUR_DELETE, "DELETE /probe-monitors/behaviours/{k}"),
    **_routes(R_CONFIRM_POSITIVE, "POST /probe-monitors/datasets/{id}/confirm-positive"),
    **_routes(R_SPOT_TEXT, "GET /probe-monitors/spot-checks/{id}/items",
              "GET /probe-monitors/spot-checks/{id}/disagreements.jsonl"),
    **_routes(R_SPOT_ADMIN, "POST /probe-monitors/spot-checks/{id}/pin", "DELETE /probe-monitors/spot-checks/{id}"),
    **_routes(R_BLEND_PRUNE, "POST /probe-monitors/blend-reports/{id}/prune-rows"),
    **_routes(R_BLEND_BYTES, "GET /probe-monitors/blend-reports/{id}/report.json"),
    **_routes(R_FAMILY_ZIP, "GET /probe-monitors/behaviours/{k}/family/export"),
    **_routes(R_CODE_RULE_DECIDE, "POST /code-rules/{sha}/revoke", "POST /code-rules/{sha}/approve"),
    **_routes(R_CODE_RULE_READ, "GET /code-rules/{sha}"),
}

HTTP = {"get", "post", "put", "patch", "delete"}


def _served(app=None) -> set[tuple[str, str]]:
    if app is None:
        from src.main import app
    return {
        (method.upper(), normalise(path))
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method in HTTP
    }


@pytest.fixture(scope="module")
def served() -> set[tuple[str, str]]:
    return _served()


@pytest.fixture(scope="module")
def covered() -> dict[tuple[str, str], set[str]]:
    mp = pytest.MonkeyPatch()
    try:
        # Every conditionally registered tool on (get_approval_status exists only with steering
        # approval): a route is covered if ANY deployable build reaches it.
        mp.setenv("MCP_STEERING_APPROVAL", "true")
        registered = registered_tools(mp)
    finally:
        mp.undo()
    return called_routes(registered, "mistudio", API)


def uncovered(served, covered) -> set[tuple[str, str]]:
    return {key for key in served if key not in covered and key not in MISTUDIO_ROUTE_EXEMPT}


class TestTheProbeCanSee:
    """A blind probe agrees with everything: each half must see a known-true value first."""

    def test_the_served_table_is_populated_and_has_the_new_route(self, served):
        assert len(served) > 300, len(served)
        assert ("GET", f"{API}/probe-monitors/datasets/{{}}") in served

    def test_known_tools_map_to_their_routes(self, covered):
        assert "list_probe_monitors" in covered[("GET", f"{API}/probe-monitors/probes")]
        assert "start_probe_training_run" in covered[("POST", f"{API}/probe-monitors/runs")]
        # Through a helper that receives the path as an argument:
        assert "steer_sweep" in covered[("POST", f"{API}/steering/async/sweep")]
        # The conditional tool is reached:
        assert "get_approval_status" in covered[("GET", f"{API}/mcp/approvals/{{}}")]

    def test_every_client_call_path_was_readable(self):
        assert unreadable_calls() == []

    def test_only_the_howto_tool_calls_nothing(self):
        silent = sorted(name for name, routes in tool_routes().items() if not routes)
        assert silent == ["mistudio_howto"], silent


class TestEveryServedRouteIsAccountedFor:
    def test_every_served_route_has_a_tool_or_a_reasoned_exemption(self, served, covered):
        missing = sorted(uncovered(served, covered))
        assert not missing, (
            f"{len(missing)} SERVED miStudio routes have no registered MCP tool and no exemption. "
            "Add a tool, or a reason in MISTUDIO_ROUTE_EXEMPT:\n  "
            + "\n  ".join(f"{m} {p}" for m, p in missing)
        )

    def test_no_exemption_names_a_route_nobody_serves(self, served):
        stale = sorted(key for key in MISTUDIO_ROUTE_EXEMPT if key not in served)
        assert not stale, f"exemptions for routes the app does not serve: {stale}"

    def test_no_route_is_both_called_and_exempted(self, covered):
        both = sorted(set(MISTUDIO_ROUTE_EXEMPT) & set(covered))
        assert not both, (
            f"exempted routes a registered tool now calls — remove the exemption: "
            f"{[(k, sorted(covered[k])) for k in both]}"
        )

    def test_every_exemption_has_a_reason(self):
        thin = [key for key, reason in MISTUDIO_ROUTE_EXEMPT.items() if len(reason.strip()) < 40]
        assert not thin, thin

    def test_every_covered_route_is_served(self, served, covered):
        """The other direction (test_mcp_tools_call_real_routes.py checks it over every module;
        this checks it over the REGISTERED set the parity is computed from)."""
        dangling = sorted((k, sorted(v)) for k, v in covered.items() if k not in served)
        assert not dangling, dangling


def test_a_route_with_no_tool_IS_reported():
    """Negative control baked in: a served route nobody calls must surface as uncovered."""

    class _App:
        @staticmethod
        def openapi():
            return {"paths": {f"{API}/__parity_dummy": {"get": {}}}}

    assert uncovered(_served(_App), {}) == {("GET", f"{API}/__parity_dummy")}
