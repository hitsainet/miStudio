"""Probe monitor tools (category: probe_monitors) — 033 FR-9.

⚠ THESE SHIP WITH THE FEATURE, NOT AFTER IT. This server once had 16
`millm_circuit_*` tools that were fully implemented, unit-tested and documented in
`docs/mcp-contract.md` while registered NOWHERE — every test passed by importing the module
directly, so the suite was green and the docs said ✅ while no agent could call the feature.
`tests/unit/test_reachability.py` is the harness that guards it, and this module is registered in
`tools/__init__.CATEGORY_MODULES`, `config.VALID_CATEGORIES` and `config.DEFAULT_CATEGORIES`
before anything here is described as done.

⚠ AND EVERY ROUTE THESE TOOLS CALL EXISTS. A tool posting to a path nobody serves is the same
defect wearing a different hat: the agent gets a 404 it cannot interpret, and the contract says the
capability is there. Scope: 033's build/export/publish/recalibrate routes, 032's readers, and
(2026-10-19, MCP gap closure) the whole probe lifecycle before them — dataset views, training runs,
evaluation, offline scoring and judge baselines. `test_mcp_route_parity.py` holds every served
probe route to a tool or a written reason.

WHAT AN AGENT NEEDS TO KNOW, AND WHY IT IS IN THE DESCRIPTIONS RATHER THAN A GUIDE: a probe is a
CORRELATIONAL readout with an evidence rung, and the rung is the whole point. An agent that exports
a rung-1 probe and serves it as a monitor has made a claim the evidence does not support, so the
acknowledgement parameter says so in as many words.
"""

import asyncio
import inspect
import socket
from typing import Annotated, Any, Dict, List, Literal, Optional

import httpx
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from ..client import BackendError, MiWorkbenchClient
from ..config import MCPSettings
from ..dataworks_client import DataworksClient
from ..patterns import HEX64
from ..products import DATAWORKS

#: The `gpu` parameter of `start_probe_training_run`. A probe run places on ONE card
#: (`resolve_gpu_request(..., can_split=False)`), so "all" is refused — stated, because an agent
#: cannot infer it from a string-typed field.
RUN_GPU_PARAMETER = (
    '"auto" (the default) runs on the GPU with the most free memory when the job starts. A GPU '
    "UUID (or index) from /api/v1/system/gpu-list runs on that card; a named card that cannot "
    'take the job is refused, never swapped for another. A probe run always runs on ONE card: '
    '"all" is refused'
)

#: `score_probe_monitor_batch`'s budget. Each input is its own queued GPU forward pass (the
#: route scores one input), so the cap bounds how much of the queue one call can claim.

#: What the two GPU-starting tools say about approval — repeated because an agent reading one
#: tool's description does not see the other's.
APPROVAL_NOTE = (
    "When operator approval is on (MCP_PROBE_GPU_APPROVAL, the default) the FIRST call files an "
    "approval request in miWorkbench's banner and returns `{approval_request_id, status: "
    '"pending_approval"}` — nothing is queued. Re-call with the SAME arguments plus '
    "`approval_request_id` once the operator approved: the approval is consumed (once, for "
    "exactly that payload) and the job is submitted. A re-call before approval answers "
    '`status: "pending"`; denied/expired/consumed approvals need a new first call'
)


# Descriptions that interpolate a constant are module constants handed to `@mcp.tool(description=…)`:
# an expression after `def` is not a docstring, and MCPServer would register the tool with none.
START_RUN_DESCRIPTION = inspect.cleandoc("""Train probe monitors: capture activations and fit a
    linear detector per rule and layer. A GPU RUN of minutes to hours; returns `{id, task_id}`
    (202).

    One active run per (model, training view) — a second is refused 409. Poll
    `get_probe_training_run(id)`; cancel with `cancel_probe_training_run`. The probes it produces
    appear in `list_probe_monitors(run_id=…)`.

    The first call forecasts the run's host memory on the CPU (`forecast_probe_training_run`). One that
    will not fit is refused here, naming the figure, before any approval or GPU time; otherwise the
    answer carries the forecast as `memory_forecast`, for the operator deciding. The run checks again
    when it starts, before its captures.

    It also forecasts the run's TIME (`time_forecast`, ENH-006 G15): each stage's work (rows, tokens,
    probes, calibration and evaluation rows after `eval_max_rows`) times the seconds per unit that stage
    took on recent completed runs of the same model and scope. A range with its basis (which runs, which
    units) and `summary`, the one line filed on the approval card the operator decides from, with the
    evaluation rows it assumes. With no comparable run it says so and gives no number.
    """) + "\n\n" + APPROVAL_NOTE + "."


def time_forecast_card(time_forecast: Dict[str, Any]) -> Dict[str, Any]:
    """What the approval request carries for the operator (ENH-006 G15): the time forecast's one-line
    summary and the evaluation rows it assumes. Small on purpose: the whole forecast is on the agent's
    answer. Built here, not in `src.services`: the MCP pod reaches the backend over HTTP only."""
    card: Dict[str, Any] = {
        "forecast": bool(time_forecast.get("forecast")),
        "summary": str(time_forecast.get("summary") or time_forecast.get("reason") or "No time forecast."),
        "eval_rows": time_forecast.get("eval_rows"),
        "eval_max_rows": time_forecast.get("eval_max_rows"),
        "eval_rows_by_set": time_forecast.get("eval_rows_by_set") or {},
    }
    if card["forecast"]:
        card["seconds"] = time_forecast.get("seconds")
        card["complete"] = time_forecast.get("complete")
    return card

#: Tokenising a large view takes longer than the client's 60 s default. A timeout here costs a forecast,
#: not a run: `start_probe_training_run` reports it unchecked and carries on.
FORECAST_TIMEOUT_S = 300.0

EVALUATE_DESCRIPTION = inspect.cleandoc("""Evaluate an existing probe on more sets — a GPU job;
    returns a task id (202).

    This is how a probe climbs the ladder: rung 1 needs an in-distribution held-out set. Rung 2 needs
    FOUR things (ENH-006 D82, D84): (1) every out-of-distribution set's AUROC interval above 0.5;
    (2) on each of those sets the probe, AT ITS SHIPPED BAR, catches at least half the positives
    (`rung2_min_recall`); (3) an everyday-text set (role `everyday_text`, ordinary prose, at least
    1,000 rows by default) and (4) a nonsense-text set (role `nonsense_text`; make one with
    `create_nonsense_text_dataset`) on each of which its false alarms at the shipped bar are at most
    twice its target rate. Refused (409) when the probe's recorded precision or render no
    longer matches the model. Read per-row scores afterwards with `get_probe_evaluation_scores`.
    """) + "\n\n" + APPROVAL_NOTE + "."

SCORE_BATCH_DESCRIPTION = inspect.cleandoc("""Score several inputs offline with one probe: one
    queued forward pass PER INPUT.

    Returns `{"tasks": [{"index", "task_id"}], "submitted"}`; collect each with
    `get_probe_score_result`. ONE request queues them all; the cap is the deployment's setting
    (`probe_score_batch_cap`), enforced by miWorkbench — over it the call is refused naming the
    setting. Each input claims a place in the GPU queue: for thousands of rows, register a dataset
    view and use `evaluate_probe_monitor`, or score on the served model with `miserver_score_probes`.
    """)


START_JUDGE_DESCRIPTION = inspect.cleandoc("""Queue the LLM-monitor BASELINE: an LLM judge
    labels the same eval rows the probe was measured on, so the probe's AUROC has something to be
    compared with (rung 3 says "compared with a judge", not "beats one").

    The judge's instructions are a version from the judge-prompt library (`prompt_id`; see
    `list_judge_prompts`), recorded on the run with its hash. Rung 3 counts the run only when the
    prompt's behaviour equals the probe's (`counts_toward_rung` in the answer says which). CPU in
    miWorkbench, but every row is a call to the served judge model: thousands of calls for a full run.
    `max_rows_per_set` bounds them. Cancel with `cancel_probe_judge_run`.
    """) + "\n\nWhen the judge endpoint is miServer: " + APPROVAL_NOTE + "."


#: Module level so tests stub it and assert its calls (count and arguments). 042 FTID §3.7.
_resolve = socket.getaddrinfo
#: Each name lookup of the judge gate is bounded; a slow or failed lookup GATES (fails closed).
JUDGE_RESOLVE_TIMEOUT_S = 2.0
#: Said in the tool result when no miServer address is configured (standalone miWorkbench).
NAME_STEM_ONLY = "name_stem_only"
NAME_STEM_ONLY_REASON = "no miServer address configured; IP-literal endpoints are not checked"


def _addresses(infos: Any) -> set[str]:
    return {info[4][0] for info in infos or () if len(info) > 4 and info[4]}


async def judge_endpoint_is_miserver(endpoint: str, settings: Any) -> tuple[bool, Optional[str]]:
    """Whether a judge endpoint is miServer (miLLM before ENH-009), as `(gated, note)` (042 FR-042.38; FTID §3.7;
    CF-9, CF-25 m14; TPD-12, TPD-30). It FAILS CLOSED, because a judge run on miServer's card queues every
    live chat behind it and may evict the served model:

    * no host in the endpoint (``""``, ``"not a url"``, scheme-less ``miserver-backend:8000/v1``) → gated;
    * the host equals the configured miServer host (``MISERVER_API_URL``), or
      is named for miServer under either stem (``miserver``, ``millm``) → gated;
    * no miServer address configured → NOT gated, note ``name_stem_only`` (a standalone miWorkbench has no
      address to compare an IP literal with; ``_resolve`` is not called);
    * otherwise both hosts are resolved (each bounded by ``JUDGE_RESOLVE_TIMEOUT_S``): a failed or slow
      lookup → gated; shared addresses → gated (an IP literal of miServer's own address); else not gated.

    The endpoint is forwarded unchanged after approval. Replaces ``judge_endpoint_is_millm``, which matched
    only ``"millm" in host`` and let every host-less endpoint through.
    """
    from urllib.parse import urlparse

    from src.core import legacy_names

    host = (urlparse(endpoint or "").hostname or "").lower()
    if not host:
        return True, None
    configured_url = getattr(settings, "miserver_api_url", "") or ""
    configured = (urlparse(configured_url).hostname or "").lower() if configured_url else ""
    if (configured and host == configured) or legacy_names.names_sibling(host, "miserver"):
        return True, None
    if not configured:
        return False, NAME_STEM_ONLY
    try:
        mine = await asyncio.wait_for(asyncio.to_thread(_resolve, host, None), JUDGE_RESOLVE_TIMEOUT_S)
        theirs = await asyncio.wait_for(asyncio.to_thread(_resolve, configured, None), JUDGE_RESOLVE_TIMEOUT_S)
    except Exception:  # noqa: BLE001 — any lookup failure or timeout gates
        return True, None
    if _addresses(mine) & _addresses(theirs):
        return True, None
    return False, None


def _with_gate_note(result: Any, note: Optional[str]) -> Any:
    """Add the standalone note to a tool result (a dict); anything else passes through unchanged."""
    if note == NAME_STEM_ONLY and isinstance(result, dict):
        return {**result, "judge_gate": NAME_STEM_ONLY, "judge_gate_reason": NAME_STEM_ONLY_REASON}
    return result


class ProbeScoreInput(BaseModel):
    """One input for `score_probe_monitor_batch`: exactly one of `text` or `messages`."""

    model_config = ConfigDict(extra="forbid")

    text: Optional[str] = Field(default=None, max_length=16384)
    messages: Optional[List[Dict[str, str]]] = Field(default=None, min_length=1)

#: Repeated on the two tools that take it, because an agent reading one tool's description does not
#: see the other's.
RUNG_NOTE = (
    "A probe's `rung` is what its evidence supports: 0 trained only, 1 detects on held-out data, "
    "2 detects on unseen tasks (every out-of-distribution set clears 0.5, AND at the shipped bar it catches "
    "at least half the positives on each of them, AND few false alarms on an everyday-text set and on a "
    "nonsense-text set: at least 1,000 rows each, at most 2x the target rate by default; "
    "`rung_next_step` says which is missing), 3 compared against an LLM judge. Below rung 2 an export is REFUSED "
    "unless `acknowledge_below_rung2` carries a reason, which is then written into the exported "
    "definition so whoever serves it can see the claim was made on a judgement"
)


#: `client.status_code` values that mean miDataworks itself is not there: unreachable or timed out
#: (0, `DataworksClient`), or an ingress answering for a dead backend. Its own refusals (404, 409,
#: 500) are not in this list: those reach the agent verbatim.
_DATAWORKS_DOWN = frozenset({0, 502, 503, 504})


def _dataworks_client(url: str) -> DataworksClient:
    """The client `register_probe_dataset` reads a version's manifest with. A function so a test
    can hand in a transport.

    NATIVE, LIKE `gpu_preflight`, deliberately: this category is handed only the miWorkbench client, and
    the check must work whichever `dataworks_*` tools the server enables, so it builds its own client
    from `MIDATAWORKS_API_URL`. The product row's identity goes with it, so miDataworks records the
    read as this agent's.
    """
    return DataworksClient(url, identity=DATAWORKS.identity)


def register(mcp: MCPServer, client: MiWorkbenchClient, settings: MCPSettings) -> None:
    async def _approval_gate(
        tool_name: str, payload: Dict[str, Any], approval_request_id: Optional[str],
        forecast: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """`None` when the job may be submitted now; otherwise the structured answer to return.

        The `miserver_load_model` shape (caller-executed approval, Feature 034 FR-20): the first
        call files a request; a re-call with its id CONSUMES it — one conditional UPDATE in
        miWorkbench, so an approval is spent once and only on the payload the operator saw.

        `forecast` (ENH-006 G15) is filed BESIDE the payload for the operator to read on the card —
        never inside it, because consuming compares the payload exactly.
        """
        if not settings.probe_gpu_approval:
            return None
        if approval_request_id is None:
            body: Dict[str, Any] = {"tool_name": tool_name, "payload": payload}
            if forecast is not None:
                body["forecast"] = forecast
            created = await client.post("/mcp/approvals", json_body=body)
            return {
                "approval_request_id": created["id"],
                "status": "pending_approval",
                "hint": (
                    f"{tool_name} holds a GPU, so it needs operator approval in miWorkbench's "
                    f"banner. Re-call {tool_name} with the same arguments and this "
                    "approval_request_id once it is approved."
                ),
            }
        try:
            await client.post(
                f"/mcp/approvals/{approval_request_id}/consume",
                json_body={"tool_name": tool_name, "payload": payload},
            )
            return None
        except BackendError as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            status = detail.get("status")
            if exc.status_code == 409 and status:
                hint = {
                    "pending": "The operator has not decided yet. Re-call later with the "
                               "same approval_request_id.",
                    "consumed": "This approval was already used. Request a new one by "
                                "calling without approval_request_id.",
                }.get(status, f"Request a new approval by calling {tool_name} without "
                              "approval_request_id.")
                return {"approval_request_id": approval_request_id, "status": status,
                        "reason": detail.get("reason"), "hint": hint}
            if exc.status_code == 404:
                return {"approval_request_id": approval_request_id, "status": "not_found",
                        "hint": f"No such approval. Call {tool_name} without "
                                "approval_request_id to request one."}
            if exc.status_code == 422:
                return {"approval_request_id": approval_request_id, "status": "mismatch",
                        "reason": exc.detail,
                        "hint": "That approval is for different arguments. An approval covers "
                                "exactly the payload it names."}
            raise

    @mcp.tool()
    async def list_probe_monitors(
        run_id: Annotated[Optional[str], Field(description="Only probes from this run (pmr_…)")] = None,
        selected_only: Annotated[bool, Field(description="Only the probe each run selected as its best by validation AUROC")] = False,
        detector_digest: Annotated[Optional[str], Field(pattern=HEX64, description="Only probes whose built definition states this detector digest (64 lowercase hex)")] = None,
        decision_digest: Annotated[Optional[str], Field(pattern=HEX64, description="Only probes that state or stated this decision digest — the decision in force, any recorded revision, or the last built definition (64 lowercase hex)")] = None,
    ) -> Any:
        """List trained probe monitors with their rung, threshold, validation metrics and identity.

        The per-epoch training curve is NOT in this listing — it is a few hundred entries per probe
        and this payload is polled. `get_probe_monitor_report` returns it in full.

        IDENTITY (Feature 039). Each probe carries `detector_digest`, `decision_digest` and
        `decision_revision` — the identity of its BUILT definition — with `identity_current` (is that
        decision the one in force?), `identity_stale_reason` and `identity_source` (`built`,
        `recomputed` for a file built before digests, `not_recorded`). Null means not recorded, never
        a guess.

        JOIN BACK FROM A DIGEST. Holding a digest from an event or a file (e.g. `<decision digest>`),
        pass it as `decision_digest` (or `detector_digest`): each returned probe says in `matched`
        how — `{by, source: current|history|last_built, revisions: [...]}` — so an event judged at
        an old revision is still traced to its bar.
        """
        # ⚠ QUERY PARAMETERS ARE KEYWORD ARGUMENTS, NOT A `params=` DICT. `MiStudioClient.get`
        # takes `**params` and strips the Nones itself, so `params={...}` would send a query
        # field literally named "params" — caught by the payload half of the reachability
        # assertion, which is why that assertion checks the payload and not just the path.
        return await client.get(
            "/probe-monitors/probes", run_id=run_id, selected_only=selected_only or None,
            detector_digest=detector_digest, decision_digest=decision_digest,
        )

    @mcp.tool()
    async def get_probe_monitor_report(
        probe_id: Annotated[str, Field(description="Probe id (pm_…) from list_probe_monitors")],
        detail: Annotated[Literal["summary", "full"], Field(description="summary (default): the whole report except the ROC curves (each evaluation's and judge set's `roc`, replaced by `roc_points`) and the per-epoch training curve (`probe.val_metrics.history`). full: those too — about 115K characters of ROC per evaluation set")] = "summary",
    ) -> Any:
        """One probe's report: per-set AUROCs with confidence intervals, the rung WITH ITS
        WORDING and what would raise it, the dense/SAE counterpart, any judge runs, and (with
        `detail="full"`) the ROC and training curves.

        SUMMARY BY DEFAULT (ENH-006 G37). A full report measured 1,347,273 characters, 775,255 of
        them six evaluations' ROC curves. The summary keeps every rung, catch-rate, false-alarm,
        threshold, operating-point and transfer field — they are computed server-side from those
        curves — and lists what it left out in `omitted`. Ask for `full` only to plot a curve.

        READ THE RUNG BEFORE THE AUROC. A high AUROC on the set a probe was fitted near says little;
        the rung says which claim the numbers support.

        `everyday_text` (ENH-006 D82) is the false-alarm check rung 2 needs: per everyday-text set,
        the rows, the false alarms at the SHIPPED bar, the allowance and a plain sentence ("On
        everyday text: 0 false alarms in 1,200 (allowed up to 24)"). A probe held at rung 1 by it
        says so in `rung_reasons` and `rung_next_step` ("not yet checked on everyday text — evaluate
        on an everyday-text set", or the false-alarm rate against the allowance).

        `nonsense_text` (ENH-006 D84) is the same check on a nonsense-text set ("On nonsense text: 3 false
        alarms in 2,000 (allowed up to 40)"; "not yet checked on nonsense text — evaluate on a nonsense-text
        set"). `catch_rate` is the catch rate at the SHIPPED bar on every counted out-of-distribution set:
        per set the positives, how many fire, and a sentence ("On unseen tasks (…): catches 10% of positives
        at the shipped bar (41 of 410; needs ≥ 50%)"). Rung 2 needs all of them to pass.

        `probe` carries the identity of the built definition (`detector_digest`, `decision_digest`,
        `decision_revision`, `identity_current`, `identity_stale_reason`, `identity_source`); a
        definition built before digests has its identity recomputed on this first read.
        """
        return await client.get(f"/probe-monitors/probes/{probe_id}", detail=detail)

    @mcp.tool()
    async def build_probe_definition(
        probe_id: Annotated[str, Field(description="Probe id (pm_…)")],
        acknowledge_below_rung2_reason: Annotated[Optional[str], Field(description=f"Required to export a probe below rung 2, minimum 10 characters. {RUNG_NOTE}")] = None,
        vector_count: Annotated[int, Field(description="Test vectors to score into the definition, 8–32 (default 16). They are real forward passes, so this is a GPU job", ge=8, le=32)] = 16,
        seed: Annotated[int, Field(description="Sampling seed. The same seed gives the same rows, so two builds of one probe differ only if the implementation did")] = 1337,
    ) -> Any:
        """Build a portable `mistudio.probe-definition/v1` for a probe. Returns a `task_queue_id` (202).

        POLL `get_task_status(task_queue_id)` (ENH-006 G17): `queued` → `running` → `completed`, or
        `failed` with `error_message` — a refusal reads `refused (<http status>): <reason>`. The
        Celery `task_id` beside it is not a task-queue id. Once completed, hand the definition to
        miServer with `miserver_import_probe(mistudio_probe_id=…)` — server-side, so the document never
        passes through you; `export_probe_definition` returns the whole document.

        A GPU JOB: the definition's test vectors are scored through the same `forward_scores` path
        that produced the probe's published metrics, so a consumer reproducing them has reproduced
        the code that measured it.

        REFUSALS ARE RESULTS, not errors: an SAE probe whose dictionary has no HuggingFace home
        cannot be exported at all (publish the SAE first — no acknowledgement waives it), a run
        still in flight is refused because its rung can still change, and a probe below rung 2
        needs `acknowledge_below_rung2_reason`.

        IDENTITY (Feature 039). The built document carries an `identity` block —
        `{algorithm, detector_digest, decision_digest, decision_revision}` — and the task result
        repeats it. If the probe's bar or rung changed WHILE it built, the result is
        `status: "built_stale"` with `moved` and a `next` call that rebuilds against the bar in
        force; the file is kept, recorded as invalidated. A refusal message starts with its code
        (`decision_snapshot_stale: …`, `non_finite_number at <path>`).
        """
        body: Dict[str, Any] = {"vector_count": vector_count, "seed": seed}
        if acknowledge_below_rung2_reason:
            body["acknowledge_below_rung2"] = {"reason": acknowledge_below_rung2_reason}
        return await client.post(
            f"/probe-monitors/probes/{probe_id}/definition", json_body=body
        )

    @mcp.tool()
    async def recalibrate_probe_monitor(
        probe_id: Annotated[str, Field(description="Probe id (pm_…) to re-cut")],
        target_fpr: Annotated[float, Field(gt=0.0, lt=1.0, description="The false-positive budget to cut the new bar at, e.g. 0.005 for 0.5%")],
        preview: Annotated[bool, Field(description="Default TRUE: compute the move and report it without writing. Pass false to commit")] = True,
        reason: Annotated[str, Field(description="Why the bar is moving. Recorded on the probe so a later reader can answer 'who changed this'")] = "",
        allow_fire_on_nothing: Annotated[bool, Field(description="Permit a target so fine the bar lands above every negative, which makes the probe fire on NOTHING. Refused unless true")] = False,
        expect_decision_digest: Annotated[Optional[str], Field(pattern=HEX64, description="The `decision_digest` you READ (the preview's `current.decision_digest`). The commit is refused `decision_changed` if the decision in force is another — so you never commit over a decision you did not see")] = None,
        recut_windows: Annotated[bool, Field(description="Re-score the per-window negatives ON THE GPU (a queued job; operator approval). Only for a probe whose per-window arrays or calibration positions were never persisted")] = False,
        require_positions: Annotated[bool, Field(description="With recut_windows: the re-cut is asked FOR the calibration positions a rules-or-probe union bar needs, and is refused before the GPU when it cannot write them")] = False,
        approval_request_id: Annotated[Optional[str], Field(description="With recut_windows: the id the first call returned, once the operator approved it")] = None,
    ) -> Any:
        """Move a probe's decision threshold to another false-positive budget. No GPU, no queue —
        unless `recut_windows=true`, the GPU arm, which needs operator approval.

        A threshold is the `(1 - target_fpr)` quantile of the negative scores the run already saved
        to disk, so re-cutting it is arithmetic over an array — milliseconds, not the ~2.6-hour run
        it used to cost. That is the whole reason this tool exists: an operating point is the one
        part of a probe genuinely free to move, and it used to be the most expensive.

        PREVIEW FIRST, AND THAT IS THE DEFAULT. The response carries the per-set consequence of the
        candidate bar on every evaluation set the probe was measured on — recall, realised FPR, and
        whether the bar is above every score the set produces — beside the same figures for the bar
        currently in force. Read `transfer_proposed.caution` before committing.

        ⚠ THIS DOES NOT MAKE A THRESHOLD TRANSFER BETWEEN DISTRIBUTIONS, AND MUST NOT BE USED AS
        IF IT DID. A re-cut bar is still a quantile of the SAME calibration corpus. On the first
        shipped probe, five evaluation sets' own 1% thresholds spanned 24 points — one number
        behaves very differently on each. Lowering a bar until a probe fires on the traffic in
        front of you is not calibration, and `transfer_proposed` is there to make that visible.

        Refusals, each naming the number that caused it:
        * 409 when the probe predates calibration persistence, so there is no array to re-cut.
        * 422 when the target is finer than the sample can express — 2,000 negatives cannot
          express a rate below 1/2000. At that point the bar sits above every negative and the
          probe fires on nothing, which is why it is refused rather than returned.
        * 409 when the probe carries a per-window or per-length bar whose own negatives were never
          persisted. Those take PRECEDENCE over the global threshold, so moving only the global
          would change nothing a consumer serves. The refusal names the windows.

        Committing invalidates any cached exported definition, because the document states the
        operating point and a consumer cannot tell it has moved. A definition already PUBLISHED to
        HuggingFace cannot be reached — `published_copies_go_stale` says when that applies.

        IDENTITY (Feature 039). The preview states `current` and `proposed` each with
        `decision_digest` and `decision_revision`, the last built `detector_digest` with
        `detector_unchanged: true` (a re-cut never moves the detector), and `unchanged` — true when the
        candidate IS the decision in force, in which case a commit appends nothing
        (`committed: false, reason: "unchanged"`). Pass the preview's `current.decision_digest` as
        `expect_decision_digest` when committing: if another commit moved the decision in between,
        this one is refused `decision_changed` (409) with `expected`, `current` and a `next` preview.
        A commit returns `previous` and `current` (the replaced and the committed decision),
        `millm_copies` (miServer's imported copies, or why none can be named) and `next`: the calls
        that bring every copy up to the new decision — rebuild, re-publish, and one
        `miserver_recalibrate_probe` per imported copy, each runnable as it stands (an argument only you
        can give, such as a below-rung-2 acknowledgement, is listed under `needs`).
        """
        body: Dict[str, Any] = {
            "target_fpr": target_fpr,
            "preview": preview,
            "reason": reason,
            "allow_fire_on_nothing": allow_fire_on_nothing,
        }
        # Sent only when given: `RecalibrateRequest` is `extra="forbid"` and absent means "not checked".
        if expect_decision_digest is not None:
            body["expect_decision_digest"] = expect_decision_digest
        if not recut_windows:
            if require_positions:
                return {"error": "require_positions asks the GPU re-cut for calibration positions; pass "
                                 "recut_windows=true"}
            return await client.post(
                f"/probe-monitors/probes/{probe_id}/recalibrate", json_body=body,
            )
        # Feature 040 S3 (review R1-5): the GPU arm, behind the same operator approval as every GPU start.
        body["require_positions"] = require_positions
        held = await _approval_gate("recalibrate_probe_monitor", {"probe_id": probe_id, **body},
                                    approval_request_id)
        if held is not None:
            return held
        return await client.post(
            f"/probe-monitors/probes/{probe_id}/recalibrate", json_body=body, recut_windows=True,
        )

    @mcp.tool()
    async def export_probe_definition(
        probe_id: Annotated[str, Field(description="Probe id (pm_…) whose definition has already been built")],
    ) -> Any:
        """Fetch the built definition as JSON.

        409 when none has been built, or when a previously built one was INVALIDATED — which happens
        whenever the probe's rung or threshold changes, because the cached document states both and
        a consumer has no way to tell it has gone stale. The 409 names the `cause` (`decision`,
        `rung`, `build_race`). Build again.

        The document carries `identity` (Feature 039): `detector_digest` and `decision_digest` are
        SHA-256 over RFC 8785 projections of it, reproducible from the document alone with
        `docs/schemas/probe-identity-cases.json`. They prove the document is self-consistent, not who
        produced it.
        """
        return await client.get(f"/probe-monitors/probes/{probe_id}/definition")

    @mcp.tool()
    async def publish_probe_definition(
        probe_id: Annotated[str, Field(description="Probe id (pm_…) whose definition has been built")],
        repo_id: Annotated[str, Field(description="HuggingFace repo, owner/name. Created if absent")],
        private: Annotated[bool, Field(description="Private by default. A probe's definition names the datasets it was measured on and the concept it detects")] = True,
    ) -> Any:
        """Publish the definition, a merged manifest and a README to HuggingFace. Returns a task id.

        THE TOKEN IS NOT A PARAMETER HERE. It is resolved server-side from Settings → API Keys, so
        no credential passes through the agent transcript. A 401 comes back before any work is
        queued when none is stored.

        The repo's manifest is MERGED by name, never replaced: publishing does not remove probes
        somebody else put there. The README states what has and has NOT been checked — the
        correlational caveat included — because a reader who takes "detects on unseen tasks" for
        "reliable on my traffic" will over-trust it.
        """
        return await client.post(
            f"/probe-monitors/probes/{probe_id}/publish",
            json_body={"repo_id": repo_id, "private": private},
        )

    # ── 032/034/035: probe datasets, training runs, evaluation, scoring, judge baselines ──────
    #
    # Added 2026-10-19 (MCP gap closure). Before this the category could READ a trained probe
    # and EXPORT it, and nothing in between — no agent could register the data a probe is
    # trained on, start the run, evaluate it on new sets (the only way to rung 2), or read a
    # score back. Every one of these is a route the UI already used.

    @mcp.tool()
    async def download_dataset(
        repo_id: Annotated[str, Field(description="HuggingFace dataset repo id, owner/name", min_length=3, max_length=255)],
        config: Annotated[Optional[str], Field(description="Dataset config/subset name, if the repo has several. Each config is its own row")] = None,
        split: Annotated[Optional[str], Field(description="Only this split. Omit for every split; a large repo can be hundreds of GB, so name the one you need")] = None,
        revision: Annotated[Optional[str], Field(description="Commit, branch or tag. Resolved to a commit before anything is created, and that commit is what the row holds. Omit for the default branch's current head, which is resolved and recorded too", max_length=255)] = None,
    ) -> Any:
        """Download a published dataset from HuggingFace into miWorkbench. Returns at once with the
        new row (`id`, `status: downloading`, `revision`); poll `get_dataset(id)` until `status`
        is `ready` (`error` carries the reason), then `register_probe_dataset` takes that id.
        `cancel_dataset_download` stops it.

        The same route the UI uses. No GPU and no approval, like `download_model`: it runs on the
        `datasets` queue and never leases a card. There is no token argument: a gated repo uses
        the token stored in Settings, so no credential passes through an agent transcript.

        A row is one repo, config, split and commit. Asking for one that already exists answers
        409 naming the row, `already exists (dataset <id>)`: use that id. Nothing new is created
        and nothing is queued. A ref that does not resolve, or a gated repo with no usable token,
        is refused the same way and leaves no row behind.
        """
        body: Dict[str, Any] = {"repo_id": repo_id}
        for key, value in (("config", config), ("split", split), ("revision", revision)):
            if value is not None:
                body[key] = value
        return await client.post("/datasets/download", json_body=body)

    @mcp.tool()
    async def cancel_dataset_download(
        dataset_id: Annotated[str, Field(description="Dataset id (UUID) of the download to stop, from download_dataset or list_datasets")],
    ) -> Any:
        """Stop a dataset download (or tokenization) in progress and clean up its partial files.
        The row reads `error` and `error_message` records who asked (`cancelled by agent:mcp`). A
        dataset that is not downloading or processing is refused 400. Also stops a tokenization on the
        same row, because the route does."""
        return await client.delete(f"/datasets/{dataset_id}/cancel")

    @mcp.tool()
    async def register_probe_dataset(
        name: Annotated[str, Field(description="A name for this labelled view", min_length=1, max_length=255)],
        dataset_id: Annotated[str, Field(description="miWorkbench dataset id (UUID) of a DOWNLOADED dataset: from list_datasets, or download_dataset once get_dataset reads ready")],
        input_column: Annotated[str, Field(description="Column holding the text (or chat) to score")],
        label_column: Annotated[str, Field(description="Column holding the raw label")],
        label_mapping: Annotated[Dict[str, Literal["positive", "negative", "excluded"]], Field(description="EVERY distinct raw label value (as a string) -> positive | negative | excluded. A train/eval view needs both classes; a calibration, everyday_text or nonsense_text view must map nothing to positive")],
        role: Annotated[Literal["train", "eval", "calibration", "blend_fit", "everyday_text", "nonsense_text"], Field(description="train: what a run fits on; eval: measured on (rung 1-2); calibration: negatives that place the threshold, NOT labelled for the concept; blend_fit: labelled data a blend of probes is fitted on (never a run's training view); everyday_text: ordinary prose with NO positives of any behaviour (label_mapping {'*': 'negative'} reads every row as a negative — the wildcard is accepted on the two text roles only — and the text column may be the label column; no behaviour, sub_question or distribution) — rung 2 needs few false alarms on one, and one set serves every probe and model; nonsense_text: the same for text with NO meaning (word salad, random characters, repeated tokens, garbled code) — rung 2 needs few false alarms on one too; `create_nonsense_text_dataset` makes one and registers it in one call")] = "eval",
        split: Annotated[Optional[str], Field(description="Dataset split. REQUIRED with dataset_version_id or dataset_version_manifest")] = None,
        config: Annotated[Optional[str], Field(description="Dataset config/subset name, if the dataset has several")] = None,
        keyword_filter: Annotated[Optional[Dict[str, Any]], Field(description="{terms: [..], mode: any|all, case_sensitive}. NARROWS the set; it never assigns a label")] = None,
        pair_column: Annotated[Optional[str], Field(description="Column pairing rows (minimal pairs); optional")] = None,
        distribution: Annotated[Optional[Literal["in_distribution", "out_of_distribution"]], Field(description="EVAL views only: whether the set is out-of-distribution for the concept — rung 2 turns on it")] = None,
        dataset_version_manifest: Annotated[Optional[Dict[str, Any]], Field(description="A miDataworks `midataworks.dataset-version` manifest (dataworks_get_version_manifest returns it). Validated against the vendored schema; the split's row count must equal the downloaded split's, or the view is REFUSED. Prefer dataset_version_id, which saves carrying the document; give one or the other")] = None,
        dataset_version_id: Annotated[Optional[str], Field(description="A miDataworks dataset version id. miWorkbench fetches that version's manifest from miDataworks and verifies it exactly as dataset_version_manifest is verified. With miDataworks absent the answer is {'unavailable': 'dataworks', ...} and NO view is created", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")] = None,
        behaviour: Annotated[Optional[str], Field(description="The behaviour key this view's POSITIVE class is, e.g. '<behaviour-key>'")] = None,
        group_columns: Annotated[Optional[List[str]], Field(description="Grouping columns of the split to declare now (read and snapshotted), so evaluations on this view can be broken down by them")] = None,
        sub_question: Annotated[Optional[str], Field(description="A sub-question key of the behaviour version this view cites (needs behaviour)")] = None,
        behaviour_version: Annotated[Optional[int], Field(description="The behaviour version sub_question is checked against; default the current one", ge=1)] = None,
        text_column: Annotated[Optional[str], Field(description="The column rules read when it differs from input_column")] = None,
        rule_columns: Annotated[Optional[Dict[str, str]], Field(description="Producer rule verdicts: {rule_match: <column>, rule_veto?: <column>, rule_set_id: '<name>@<version>', rule_set_digest: <sha256>}; the set must be in the manifest's rule-set listing")] = None,
        span_column: Annotated[Optional[str], Field(description="Column holding, per row, the evidence's character offsets in the input text: [[start, end], ...] (code points, half-open; null or [] for none), e.g. a rule set's span output such as r_ssn_spans. Read once into the view's evidence_spans, which recommend_probe_rules reads to say which pooling rules fit; without it nothing is recommended", min_length=1, max_length=255)] = None,
    ) -> Any:
        """Register a label-mapped VIEW over a downloaded dataset — what a probe run trains on,
        is evaluated on, or calibrates its threshold from.

        The class counts are computed NOW and stored; a view that cannot be scored (no positives
        or no negatives after mapping and filtering) is refused here, naming both counts.

        **The miDataworks path (Feature 034):** name the version as `dataset_version_id` together
        with the `split`; miWorkbench fetches its handoff manifest (or pass the document itself as
        `dataset_version_manifest`, from `dataworks_get_version_manifest`, never both). miWorkbench
        validates it and REFUSES the view (409) when the manifest's row count for that split
        differs from what was downloaded — so a probe is never trained on rows other than the ones
        the manifest describes. The verified facts are stored on the view as `dataset_version`.

        A named version is never skipped: when miDataworks is not configured or does not answer,
        this returns `{"unavailable": "dataworks", "reason": ...}` and registers nothing. Report the
        reason; do not retry in a loop. With neither argument the view is plain, and
        `dataset_version` reads null.

        An agent's registration never confirms what the positive labels mean, with or without a manifest:
        `positive_labels_confirmed` reads null and the answer's `positive_labels` says how to confirm it. Only
        the operator can, on the Probes screen, or by a miDataworks detector-set send, which carries the
        operator's approval. A probe trained on an unconfirmed view cannot be made primary.
        """
        # Arguments first, as in every proxied tool: a payload mistake must not read as "miDataworks
        # is down", and costs no fetch.
        if dataset_version_id is not None:
            if dataset_version_manifest is not None:
                return {"refused": "manifest_and_version_id",
                        "reason": "Give dataset_version_id or dataset_version_manifest, not both: "
                                  "the two could describe different versions. Nothing was registered."}
            if not split:
                return {"refused": "split_required",
                        "reason": "A dataset-version manifest is checked per split; name the split. "
                                  "Nothing was fetched or registered."}
            url = settings.midataworks_api_url
            if not url:
                return {"unavailable": DATAWORKS.name,
                        "reason": f"{DATAWORKS.url_env} is not configured on this MCP server, so "
                                  f"version {dataset_version_id} cannot be read and the view was "
                                  "NOT registered."}
            dw = None
            try:
                dw = _dataworks_client(url)
                fetched = await dw.get(f"/api/v1/versions/{dataset_version_id}/handoff-manifest")
            except httpx.InvalidURL as exc:
                # Not an `httpx.HTTPError`, so the client does not turn it into a BackendError.
                return {"unavailable": DATAWORKS.name,
                        "reason": f"malformed {DATAWORKS.url_env}: {exc}. The view was NOT registered."}
            except BackendError as exc:
                if exc.status_code in _DATAWORKS_DOWN:
                    return {"unavailable": DATAWORKS.name,
                            "reason": f"{exc}. The view was NOT registered."}
                raise
            finally:
                if dw is not None:
                    await dw.close()
            # `{}` is what the client makes of an empty 2xx. Sent on, it would be the route's 422,
            # but this names the real cause: miDataworks gave no manifest.
            if not fetched or not isinstance(fetched, dict):
                return {"refused": "manifest_unreadable",
                        "reason": f"miDataworks answered version {dataset_version_id} with no manifest "
                                  f"document (got {type(fetched).__name__}). Nothing was registered."}
            dataset_version_manifest = fetched

        body: Dict[str, Any] = {
            "name": name,
            "dataset_id": dataset_id,
            "input_column": input_column,
            "label_column": label_column,
            "label_mapping": dict(label_mapping),
            "role": role,
        }
        for key, value in (("split", split), ("config", config),
                           ("keyword_filter", keyword_filter), ("pair_column", pair_column),
                           ("distribution", distribution),
                           ("dataset_version_manifest", dataset_version_manifest),
                           ("behaviour", behaviour), ("group_columns", group_columns),
                           ("sub_question", sub_question), ("behaviour_version", behaviour_version),
                           ("text_column", text_column), ("rule_columns", rule_columns),
                           ("span_column", span_column)):
            if value is not None:
                body[key] = value
        view = await client.post("/probe-monitors/datasets", json_body=body)
        if behaviour and role != "calibration" and isinstance(view, dict) and not view.get("positive_labels_confirmed"):
            return {**view, "positive_labels": {
                "confirmed": False,
                "next": f"The operator confirms that these positive labels are {behaviour!r} on miWorkbench's Probes "
                        "screen (Datasets, Confirm positive labels). An agent cannot. Until then this view trains "
                        "and evaluates as usual, and a probe trained on it cannot be made primary "
                        "(training_view_unconfirmed).",
            }}
        return view

    @mcp.tool()
    async def create_nonsense_text_dataset(
        rows: Annotated[int, Field(description="Rows to generate (default 2,000; rung 2 counts a set of at least 1,000 by default)", ge=100, le=50_000)] = 2000,
        seed: Annotated[int, Field(description="The same seed gives the same rows, byte for byte", ge=0, le=2**31 - 1)] = 0,
        name: Annotated[Optional[str], Field(description="Dataset and view name; default nonsense-text-seed<seed>-<rows>")] = None,
        prose_dataset_id: Annotated[Optional[str], Field(description="A downloaded dataset of real prose (e.g. OpenWebText) whose sentences are shuffled into one kind; without it that kind is skipped and `manifest.skipped` says so")] = None,
        prose_column: Annotated[str, Field(description="The prose dataset's text column")] = "text",
        register: Annotated[bool, Field(description="Also register its `nonsense_text` view (default) — one call, ready to evaluate")] = True,
    ) -> Any:
        """Generate a NONSENSE-TEXT set and register its view — no GPU, no download (ENH-006 D84).

        Rung 2 needs few false alarms on nonsense text at the probe's shipped bar (at most 2x its target rate,
        on a set of at least 1,000 rows, by default). This writes a miWorkbench dataset (columns `text` and
        `kind`) mixing word salad from common words, random characters, one token repeated, garbled code,
        digit noise that is never SSN-shaped (no run of 9 digits, no 3-2-4 grouping), mixed scripts and — with
        `prose_dataset_id` — shuffled sentences of real prose, at lengths from a token to about 4,000. With
        `register` it also creates the `nonsense_text` view (every row a negative, grouped by `kind` so a
        breakdown shows which kind fires). Returns `{dataset, view, manifest, next}`; then call
        `evaluate_probe_monitor(probe_id, dataset_ids=[view.id])` — which waits for the operator's approval.
        One set serves every probe and every model.
        """
        body: Dict[str, Any] = {"rows": rows, "seed": seed, "prose_column": prose_column, "register_view": register}
        if name is not None:
            body["name"] = name
        if prose_dataset_id is not None:
            body["prose_dataset_id"] = prose_dataset_id
        return await client.post("/probe-monitors/nonsense-text", json_body=body)

    @mcp.tool()
    async def list_probe_datasets(
        role: Annotated[Optional[Literal["train", "eval", "calibration", "blend_fit", "everyday_text", "nonsense_text"]], Field(description="Only views of this role")] = None,
        include_archived: Annotated[bool, Field(description="Also list ARCHIVED views (hidden by default)")] = False,
        search: Annotated[Optional[str], Field(description="Only views whose name or id contains this text, case-insensitive (e.g. a stack's common name part)", max_length=200)] = None,
        behaviour: Annotated[Optional[str], Field(description="Only views of this behaviour key", max_length=64)] = None,
        offset: Annotated[int, Field(description="Views to skip (of the matching views)", ge=0)] = 0,
        limit: Annotated[int, Field(description="Views per page, 1-200", ge=1, le=200)] = 50,
        detail: Annotated[Literal["summary", "full"], Field(description="summary (default): name, role, behaviour, counts, column names and the version each came from; full: every field `get_probe_dataset` gives")] = "summary",
    ) -> Any:
        """List registered probe dataset views, newest first, a page at a time:
        `{total, offset, limit, detail, items}`. `total` counts every matching view, so
        `offset + len(items) < total` means there are more pages.

        Narrow with `search` (name or id), `role` and `behaviour` before paging. A `train` view is
        what a run fits on; `eval` views are what rungs 1-2 are measured on; a `calibration` view
        supplies the negatives that place the threshold; an `everyday_text` view is ordinary prose
        and a `nonsense_text` view text with no meaning, both of which rung 2 checks false alarms
        on. Each summary leaves out the column snapshots' paths and digests, the manifest's
        commit and digest and the positive class's wording: read one view whole with
        `get_probe_dataset`. Archived views are hidden unless `include_archived`; an archived
        view cannot be used by a new run."""
        return await client.get(
            "/probe-monitors/datasets-page", role=role, include_archived=include_archived or None,
            search=search or None, behaviour=behaviour, offset=offset, limit=limit, detail=detail,
        )

    @mcp.tool()
    async def get_probe_dataset(
        dataset_id: Annotated[str, Field(description="Probe dataset view id (from register/list)")],
    ) -> Any:
        """One probe dataset view: its mapping, filter, class counts, and — when it was
        registered with a miDataworks manifest — the `dataset_version` facts that were verified."""
        return await client.get(f"/probe-monitors/datasets/{dataset_id}")

    @mcp.tool()
    async def get_probe_dataset_rows(
        dataset_id: Annotated[str, Field(description="Probe dataset view id (from register/list)")],
        offset: Annotated[int, Field(description="Rows to skip (of the matching rows)", ge=0)] = 0,
        limit: Annotated[int, Field(description="Rows per page, 1-200", ge=1, le=200)] = 50,
        columns: Annotated[Optional[List[str]], Field(description="OTHER columns of the download to return. The view's input, label and pair columns are always returned. Omit for every column; [] for none")] = None,
        label: Annotated[Optional[Literal["positive", "negative", "excluded", "filtered_out", "unparseable"]], Field(description="Only rows the view's mapping and filter turn into this")] = None,
        search: Annotated[Optional[str], Field(description="Only rows whose input contains this text (case-insensitive)", max_length=500)] = None,
    ) -> Any:
        """Read a probe dataset view's ROWS, paged and projected: each row's position in the
        downloaded split, its raw label, the label the view's mapping gives it (`outcome`:
        positive | negative | excluded | filtered_out | unparseable) and the requested columns.

        Read-only. Use `columns` to fetch only what you need — two columns of a view in one call
        per page, instead of paging every column. The view's stored `counts` come back with each
        page; a `label` filter also returns `recount` and `counts_agree` (false means the
        download changed since the view was registered)."""
        params: Dict[str, Any] = {"offset": offset, "limit": limit}
        if columns is not None:
            params["columns"] = ",".join(columns)
        if label is not None:
            params["label"] = label
        if search:
            params["search"] = search
        return await client.get(f"/probe-monitors/datasets/{dataset_id}/rows", **params)

    @mcp.tool()
    async def archive_probe_dataset(
        dataset_id: Annotated[str, Field(description="Probe dataset view id to archive")],
        reason: Annotated[Optional[str], Field(description="Why, kept on the view (optional)", max_length=2000)] = None,
    ) -> Any:
        """ARCHIVE a probe dataset view: hide it from listings and from new runs, evaluations
        and judge baselines. Nothing is deleted — every run, probe and evaluation that already
        uses it keeps working — and `unarchive_probe_dataset` undoes it. Recorded as archived by
        this agent.

        There is deliberately NO delete tool: permanent deletion of an archived view is the
        operator's alone, from miWorkbench's Probes screen."""
        body = {"reason": reason} if reason else None
        return await client.post(f"/probe-monitors/datasets/{dataset_id}/archive", json_body=body)

    @mcp.tool()
    async def unarchive_probe_dataset(
        dataset_id: Annotated[str, Field(description="Archived probe dataset view id")],
    ) -> Any:
        """Unarchive a probe dataset view: it is listed again by `list_probe_datasets` and can
        be used by new training runs, evaluations and judge baselines. Idempotent."""
        return await client.post(f"/probe-monitors/datasets/{dataset_id}/unarchive")

    @mcp.tool(description=START_RUN_DESCRIPTION)
    async def start_probe_training_run(
        model_id: Annotated[str, Field(description="miWorkbench model id (m_xxxxxxxx)")],
        train_dataset_id: Annotated[str, Field(description="A probe dataset view with role 'train'")],
        eval_dataset_ids: Annotated[Optional[List[str]], Field(description="Eval views to measure every probe on. Never the training view (that would measure memorisation) and never a calibration view")] = None,
        calibration_dataset_id: Annotated[Optional[str], Field(description="A calibration view whose negatives place the threshold at config.target_fpr")] = None,
        config: Annotated[Optional[Dict[str, Any]], Field(description="ProbeRunConfig, passed verbatim: layers | stride, rules, scope, skip_template_prefix (default true: the chat template's fixed prefix is left out of the all/prompt/last_user windows, so max and short rolling windows are not floored by it), max_length, val_fraction, seed, top_n_layers, rolling_windows, sae_variant, sae_k, target_fpr, calibration_max_rows, eval_max_rows (a seeded sample of each evaluation view; REQUIRED when a view holds more than 20,000 rows, e.g. an everyday_text corpus), batch_size, acknowledge_rule_choice (the reason, recorded on the run, for omitting a rule recommend_probe_rules recommends for the training view; without it such a run is refused). Omit for the defaults")] = None,
        gpu: Annotated[str, Field(description=RUN_GPU_PARAMETER)] = "auto",
        approval_request_id: Annotated[Optional[str], Field(description="The id a first call returned with status pending_approval; pass it once the operator approved")] = None,
        behaviour: Annotated[Optional[str], Field(description="What the probes detect, as a short key (lower-case words joined by hyphens, e.g. '<behaviour-key>'). Rung 3 counts only judge runs whose prompt judges this same key; omit it and no judge run can raise these probes past rung 2 until it is set with set_probe_behaviour")] = None,
        behaviour_label: Annotated[Optional[str], Field(description="The behaviour in words (the label a person reads), e.g. '<behaviour label>'")] = None,
        selection_rule: Annotated[Optional[Literal["auroc", "recall_at_decision"]], Field(description="How the run picks its selected probe: recall at each probe's calibrated decision on the out-of-distribution sets (the default), or the older AUROC ranking. Under either rule a probe with validation AUROC below 0.55, or whose bar fires on more than 10x its target false-positive rate on the calibration negatives, is never selected; if none qualifies the run selects none and `selection.none_selected` says why. Merged into config")] = None,
    ) -> Any:
        if selection_rule is not None:
            config = {**(config or {}), "selection_rule": selection_rule}
        payload: Dict[str, Any] = {
            "model_id": model_id,
            "train_dataset_id": train_dataset_id,
            "eval_dataset_ids": list(eval_dataset_ids or []),
            "gpu": gpu,
        }
        if behaviour is not None:
            payload["behaviour"] = behaviour
        if behaviour_label is not None:
            payload["behaviour_label"] = behaviour_label
        if calibration_dataset_id is not None:
            payload["calibration_dataset_id"] = calibration_dataset_id
        if config is not None:
            payload["config"] = config
        # On the first call only: the re-call spends an approval the operator gave for this payload,
        # and the run checks itself when it starts.
        forecast: Optional[Dict[str, Any]] = None
        rule_recommendation: Optional[Dict[str, Any]] = None
        if approval_request_id is None:
            try:
                forecast = await client.post(
                    "/probe-monitors/runs/forecast", json_body=payload, timeout=FORECAST_TIMEOUT_S
                )
            except BackendError as exc:
                forecast = {"checked": False, "fits": None, "reason": f"the forecast could not run: {exc}"}
            # ENH-006 G15: the time forecast rides on the forecast route's answer; it is reported beside the
            # memory forecast and filed on the approval the operator decides from.
            time_forecast = forecast.get("time_forecast") or {
                "forecast": False,
                "summary": f"No time forecast: {forecast.get('reason') or 'the forecast route gave none'}.",
            }
            rule_recommendation = forecast.get("rule_recommendation")
            forecast = {key: value for key, value in forecast.items()
                        if key not in ("time_forecast", "rule_recommendation")}
            if forecast.get("fits") is False:
                return {"refused": "probe_training_too_large", "reason": forecast.get("message"),
                        "memory_forecast": forecast, "time_forecast": time_forecast}
            # ENH-006 G39: the refusal `POST /runs` would give, said before an approval is spent on it.
            rule_refusal = (rule_recommendation or {}).get("refusal")
            if rule_refusal:
                return {"refused": rule_refusal["code"], "reason": rule_refusal["message"],
                        "missing": rule_refusal["missing"], "rule_recommendation": rule_recommendation}
        extra = {} if forecast is None else {"memory_forecast": forecast, "time_forecast": time_forecast}
        if rule_recommendation is not None:
            extra["rule_recommendation"] = rule_recommendation
        held = await _approval_gate(
            "start_probe_training_run", payload, approval_request_id,
            forecast=None if forecast is None else {"time": time_forecast_card(time_forecast)},
        )
        if held is not None:
            return {**held, **extra}
        started = await client.post("/probe-monitors/runs", json_body=payload)
        return {**started, **extra}

    @mcp.tool()
    async def forecast_probe_training_run(
        model_id: Annotated[str, Field(description="miWorkbench model id (m_xxxxxxxx)")],
        train_dataset_id: Annotated[str, Field(description="A probe dataset view with role 'train'")],
        config: Annotated[Optional[Dict[str, Any]], Field(description="ProbeRunConfig, as start_probe_training_run takes it: layers | stride, scope, max_length, val_fraction, seed, top_n_layers, rules, rolling_windows, calibration_max_rows, eval_max_rows")] = None,
        eval_dataset_ids: Annotated[Optional[List[str]], Field(description="The eval views the run will name, so the time forecast counts its evaluation rows; omitted, evaluating counts 0 rows")] = None,
        calibration_dataset_id: Annotated[Optional[str], Field(description="The calibration view the run will name, for the time forecast")] = None,
        gpu: Annotated[Optional[str], Field(description="The card the run will ask for, as start_probe_training_run takes it; the time forecast then prefers runs measured on that card. " + RUN_GPU_PARAMETER)] = None,
    ) -> Any:
        """Will this probe run fit in host memory, and how long will it take? Tokenises every row on the
        CPU with the model's tokenizer and the run's own render, split and scope. No GPU, no approval,
        nothing created. Can take minutes on a large view.

        `time_forecast` (ENH-006 G15): each stage's work times the seconds per unit that stage took on
        recent completed runs of the same model and scope — `seconds` low/typical/high, `stages` with
        every rate, `basis` naming the runs and units, `summary` in one line. `forecast: false` means no
        comparable run was measured, and no number is given.

        `fits`, `needed_gib` against `usable_gib` (available memory now, less the safety margin and other
        jobs' reserves), `arithmetic` showing each term, `based_on` saying what the figure rests on, and
        `by_top_n`: one layer's batches are held at a time, so more chosen layers cost disk for their
        token files and not required RAM, and the table gives both. `checked: false` with a `reason`
        means no forecast could be made, and is not a yes.
        """
        body: Dict[str, Any] = {"model_id": model_id, "train_dataset_id": train_dataset_id}
        if config is not None:
            body["config"] = config
        if eval_dataset_ids:
            body["eval_dataset_ids"] = list(eval_dataset_ids)
        if calibration_dataset_id is not None:
            body["calibration_dataset_id"] = calibration_dataset_id
        if gpu is not None:
            body["gpu"] = gpu
        return await client.post("/probe-monitors/runs/forecast", json_body=body, timeout=FORECAST_TIMEOUT_S)

    @mcp.tool()
    async def recommend_probe_rules(
        train_dataset_id: Annotated[str, Field(description="The probe dataset view a run would train on")],
        model_id: Annotated[Optional[str], Field(description="Measure the rolling window in this model's tokens (CPU, tokenizer only). Without it the window is not sized")] = None,
    ) -> Any:
        """Which pooling rules fit a training view, from where its evidence sits — ask before
        `start_probe_training_run`.

        Reads the view's span column (character offsets of the evidence in each row; miDataworks sends
        one from a rule set's span output): the evidence's share of the row, where it starts, how much the
        row length spreads, and how many negatives carry the same surface form. A LOCAL pattern (an SSN in
        a long text) gets `max` and `rolling_mean_max` with a window near the evidence's width in tokens;
        a WHOLE-TEXT behaviour gets `mean`; in between, both shapes. `evidence` has the numbers,
        `thresholds` the cuts.

        A run that omits a recommended rule is REFUSED (`rule_choice_omits_recommendation`) unless its
        config passes `acknowledge_rule_choice` with the reason, which is recorded on the run. A view
        without a span column gets `rules: []` and a reason: nothing is recommended and nothing refused.
        """
        return await client.get(f"/probe-monitors/datasets/{train_dataset_id}/rule-recommendation",
                                model_id=model_id)

    @mcp.tool()
    async def list_probe_training_runs(
        status: Annotated[Optional[Literal["pending", "running", "completed", "failed", "cancelled"]], Field(description="Only runs in this state")] = None,
        train_dataset_id: Annotated[Optional[str], Field(description="Only runs whose TRAINING set is this probe dataset view")] = None,
        model_id: Annotated[Optional[str], Field(description="Only runs of this model (m_...)")] = None,
        behaviour: Annotated[Optional[str], Field(description="Only runs of this behaviour key", max_length=64)] = None,
        search: Annotated[Optional[str], Field(description="Only runs whose id or behaviour label contains this text, case-insensitive", max_length=200)] = None,
        offset: Annotated[int, Field(description="Runs to skip (of the matching runs)", ge=0)] = 0,
        limit: Annotated[int, Field(description="Runs per page, 1-200", ge=1, le=200)] = 50,
        detail: Annotated[Literal["summary", "full"], Field(description="summary (default): ids, behaviour, status, stage, progress and ETA (with eta_provisional), error, card, target_fpr and the selected probe; full: every field, config and environment included")] = "summary",
    ) -> Any:
        """List probe training runs, newest first, a page at a time:
        `{total, offset, limit, detail, items}`. `total` counts every matching run, so
        `offset + len(items) < total` means there are more pages.

        Narrow with `status`, `train_dataset_id` (every run trained on one view, 035 XR-4),
        `model_id`, `behaviour` and `search` before paging. A summary leaves out each run's
        config, environment (memory forecast, stage timings, GPU memory, selection) and layer grid:
        read one run whole with `get_probe_training_run`."""
        return await client.get(
            "/probe-monitors/runs-page", status=status, train_dataset_id=train_dataset_id,
            model_id=model_id, behaviour=behaviour, search=search or None, offset=offset, limit=limit,
            detail=detail,
        )

    @mcp.tool()
    async def get_probe_training_run(
        run_id: Annotated[str, Field(description="Probe run id (pmr_…)")],
    ) -> Any:
        """One probe run: status, stage, progress, its config echoed verbatim and the environment
        it ran in (model precision included).

        `progress_detail` (runs from 2026-10-21) is the live reading behind `progress`: `stage`,
        `done`/`total` in `unit` (batches, probes, sets), `current` (the item inside the stage — a
        probe's epochs, a set's batches), `rate_per_second` and `eta_seconds` for the stage. Null
        until the run reports; rate and ETA stay null until measured. `eta_provisional: true`
        (ENH-006 G23) means the ETA rests on under 25 reports or under two minutes of the stage —
        an early guess, often hours off on a set whose first batches are its longest; do not read
        it as an overrun.

        `environment.gpu_memory` (runs from 2026-10-21) is the GPU memory each stage took:
        `stages.<name>.peak_allocated_mib` / `peak_reserved_mib`, the run's overall peaks and the
        card's `gpu_uuid` — size the next run from it. `measured: false` (null peaks) means the
        model ran on no CUDA device. Judge runs call an LLM endpoint and use no miWorkbench GPU."""
        return await client.get(f"/probe-monitors/runs/{run_id}")

    @mcp.tool()
    async def cancel_probe_training_run(
        run_id: Annotated[str, Field(description="Probe run id (pmr_…) to stop")],
    ) -> Any:
        """Request a COOPERATIVE stop: the worker stops at its next stage boundary, so poll
        `get_probe_training_run` until it reads cancelled. 409 when the run is already finished."""
        return await client.post(f"/probe-monitors/runs/{run_id}/cancel")

    @mcp.tool()
    async def delete_probe_training_run(
        run_id: Annotated[str, Field(description="Probe run id (pmr_…) to delete")],
        acknowledge_permanent: Annotated[bool, Field(description="Delete the run, EVERY probe it produced and their artifacts. Irreversible; retraining costs the whole GPU run again")] = False,
    ) -> Any:
        """Delete a probe run, its probes and their artifacts. A RUNNING run is stopped first.

        Refuses unless `acknowledge_permanent=True`. Probes already EXPORTED or imported into
        miServer keep serving there; this removes miWorkbench's record and evidence for them."""
        if not acknowledge_permanent:
            return {
                "refused": "delete_not_acknowledged",
                "run_id": run_id,
                "message": (
                    f"Deleting {run_id} removes every probe it produced and their artifacts, "
                    "irreversibly. Re-call with acknowledge_permanent=true if that is intended."
                ),
            }
        return await client.delete(f"/probe-monitors/runs/{run_id}")

    @mcp.tool(description=EVALUATE_DESCRIPTION)
    async def evaluate_probe_monitor(
        probe_id: Annotated[str, Field(description="Probe id (pm_…)")],
        dataset_ids: Annotated[List[str], Field(description="Eval probe dataset views to measure the probe on — every out-of-distribution set (AUROC and the catch rate at the shipped bar), plus an everyday_text view and a nonsense_text view (false alarms at the shipped bar), is how rung 2 is reached", min_length=1)],
        approval_request_id: Annotated[Optional[str], Field(description="The id a first call returned with status pending_approval; pass it once the operator approved")] = None,
        sample_rows: Annotated[Optional[int], Field(description="Score a seeded sample of this many rows of each view (the seed is the view and the size, so every probe evaluated at this size scores the same rows). REQUIRED for a calibration view", ge=100, le=200000)] = None,
    ) -> Any:
        payload: Dict[str, Any] = {"probe_id": probe_id, "dataset_ids": list(dataset_ids)}
        if sample_rows is not None:
            # Feature 041 T20: part of what the operator approves — the approval binds the sample.
            payload["sample_rows"] = sample_rows
        held = await _approval_gate("evaluate_probe_monitor", payload, approval_request_id)
        if held is not None:
            return held
        # The route's body IS the list (FastAPI reads a sole `List[str]` parameter as the body).
        return await client.post(
            f"/probe-monitors/probes/{probe_id}/evaluate", json_body=list(dataset_ids),
            sample_rows=sample_rows,
        )

    @mcp.tool()
    async def get_probe_evaluation_scores(
        probe_id: Annotated[str, Field(description="Probe id (pm_…)")],
        evaluation_id: Annotated[str, Field(description="Evaluation id, from the probe's report (get_probe_monitor_report)")],
        offset: Annotated[int, Field(description="First row to return", ge=0)] = 0,
        limit: Annotated[int, Field(description="Rows per page (1-50000)", ge=1, le=50000)] = 1000,
    ) -> Any:
        """One evaluation's PER-ROW scores, paged in stored order, with the label and the source
        dataset's revision (035 XR-2).

        Page with `next_offset` until it is null. 409 `positions_not_recorded` for an evaluation
        recorded before positions were saved — re-evaluate the probe on that set (the refusal
        names the call); scores are never served from replayed positions. A refused evaluation is
        served with `reported_auroc: null`."""
        return await client.get(
            f"/probe-monitors/probes/{probe_id}/evaluations/{evaluation_id}/scores",
            offset=offset, limit=limit,
        )

    @mcp.tool()
    async def score_probe_monitor(
        probe_id: Annotated[str, Field(description="Probe id (pm_…)")],
        text: Annotated[Optional[str], Field(description="Plain text to score (max 16384 chars). Exactly one of text or messages", max_length=16384)] = None,
        messages: Annotated[Optional[List[Dict[str, str]]], Field(description="A chat [{role, content}] to score, rendered as the probe was trained. Exactly one of text or messages")] = None,
    ) -> Any:
        """Score ONE input offline with a probe — a queued forward pass; returns a task id (202).
        Collect it with `get_probe_score_result(probe_id, task_id)`. Over the length cap the
        input is REFUSED, never truncated."""
        if (text is None) == (messages is None):
            return {"error": "exactly_one_input", "message": "give exactly one of text or messages"}
        body: Dict[str, Any] = {"text": text} if text is not None else {"messages": messages}
        return await client.post(f"/probe-monitors/probes/{probe_id}/score", json_body=body)

    @mcp.tool(description=SCORE_BATCH_DESCRIPTION)
    async def score_probe_monitor_batch(
        probe_id: Annotated[str, Field(description="Probe id (pm_…)")],
        inputs: Annotated[List[ProbeScoreInput], Field(description="Inputs, each exactly one of text or messages (at most the deployment's probe_score_batch_cap)", min_length=1)],
    ) -> Any:
        # ⚠ ONE POST, AND NO CAP HERE (Feature 041 T19): the cap was an MCP-side literal, so a REST caller was
        # unbounded; miWorkbench now refuses over the runtime setting and the refusal is passed through.
        bodies = []
        for index, item in enumerate(inputs):
            if (item.text is None) == (item.messages is None):
                return {"error": "exactly_one_input",
                        "message": f"inputs[{index}]: give exactly one of text or messages"}
            bodies.append({"text": item.text} if item.text is not None
                          else {"messages": item.messages})
        return await client.post(f"/probe-monitors/probes/{probe_id}/score-batch", json_body={"inputs": bodies})

    @mcp.tool()
    async def get_probe_score_result(
        probe_id: Annotated[str, Field(description="Probe id (pm_…) the score was requested for")],
        task_id: Annotated[str, Field(description="Task id from score_probe_monitor(_batch)")],
    ) -> Any:
        """Collect an offline probe score. `status` is Celery's: SUCCESS carries `result`;
        FAILURE carries the reason (the probe did NOT run — not "scored nothing"); PENDING means
        queued OR unknown, which this endpoint cannot tell apart."""
        return await client.get(f"/probe-monitors/probes/{probe_id}/score/{task_id}")

    @mcp.tool(description=START_JUDGE_DESCRIPTION)
    async def start_probe_judge_run(
        endpoint: Annotated[str, Field(description="OpenAI-compatible base URL of the LLM judge (e.g. a miServer /v1)")],
        model: Annotated[str, Field(description="The judge model name at that endpoint")],
        dataset_ids: Annotated[List[str], Field(description="Eval probe dataset views to judge", min_length=1)],
        probe_id: Annotated[Optional[str], Field(description="The probe to compare against (pm_…) — what lets the probe reach rung 3, when the prompt judges the probe's behaviour")] = None,
        prompt_id: Annotated[Optional[str], Field(description="The judge prompt version to use (jp_…, from list_judge_prompts). Omit for stakes-rating/v1, the prompt every run used before the library. Choose one whose behaviour matches the probe's, or the run is a baseline that never raises the rung")] = None,
        max_rows_per_set: Annotated[int, Field(description="Rows judged per set (20-50000); each is one judge call", ge=20, le=50000)] = 2000,
        parse_failure_limit: Annotated[Optional[float], Field(description="Fail the run when more than this fraction of judge answers cannot be parsed (0-1, exclusive); omit for the default", gt=0.0, lt=1.0)] = None,
        approval_request_id: Annotated[Optional[str], Field(description="The id a first call returned with status pending_approval (only when the judge is served by miServer); pass it once the operator approved")] = None,
    ) -> Any:
        body: Dict[str, Any] = {
            "endpoint": endpoint,
            "model": model,
            "dataset_ids": list(dataset_ids),
            "max_rows_per_set": max_rows_per_set,
        }
        if probe_id is not None:
            body["probe_id"] = probe_id
        if prompt_id is not None:
            body["prompt_id"] = prompt_id
        if parse_failure_limit is not None:
            body["parse_failure_limit"] = parse_failure_limit
        # ⚠ GATED WHEN THE JUDGE IS miServer (2026-10-21; it was not gated at all). Every row is a
        # generation on the GPU that serves miServer's users, and a judge run asks miServer to load the
        # judge model if it is not loaded — evicting what is served. Same caller-executed approval
        # as a probe training run, under the same switch (MCP_PROBE_GPU_APPROVAL).
        gated, note = await judge_endpoint_is_miserver(endpoint, settings)
        if gated:
            held = await _approval_gate("start_probe_judge_run", body, approval_request_id)
            if held is not None:
                return held
        return _with_gate_note(await client.post("/probe-monitors/judge-runs", json_body=body), note)

    @mcp.tool()
    async def start_probe_spot_check(
        behaviour: Annotated[str, Field(description="Catalogued behaviour key to audit, e.g. '<behaviour-key>'")],
        prompt_id: Annotated[str, Field(description="Judge prompt version (jp_…) whose behaviour IS this behaviour")],
        endpoint: Annotated[str, Field(description="OpenAI-compatible base URL of the judge (e.g. a miServer /v1)")],
        model: Annotated[str, Field(description="The judge model name at that endpoint")],
        size: Annotated[int, Field(description="Recorded turns to sample (1-500), split over fired / near threshold / not fired", ge=1, le=500)] = 100,
        window: Annotated[str, Field(description="Which probe window to sample on: all, prompt, response or last_user")] = "all",
        seed: Annotated[Optional[int], Field(description="Seed for the sample; the same seed and spec give the same turns", ge=0)] = None,
        approval_request_id: Annotated[Optional[str], Field(description="The id a first call returned with status pending_approval (only when the judge is miServer); pass it once the operator approved")] = None,
    ) -> Any:
        """Spot-check a behaviour's live probes against a library judge (Feature 038): miServer returns a seeded,
        stratified sample of RECORDED session turns (transcripts are opt-in in miServer, so a short sample is
        reported, never padded), the judge is shown the window each probe scored, and the report gives a
        2 × 2 table, agreement with a confidence interval and kappa per probe. The judge is a second
        opinion, not ground truth. Refused (503) when miServer is not connected or lacks its Feature 036
        session routes. When the judge is miServer it waits for operator approval first: live chats queue
        behind every judge call."""
        body: Dict[str, Any] = {
            "behaviour": behaviour, "prompt_id": prompt_id, "endpoint": endpoint, "model": model,
            "size": size, "window": window,
        }
        if seed is not None:
            body["seed"] = seed
        # D34: gated exactly like start_probe_judge_run when the judge is served by miServer.
        # Removing this gate fails test_mcp_spot_check_gate.py.
        gated, note = await judge_endpoint_is_miserver(endpoint, settings)
        if gated:
            held = await _approval_gate("start_probe_spot_check", body, approval_request_id)
            if held is not None:
                return held
        return _with_gate_note(await client.post("/probe-monitors/spot-checks", json_body=body), note)

    @mcp.tool()
    async def get_probe_spot_check(
        spot_check_id: Annotated[str, Field(description="Spot-check id (psc_…)")],
    ) -> Any:
        """One spot-check: status, progress, the sample spec and what miServer returned (ids only), and the
        report — per compared probe the 2 × 2 table (probe fired vs judge positive), agreement with its
        bootstrap confidence interval, kappa, shortfall per stratum. No turn text (that is the operator's
        export)."""
        return await client.get(f"/probe-monitors/spot-checks/{spot_check_id}")

    @mcp.tool()
    async def list_probe_spot_checks(
        behaviour: Annotated[Optional[str], Field(description="Only checks of this behaviour key")] = None,
    ) -> Any:
        """List behaviour spot-checks, newest first, with status, progress, whether a run is partial,
        and each report (2 × 2 table, agreement, kappa per compared probe). Filter by behaviour key."""
        return await client.get("/probe-monitors/spot-checks", behaviour=behaviour)

    @mcp.tool()
    async def cancel_probe_spot_check(
        spot_check_id: Annotated[str, Field(description="Spot-check id (psc_…)")],
    ) -> Any:
        """Stop a running spot-check between turns; the turns judged so far are kept as a partial run."""
        return await client.post(f"/probe-monitors/spot-checks/{spot_check_id}/cancel")

    @mcp.tool()
    async def get_spot_check_availability() -> Any:
        """Whether miServer is connected and serves the recorded-session routes a spot-check needs
        (miServer Feature 036: the seeded sample and the turn transcript), and the plain reason when not —
        call this before start_probe_spot_check."""
        return await client.get("/probe-monitors/spot-checks/availability")

    @mcp.tool()
    async def list_probe_judge_runs(
        probe_id: Annotated[Optional[str], Field(description="Only judge runs compared against this probe (pm_…)")] = None,
    ) -> Any:
        """List judge baseline runs with status, progress, parse failures and metrics. A run
        whose parse failures exceed its limit FAILS rather than reporting metrics over the rows
        the judge happened to answer.

        A live run's `progress_detail` says how far it has got: `rows_done` of `rows_total` (every
        set, capped at max_rows_per_set each), the current `set_name` with `set_rows_done` of
        `set_rows_total`, `rate_per_second`, `eta_seconds` (`eta_provisional: true` while it rests
        on under 25 rows or two minutes) and `parse_failures` so far. Updated every few seconds;
        poll this rather than reading the endpoint's request log.

        `metrics.per_set.<id>.at_probe_fpr` reads the judge at the probe's calibrated false-positive
        rate. `reachable: false` (recall null) means the judge cannot run that strictly — a yes/no
        judge has one operating point — and `reason` states its own rate and recall instead; it
        is not a recall of 0. `metrics.rows_judged` is every row a judge call was made for."""
        return await client.get("/probe-monitors/judge-runs", probe_id=probe_id)

    @mcp.tool()
    async def cancel_probe_judge_run(
        judge_run_id: Annotated[str, Field(description="Judge run id to stop")],
    ) -> Any:
        """Stop a judge baseline — every remaining row is a call to the served judge model, so
        stopping one nobody is waiting for saves real work. Cooperative; 409 when it already
        finished."""
        return await client.post(f"/probe-monitors/judge-runs/{judge_run_id}/cancel")
