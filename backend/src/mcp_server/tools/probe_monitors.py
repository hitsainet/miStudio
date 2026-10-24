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

import inspect
from typing import Annotated, Any, Dict, List, Literal, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from ..patterns import HEX64
from ..client import BackendError, MiStudioClient
from ..config import MCPSettings

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
    "approval request in miStudio's banner and returns `{approval_request_id, status: "
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
    """) + "\n\n" + APPROVAL_NOTE + "."

EVALUATE_DESCRIPTION = inspect.cleandoc("""Evaluate an existing probe on more sets — a GPU job;
    returns a task id (202).

    This is how a probe climbs the ladder: rung 1 needs an in-distribution held-out set, rung 2
    an out-of-distribution one. Refused (409) when the probe's recorded precision or render no
    longer matches the model. Read per-row scores afterwards with `get_probe_evaluation_scores`.
    """) + "\n\n" + APPROVAL_NOTE + "."

SCORE_BATCH_DESCRIPTION = inspect.cleandoc("""Score several inputs offline with one probe: one
    queued forward pass PER INPUT.

    Returns `{"tasks": [{"index", "task_id"}], "submitted"}`; collect each with
    `get_probe_score_result`. ONE request queues them all; the cap is the deployment's setting
    (`probe_score_batch_cap`), enforced by miStudio — over it the call is refused naming the
    setting. Each input claims a place in the GPU queue: for thousands of rows, register a dataset
    view and use `evaluate_probe_monitor`, or score on the served model with `millm_score_probes`.
    """)


START_JUDGE_DESCRIPTION = inspect.cleandoc("""Queue the LLM-monitor BASELINE: an LLM judge
    labels the same eval rows the probe was measured on, so the probe's AUROC has something to be
    compared with (rung 3 says "compared with a judge", not "beats one").

    The judge's instructions are a version from the judge-prompt library (`prompt_id`; see
    `list_judge_prompts`), recorded on the run with its hash. Rung 3 counts the run only when the
    prompt's behaviour equals the probe's (`counts_toward_rung` in the answer says which). CPU in
    miStudio, but every row is a call to the served judge model: thousands of calls for a full run.
    `max_rows_per_set` bounds them. Cancel with `cancel_probe_judge_run`.
    """) + "\n\nWhen the judge endpoint is miLLM: " + APPROVAL_NOTE + "."


def judge_endpoint_is_millm(endpoint: str, millm_api_url: str) -> bool:
    """Whether a judge endpoint is miLLM: the configured MILLM_API_URL's host, or a host named for
    it. FAILS CLOSED on the name — a judge reached through another miLLM hostname is still the
    GPU that serves miLLM's users."""
    from urllib.parse import urlparse

    host = (urlparse(endpoint).hostname or "").lower()
    if not host:
        return False
    configured = (urlparse(millm_api_url).hostname or "").lower() if millm_api_url else ""
    return (configured != "" and host == configured) or "millm" in host


class ProbeScoreInput(BaseModel):
    """One input for `score_probe_monitor_batch`: exactly one of `text` or `messages`."""

    model_config = ConfigDict(extra="forbid")

    text: Optional[str] = Field(default=None, max_length=16384)
    messages: Optional[List[Dict[str, str]]] = Field(default=None, min_length=1)

#: Repeated on the two tools that take it, because an agent reading one tool's description does not
#: see the other's.
RUNG_NOTE = (
    "A probe's `rung` is what its evidence supports: 0 trained only, 1 detects on held-out data, "
    "2 detects on unseen tasks, 3 compared against an LLM judge. Below rung 2 an export is REFUSED "
    "unless `acknowledge_below_rung2` carries a reason, which is then written into the exported "
    "definition so whoever serves it can see the claim was made on a judgement"
)


def register(mcp: MCPServer, client: MiStudioClient, settings: MCPSettings) -> None:
    async def _approval_gate(
        tool_name: str, payload: Dict[str, Any], approval_request_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """`None` when the job may be submitted now; otherwise the structured answer to return.

        The `millm_load_model` shape (caller-executed approval, Feature 034 FR-20): the first
        call files a request; a re-call with its id CONSUMES it — one conditional UPDATE in
        miStudio, so an approval is spent once and only on the payload the operator saw.
        """
        if not settings.probe_gpu_approval:
            return None
        if approval_request_id is None:
            created = await client.post(
                "/mcp/approvals", json_body={"tool_name": tool_name, "payload": payload}
            )
            return {
                "approval_request_id": created["id"],
                "status": "pending_approval",
                "hint": (
                    f"{tool_name} holds a GPU, so it needs operator approval in miStudio's "
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
    ) -> Any:
        """One probe's full report: per-set AUROCs with confidence intervals, the rung WITH ITS
        WORDING and what would raise it, the dense/SAE counterpart, any judge runs, and the
        training curve.

        READ THE RUNG BEFORE THE AUROC. A high AUROC on the set a probe was fitted near says little;
        the rung says which claim the numbers support.

        `probe` carries the identity of the built definition (`detector_digest`, `decision_digest`,
        `decision_revision`, `identity_current`, `identity_stale_reason`, `identity_source`); a
        definition built before digests has its identity recomputed on this first read.
        """
        return await client.get(f"/probe-monitors/probes/{probe_id}")

    @mcp.tool()
    async def build_probe_definition(
        probe_id: Annotated[str, Field(description="Probe id (pm_…)")],
        acknowledge_below_rung2_reason: Annotated[Optional[str], Field(description=f"Required to export a probe below rung 2, minimum 10 characters. {RUNG_NOTE}")] = None,
        vector_count: Annotated[int, Field(description="Test vectors to score into the definition, 8–32 (default 16). They are real forward passes, so this is a GPU job", ge=8, le=32)] = 16,
        seed: Annotated[int, Field(description="Sampling seed. The same seed gives the same rows, so two builds of one probe differ only if the implementation did")] = 1337,
    ) -> Any:
        """Build a portable `mistudio.probe-definition/v1` for a probe. Returns a task id (202).

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
    ) -> Any:
        """Move a probe's decision threshold to another false-positive budget. No GPU, no queue.

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
        `millm_copies` (miLLM's imported copies, or why none can be named) and `next`: the calls
        that bring every copy up to the new decision — rebuild, re-publish, and one
        `millm_recalibrate_probe` per imported copy, each runnable as it stands (an argument only you
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
        return await client.post(
            f"/probe-monitors/probes/{probe_id}/recalibrate", json_body=body,
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
    async def register_probe_dataset(
        name: Annotated[str, Field(description="A name for this labelled view", min_length=1, max_length=255)],
        dataset_id: Annotated[str, Field(description="miStudio dataset id (UUID) of a DOWNLOADED dataset")],
        input_column: Annotated[str, Field(description="Column holding the text (or chat) to score")],
        label_column: Annotated[str, Field(description="Column holding the raw label")],
        label_mapping: Annotated[Dict[str, Literal["positive", "negative", "excluded"]], Field(description="EVERY distinct raw label value (as a string) -> positive | negative | excluded. A train/eval view needs both classes; a calibration view must map nothing to positive")],
        role: Annotated[Literal["train", "eval", "calibration", "blend_fit"], Field(description="train: what a run fits on; eval: measured on (rung 1-2); calibration: negatives that place the threshold, NOT labelled for the concept; blend_fit: labelled data a blend of probes is fitted on (never a run's training view)")] = "eval",
        split: Annotated[Optional[str], Field(description="Dataset split. REQUIRED when a dataset_version_manifest is given")] = None,
        config: Annotated[Optional[str], Field(description="Dataset config/subset name, if the dataset has several")] = None,
        keyword_filter: Annotated[Optional[Dict[str, Any]], Field(description="{terms: [..], mode: any|all, case_sensitive}. NARROWS the set; it never assigns a label")] = None,
        pair_column: Annotated[Optional[str], Field(description="Column pairing rows (minimal pairs); optional")] = None,
        distribution: Annotated[Optional[Literal["in_distribution", "out_of_distribution"]], Field(description="EVAL views only: whether the set is out-of-distribution for the concept — rung 2 turns on it")] = None,
        dataset_version_manifest: Annotated[Optional[Dict[str, Any]], Field(description="A miDataworks `midataworks.dataset-version` manifest (dataworks_get_version_manifest returns it). Validated against the vendored schema; the split's row count must equal the downloaded split's, or the view is REFUSED")] = None,
        behaviour: Annotated[Optional[str], Field(description="The behaviour key this view's POSITIVE class is, e.g. '<behaviour-key>'")] = None,
        group_columns: Annotated[Optional[List[str]], Field(description="Grouping columns of the split to declare now (read and snapshotted), so evaluations on this view can be broken down by them")] = None,
        sub_question: Annotated[Optional[str], Field(description="A sub-question key of the behaviour version this view cites (needs behaviour)")] = None,
        behaviour_version: Annotated[Optional[int], Field(description="The behaviour version sub_question is checked against; default the current one", ge=1)] = None,
    ) -> Any:
        """Register a label-mapped VIEW over a downloaded dataset — what a probe run trains on,
        is evaluated on, or calibrates its threshold from.

        The class counts are computed NOW and stored; a view that cannot be scored (no positives
        or no negatives after mapping and filtering) is refused here, naming both counts.

        **The miDataworks path (Feature 034):** pass the version's handoff manifest as
        `dataset_version_manifest` (from `dataworks_get_version_manifest`) together with the
        `split`. miStudio validates it and REFUSES the view (409) when the manifest's row count
        for that split differs from what was downloaded — so a probe is never trained on rows
        other than the ones the manifest describes. The verified facts are stored on the view as
        `dataset_version`.
        """
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
                           ("sub_question", sub_question), ("behaviour_version", behaviour_version)):
            if value is not None:
                body[key] = value
        return await client.post("/probe-monitors/datasets", json_body=body)

    @mcp.tool()
    async def list_probe_datasets(
        role: Annotated[Optional[Literal["train", "eval", "calibration"]], Field(description="Only views of this role")] = None,
        include_archived: Annotated[bool, Field(description="Also list ARCHIVED views (hidden by default)")] = False,
    ) -> Any:
        """List registered probe dataset views, newest first, with their class counts and
        role. A `train` view is what a run fits on; `eval` views are what rungs 1-2 are measured
        on; a `calibration` view supplies the negatives that place the threshold. Archived views
        are hidden unless `include_archived`; an archived view cannot be used by a new run."""
        return await client.get(
            "/probe-monitors/datasets", role=role, include_archived=include_archived or None
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
        operator's alone, from miStudio's Probes screen."""
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
        model_id: Annotated[str, Field(description="miStudio model id (m_xxxxxxxx)")],
        train_dataset_id: Annotated[str, Field(description="A probe dataset view with role 'train'")],
        eval_dataset_ids: Annotated[Optional[List[str]], Field(description="Eval views to measure every probe on. Never the training view (that would measure memorisation) and never a calibration view")] = None,
        calibration_dataset_id: Annotated[Optional[str], Field(description="A calibration view whose negatives place the threshold at config.target_fpr")] = None,
        config: Annotated[Optional[Dict[str, Any]], Field(description="ProbeRunConfig, passed verbatim: layers | stride, rules, scope, max_length, val_fraction, seed, top_n_layers, rolling_windows, sae_variant, sae_k, target_fpr, calibration_max_rows, batch_size. Omit for the defaults")] = None,
        gpu: Annotated[str, Field(description=RUN_GPU_PARAMETER)] = "auto",
        approval_request_id: Annotated[Optional[str], Field(description="The id a first call returned with status pending_approval; pass it once the operator approved")] = None,
        behaviour: Annotated[Optional[str], Field(description="What the probes detect, as a short key (lower-case words joined by hyphens, e.g. '<behaviour-key>'). Rung 3 counts only judge runs whose prompt judges this same key; omit it and no judge run can raise these probes past rung 2 until it is set with set_probe_behaviour")] = None,
        behaviour_label: Annotated[Optional[str], Field(description="The behaviour in words (the label a person reads), e.g. '<behaviour label>'")] = None,
        selection_rule: Annotated[Optional[Literal["auroc", "recall_at_decision"]], Field(description="How the run picks its selected probe: recall at each probe's calibrated decision on the out-of-distribution sets (the default), or the older AUROC ranking. Merged into config")] = None,
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
        held = await _approval_gate("start_probe_training_run", payload, approval_request_id)
        if held is not None:
            return held
        return await client.post("/probe-monitors/runs", json_body=payload)

    @mcp.tool()
    async def list_probe_training_runs(
        status: Annotated[Optional[str], Field(description="pending | running | completed | failed | cancelled")] = None,
        train_dataset_id: Annotated[Optional[str], Field(description="Only runs whose TRAINING set is this probe dataset view")] = None,
        limit: Annotated[int, Field(description="At most this many, newest first (1-200)", ge=1, le=200)] = 50,
    ) -> Any:
        """List probe training runs, newest first, with status and stage. Filter by
        `train_dataset_id` to find every run trained on one view (035 XR-4) — the listing is
        capped at 200, so an unfiltered scan misses older runs."""
        return await client.get(
            "/probe-monitors/runs", status=status, train_dataset_id=train_dataset_id, limit=limit
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
        until the run reports; rate and ETA stay null until measured.

        `environment.gpu_memory` (runs from 2026-10-21) is the GPU memory each stage took:
        `stages.<name>.peak_allocated_mib` / `peak_reserved_mib`, the run's overall peaks and the
        card's `gpu_uuid` — size the next run from it. `measured: false` (null peaks) means the
        model ran on no CUDA device. Judge runs call an LLM endpoint and use no miStudio GPU."""
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
        miLLM keep serving there; this removes miStudio's record and evidence for them."""
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
        dataset_ids: Annotated[List[str], Field(description="Eval probe dataset views to measure the probe on — an out-of-distribution set is how rung 2 is reached", min_length=1)],
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
        # unbounded; miStudio now refuses over the runtime setting and the refusal is passed through.
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
        endpoint: Annotated[str, Field(description="OpenAI-compatible base URL of the LLM judge (e.g. a miLLM /v1)")],
        model: Annotated[str, Field(description="The judge model name at that endpoint")],
        dataset_ids: Annotated[List[str], Field(description="Eval probe dataset views to judge", min_length=1)],
        probe_id: Annotated[Optional[str], Field(description="The probe to compare against (pm_…) — what lets the probe reach rung 3, when the prompt judges the probe's behaviour")] = None,
        prompt_id: Annotated[Optional[str], Field(description="The judge prompt version to use (jp_…, from list_judge_prompts). Omit for stakes-rating/v1, the prompt every run used before the library. Choose one whose behaviour matches the probe's, or the run is a baseline that never raises the rung")] = None,
        max_rows_per_set: Annotated[int, Field(description="Rows judged per set (20-50000); each is one judge call", ge=20, le=50000)] = 2000,
        parse_failure_limit: Annotated[Optional[float], Field(description="Fail the run when more than this fraction of judge answers cannot be parsed (0-1, exclusive); omit for the default", gt=0.0, lt=1.0)] = None,
        approval_request_id: Annotated[Optional[str], Field(description="The id a first call returned with status pending_approval (only when the judge is served by miLLM); pass it once the operator approved")] = None,
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
        # ⚠ GATED WHEN THE JUDGE IS miLLM (2026-10-21; it was not gated at all). Every row is a
        # generation on the GPU that serves miLLM's users, and a judge run asks miLLM to load the
        # judge model if it is not loaded — evicting what is served. Same caller-executed approval
        # as a probe training run, under the same switch (MCP_PROBE_GPU_APPROVAL).
        if judge_endpoint_is_millm(endpoint, settings.millm_api_url):
            held = await _approval_gate("start_probe_judge_run", body, approval_request_id)
            if held is not None:
                return held
        return await client.post("/probe-monitors/judge-runs", json_body=body)

    @mcp.tool()
    async def start_probe_spot_check(
        behaviour: Annotated[str, Field(description="Catalogued behaviour key to audit, e.g. '<behaviour-key>'")],
        prompt_id: Annotated[str, Field(description="Judge prompt version (jp_…) whose behaviour IS this behaviour")],
        endpoint: Annotated[str, Field(description="OpenAI-compatible base URL of the judge (e.g. a miLLM /v1)")],
        model: Annotated[str, Field(description="The judge model name at that endpoint")],
        size: Annotated[int, Field(description="Recorded turns to sample (1-500), split over fired / near threshold / not fired", ge=1, le=500)] = 100,
        window: Annotated[str, Field(description="Which probe window to sample on: all, prompt, response or last_user")] = "all",
        seed: Annotated[Optional[int], Field(description="Seed for the sample; the same seed and spec give the same turns", ge=0)] = None,
        approval_request_id: Annotated[Optional[str], Field(description="The id a first call returned with status pending_approval (only when the judge is miLLM); pass it once the operator approved")] = None,
    ) -> Any:
        """Spot-check a behaviour's live probes against a library judge (Feature 038): miLLM returns a seeded,
        stratified sample of RECORDED session turns (transcripts are opt-in in miLLM, so a short sample is
        reported, never padded), the judge is shown the window each probe scored, and the report gives a
        2 × 2 table, agreement with a confidence interval and kappa per probe. The judge is a second
        opinion, not ground truth. Refused (503) when miLLM is not connected or lacks its Feature 036
        session routes. When the judge is miLLM it waits for operator approval first: live chats queue
        behind every judge call."""
        body: Dict[str, Any] = {
            "behaviour": behaviour, "prompt_id": prompt_id, "endpoint": endpoint, "model": model,
            "size": size, "window": window,
        }
        if seed is not None:
            body["seed"] = seed
        # D34: gated exactly like start_probe_judge_run when the judge is served by miLLM.
        # Removing this gate fails test_mcp_spot_check_gate.py.
        if judge_endpoint_is_millm(endpoint, settings.millm_api_url):
            held = await _approval_gate("start_probe_spot_check", body, approval_request_id)
            if held is not None:
                return held
        return await client.post("/probe-monitors/spot-checks", json_body=body)

    @mcp.tool()
    async def get_probe_spot_check(
        spot_check_id: Annotated[str, Field(description="Spot-check id (psc_…)")],
    ) -> Any:
        """One spot-check: status, progress, the sample spec and what miLLM returned (ids only), and the
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
        """Whether miLLM is connected and serves the recorded-session routes a spot-check needs
        (miLLM Feature 036: the seeded sample and the turn transcript), and the plain reason when not —
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
        `set_rows_total`, `rate_per_second`, `eta_seconds` and `parse_failures` so far. Updated
        every few seconds; poll this rather than reading the endpoint's request log.

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
