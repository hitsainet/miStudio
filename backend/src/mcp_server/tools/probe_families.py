"""Probe-family and blend tools (category: probe_monitors) — Feature 041; FTDD §5.6.

Pure HTTP through the miStudio client (IDL-54): this module imports nothing from the backend, because the
`mistudio-mcp` pod has no database (miStudio memory `mcp-server-is-a-separate-deployment`; guarded by
`test_mcp_tools_stay_on_http.py` and `test_mcp_server_starts_without_the_backend.py`).

Every result and refusal is data: a `code`, the reasons as fields, and a `next` list of calls an agent can
run as they stand. Descriptions use `<behaviour-key>` placeholders only (no behaviour is named here).
"""

from typing import Annotated, Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import MiStudioClient
from ..config import MCPSettings

KEY = Annotated[str, Field(description="Behaviour key, e.g. '<behaviour-key>'")]


def register(mcp: MCPServer, client: MiStudioClient, settings: MCPSettings) -> None:  # noqa: ARG001
    @mcp.tool()
    async def get_probe_family(
        key: KEY,
        model: Annotated[Optional[str], Field(description="miStudio model id (m_…) whose members to list; required when the behaviour has probes on more than one model (refused model_required, listing them)")] = None,
        member_ids: Annotated[Optional[List[str]], Field(description="Explicit members (probe ids, any role, one model) instead of the default family")] = None,
        include_retired: Annotated[bool, Field(description="Also list retired members")] = False,
        offset: Annotated[int, Field(description="First member to return", ge=0)] = 0,
        limit: Annotated[Optional[int], Field(description="Members per page (default and most: the deployment setting)", ge=1)] = None,
    ) -> Any:
        """A behaviour's probe family on one model: its primary and candidate probes (retired on request)
        with each member's facts (precision, render form, scope, layer, rule, sub-question, rung,
        threshold, decision revision, digests, training dataset version, whether a definition is built),
        flags where a member differs from the family's majority, and coverage of every sub-question the
        current behaviour version declares — a `missing` one carries the `next` call that trains it
        (approval-gated) or registers its training view first. Read-only; nothing is started."""
        return await client.get(
            f"/probe-monitors/behaviours/{key}/family", model=model, member_ids=member_ids,
            include_retired=include_retired, offset=offset, limit=limit,
        )

    # ── S2: blend evaluation (CPU on stored evidence; no approval gate) ────────────────────────

    @mcp.tool()
    async def evaluate_probe_blends(
        behaviour_key: KEY,
        model: Annotated[Optional[str], Field(description="miStudio model id (m_…); required when the behaviour has probes on more than one")] = None,
        dataset_version: Annotated[Optional[str], Field(description="A miDataworks dataset-version id: the blend_fit, test and calibration views are resolved from the views registered from it")] = None,
        member_ids: Annotated[Optional[List[str]], Field(description="Explicit members (probe ids) instead of the default family (primary + candidates)")] = None,
        views: Annotated[Optional[Dict[str, Any]], Field(description="Named views instead of resolution: {blend_fit: view id, tests: [view ids], calibration: view id}")] = None,
        candidates: Annotated[Optional[List[str]], Field(description="Candidate names or kinds (stack, max, any_of, all_of, k_of_n, best_single, k_of_n:<k>, and rules_or_<name> when the version has a rule set); default all")] = None,
        folds: Annotated[Optional[int], Field(description="Cross-validation folds (default: the deployment setting)", ge=2)] = None,
        group_column: Annotated[Optional[str], Field(description="A declared column of the blend_fit view whose groups never straddle folds (default: the first declared held-out column)")] = None,
        target_fpr: Annotated[Optional[float], Field(description="False-positive rate each candidate is calibrated to (default: the behaviour's)", gt=0, lt=1)] = None,
        seed: Annotated[int, Field(description="Seed of the folds and fits; the same request and seed give the same bytes", ge=0)] = 0,
        calibration_sample_rows: Annotated[Optional[int], Field(description="The calibration sample size every member was evaluated at (default: the behaviour's calibration floor, else the deployment default)", ge=100, le=200000)] = None,
    ) -> Any:
        """Evaluate blends of a behaviour's probe family the way they would be deployed: fit on the
        blend_fit view (grouped cross-validation), calibrate each candidate on the shared calibration
        sample, and measure it on every test view, ranked by the worst held-out recall. Returns at once
        with `report_id`, the resolved members, views and candidates; poll `get_blend_report`. Refused at
        the call when a member lacks stored scores (`member_scores_missing` lists the exact
        `evaluate_probe_monitor` calls — never started here), when members differ in precision, render or
        scope, or when the blend_fit view overlaps a training or test view. A blend has no rung."""
        body: Dict[str, Any] = {"behaviour_key": behaviour_key, "seed": seed}
        for name, value in (("model", model), ("dataset_version", dataset_version), ("member_ids", member_ids),
                            ("views", views), ("candidates", candidates), ("folds", folds),
                            ("group_column", group_column), ("target_fpr", target_fpr),
                            ("calibration_sample_rows", calibration_sample_rows)):
            if value is not None:
                body[name] = value
        return await client.post("/probe-monitors/blend-reports", json_body=body)

    @mcp.tool()
    async def get_blend_report(
        report_id: Annotated[str, Field(description="Blend report id (pbr_…)")],
    ) -> Any:
        """One blend report: status (queued, running, done, failed, cancelled), `stale` with the reasons
        (computed on read: a member re-cut, retired or deleted, a new behaviour version) and the
        `rerun_blend_report` call, and — when done — the report: ranked candidates with in-fold and test
        figures, Wilson intervals, per-subgroup figures, success-criterion verdicts and the members' own
        figures. `has_rung` is false: each armed probe keeps its own rung."""
        return await client.get(f"/probe-monitors/blend-reports/{report_id}")

    @mcp.tool()
    async def list_blend_reports(
        behaviour_key: KEY,
        model: Annotated[Optional[str], Field(description="Only reports on this model")] = None,
        offset: Annotated[int, Field(description="First report to return", ge=0)] = 0,
        limit: Annotated[Optional[int], Field(description="Reports per page (default and most: the deployment setting)", ge=1)] = None,
    ) -> Any:
        """A behaviour's blend reports, newest first, each with its status and `stale`."""
        return await client.get("/probe-monitors/blend-reports", behaviour_key=behaviour_key, model=model,
                                offset=offset, limit=limit)

    @mcp.tool()
    async def get_blend_report_scores(
        report_id: Annotated[str, Field(description="Blend report id (pbr_…)")],
        candidate: Annotated[str, Field(description="A candidate name of the report (e.g. stack, k_of_n:3)")],
        view_id: Annotated[str, Field(description="One of the report's views")],
        offset: Annotated[int, Field(description="First row", ge=0)] = 0,
        limit: Annotated[int, Field(description="Rows per page", ge=1)] = 10000,
    ) -> Any:
        """One candidate's per-row values on one view, in split-position order: {position, value, verdict,
        label}. Refused `rows_pruned` after the operator pruned the report's rows."""
        return await client.get(f"/probe-monitors/blend-reports/{report_id}/scores", candidate=candidate,
                                view_id=view_id, offset=offset, limit=limit)

    @mcp.tool()
    async def rerun_blend_report(
        report_id: Annotated[str, Field(description="Blend report id (pbr_…)")],
    ) -> Any:
        """Run a report's request again against today's members and definition; the new report and the old
        one link to each other. The old report's bytes never change."""
        return await client.post(f"/probe-monitors/blend-reports/{report_id}/rerun")

    @mcp.tool()
    async def cancel_blend_evaluation(
        report_id: Annotated[str, Field(description="Blend report id (pbr_…)")],
    ) -> Any:
        """Stop a queued or running blend evaluation (it stops between folds or candidates; its inputs are
        kept)."""
        return await client.post(f"/probe-monitors/blend-reports/{report_id}/cancel")

    @mcp.tool()
    async def set_suggested_blend(
        key: KEY,
        model: Annotated[str, Field(description="miStudio model id (m_…)")],
        report_id: Annotated[str, Field(description="A done, current blend report of this behaviour and model")],
        candidate: Annotated[str, Field(description="The candidate of that report to suggest")],
        reason: Annotated[Optional[str], Field(description="Why, in your words (kept in the history)")] = None,
    ) -> Any:
        """Record ADVICE for an application: which blend of armed probes to use, sliced from a report. It
        binds nothing — miLLM never reads it and no probe changes; the application blends. Refused for a
        stale or unfinished report. Who marked it and why are kept in an append-only history."""
        body: Dict[str, Any] = {"model": model, "report_id": report_id, "candidate": candidate}
        if reason is not None:
            body["reason"] = reason
        return await client.put(f"/probe-monitors/behaviours/{key}/suggested-blend", json_body=body)

    @mcp.tool()
    async def get_suggested_blend(
        key: KEY,
        model: Annotated[str, Field(description="miStudio model id (m_…)")],
        include_history: Annotated[bool, Field(description="Also return the history of markings, newest first")] = False,
    ) -> Any:
        """The current suggested blend of a behaviour on a model (advisory; `stale` when its report is), and
        with `include_history` every earlier marking with who made it and why."""
        if include_history:
            return await client.get(f"/probe-monitors/behaviours/{key}/suggested-blend/history", model=model)
        return await client.get(f"/probe-monitors/behaviours/{key}/suggested-blend", model=model)

    # ── S3: family scoring, export and publish ────────────────────────────────────────────────

    def _score_body(model, member_ids, inputs, blend_report_id, candidates, include_traces) -> Dict[str, Any]:
        body: Dict[str, Any] = {"inputs": inputs, "include_traces": include_traces}
        for name, value in (("model", model), ("member_ids", member_ids), ("blend_report_id", blend_report_id),
                            ("candidates", candidates)):
            if value is not None:
                body[name] = value
        return body

    @mcp.tool()
    async def score_probe_family(
        key: KEY,
        text: Annotated[Optional[str], Field(description="Plain text to score (max 16384 chars). Exactly one of text or messages", max_length=16384)] = None,
        messages: Annotated[Optional[List[Dict[str, str]]], Field(description="A chat [{role, content}]. Exactly one of text or messages")] = None,
        model: Annotated[Optional[str], Field(description="miStudio model id; required when the behaviour has probes on more than one")] = None,
        member_ids: Annotated[Optional[List[str]], Field(description="Explicit members instead of the default family (or the report's members)")] = None,
        blend_report_id: Annotated[Optional[str], Field(description="A done blend report whose candidates to apply")] = None,
        candidates: Annotated[Optional[List[str]], Field(description="Candidates of that report to apply (their recorded weights, k and threshold)")] = None,
        include_traces: Annotated[bool, Field(description="Also return each member's per-token scores")] = False,
    ) -> Any:
        """Score ONE input with the whole family — a queued forward pass with ONE model load; returns a task
        id. Collect with `get_probe_family_score_result`: each member's score and verdict, each named blend
        candidate's value and verdict (with the report's recorded parameters, a rules-or candidate at its
        union bar), and the behaviour's rule-set verdict (rule_match, rule_veto, unknown with its reason;
        no_rule_set when the version has none)."""
        if (text is None) == (messages is None):
            return {"error": "exactly_one_input", "message": "give exactly one of text or messages"}
        item = {"text": text} if text is not None else {"messages": messages}
        return await client.post(f"/probe-monitors/behaviours/{key}/family/score-batch",
                                 json_body=_score_body(model, member_ids, [item], blend_report_id, candidates,
                                                       include_traces))

    @mcp.tool()
    async def score_probe_family_batch(
        key: KEY,
        inputs: Annotated[List[Dict[str, Any]], Field(description="Inputs, each {text} or {messages}; members x inputs is capped by the deployment setting", min_length=1)],
        model: Annotated[Optional[str], Field(description="miStudio model id; required when the behaviour has probes on more than one")] = None,
        member_ids: Annotated[Optional[List[str]], Field(description="Explicit members instead of the default family")] = None,
        blend_report_id: Annotated[Optional[str], Field(description="A done blend report whose candidates to apply")] = None,
        candidates: Annotated[Optional[List[str]], Field(description="Candidates of that report to apply")] = None,
        include_traces: Annotated[bool, Field(description="Also return each member's per-token scores")] = False,
    ) -> Any:
        """Score several inputs with the family: one queued task per input (an input is never half
        scored). Over the cap (members x inputs) the call is refused naming both numbers and the setting."""
        return await client.post(f"/probe-monitors/behaviours/{key}/family/score-batch",
                                 json_body=_score_body(model, member_ids, list(inputs), blend_report_id, candidates,
                                                       include_traces))

    @mcp.tool()
    async def get_probe_family_score_result(
        key: KEY,
        task_id: Annotated[str, Field(description="A task id from score_probe_family(_batch)")],
    ) -> Any:
        """Collect a family score. SUCCESS carries every member's score, the candidates' values and verdicts;
        FAILURE carries the reason; PENDING means queued or unknown. A task of another behaviour is 404."""
        return await client.get(f"/probe-monitors/behaviours/{key}/family/score/{task_id}")

    @mcp.tool()
    async def export_probe_family(
        key: KEY,
        model: Annotated[Optional[str], Field(description="miStudio model id; required when the behaviour has probes on more than one")] = None,
        member_ids: Annotated[Optional[List[str]], Field(description="Explicit members instead of the default family")] = None,
    ) -> Any:
        """Export the family, ALL OR NOTHING: each member's built definition with its single-export route,
        sha256 and size, and the archive's route (a ZIP of the untouched files, for the operator). If any member
        is blocked — no current built definition, a missing file, colliding names, members on different
        models — nothing is exported and every blocker is listed with the call that clears it."""
        return await client.get(f"/probe-monitors/behaviours/{key}/family/export-index", model=model,
                                member_ids=member_ids)

    @mcp.tool()
    async def publish_probe_family(
        key: KEY,
        repo_id: Annotated[str, Field(description="Hugging Face repository (owner/name)", min_length=1, max_length=200)],
        model: Annotated[Optional[str], Field(description="miStudio model id; required when the behaviour has probes on more than one")] = None,
        member_ids: Annotated[Optional[List[str]], Field(description="Explicit members instead of the default family")] = None,
        private: Annotated[bool, Field(description="Create the repository private")] = True,
    ) -> Any:
        """Publish every member's definition in ONE Hugging Face commit (manifest merged, never replaced; one
        README section per probe). All or nothing, as export_probe_family. The token is NOT a parameter: it is
        resolved server-side from Settings → API Keys and checked before anything is queued."""
        body: Dict[str, Any] = {"repo_id": repo_id, "private": private}
        if model is not None:
            body["model"] = model
        if member_ids is not None:
            body["member_ids"] = member_ids
        return await client.post(f"/probe-monitors/behaviours/{key}/family/publish", json_body=body)
