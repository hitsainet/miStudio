"""Behaviour-detector report tools (category: probe_monitors) — Feature 040 S2, FR-39; FTDD §6.7.

Pure HTTP through the miStudio client (IDL-54): this module imports nothing from the backend, because the
`mistudio-mcp` pod has no database (miStudio memory `mcp-server-is-a-separate-deployment`). Every tool is
CPU-only reading of stored evidence — none calls `_approval_gate`; the only GPU work an agent is pointed
at (`evaluate_probe_monitor`, a `recalibrate_probe_monitor` re-cut) stays behind the existing gate.

Every result and refusal is data: a `code` or `verdict`, the reasons as fields, and a `next` list of
calls that advance or unblock it (FR-40). Descriptions use `<behaviour-key>` placeholders only.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import MiStudioClient
from ..config import MCPSettings

KEY = Annotated[str, Field(description="Behaviour key, e.g. '<behaviour-key>'")]


def register(mcp: MCPServer, client: MiStudioClient, settings: MCPSettings) -> None:  # noqa: ARG001
    @mcp.tool()
    async def update_probe_dataset_columns(
        dataset_id: Annotated[str, Field(description="Probe dataset view id (pmd_…)")],
        group_columns: Annotated[Optional[List[str]], Field(description="EVERY grouping column the view should declare: the ones already declared plus the new ones (additive — a removal or rename is refused column_not_additive)", min_length=1)] = None,
        text_column: Annotated[Optional[str], Field(description="The column rules read when it differs from the view's input column (declared once)")] = None,
        rule_columns: Annotated[Optional[Dict[str, str]], Field(description="Producer rule verdicts the view carries: {rule_match: <column>, rule_veto?: <column>, rule_set_id: '<name>@<version>', rule_set_digest: <sha256>} — the set must be in the view's manifest rule-set listing (declared once)")] = None,
    ) -> Any:
        """Declare columns on a registered view: grouping columns (for example a kind or source column) so its
        evaluations can be broken down by them; the text column rules read; producer rule verdict columns to
        compare with miStudio's (never preferred). Additive. New columns are read from the downloaded split
        and snapshotted by row position; a column with too many distinct values (a row id) is refused."""
        body: Dict[str, Any] = {}
        if group_columns is not None:
            body["group_columns"] = list(group_columns)
        if text_column is not None:
            body["text_column"] = text_column
        if rule_columns is not None:
            body["rule_columns"] = dict(rule_columns)
        if not body:
            return {"error": "send group_columns, text_column or rule_columns"}
        return await client.patch(f"/probe-monitors/datasets/{dataset_id}/columns", json_body=body)

    @mcp.tool()
    async def get_probe_breakdown(
        probe_id: Annotated[str, Field(description="Probe id (pm_…)")],
        evaluation_id: Annotated[str, Field(description="Evaluation id of this probe (from get_probe_monitor_report)")],
        by_column: Annotated[str, Field(description="A grouping column declared on the evaluated view")],
        filters: Annotated[Optional[Dict[str, str]], Field(description="Only rows whose declared column equals the value, {column: value}")] = None,
        mode: Annotated[Literal["recorded", "target_fpr"], Field(description="recorded: the decision a deployment applies (window bar, window bands, probe bands, global bar per row). target_fpr: a bar cut on the CALIBRATION negatives at target_fpr")] = "recorded",
        target_fpr: Annotated[Optional[float], Field(description="With mode=target_fpr: the false-positive rate to cut at", gt=0, lt=1)] = None,
        min_rows: Annotated[Optional[int], Field(description="Groups with fewer rows are flagged too_small (default: the deployment setting)", ge=1)] = None,
        limit: Annotated[Optional[int], Field(description="Groups per page", ge=1)] = None,
        cursor: Annotated[Optional[str], Field(description="next_cursor of the previous page")] = None,
    ) -> Any:
        """Recall and false-positive rate of one evaluation, per value of one declared column, at the
        decision a deployment applies — from stored arrays, no GPU. Small groups are shown and flagged
        too_small, never dropped; the body names its inputs (inputs_digest), the decision revision and
        how many rows each decision rule covered."""
        params: Dict[str, Any] = {"by_column": by_column, "mode": mode, "target_fpr": target_fpr,
                                  "min_rows": min_rows, "limit": limit, "cursor": cursor}
        if filters:
            params["filter"] = [f"{column}={value}" for column, value in filters.items()]
        return await client.get(f"/probe-monitors/probes/{probe_id}/evaluations/{evaluation_id}/breakdown",
                                **params)

    @mcp.tool()
    async def get_detector_success_report(
        key: KEY,
        model_id: Annotated[Optional[str], Field(description="The model judged; default the only model with a probe of the behaviour")] = None,
        version: Annotated[Optional[int], Field(description="Behaviour version whose criteria are judged; default the current one", ge=1)] = None,
        dataset_version: Annotated[Optional[str], Field(description="A miDataworks dataset-version id: every view registered from it")] = None,
        view_ids: Annotated[Optional[List[str]], Field(description="Explicit views instead of a dataset version")] = None,
        probe_id: Annotated[Optional[str], Field(description="The probe judged; default the behaviour's primary, else the only matching run's selected probe")] = None,
        blend_report_id: Annotated[Optional[str], Field(description="A blend report (pbr_…) whose candidate judges the blend_alone criteria (with candidate); default the model's suggested blend")] = None,
        candidate: Annotated[Optional[str], Field(description="The candidate of blend_report_id")] = None,
    ) -> Any:
        """Does this behaviour's detector meet its own bar? Judges every success criterion of the version
        at the decision a deployment applies: per criterion the verdict (met, not_met, not_assessable with
        reasons), counts, Wilson interval and the calibration it rests on; one overall verdict. CPU only,
        no approval. Stored immutably: the same inputs return the same report id; `stale` says which input
        moved since."""
        body: Dict[str, Any] = {}
        for name, value in (("model_id", model_id), ("version", version), ("dataset_version", dataset_version),
                            ("view_ids", view_ids), ("probe_id", probe_id), ("blend_report_id", blend_report_id),
                            ("candidate", candidate)):
            if value is not None:
                body[name] = value
        return await client.post(f"/probe-monitors/behaviours/{key}/success-reports", json_body=body)

    @mcp.tool()
    async def list_detector_success_reports(
        key: KEY,
        limit: Annotated[Optional[int], Field(description="Reports per page", ge=1)] = None,
        cursor: Annotated[Optional[str], Field(description="next_cursor of the previous page")] = None,
    ) -> Any:
        """The behaviour's stored success reports, newest first: id, version judged, model, overall verdict."""
        return await client.get(f"/probe-monitors/behaviours/{key}/success-reports", limit=limit, cursor=cursor)

    @mcp.tool()
    async def read_detector_success_report(
        report_id: Annotated[str, Field(description="Success report id (dsr_…)")],
    ) -> Any:
        """One stored success report exactly as it was built, with `stale` computed now: which inputs moved
        (a newer behaviour version, a removed probe or view) and the call that rebuilds it."""
        return await client.get(f"/probe-monitors/success-reports/{report_id}")

    @mcp.tool()
    async def compare_detectors_across_models(
        key: KEY,
        version: Annotated[Optional[int], Field(description="Behaviour version; default the current one", ge=1)] = None,
        dataset_version: Annotated[Optional[str], Field(description="A miDataworks dataset-version id")] = None,
        view_ids: Annotated[Optional[List[str]], Field(description="Explicit views instead of a dataset version")] = None,
        limit: Annotated[Optional[int], Field(description="Models per page", ge=1)] = None,
        cursor: Annotated[Optional[str], Field(description="next_cursor of the previous page")] = None,
    ) -> Any:
        """The behaviour's detectors on each model side by side: one success report per model (the
        definition's models, else every model with a probe of it), criteria verdicts in adjacent columns.
        Models trained on different dataset versions are refused dataset_identity_mismatch."""
        return await client.get(f"/probe-monitors/behaviours/{key}/model-comparison", version=version,
                                dataset_version=dataset_version, view_ids=view_ids, limit=limit, cursor=cursor)
