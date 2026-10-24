"""Behaviour catalogue tools (category: probe_monitors) — Feature 037 FR-14; ENH-005 D20–D27.

Registered beside `probe_monitors` and `judge_prompts` in `tools/__init__.CATEGORY_MODULES`, so the
category's deployment switches cover them. Pure HTTP through the miStudio client (IDL-54; memory:
mcp-server-is-a-separate-deployment). Every catalogue route is called by exactly one tool here, except the
two the operator keeps (`test_mcp_route_parity` records why): deleting a behaviour, and confirming that a
view's positive labels are a behaviour (D48).

WHAT AN AGENT NEEDS TO KNOW, in the descriptions because that is what an agent reads:

* a behaviour KEY (`<behaviour-key>`) is already written on judge prompts, runs and probes; the catalogue RECORDS what
  it means. Cataloguing a key changes no probe, run or prompt — they are adopted by key;
* who asked is recorded from this server's `X-MiStudio-Agent` header ("agent:mcp"), so these writes are
  attributed to the agent in the behaviour's history;
* no approval gate: catalogue writes are metadata, not GPU or served-model work (FTDD 037 D-T3). The
  rules still hold — a probe can be primary only with a matching key and a confirmed training view.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import MiStudioClient
from ..config import MCPSettings

KEY = Annotated[str, Field(description="Behaviour key: lower-case words joined by hyphens, e.g. '<behaviour-key>'")]


def _definition(name, positive_meaning, description, negative_meaning, target_fpr) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    for field, value in (
        ("name", name),
        ("positive_meaning", positive_meaning),
        ("description", description),
        ("negative_meaning", negative_meaning),
        ("target_fpr", target_fpr),
    ):
        if value is not None:
            body[field] = value
    return body


# ── Feature 040 FR-1: the detector fields (shapes checked by the server, which names the field and
# the rule it breaks; this module stays pure HTTP and imports no backend schema). ──────────────────
SUB_QUESTIONS = Annotated[Optional[List[Dict[str, Any]]], Field(description=(
    "Sub-questions of the behaviour: [{key, question, positive_meaning}]; `key` is a slug like "
    "'<sub-question-key>', unique in the version"))]
SUCCESS_CRITERIA = Annotated[Optional[List[Dict[str, Any]]], Field(description=(
    "The bar the detector must meet: [{id, detector, subset, held_out_test?, target_fpr?, min_recall, "
    "probe_id?}]. detector: rules_or_blend | blend_alone | best_single | rules_alone | named_probe "
    "(named_probe needs probe_id). subset: all | rule_invisible_positives | rule_visible_positives | "
    "{column, value}. held_out_test names a column in held_out_columns (absent: judged in distribution). "
    "A criterion without target_fpr inherits the version's"))]
HELD_OUT_COLUMNS = Annotated[Optional[List[str]], Field(description=(
    "Column names that define 'held out' for this behaviour, e.g. ['<column>']"))]
MODELS = Annotated[Optional[List[Dict[str, Any]]], Field(description=(
    "Models the behaviour is judged on: [{model_id}] (a miStudio model id, or a Hugging Face repo id not "
    "yet downloaded, which saves with a warning). Absent: every model with a probe of the behaviour"))]
CALIBRATION_FLOOR = Annotated[Optional[int], Field(description=(
    "Fewest calibration negatives a calibration of this behaviour may use; overrides the deployment's "
    "setting"), ge=1)]
RULE_SET = Annotated[Optional[Dict[str, Any]], Field(description=(
    "The behaviour's rule set: a miDataworks reference {name, version, digest} (fetched once now), or "
    "{document: <midataworks.rule-set/v1 document>} inline. Stored and reported 'unvalidated' until "
    "the rule engine is available"))]
DETECTOR_FIELD_NAMES = ("sub_questions", "success_criteria", "held_out_columns", "models",
                        "calibration_floor", "rule_set")


def _with_detector_fields(body: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    for field in DETECTOR_FIELD_NAMES:
        if values.get(field) is not None:
            body[field] = values[field]
    return body


def register(mcp: MCPServer, client: MiStudioClient, settings: MCPSettings) -> None:  # noqa: ARG001
    @mcp.tool()
    async def list_behaviours(
        include_uncatalogued: Annotated[bool, Field(description="Also list keys in use that have no catalogue record")] = True,
        include_archived: Annotated[bool, Field(description="Also list archived behaviours")] = False,
        limit: Annotated[Optional[int], Field(description="Behaviours per page (default: the deployment's page size, 100)", ge=1)] = None,
        cursor: Annotated[Optional[str], Field(description="`next_cursor` of the previous page; valid only with the same filters")] = None,
        q: Annotated[Optional[str], Field(description="Only behaviours whose key or name contains this text")] = None,
        status: Annotated[Optional[Literal["draft", "validated", "monitoring", "unknown", "archived"]], Field(description="Only behaviours with this computed status")] = None,
    ) -> Any:
        """List the behaviour catalogue, one page at a time, in key order: every catalogued behaviour (key,
        name, meaning, version, archived, how many prompts/runs/probes/views use its key) and every
        UNCATALOGUED key already in use, with its counts. Follow `next_cursor` until it is null; `total` is
        the number of matches (`total_exact` false when a status filter could not count them all). An
        uncatalogued key is valid; cataloguing it records its meaning."""
        return await client.get(
            "/probe-monitors/behaviours",
            include_uncatalogued=include_uncatalogued,
            include_archived=include_archived,
            limit=limit,
            cursor=cursor,
            q=q,
            status=status,
        )

    @mcp.tool()
    async def get_behaviour(key: KEY) -> Any:
        """One behaviour: its current meaning and every judge prompt, run, probe (with its role: primary,
        candidate or retired) and dataset view that uses its key."""
        return await client.get(f"/probe-monitors/behaviours/{key}")

    @mcp.tool()
    async def get_behaviour_checklist(key: KEY) -> Any:
        """The behaviour's eight-step checklist (defined, data, trained, rung 2, judge prompt, rung 3,
        exported, armed in miLLM), each step done / not_done / unknown with its reason and the ids that
        make it so, plus the computed status (draft, validated, monitoring, unknown, archived).
        "unknown" means miLLM could not say — it is not "not done"."""
        return await client.get(f"/probe-monitors/behaviours/{key}/checklist")

    @mcp.tool()
    async def list_behaviour_versions(key: KEY) -> Any:
        """Every version of a behaviour's definition text, newest first. Versions never change; an exported
        probe states the version current when it was exported."""
        return await client.get(f"/probe-monitors/behaviours/{key}/versions")

    @mcp.tool()
    async def get_behaviour_history(
        key: KEY,
        limit: Annotated[int, Field(description="Entries per page", ge=1, le=500)] = 100,
        offset: Annotated[int, Field(description="Entries to skip", ge=0)] = 0,
    ) -> Any:
        """The behaviour's append-only history: role changes, view links, confirmations, probe key changes,
        archive and delete — who did each (operator or agent), when, and the old and new value."""
        return await client.get(f"/probe-monitors/behaviours/{key}/history", limit=limit, offset=offset)

    @mcp.tool()
    async def create_behaviour(
        key: KEY,
        name: Annotated[str, Field(description="What a person reads, e.g. '<behaviour label>'")],
        positive_meaning: Annotated[str, Field(description="What counts as POSITIVE for this behaviour, in plain words")],
        description: Annotated[Optional[str], Field(description="Longer notes")] = None,
        negative_meaning: Annotated[Optional[str], Field(description="What counts as negative")] = None,
        target_fpr: Annotated[Optional[float], Field(description="Target false-positive rate for its probes, e.g. 0.01", gt=0, lt=1)] = None,
        sub_questions: SUB_QUESTIONS = None,
        success_criteria: SUCCESS_CRITERIA = None,
        held_out_columns: HELD_OUT_COLUMNS = None,
        models: MODELS = None,
        calibration_floor: CALIBRATION_FLOOR = None,
        rule_set: RULE_SET = None,
    ) -> Any:
        """Catalogue a behaviour key with version 1 of its meaning and, optionally, its detector data
        (sub-questions, success criteria, held-out columns, models, calibration floor, rule set). Prompts,
        runs and probes already carrying the key are adopted unchanged. Refused (409) when the key is
        already catalogued — use new_behaviour_version. A refusal names the field and the rule; things
        that may not exist yet (a model not downloaded, a column no view carries) come back as `warnings`,
        each with a `next` call."""
        body = _definition(name, positive_meaning, description, negative_meaning, target_fpr)
        body["key"] = key
        _with_detector_fields(body, {
            "sub_questions": sub_questions, "success_criteria": success_criteria,
            "held_out_columns": held_out_columns, "models": models,
            "calibration_floor": calibration_floor, "rule_set": rule_set,
        })
        return await client.post("/probe-monitors/behaviours", json_body=body)

    @mcp.tool()
    async def new_behaviour_version(
        key: KEY,
        base_version: Annotated[Optional[int], Field(description="The version this change is against (the current one). Given: every field you leave out carries over from it. Omitted: the call replaces every field, and name and positive_meaning are required", ge=1)] = None,
        name: Annotated[Optional[str], Field(description="Name in this version")] = None,
        positive_meaning: Annotated[Optional[str], Field(description="What counts as positive, in this version")] = None,
        description: Annotated[Optional[str], Field(description="Longer notes")] = None,
        negative_meaning: Annotated[Optional[str], Field(description="What counts as negative")] = None,
        target_fpr: Annotated[Optional[float], Field(description="Target false-positive rate", gt=0, lt=1)] = None,
        sub_questions: SUB_QUESTIONS = None,
        success_criteria: SUCCESS_CRITERIA = None,
        held_out_columns: HELD_OUT_COLUMNS = None,
        models: MODELS = None,
        calibration_floor: CALIBRATION_FLOOR = None,
        rule_set: RULE_SET = None,
        remove_fields: Annotated[Optional[List[Literal["description", "negative_meaning", "target_fpr", "sub_questions", "success_criteria", "held_out_columns", "models", "calibration_floor", "rule_set"]]], Field(description="With base_version: fields to REMOVE in the new version (sent as null)")] = None,
    ) -> Any:
        """A new version of a behaviour's definition. With `base_version` (the current version), send only
        what changes: everything else carries over, `remove_fields` drops fields, and the result carries a
        field-level `diff`. A base that is no longer current is refused `stale_base_version` naming the
        current one. Without `base_version`, the call replaces every field (name and positive_meaning
        required). Earlier versions stay readable; probes already exported keep the version they were
        exported with."""
        body = _definition(name, positive_meaning, description, negative_meaning, target_fpr)
        if base_version is not None:
            body["base_version"] = base_version
        _with_detector_fields(body, {
            "sub_questions": sub_questions, "success_criteria": success_criteria,
            "held_out_columns": held_out_columns, "models": models,
            "calibration_floor": calibration_floor, "rule_set": rule_set,
        })
        for field in remove_fields or []:
            body[field] = None
        return await client.post(f"/probe-monitors/behaviours/{key}/versions", json_body=body)

    @mcp.tool()
    async def set_behaviour_probe_role(
        key: KEY,
        probe_id: Annotated[str, Field(description="Probe id (pm_…) whose own behaviour key is `key`")],
        role: Annotated[str, Field(description="'primary', 'candidate' or 'retired'")],
        reason: Annotated[Optional[str], Field(description="Why, kept in the history")] = None,
    ) -> Any:
        """Set a probe's role in its behaviour. At most one primary per behaviour. A probe can be made
        primary only when its key equals the behaviour and its training view's positive labels have been
        confirmed as this behaviour (by the operator, or by a miDataworks send naming it)."""
        body: Dict[str, Any] = {"role": role}
        if reason is not None:
            body["reason"] = reason
        return await client.put(f"/probe-monitors/behaviours/{key}/probes/{probe_id}/role", json_body=body)

    @mcp.tool()
    async def link_behaviour_view(
        key: KEY,
        dataset_id: Annotated[str, Field(description="Probe dataset view id (pmd_…)")],
        use: Annotated[Optional[str], Field(description="'train', 'eval' or 'calibration'; default the view's own role")] = None,
        reason: Annotated[Optional[str], Field(description="Why, kept in the history")] = None,
    ) -> Any:
        """Link a probe dataset view to a behaviour explicitly. A view whose own behaviour is the key is
        already listed and needs no link."""
        body: Dict[str, Any] = {}
        if use is not None:
            body["use"] = use
        if reason is not None:
            body["reason"] = reason
        return await client.post(f"/probe-monitors/behaviours/{key}/views/{dataset_id}", json_body=body)

    @mcp.tool()
    async def unlink_behaviour_view(
        key: KEY,
        dataset_id: Annotated[str, Field(description="Probe dataset view id (pmd_…)")],
    ) -> Any:
        """Remove an explicit view link (the history keeps it)."""
        return await client.delete(f"/probe-monitors/behaviours/{key}/views/{dataset_id}")

    @mcp.tool()
    async def archive_behaviour(
        key: KEY,
        reason: Annotated[Optional[str], Field(description="Why, kept in the history")] = None,
    ) -> Any:
        """Archive a behaviour: hidden from the list, roles frozen. Every record keeps its key and its
        probes still export it."""
        return await client.post(
            f"/probe-monitors/behaviours/{key}/archive",
            json_body={"reason": reason} if reason is not None else {},
        )

    @mcp.tool()
    async def unarchive_behaviour(key: KEY) -> Any:
        """Undo archive_behaviour."""
        return await client.post(f"/probe-monitors/behaviours/{key}/unarchive")
