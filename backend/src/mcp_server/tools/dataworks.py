"""
miDataworks proxy (category: dataworks) — find data, inspect a version and its handoff manifest,
label it, publish it, send a detector set to miStudio, follow the job (Feature 034, FR-5–FR-14).

A CURATED SUBSET, not a mirror: these thirteen tools are the loop an agent working in miStudio
needs. Everything else is on miDataworks' own MCP server.

Rules every tool here keeps:

1. **miDataworks holds every approval** (C6). A gated action — any Hugging Face push (publishing,
   a detector-set send), deleting a version, labelling more than miDataworks' agent window — is
   answered `202` with `{approval_id, status: "pending", action, request_digest, expires_at,
   hint}`. That answer comes back UNCHANGED; the operator approves in miDataworks, which then runs
   the stored request exactly once. Poll with `dataworks_get_approval_status`. Nothing here
   approves, skips an approval, counts rows to decide one, or re-submits after one; nothing here
   writes to miStudio's own approval queue.
2. **"Who" is the header, never an argument.** Every request carries
   `X-Dataworks-Agent: agent:mistudio-mcp`; miDataworks records that and ignores any body value.
3. **No credentials.** The Hugging Face token stays in miDataworks' worker.
4. **Arguments are checked before the gate**, so a payload error is never reported as
   "miDataworks is down". When it IS down: `{"unavailable": "dataworks", "reason": …}` — report
   it, do not loop.
5. **Errors are miDataworks' own** `{code, message, details}`, verbatim.
"""

from typing import Annotated, Any, Literal, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ..context import ExternalContext
from ..dataworks_client import DataworksClient
from ..health_gate import HealthGate, gated

PRODUCT = "dataworks"
POLL_HINT = (
    "miDataworks holds this for the operator's approval and runs it ONCE when approved. Poll "
    "dataworks_get_approval_status(approval_id); do not re-submit."
)

#: Option keys `dataworks_start_label_run` refuses: they are set by the tool's own arguments, or
#: are "who" (the header carries it; miDataworks ignores a body value anyway).
_RESERVED_RUN_KEYS = frozenset({"input_version_id", "role", "who", "requested_by"})


def with_poll_hint(result: Any) -> Any:
    """A pending-approval answer passes through unchanged, gaining a `hint` only if it has none."""
    if isinstance(result, dict) and result.get("approval_id") and result.get("status") == "pending":
        if not result.get("hint"):
            return {**result, "hint": POLL_HINT}
    return result


def register(
    mcp: FastMCP, dw: DataworksClient, gate: HealthGate, ctx: ExternalContext
) -> None:
    async def _gate() -> Optional[dict]:
        ok, reason = await gate.check(PRODUCT)
        return None if ok else {"unavailable": PRODUCT, "reason": reason}

    # ── read ──────────────────────────────────────────────────────────────────────────────

    @mcp.tool()
    @gated(gate, PRODUCT)
    async def dataworks_list_datasets(
        q: Annotated[Optional[str], Field(description="Filter by name", max_length=100)] = None,
        page: Annotated[int, Field(description="Page number, from 1", ge=1)] = 1,
        limit: Annotated[int, Field(description="Items per page", ge=1, le=200)] = 50,
    ) -> Any:
        """miDataworks' datasets, paged — step 1 of the loop: find the data. Inspect a version
        with dataworks_get_version."""
        return await dw.get("/api/v1/datasets", q=q, page=page, limit=limit)

    @mcp.tool()
    @gated(gate, PRODUCT)
    async def dataworks_get_version(
        version_id: Annotated[str, Field(description="Version id", min_length=1)],
    ) -> Any:
        """One dataset version: state, row counts per split, recipe, labelers and caveats.
        Inspect it before labelling, publishing or registering it in miStudio."""
        return await dw.get(f"/api/v1/versions/{version_id}")

    @mcp.tool()
    @gated(gate, PRODUCT)
    async def dataworks_get_version_manifest(
        version_id: Annotated[str, Field(description="Version id", min_length=1)],
    ) -> Any:
        """The version's handoff manifest (`midataworks.dataset-version/v1`) — the document
        miStudio verifies when a probe dataset is registered with `dataset_version_manifest`
        (split row counts, version identity, publication). Returned verbatim."""
        return await dw.get(f"/api/v1/versions/{version_id}/handoff-manifest")

    @mcp.tool()
    @gated(gate, PRODUCT)
    async def dataworks_list_detector_sets(
        archived: Annotated[Optional[bool], Field(description="True for archived sets only; omit for active")] = None,
        page: Annotated[int, Field(description="Page number, from 1", ge=1)] = 1,
        limit: Annotated[int, Field(description="Items per page", ge=1, le=200)] = 50,
    ) -> Any:
        """Detector sets — the four-role bundles (train, in-distribution test, out-of-
        distribution eval, calibration negatives) a probe is trained and judged on."""
        return await dw.get("/api/v1/detector-sets", archived=archived, page=page, limit=limit)

    @mcp.tool()
    @gated(gate, PRODUCT)
    async def dataworks_get_detector_set(
        set_id: Annotated[str, Field(description="Detector set id", min_length=1)],
    ) -> Any:
        """One detector set as miDataworks reports it: its roles and the version behind each."""
        return await dw.get(f"/api/v1/detector-sets/{set_id}")

    # ── act ───────────────────────────────────────────────────────────────────────────────

    @mcp.tool()
    async def dataworks_start_label_run(
        input_version_id: Annotated[str, Field(description="The version to label", min_length=1, max_length=64)],
        role: Annotated[Literal["classifier", "judge", "probe", "features"], Field(description="Which labeller: a classifier, an LLM judge, a miLLM probe, or a feature-tag run")],
        options: Annotated[Optional[dict[str, Any]], Field(description="The rest of miDataworks' label-run request, verbatim: template_id, rubric_id, question, field_map, probe, features, thresholds, sampling, …")] = None,
    ) -> Any:
        """Start labelling a version. Over miDataworks' agent window (5,000 rows per version
        per 24 h) it is held for the operator's approval: the answer is `{approval_id,
        status: "pending", …}` — poll dataworks_get_approval_status. miDataworks counts the
        rows, not this tool. Returns the run (or the pending approval); follow the job with
        dataworks_job_status."""
        reserved = sorted(_RESERVED_RUN_KEYS & set(options or {}))
        if reserved:
            return {"error": "invalid_options",
                    "message": f"options may not set {reserved}: input_version_id and role are "
                               "this tool's own arguments, and miDataworks takes 'who' from the "
                               "agent header."}
        unavailable = await _gate()
        if unavailable:
            return unavailable
        body = {**(options or {}), "input_version_id": input_version_id, "role": role}
        return with_poll_hint(await dw.post("/api/v1/label-runs", json_body=body))

    @mcp.tool()
    @gated(gate, PRODUCT)
    async def dataworks_build_version_files(
        version_id: Annotated[str, Field(description="The version to build files for", min_length=1)],
        label_column: Annotated[Optional[str], Field(description="Which label column the files carry; omit for the version's own", max_length=200)] = None,
    ) -> Any:
        """Build the version's Parquet files (one per split) — the step before publishing.
        Returns `{build_id, job_id, reused, status}`; pass `build_id` to
        dataworks_publish_version once the build's job has finished. A matching existing build
        is reused (no new job). Not gated."""
        body = {"label_column": label_column} if label_column is not None else {}
        return await dw.post(f"/api/v1/versions/{version_id}/publish-builds", json_body=body)

    @mcp.tool()
    @gated(gate, PRODUCT)
    async def dataworks_publish_version(
        version_id: Annotated[str, Field(description="The version to publish", min_length=1, max_length=64)],
        build_id: Annotated[str, Field(description="The build_id dataworks_build_version_files returned", min_length=1, max_length=40)],
        repo_id: Annotated[str, Field(description="Hugging Face dataset repository, owner/name")],
        card_prose: Annotated[str, Field(description="Your prose for the dataset card; miDataworks writes the facts", max_length=100_000)] = "",
        visibility: Annotated[Literal["private", "public"], Field(description="Repository visibility; private unless you say otherwise")] = "private",
    ) -> Any:
        """Publish a built version to the Hugging Face Hub. ALWAYS held for the operator's
        approval in miDataworks: the answer is `{approval_id, status: "pending",
        request_digest, …}`. miDataworks runs it once when approved; poll
        dataworks_get_approval_status and never re-submit."""
        body = {"version_id": version_id, "build_id": build_id, "repo_id": repo_id,
                "card_prose": card_prose, "visibility": visibility}
        return with_poll_hint(await dw.post("/api/v1/publishes", json_body=body))

    @mcp.tool()
    @gated(gate, PRODUCT)
    async def dataworks_send_detector_set(
        set_id: Annotated[str, Field(description="The detector set to send", min_length=1)],
        repositories: Annotated[Optional[dict[str, str]], Field(description="version id → owner/name repository; versions not named get <namespace>/<dataset>-v<n>")] = None,
        namespace: Annotated[Optional[str], Field(description="Hub namespace for versions not named in repositories")] = None,
        visibility: Annotated[Literal["private", "public"], Field(description="Repository visibility")] = "private",
    ) -> Any:
        """Send a detector set to miStudio: publish each version to the Hub, download it into
        miStudio and register every role. **ONE approval for the whole send**, which
        miDataworks binds to a digest of exactly what will be published and registered (P-06):
        the answer is `{approval_id, status: "pending", request_digest, …}`. miDataworks runs
        the stored send once when approved; this tool never re-submits. Poll
        dataworks_get_approval_status, then follow the job."""
        body: dict[str, Any] = {"repositories": dict(repositories or {}),
                                "visibility": visibility}
        if namespace is not None:
            body["namespace"] = namespace
        return with_poll_hint(
            await dw.post(f"/api/v1/detector-sets/{set_id}/send", json_body=body)
        )

    @mcp.tool()
    async def dataworks_delete_version(
        version_id: Annotated[str, Field(description="The version to delete", min_length=1)],
        reason: Annotated[str, Field(description="Why — recorded on the tombstone (1-2000 chars)", min_length=1, max_length=2000)],
    ) -> Any:
        """Delete (tombstone) a version. ALWAYS held for the operator's approval in
        miDataworks; the version's record and manifest outlive its rows. The reason is
        recorded."""
        if not reason.strip():
            return {"error": "invalid_reason", "message": "a deletion needs a reason"}
        unavailable = await _gate()
        if unavailable:
            return unavailable
        return with_poll_hint(
            await dw.delete(f"/api/v1/versions/{version_id}", json_body={"reason": reason})
        )

    # ── jobs and approvals ───────────────────────────────────────────────────────────────────

    @mcp.tool()
    @gated(gate, PRODUCT)
    async def dataworks_job_status(
        job_id: Annotated[str, Field(description="Job id from an act tool", min_length=1)],
    ) -> Any:
        """A miDataworks job's state, progress and result. Every act tool starts one."""
        return await dw.get(f"/api/v1/jobs/{job_id}")

    @mcp.tool()
    @gated(gate, PRODUCT)
    async def dataworks_cancel_job(
        job_id: Annotated[str, Field(description="The job to cancel", min_length=1)],
    ) -> Any:
        """Cancel a job you started. Cancellation is cooperative: poll dataworks_job_status for
        the job to reach `cancelled`."""
        return await dw.post(f"/api/v1/jobs/{job_id}/cancel")

    @mcp.tool()
    @gated(gate, PRODUCT)
    async def dataworks_get_approval_status(
        approval_id: Annotated[str, Field(description="The approval_id from a pending answer", min_length=1)],
    ) -> Any:
        """Where a held action stands, verbatim from miDataworks: `pending`, `executing`,
        `executed` (with `result_kind`, `result_id`), `failed` (with `error`), `rejected` (with
        `reason`) or `expired` (unanswered for 24 hours — request it again). Only the operator
        decides; an agent cannot approve or reject."""
        return await dw.get(f"/api/v1/approvals/{approval_id}")
