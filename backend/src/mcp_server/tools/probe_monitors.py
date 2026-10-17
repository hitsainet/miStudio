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
capability is there. Scope here is exactly the five routes 033 phases 2–4 added plus 032's two
readers.

WHAT AN AGENT NEEDS TO KNOW, AND WHY IT IS IN THE DESCRIPTIONS RATHER THAN A GUIDE: a probe is a
CORRELATIONAL readout with an evidence rung, and the rung is the whole point. An agent that exports
a rung-1 probe and serves it as a monitor has made a claim the evidence does not support, so the
acknowledgement parameter says so in as many words.
"""

from typing import Annotated, Any, Dict, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ..client import MiStudioClient
from ..config import MCPSettings

#: Repeated on the two tools that take it, because an agent reading one tool's description does not
#: see the other's.
RUNG_NOTE = (
    "A probe's `rung` is what its evidence supports: 0 trained only, 1 detects on held-out data, "
    "2 detects on unseen tasks, 3 compared against an LLM judge. Below rung 2 an export is REFUSED "
    "unless `acknowledge_below_rung2` carries a reason, which is then written into the exported "
    "definition so whoever serves it can see the claim was made on a judgement"
)


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def list_probe_monitors(
        run_id: Annotated[Optional[str], Field(description="Only probes from this run (pmr_…)")] = None,
        selected_only: Annotated[bool, Field(description="Only the probe each run selected as its best by validation AUROC")] = False,
    ) -> Any:
        """List trained probe monitors with their rung, threshold and validation metrics.

        The per-epoch training curve is NOT in this listing — it is a few hundred entries per probe
        and this payload is polled. `get_probe_monitor_report` returns it in full.
        """
        # ⚠ QUERY PARAMETERS ARE KEYWORD ARGUMENTS, NOT A `params=` DICT. `MiStudioClient.get`
        # takes `**params` and strips the Nones itself, so `params={...}` would send a query
        # field literally named "params" — caught by the payload half of the reachability
        # assertion, which is why that assertion checks the payload and not just the path.
        return await client.get(
            "/probe-monitors/probes", run_id=run_id, selected_only=selected_only or None
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
        """
        return await client.post(
            f"/probe-monitors/probes/{probe_id}/recalibrate",
            json_body={
                "target_fpr": target_fpr,
                "preview": preview,
                "reason": reason,
                "allow_fire_on_nothing": allow_fire_on_nothing,
            },
        )

    @mcp.tool()
    async def export_probe_definition(
        probe_id: Annotated[str, Field(description="Probe id (pm_…) whose definition has already been built")],
    ) -> Any:
        """Fetch the built definition as JSON.

        409 when none has been built, or when a previously built one was INVALIDATED — which happens
        whenever the probe's rung or threshold changes, because the cached document states both and
        a consumer has no way to tell it has gone stale. Build again.
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
