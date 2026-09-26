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
