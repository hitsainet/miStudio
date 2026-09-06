"""Labeling write-back tools (category: labeling) — Feature 010 provenance rules."""

from typing import Annotated, Any, List, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def update_feature_label(
        feature_id: Annotated[str, Field(description="Feature row id from search_features/get_feature_groups")],
        name: Annotated[Optional[str], Field(description="Human-readable name")] = None,
        category: Annotated[Optional[str], Field(description="Filter by label category")] = None,
        description: Annotated[Optional[str], Field(description="Longer free-text description")] = None,
        notes: Annotated[Optional[str], Field(description="Free-text evidence notes stored with the label")] = None,
        override_protected: Annotated[bool, Field(description="Overwrite an aqua-starred (completed) label. The previous label is NOT recoverable")] = False,
    ) -> Any:
        """Update a feature's label. Writes carry label_source='mcp_agent' provenance.

        Aqua-starred features hold protected (completed enhanced) labels: editing
        their name/category/description returns 409 PROTECTED_LABEL unless
        override_protected=true — only override with strong steering evidence.
        Convention: append evidence to notes as
        '[MCP <date>] evidence: experiment <id> — <one-line summary>'.
        """
        body: dict[str, Any] = {"label_source": "mcp_agent", "override_protected": override_protected}
        if name is not None:
            body["name"] = name
        if category is not None:
            body["category"] = category
        if description is not None:
            body["description"] = description
        if notes is not None:
            body["notes"] = notes
        return await client.patch(f"/features/{feature_id}", json_body=body)

    @mcp.tool()
    async def run_enhanced_labeling(feature_id: Annotated[str, Field(description="Feature row id from search_features/get_feature_groups")]) -> Any:
        """Trigger two-pass enhanced LLM labeling for one feature (background job;
        uses the labeling backend configured in Settings). Poll get_enhanced_label."""
        return await client.post(f"/features/{feature_id}/label/enhanced", json_body={})

    @mcp.tool()
    async def get_enhanced_label(feature_id: Annotated[str, Field(description="Feature row id from search_features/get_feature_groups")]) -> Any:
        """Latest enhanced-labeling job + synthesized label for a feature."""
        return await client.get(f"/features/{feature_id}/label/enhanced/latest")

    # ── prompt-template optimization ─────────────────────────────────────────

    @mcp.tool()
    async def list_labeling_templates(
        search: Annotated[Optional[str], Field(description="Free-text filter over template name/description")] = None,
        limit: Annotated[int, Field(description="Max rows to return")] = 50,
    ) -> Any:
        """List labeling prompt templates — the variable a trial tests.

        Use the returned ids with run_labeling_trial. A template flagged
        is_detection_template is a SCORING template and is refused as a trial
        subject: it is the ruler, not the thing being measured.
        """
        return await client.get(
            "/labeling-prompt-templates", search=search, limit=min(limit, 100))

    @mcp.tool()
    async def run_labeling_trial(
        extraction_job_id: Annotated[str, Field(description="Extraction whose features form the panel")],
        feature_ids: Annotated[List[str], Field(description="The fixed panel — 1 to 200 feature ids, all from this extraction")],
        prompt_template_id: Annotated[Optional[str], Field(description="Template to test; omit to use the default template")] = None,
        name: Annotated[Optional[str], Field(description="Short label for this run, e.g. 'baseline' or 'v2-negatives'")] = None,
        labeling_method: Annotated[str, Field(description="'openai', 'openai_compatible' or 'local'")] = "openai_compatible",
        openai_compatible_endpoint: Annotated[Optional[str], Field(description="OpenAI-compatible endpoint URL including /v1")] = None,
        openai_compatible_model: Annotated[Optional[str], Field(description="Model name at that endpoint")] = None,
    ) -> Any:
        """Run ONE prompt template over a fixed feature panel.

        **NO FEATURE ROW IS WRITTEN.** This is a measurement, not a labeling run —
        the labels it produces live only in the trial record, so running several
        template variants cannot overwrite the labels being compared against.
        Contrast update_feature_label above, which does persist.

        Panel identity is content-addressed from (extraction, sorted feature ids),
        so two trials over the same panel are comparable by construction and
        compare_labeling_trials refuses a mismatched pair. Returns a
        trial_run_id; poll get_labeling_trial.
        """
        body: dict = {
            "extraction_job_id": extraction_job_id,
            "feature_ids": feature_ids,
            "labeling_method": labeling_method,
        }
        if prompt_template_id is not None:
            body["prompt_template_id"] = prompt_template_id
        if name is not None:
            body["name"] = name
        if openai_compatible_endpoint is not None:
            body["openai_compatible_endpoint"] = openai_compatible_endpoint
        if openai_compatible_model is not None:
            body["openai_compatible_model"] = openai_compatible_model
        return await client.post("/labeling/trials", json_body=body)

    @mcp.tool()
    async def get_labeling_trial(
        trial_run_id: Annotated[str, Field(description="Trial id (ltr_xxxxxxxxxxxx) from run_labeling_trial")],
    ) -> Any:
        """One trial's full record: the frozen template, the panel, every label.

        The template body is stored as a FROZEN COPY, not a reference — templates
        are editable, so a run holding only an id would silently re-describe
        itself if someone tuned the template mid-experiment.
        """
        return await client.get(f"/labeling/trials/{trial_run_id}")

    @mcp.tool()
    async def list_labeling_trials(
        extraction_job_id: Annotated[Optional[str], Field(description="Filter to one extraction")] = None,
        panel_id: Annotated[Optional[str], Field(description="Filter to one panel — the way to find every variant tested on it")] = None,
        prompt_template_id: Annotated[Optional[str], Field(description="Filter to one template")] = None,
        limit: Annotated[int, Field(description="Max rows to return")] = 50,
    ) -> Any:
        """List trials. Filter by panel_id to find every variant run on one panel."""
        return await client.get(
            "/labeling/trials", extraction_job_id=extraction_job_id,
            panel_id=panel_id, prompt_template_id=prompt_template_id,
            limit=min(limit, 100))

    @mcp.tool()
    async def compare_labeling_trials(
        run_a: Annotated[str, Field(description="Baseline trial id")],
        run_b: Annotated[str, Field(description="Candidate trial id")],
    ) -> Any:
        """Compare two trials over the SAME panel, per feature.

        Refuses rather than guesses. Two runs over different panels return 409 —
        comparing them would produce a number that looks like a template
        difference and is not one. Zero overlapping features returns no verdict:
        comparing nothing is not comparing. If every overlapping feature errored
        in one arm the verdict is 'inconclusive', never 'identical' — failed
        labels stringify the same way and would otherwise read as agreement.
        """
        return await client.get(f"/labeling/trials/compare/{run_a}/{run_b}")
