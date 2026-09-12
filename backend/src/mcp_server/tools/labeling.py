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

    # ── coverage and resume ──────────────────────────────────────────────────

    @mcp.tool()
    async def get_labeling_coverage(
        extraction_job_id: Annotated[str, Field(description="Extraction id from list_extractions")],
        prompt_fingerprint: Annotated[Optional[str], Field(description="With judge_model, also counts verdicts from a DIFFERENT judge as stale. Get it from list_labeling_templates.")] = None,
        judge_model: Annotated[Optional[str], Field(description="The model that produced the existing verdicts, e.g. 'gemma-4-31b-GGUF:IQ4_XS'")] = None,
        resume_limit: Annotated[int, Field(description="How many ids to return for the next batch (max 2000)")] = 2000,
    ) -> Any:
        """What still needs labeling in an extraction, and why the failures failed.

        Returns counts by outcome, a breakdown of failure reasons with real
        example features, and the feature ids a resume would take next.

        `adjudicated` includes features the judge honestly found
        UNINTERPRETABLE. That is a verdict, not a gap, and re-deriving it costs
        ~8 s each for the same answer.

        Read `failure_reasons` before retrying in bulk: 14,560 failures is ~32
        GPU-hours, and the breakdown says in one line whether they are all one
        recoverable cause. `(no reason recorded)` means the failure predates
        per-feature error capture — sample before committing.
        """
        return await client.get(
            f"/labeling/{extraction_job_id}/coverage",
            prompt_fingerprint=prompt_fingerprint,
            judge_model=judge_model,
            resume_limit=min(resume_limit, 2000),
        )

    @mcp.tool()
    async def resume_labeling(
        extraction_job_id: Annotated[str, Field(description="Extraction id from list_extractions")],
        labeling_method: Annotated[str, Field(description="openai | openai_compatible | local")],
        limit: Annotated[int, Field(description="How many features this ONE batch labels (max 2000)")] = 2000,
        only: Annotated[Optional[str], Field(description="'failed' to retry failures only, 'pending' for never-attempted only. Omit for both.")] = None,
        openai_model: Annotated[Optional[str], Field(description="Model name for labeling_method=openai")] = None,
        openai_compatible_endpoint: Annotated[Optional[str], Field(description="Base URL for labeling_method=openai_compatible")] = None,
        openai_compatible_model: Annotated[Optional[str], Field(description="Model name for labeling_method=openai_compatible")] = None,
        local_model: Annotated[Optional[str], Field(description="Model name for labeling_method=local")] = None,
        prompt_template_id: Annotated[Optional[str], Field(description="Template id from list_labeling_templates")] = None,
    ) -> Any:
        """Label ONE batch of the features that still need it.

        Two calls: coverage names the ids, the panel route labels them. Nothing
        already adjudicated is touched.

        ONE BATCH. At ~8 s/feature a full 2000 is ~4.4 hours, and finishing a
        53k-feature extraction takes ~27 of these. Use
        start_labeling_resume_sweep for that rather than calling this in a loop.

        Take a sample first when retrying failures: `only='failed', limit=20`
        answers "do these still fail?" in about three minutes, against ~32
        GPU-hours for all of them.
        """
        coverage = await client.get(
            f"/labeling/{extraction_job_id}/coverage",
            resume_limit=min(limit, 2000),
            only=only,
        )
        feature_ids = coverage.get("resume_feature_ids") or []
        if not feature_ids:
            return {"labeled": 0, "reason": "nothing left to label", "coverage": coverage}

        body = {
            "extraction_job_id": extraction_job_id,
            "labeling_method": labeling_method,
            "feature_ids": feature_ids,
        }
        for key, value in (
            ("openai_model", openai_model),
            ("openai_compatible_endpoint", openai_compatible_endpoint),
            ("openai_compatible_model", openai_compatible_model),
            ("local_model", local_model),
            ("prompt_template_id", prompt_template_id),
        ):
            if value is not None:
                body[key] = value
        return await client.post("/labeling/panel", json_body=body)

    @mcp.tool()
    async def start_labeling_resume_sweep(
        extraction_job_id: Annotated[str, Field(description="Extraction id from list_extractions")],
        max_batches: Annotated[int, Field(description="REQUIRED ceiling. Each batch is ~4.4 GPU-hours at the default size; there is no unlimited option.")],
        labeling_method: Annotated[str, Field(description="openai | openai_compatible | local")],
        batch_size: Annotated[int, Field(description="Features per batch (max 2000)")] = 2000,
        openai_model: Annotated[Optional[str], Field(description="Model name for labeling_method=openai")] = None,
        openai_compatible_endpoint: Annotated[Optional[str], Field(description="Base URL for labeling_method=openai_compatible")] = None,
        openai_compatible_model: Annotated[Optional[str], Field(description="Model name for labeling_method=openai_compatible")] = None,
        local_model: Annotated[Optional[str], Field(description="Model name for labeling_method=local")] = None,
        prompt_template_id: Annotated[Optional[str], Field(description="Template id from list_labeling_templates")] = None,
        prompt_fingerprint: Annotated[Optional[str], Field(description="Pass BOTH this and judge_model, or neither. The fingerprint you sized the sweep against — from list_labeling_templates.")] = None,
        judge_model: Annotated[Optional[str], Field(description="Pass BOTH this and prompt_fingerprint, or neither. The judge you sized the sweep against.")] = None,
    ) -> Any:
        """Run several resume batches back to back, one Celery task each.

        `max_batches` IS REQUIRED AND HAS NO DEFAULT, here or anywhere below it.
        A sweep of 27 batches is roughly 59 GPU-hours; an agent must state how
        much it intends to spend rather than inherit an unbounded default.

        Check get_labeling_coverage first and divide `remaining` by `batch_size`
        to size it. Poll get_labeling_resume_sweep; stop it with
        cancel_labeling_resume_sweep, which is cooperative — the batch in flight
        finishes and nothing further is queued.

        IF YOU SIZED WITH A PREDICATE, PASS IT. `get_labeling_coverage` accepts
        `judge_model` and `prompt_fingerprint`, and `remaining` counts verdicts
        that are STALE for that judge and template as well as the untried ones.
        A sweep started without them selects only untried features — so it takes
        a different set than the one you were quoted, and after a template edit
        that set can be twenty times smaller. It finishes early having done
        almost nothing, or works the wrong backlog while the one you booked it
        for is never touched.

        BOTH OR NEITHER. A fingerprint alone marks every same-template/
        different-model verdict as fresh; a judge alone cannot see a template
        edit. Sending one makes the staleness test true for every row, which
        selects the entire extraction — the server refuses it rather than
        quietly booking that.

        The pair is FROZEN onto the sweep, so editing the template while it runs
        cannot change what the remaining batches select.
        """
        config: dict = {
            "extraction_job_id": extraction_job_id,
            "labeling_method": labeling_method,
        }
        for key, value in (
            ("openai_model", openai_model),
            ("openai_compatible_endpoint", openai_compatible_endpoint),
            ("openai_compatible_model", openai_compatible_model),
            ("local_model", local_model),
            ("prompt_template_id", prompt_template_id),
        ):
            if value is not None:
                config[key] = value
        body: dict = {
            "max_batches": max_batches,
            "batch_size": min(batch_size, 2000),
            "config": config,
        }
        # Sent at the TOP level, beside max_batches, not inside `config`: the
        # endpoint freezes them onto the sweep row itself, and burying them in
        # the judge config would make them look like judge settings.
        if prompt_fingerprint is not None:
            body["prompt_fingerprint"] = prompt_fingerprint
        if judge_model is not None:
            body["judge_model"] = judge_model
        return await client.post(
            f"/labeling/{extraction_job_id}/resume-sweep",
            json_body=body,
        )

    @mcp.tool()
    async def get_labeling_resume_sweep(
        sweep_id: Annotated[str, Field(description="Sweep id from start_labeling_resume_sweep")],
    ) -> Any:
        """How far a resume sweep has got.

        `features_labeled` and `features_failed` count what the batches actually
        WROTE, not how many batches ran — a sweep whose every batch failed does
        not look complete.
        """
        return await client.get(f"/labeling/resume-sweeps/{sweep_id}")

    @mcp.tool()
    async def cancel_labeling_resume_sweep(
        sweep_id: Annotated[str, Field(description="Sweep id from start_labeling_resume_sweep")],
    ) -> Any:
        """Stop a sweep after its current batch.

        Cooperative, and it has to be: the worker pool is solo, so a hard revoke
        signals a pool child that does not exist. The batch in flight finishes —
        up to ~4.4 hours — and nothing further is queued.
        """
        return await client.post(f"/labeling/resume-sweeps/{sweep_id}/cancel", json_body={})

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

        TWO VERDICTS, AND THEY ANSWER DIFFERENT QUESTIONS.

        `verdict` ('b_differs' | 'identical') is a LABEL-STRING diff: did the
        wording move? It says nothing about quality — a template that renames
        everything scores the same as one that fixes everything.

        `detection_delta` is the real instrument: a PAIRED bootstrap over the
        two arms' detection scores, with `mean_delta`, a confidence interval,
        and `minimum_detectable_effect` — the smallest effect this panel could
        have resolved, so a null result can be read honestly. Both arms saw the
        same features, passages and judge, so everything except the prompt
        cancels; that pairing is what makes ~30 features enough.

        It is `null` when neither trial carried detection scores ("not
        measured", which is not the same as "no difference"). It carries a
        `reason` and no verdict when the two arms were scored under different
        detection-prompt versions (a moved ruler is not a comparison), when
        either judge failed its sanity gate (the template was never measured —
        do not blame it), or when `confounded_by_coverage` is set because the
        arms scored materially different numbers of features (the arm that
        answered fewer was graded on the subset it chose).
        """
        return await client.get(f"/labeling/trials/compare/{run_a}/{run_b}")
