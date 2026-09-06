"""J-space lens tools (category: jlens) — BR-027 full MCP parity.

Every J-space capability reachable in the workbench must be reachable by an
agent, and the tools ship WITH the feature that creates them rather than being
batched into a final one. That sequencing is not a preference: this server once
shipped 16 tools that were fully implemented, unit-tested and documented in the
contract while registered nowhere, and every test passed by importing the module
directly. `tests/unit/test_reachability.py` is the harness that now guards it.

Scope here is what EXISTS as a route. A tool calling a route that does not
exist is the same defect in a new place, so tools land as their endpoints do.
"""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def cancel_jlens_task(
        task_id: Annotated[str, Field(description="Celery task id returned by fit_jlens_artifact or acquire_jlens_artifact (NOT the tq_ row id)")],
    ) -> Any:
        """Stop a running or queued J-space task.

        COOPERATIVE, BECAUSE REVOKE DOES NOT WORK HERE. The GPU worker runs
        `--pool=solo` (CUDA and fork do not mix), and Celery's
        `revoke(terminate=True)` only signals a pool child — solo has none.
        A solo worker busy in a task is not reading the control queue either,
        so the revoke is never delivered: it returns cleanly, changes nothing,
        and the worker does not appear in `inspect()`. Verified on hardware
        against a running gemma-4-12B fit, which needed a SIGKILL on the PID.

        So this writes the request to the task's row and the task stops itself
        at its next checkpoint — ONE PROMPT for a fit, which is seconds on a
        small model and MINUTES on a large one. `was_running: false` means it
        had not started and never will, which is immediate.

        Anything already completed/failed/cancelled returns `cancelled: false`
        with the reason, rather than pretending to act.
        """
        return await client.post(f"/jlens/tasks/{task_id}/cancel")

    @mcp.tool()
    async def list_jlens_artifacts() -> Any:
        """List J-lens artifacts present in the mounted registry.

        PRESENCE, NOT VALIDITY. An artifact appearing here has not been
        validated — run validate_jlens_artifact before trusting one. The
        consumer's lens loading fails at request time WITHOUT RAISING, so an
        unvalidated artifact presents as a feature that quietly returns
        nothing rather than as an error.
        """
        return await client.get("/jlens/artifacts")

    @mcp.tool()
    async def validate_jlens_artifact(
        slug: Annotated[str, Field(description="Artifact slug from list_jlens_artifacts")],
        d_model: Annotated[int, Field(description="Model hidden size the artifact was fitted for")],
        n_layers: Annotated[int, Field(description="Layer count the artifact should cover")],
        n_vocab: Annotated[int, Field(description="Model vocabulary size — the envelope bound is derived from it, so a wrong value makes the check meaningless")],
    ) -> Any:
        """Run the BR-030 validation suite against one artifact.

        Reports all six classes individually. `passed` is FAIL-CLOSED: the
        three checks needing a loaded model or a running consumer report
        NOT_RUN from here, so `passed` is False and that is the honest answer,
        not a defect — "we did not check" must never read like "we checked and
        it was fine".

        The model's dimensions are required rather than looked up because the
        envelope bound must come from the model the artifact was fitted for.
        The required-vs-materialised ratio scales with vocabulary, so a bound
        derived from the wrong model passes while missing a real
        materialisation.
        """
        return await client.post(
            f"/jlens/artifacts/{slug}/validate",
            d_model=d_model,
            n_layers=n_layers,
            n_vocab=n_vocab,
        )

    @mcp.tool()
    async def get_jlens_band_report(
        slug: Annotated[str, Field(description="Artifact slug from list_jlens_artifacts")],
    ) -> Any:
        """This model's OWN sensory / workspace / motor boundaries, or null.

        A null result means no band report has been computed for this model,
        and NOTHING should be inferred about where its bands lie. The published
        boundaries in the literature were measured on one specific model and do
        not transfer — miStudio has no default and will not supply one.

        The report also carries the per-layer profile, including next-token
        agreement. That figure is DESCRIPTIVE. Do not rank or gate on it: the
        J-lens is deliberately worse than the logit lens on agreement through
        most of the network (BR-004).
        """
        return await client.get(f"/jlens/artifacts/{slug}/band-report")

    @mcp.tool()
    async def get_jlens_gate(
        slug: Annotated[str, Field(description="Artifact slug from list_jlens_artifacts")],
    ) -> Any:
        """The recorded Phase-0 GO / NO-GO / GO-AT-LARGER-SCALE decision, or null.

        NO_GO is a complete, publishable outcome rather than a failure — it
        means the full workspace claim set did not replicate at this scale, and
        it BLOCKS product surface beyond the readout viewer (BR-003).

        Null means no decision has been recorded yet, which is not the same as
        GO and must not be read as one.
        """
        return await client.get(f"/jlens/artifacts/{slug}/gate")

    @mcp.tool()
    async def jlens_readout(
        model_id: Annotated[str, Field(description="miStudio model id (m_xxxxxxxx)")],
        prompt: Annotated[str, Field(description="Text to read out, max 8000 characters")],
        types: Annotated[Optional[List[str]], Field(description="LOGIT_LENS and/or JACOBIAN_LENS. Defaults to LOGIT_LENS, which needs no artifact")] = None,
        layers: Annotated[Optional[List[int]], Field(description="Absolute layer indices; omit for every layer")] = None,
        top_n: Annotated[int, Field(description="Readout depth per cell")] = 8,
        artifact_id: Annotated[Optional[str], Field(description="Required for JACOBIAN_LENS; must be the artifact fitted for THIS model's weights")] = None,
    ) -> Any:
        """QUEUE a readout of what a model is poised to say per layer and position.

        Returns a TASK ID, not a readout. Poll `get_jlens_readout(task_id)`.
        The readout is asynchronous because it needs the whole model resident
        for a forward pass — bound synchronously it exceeded the ingress
        timeout on a real model — so the first readout for a model takes about
        a minute and subsequent ones are fast.

        RUNG 0. A concept appearing in a readout is NOT a causal claim — it says
        the direction was present, not that the model used it. Raising the rung
        takes a coordinate swap with a matched control.

        Three limits worth stating before interpreting a result: readouts only
        surface concepts with SINGLE-TOKEN names; a readout that resists
        interpretation is not a null result; and absence of a signal is not
        evidence that the computation did not occur.

        The logit lens needs no artifact and works on any downloaded model.
        JACOBIAN_LENS requires a validated artifact fitted for these exact
        weights and is REFUSED without one — it is never silently answered with
        logit data under a Jacobian label.
        """
        body: dict[str, Any] = {"model_id": model_id, "prompt": prompt, "top_n": top_n}
        if types:
            body["types"] = types
        if layers:
            body["layers"] = layers
        if artifact_id:
            body["artifact_id"] = artifact_id
        return await client.post("/jlens/readout", json_body=body)

    @mcp.tool()
    async def get_jlens_readout(
        task_id: Annotated[str, Field(description="Task id returned by jlens_readout")],
    ) -> Any:
        """Poll a queued readout.

        `readout` is null until `status` is SUCCESS — a PENDING or PROGRESS
        task is NOT an empty readout, and reading it as one is exactly the
        confusion this feature exists to prevent. A FAILURE reports its reason.
        """
        return await client.get(f"/jlens/readout/{task_id}")

    @mcp.tool()
    async def annotate_jlens_feature(
        model_id: Annotated[str, Field(description="miStudio model id")],
        sae_id: Annotated[str, Field(description="SAE the feature belongs to")],
        feature_id: Annotated[str, Field(description="Feature id being annotated")],
        layer: Annotated[int, Field(description="Layer the feature lives at")],
        label_tokens: Annotated[Optional[List[str]], Field(description="The feature's existing label, to compare the readout against. Omit and no disagreement is computed")] = None,
        top_k: Annotated[int, Field(description="How many readout tokens to return")] = 8,
        direction: Annotated[Optional[List[float]], Field(description="Explicit d_model decoder direction. OMIT IT and the server resolves this feature's decoder column from sae_id — which is what makes this callable without shipping thousands of floats")] = None,
    ) -> Any:
        """Describe an SAE feature in J-space: what it pushes TOWARD.

        TWO INDEPENDENT FIELDS, and the second one matters. `lens_kurtosis` is
        geometric; `workspace_class` is behavioural. High kurtosis ALONE is not
        workspace alignment — a MOTOR feature is sharp too, so classifying on
        kurtosis would call every motor feature a workspace feature.

        `workspace_class` is UNKNOWN unless a band report exists for this
        model. That is a real answer, not a failure: without boundaries
        measured here there is no principled middle of the stack.

        RUNG 0. An annotation is an observation about a direction, not a claim
        that the feature causes anything.
        """
        body: dict[str, Any] = {
            "model_id": model_id,
            "sae_id": sae_id,
            "feature_id": feature_id,
            "layer": layer,
            "top_k": top_k,
        }
        if direction:
            body["direction"] = direction
        if label_tokens:
            body["label_tokens"] = label_tokens
        return await client.post("/jlens/annotate", json_body=body)

    @mcp.tool()
    async def create_jlens_watchlist(
        name: Annotated[str, Field(description="Watchlist name")],
        artifact_ref: Annotated[str, Field(description="The artifact its directions live in. Lens coordinates are artifact-specific and mean nothing elsewhere")],
        scoring_definition: Annotated[str, Field(description="HOW the score is computed. REQUIRED: a threshold applied to a differently computed score is a different detector, and the consumer cannot notice")],
        concepts: Annotated[List[Dict[str, Any]], Field(description="[{token, threshold}] pairs")],
        control_set: Annotated[Optional[List[str]], Field(description="Unrelated concrete nouns the score is measured against")] = None,
    ) -> Any:
        """Author a watchlist for miLLM to evaluate per token at inference.

        miStudio EMITS; runtime evaluation is miLLM's plane.

        A watchlist is a detector definition, not a list of words: directions,
        thresholds and the scoring definition travel together or none of them
        mean anything. Missing either the scoring definition or the artifact
        reference is refused rather than exported and discovered later.
        """
        body: dict[str, Any] = {
            "name": name,
            "artifact_ref": artifact_ref,
            "scoring_definition": scoring_definition,
            "concepts": concepts,
        }
        if control_set:
            body["control_set"] = control_set
        return await client.post("/jlens/watchlists", json_body=body)

    @mcp.tool()
    async def jlens_cost_estimate(
        operation: Annotated[str, Field(description="artifact_construction | readout | decomposition | annotation_sweep | intervention_run | template_lens_build")],
        d_model: Annotated[int, Field(description="Model hidden size")],
        n_layers: Annotated[int, Field(description="Layer count")],
        n_positions: Annotated[int, Field(description="Prompt length in tokens")] = 1,
        n_prompts: Annotated[int, Field(description="Corpus size, for a fit")] = 1,
        n_features: Annotated[int, Field(description="Dictionary size, for a sweep")] = 1,
    ) -> Any:
        """Estimate an operation's cost BEFORE committing to it.

        CALL THIS FIRST for anything larger than a single readout. An
        annotation sweep over a 32k-feature dictionary and one readout differ by
        orders of magnitude, and there is no way to tell them apart from the
        request alone.

        Estimates are ORDER-OF-MAGNITUDE and carry their basis. An unknown
        operation is an error rather than a cheap default — a small number would
        invite exactly the run it should warn about.
        """
        return await client.get(
            "/jlens/cost-estimate",
            operation=operation,
            d_model=d_model,
            n_layers=n_layers,
            n_positions=n_positions,
            n_prompts=n_prompts,
            n_features=n_features,
        )

    @mcp.tool()
    async def get_jlens_replication_report(
        slug: Annotated[str, Field(description="Artifact slug")],
    ) -> Any:
        """The recorded replication report, or null (BR-001).

        Published whether favourable or not. A partial run reports as partial —
        `complete: false` with the missing evaluation sets named — rather than
        as a clean table over whatever happened to finish.
        """
        return await client.get(f"/jlens/reports/replication?slug={slug}")

    @mcp.tool()
    async def run_jlens_intervention(
        model_id: Annotated[str, Field(description="miStudio model id")],
        prompt: Annotated[str, Field(description="Text to intervene on")],
        primitive: Annotated[str, Field(description="additive (steer along a direction) | projective_ablation (remove a direction's component) | coordinate_swap (EXCHANGE two tokens' coordinates; needs target_token). dynamic_topk_ablation is REFUSED — it needs lens coordinates this path does not compute, and it used to run an additive steer under its own name")],
        layers: Annotated[List[int], Field(description="Absolute layer indices to act at. DISTINCT — a repeat registers a second hook that perturbs the output of the first, so [9,9,9] at strength 1.0 applies 3.0 while the recorded recipe still says 1.0. Above a quarter of the stack the result carries `over_layer_budget`: swaps and steers oversteer easily at that width on small models (BR-017 v0.2). It is a warning, not a refusal — a deliberate whole-stack intervention is a legitimate experiment")],
        control_seed: Annotated[int, Field(description="REQUIRED in practice. 'A random direction' is not a control; 'k random directions from seed s' is, and a control nobody can reconstruct is not one")],
        prompts: Annotated[Optional[List[str]], Field(description="MORE PROMPTS, one TRIAL each. BELOW FOUR TRIALS NO OUTCOME SEPARATES: a perfect intervened arm against a perfect null control still produces overlapping Wilson intervals, so a run this small can only report `separated_from_control: false` and that says nothing about the direction. The result carries `separation_attainable` and `min_trials_for_separation` so you can tell the two apart. Tens of prompts is the useful scale")] = None,
        target_token: Annotated[Optional[str], Field(description="The token whose RANK is scored in the model's output. Defaults to direction_token. REQUIRED and must DIFFER for coordinate_swap: a swap exchanges two coordinates, and one token would be an additive steer wearing a swap's name")] = None,
        direction: Annotated[Optional[List[float]], Field(description="Explicit d_model vector to act along")] = None,
        direction_token: Annotated[Optional[str], Field(description="Resolve the direction from a SINGLE token's unembedding row instead of passing d_model floats. Multi-token strings are REFUSED rather than truncated — a lens direction is defined for one token")] = None,
        strength: Annotated[float, Field(description="How far to move the residual, in ABSOLUTE units: the direction is scaled to unit norm at the point of use, so the same number means the same displacement for every token. It did not before — a raw unembedding row carries the token's own norm, which varies several-fold, so a sweep on two tokens was two different experiments wearing the same numbers, and the unit-norm control was not matched to it. IGNORED by projective_ablation and coordinate_swap, which take no strength; the result reports null for those rather than echoing your value")] = 1.0,
        k: Annotated[int, Field(description="Control size, matched to the intervention")] = 1,
        positions: Annotated[Optional[List[int]], Field(description="Token positions; defaults to the last")] = None,
        artifact_id: Annotated[Optional[str], Field(description="PROVENANCE, NOT A DIFFERENT MEASUREMENT. The perturbation always happens in the residual stream inside the running model; naming an artifact runs its publish gate (so an unvalidated lens cannot justify a finding) and FILES the result beside that lens in interventions.json, where it travels to HuggingFace and into a serving runtime. Name it only when the direction came from that lens — crediting it for a finding it played no part in is what the file exists to prevent")] = None,
    ) -> Any:
        """Run an intervention AND its size-matched control in one pass.

        THE FINDING IS `excess_over_control`, NOT `intervened_outcome`. An
        intervention that moves the output says nothing until compared with what
        a random direction of the same size does — so this tool will not run one
        without the other, and the result reports both figures alongside their
        difference so you can see the control actually ran (BR-018).

        RUNG 2. The perturbation is applied to the residual stream at
        `layers` and the forward pass CONTINUES, so what is scored is the rank
        of `target_token` in the model's own next-token distribution — its
        behaviour, not the lens's geometry. This replaced a lens-space
        displacement measure that could not see the prompt at all: the transport
        is linear, so the activation cancelled and two unrelated prompts
        returned the same number to seven significant figures.

        REPORTED AS RATES WITH INTERVALS. `intervened_top1` / `control_top1` /
        `baseline_top1` each carry hits, n and a Wilson 95% interval.
        `separated_from_control` requires the intervened and control intervals
        to be DISJOINT — a bigger rate is not a finding, and 6/10 against 5/10
        is noise. The baseline arm matters as much: an intervention that
        "achieves" what the model already did has moved nothing.

        CHECK `separation_attainable` BEFORE READING A NULL. It is false when no
        outcome at that trial count COULD have separated the intervals — below
        four trials, always. The two readings are opposite: `false` with
        `separation_attainable: true` means no effect was demonstrated; `false`
        with `separation_attainable: false` means nothing could have been, and
        the answer is more prompts rather than a different direction.

        THE CONTROL IS NORM-MATCHED BY CONSTRUCTION. Both arms move the residual
        the same distance, because the named direction is scaled to unit norm
        and the control directions already are. Records carrying
        `direction_scaling: "unit"` were produced under that rule; older ones
        without it were not, and their `strength` was multiplied by an
        unembedding row norm that nobody wrote down.

        WHEN AN ARTIFACT IS NAMED the result is recorded beside the lens in
        `interventions.json`, with the steering recipe that produced it, so the
        evidence travels with the artifact rather than living only here.

        Long-running and model-bound; poll get_task_status.
        """
        body: dict[str, Any] = {
            "model_id": model_id,
            "prompt": prompt,
            "primitive": primitive,
            "layers": layers,
            "control_seed": control_seed,
            "strength": strength,
            "k": k,
        }
        if prompts:
            body["prompts"] = prompts
        if target_token:
            body["target_token"] = target_token
        if direction:
            body["direction"] = direction
        if direction_token:
            body["direction_token"] = direction_token
        if positions:
            body["positions"] = positions
        if artifact_id:
            body["artifact_id"] = artifact_id
        return await client.post("/jlens/interventions", json_body=body)

    @mcp.tool()
    async def get_jlens_interventions(
        slug: Annotated[str, Field(description="Artifact slug from list_jlens_artifacts")],
    ) -> Any:
        """What this lens has been MEASURED to do, recorded beside the weights.

        Each record carries a `steering_recipe` — primitive, direction token,
        target token, layers, positions, strength and hook target — next to the
        rates that justify it. That is deliberate: a score alone says something
        happened without saying what to apply, so a caller can take a record
        whose intervened rate separated from its matched control and reproduce
        exactly what was tested rather than inferring a recipe from a number.

        BOUND TO THE WEIGHTS THAT WERE TESTED. Records are dropped on read if
        the lens file's digest no longer matches — a refit replaces the
        matrices, and a record describing the previous lens would attribute one
        artifact's measured behaviour to another.

        AN EMPTY LIST IS AN ANSWER: no intervention has been run against this
        lens. That is NOT the same as one that was run and moved nothing, which
        appears here with overlapping intervals.
        """
        return await client.get(f"/jlens/artifacts/{slug}/interventions")

    @mcp.tool()
    async def restore_jlens_artifact(
        slug: Annotated[str, Field(description="Artifact slug from list_jlens_artifacts")],
    ) -> Any:
        """Promote the archived artifact back into service.

        A SWAP, not a move, so this is its own undo: call it twice and you are
        back where you started, and nothing is deleted at any point.

        Publishing is otherwise last-writer-wins, and "last" means finished
        last, not best — a 400-prompt fit that never converged once published
        over a 1097-prompt fit that did, because the weaker job had been queued
        hours earlier and only got a worker once the queue drained. The publish
        guard now refuses that, but an artifact already displaced needed a shell
        rename inside the pod to recover.

        The archive is not privileged: its recorded verdict is verified against
        the file it describes before promotion, and a mismatch is refused rather
        than served.
        """
        return await client.post(f"/jlens/artifacts/{slug}/restore-superseded")

    @mcp.tool()
    async def compute_jlens_band_report(
        model_id: Annotated[str, Field(description="miStudio model id to profile")],
        prompts: Annotated[List[str], Field(description="Corpus to measure the per-layer profile over")],
        control_seed: Annotated[int, Field(description="REQUIRED. The autocorrelation null is drawn from it; a report whose control cannot be reproduced is not evidence")],
        layers: Annotated[Optional[List[int]], Field(description="Absolute layer indices; omit to use the artifact's layers, or every layer when no artifact is used")] = None,
        use_artifact: Annotated[bool, Field(description="Use the fitted lens dictionary. Effective dimensionality is a property of that dictionary; for the logit lens it is the identity and is recorded ABSENT rather than as a number")] = True,
    ) -> Any:
        """Measure this model's band profile and derive ITS OWN boundaries.

        THE ONLY THING THAT CAN MAKE BANDS APPEAR. Until this runs for a model,
        every band surface renders nothing and `annotate_jlens_feature` returns
        workspace_class UNKNOWN — the honest answer, because the published
        sensory/workspace/motor boundaries were measured on ONE specific model
        and do not transfer to another.

        There is no way to supply boundaries and there never will be. The result
        carries `has_bands: false` and `boundaries: null` when this model's
        kurtosis profile does not support a three-way split, and that null is a
        finding, not a missing value (BR-002).

        Long-running and model-bound; poll get_task_status.
        """
        body: dict[str, Any] = {
            "model_id": model_id,
            "prompts": prompts,
            "control_seed": control_seed,
            "use_artifact": use_artifact,
        }
        if layers:
            body["layers"] = layers
        return await client.post("/jlens/band-report", json_body=body)

    @mcp.tool()
    async def record_jlens_gate(
        model_id: Annotated[str, Field(description="miStudio model id the gate decision is about")],
        claim_set_replicated: Annotated[bool, Field(description="Whether the FULL workspace claim set replicated. This is the question BR-003 asks — a finding you supply, not a score computed here")],
        rationale: Annotated[str, Field(description="MANDATORY. A recorded decision without its reasoning is not a record")],
        larger_scale_indicated: Annotated[bool, Field(description="Whether the evidence suggests retrying at a larger scale. Produces GO_AT_LARGER_SCALE, which still BLOCKS at this scale")] = False,
        replication_report_id: Annotated[Optional[str], Field(description="Replication report this decision rests on")] = None,
    ) -> Any:
        """Record the Phase-0 GO / NO-GO decision (BR-003).

        REFUSES without a band report to weigh — a gate decision with no
        evidence behind it is what this gate exists to prevent.

        There is deliberately NO numeric criterion. A threshold on any single
        metric would become the definition of the gate, and the metric most
        likely to be reached for is next-token agreement, which BR-004 forbids
        being scored on at all. NO_GO is a complete, publishable outcome and
        persists exactly like GO.
        """
        return await client.post(
            "/jlens/gate",
            json_body={
                "model_id": model_id,
                "claim_set_replicated": claim_set_replicated,
                "larger_scale_indicated": larger_scale_indicated,
                "rationale": rationale,
                "replication_report_id": replication_report_id,
            },
        )

    @mcp.tool()
    async def preview_jlens_repo(
        repo_id: Annotated[str, Field(description="HuggingFace repo holding one or more lenses, e.g. 'neuronpedia/jacobian-lens'")],
        model_id: Annotated[Optional[str], Field(description="miStudio model this would be attached to. Supply it: the response then carries a per-file envelope verdict for THAT model's dimensions, which is the difference between choosing a path and guessing one")] = None,
        revision: Annotated[Optional[str], Field(description="Commit, branch or tag. Omit and the response reports the sha `main` currently resolves to — pass THAT to acquire, or the acquisition names a moving target")] = None,
        access_token: Annotated[Optional[str], Field(description="For a private repo. Falls back to the configured token, then the stored one")] = None,
    ) -> Any:
        """List the files in a repo that could be a J-lens, with sizes.

        READ-ONLY, AND SPENDING A REQUEST INSTEAD OF A DOWNLOAD. A mistyped path
        otherwise costs a multi-gigabyte fetch and a slot on the single-GPU
        queue before anything notices.

        It lists every `*.pt` / `*.safetensors`, not only conformant
        `*_jacobian_lens.pt` names: community repos publish `qwen3_8b_lens.pt`
        and `gemma2_9b_jlens.pt`, and filtering to the conformant name would
        hide exactly the repos this exists to reach.

        `has_config` is the field to read first. A file beside a `config.yaml`
        declares which weights it was fitted for, so its weight identity can be
        CHECKED; one without leaves the pairing resting on your assertion, and
        the artifact records that it does.
        """
        body: dict[str, Any] = {"repo_id": repo_id}
        if model_id:
            body["model_id"] = model_id
        if revision:
            body["revision"] = revision
        if access_token:
            body["access_token"] = access_token
        return await client.post("/jlens/acquire/preview", json_body=body)

    @mcp.tool()
    async def publish_jlens_artifact(
        model_id: Annotated[str, Field(description="miStudio model whose PUBLISHED lens to upload. A staged artifact is not published and is refused")],
        target_repo: Annotated[str, Field(description="HuggingFace repo to publish into, e.g. 'you/jacobian-lenses'")],
        access_token: Annotated[Optional[str], Field(description="A token with WRITE access. Required — the read path may run anonymously, an upload cannot, and it is refused rather than attempted with an empty credential")] = None,
        dataset: Annotated[str, Field(description="Corpus segment of the published path, per the conformance layout <model>/jlens/<dataset>/. Name the corpus the fit was drawn from; 'mistudio' is the honest default for an ad-hoc one")] = "mistudio",
        create_repo: Annotated[bool, Field(description="Create the repo if it does not exist")] = False,
        private: Annotated[bool, Field(description="Make a newly created repo private")] = False,
    ) -> Any:
        """Publish a validated lens so others can mount it. Poll get_task_status.

        WHAT TRAVELS: the checkpoint in the conformant wrapper shape, its
        `config.yaml`, and a README stating the recipe and what was checked.

        WHAT DOES NOT: `validation.json` and `acquisition.json`. The first is
        this installation's verdict on its own copy — including two classes
        recorded as DEFERRED because they need a live external consumer and have
        never been run anywhere — and shipping it invites a reader to take a
        local verdict for the lens's own. The README says so in words instead.

        Band boundaries are never included and must not be inferred from
        anything here: the published figures were measured on one specific model
        and porting them is the error this project forbids by construction.
        """
        body: dict[str, Any] = {
            "model_id": model_id,
            "target_repo": target_repo,
            "dataset": dataset,
            "create_repo": create_repo,
            "private": private,
        }
        if access_token:
            body["access_token"] = access_token
        return await client.post("/jlens/publish", json_body=body)

    @mcp.tool()
    async def acquire_jlens_artifact(
        model_id: Annotated[str, Field(description="miStudio model to attach the lens to. Its WEIGHTS MUST BE DOWNLOADED — validating an acquired lens means reading out through it, which runs a real forward pass. Refused synchronously otherwise")],
        repo_id: Annotated[str, Field(description="HuggingFace repo to take it from")],
        path_in_repo: Annotated[str, Field(description="Exact file path inside the repo. Use preview_jlens_repo to find it rather than guessing — a wrong path costs a multi-gigabyte download")],
        revision: Annotated[Optional[str], Field(description="Commit to pin. Omit and the resolved sha of `main` is used AND RECORDED, so the acquisition stays reproducible either way")] = None,
        access_token: Annotated[Optional[str], Field(description="For a private repo. Falls back to the configured token, then the stored one")] = None,
        allow_coverage_loss: Annotated[bool, Field(description="Publish even though the artifact this REPLACES covers layers the downloaded one does not. Off by default")] = False,
        allow_quality_regression: Annotated[bool, Field(description="Publish even when this lens is weaker evidence than the one it replaces. Off by default — and note the gate cannot fire when the downloaded lens declares no prompt count, so read `displaced` in the result rather than trusting silence")] = False,
    ) -> Any:
        """Adopt a lens someone else fitted. GPU-bound; poll get_task_status.

        FITTING REMAINS THE PRIMARY PATH — published lenses exist for a limited
        model set, and most models this workbench runs are not in it. This is
        the cheaper route when one does exist: minutes and a download instead of
        a GPU hour.

        WHAT IS AND IS NOT CHECKED. The artifact is validated exactly as a local
        fit is, including a real SEMANTIC readout on the same fixture, so it can
        only publish if it actually discriminates. Beyond that:

        * `weight_identity` is `verified` when the publisher's own config names
          these weights, `unverified` when they named none, and a MISMATCH is
          refused outright — a lens fitted for different weights produces a
          complete, plausible readout that is wrong.
        * `bytes_identical` says whether what we serve is bit-for-bit what they
          published.
        * the layer indexing convention and the target layer are read OFF THE
          TENSORS, because a semantic check scans every fitted layer and so
          cannot catch a foreign convention.

        It does NOT reproduce the fit. `n_prompts` and `converged` in the
        resulting recipe are the publisher's figures, marked as such.
        """
        body: dict[str, Any] = {
            "model_id": model_id,
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "allow_coverage_loss": allow_coverage_loss,
            "allow_quality_regression": allow_quality_regression,
        }
        if revision:
            body["revision"] = revision
        if access_token:
            body["access_token"] = access_token
        return await client.post("/jlens/acquire", json_body=body)

    @mcp.tool()
    async def fit_jlens_artifact(
        model_id: Annotated[str, Field(description="miStudio model id to fit a lens for")],
        prompts: Annotated[List[str], Field(description="Fitting corpus. The fitter REFUSES fewer than 100 — an under-fitted lens is indistinguishable from a fitted one by inspection")],
        layers: Annotated[Optional[List[int]], Field(description="Absolute layer indices; omit for every layer")] = None,
        freeze_qk: Annotated[bool, Field(description="Freeze attention patterns. OFF by default — the paper treats freezing as an ABLATION, not the standard recipe. It slightly reduces readout quality while tending to produce directions that respond MORE strongly to intervention — an association reported by the source paper, not a validated claim about this artifact, and the reason frozen-Q/K is the suggested variant when the lens is built for intervention work rather than reading. INAPPLICABLE on layers that do not attend, and recorded per layer rather than claimed wholesale")] = False,
        corpus_name: Annotated[str, Field(description="Recorded in the artifact's recipe (BR-007) — name the corpus, do not leave it unspecified")] = "unspecified",
        convergence_delta: Annotated[Optional[float], Field(description="Relative Frobenius change in J below which the fit is settled. Omit for the fitter default (1e-3). A LOOSER value converges sooner without more corpus; the artifact records whichever was used, so two fits are never compared as though they met the same criterion")] = None,
        freeze_norms: Annotated[bool, Field(description="Freeze normalisation statistics too. Off by default: freezing makes the map exactly AFFINE, which is convenient and is not what the paper computes — its J is a local linearisation whose departure is reported rather than engineered away")] = False,
        target_layer: Annotated[str, Field(description="'penultimate' (default) or 'final'. The last block is specialised for next-token calibration and adds noise, per BRD A.2. Layers ABOVE the target are REFUSED — their gradient to it is zero by causality, and a zero lens reads out as confident uniform noise")] = "penultimate",
        allow_coverage_loss: Annotated[bool, Field(description="Publish even though the EXISTING artifact covers layers this fit does not. Off by default and refused otherwise — a 16-layer lens was once destroyed by a 9-layer refit with no warning. Losing coverage must be a decision")] = False,
        allow_quality_regression: Annotated[bool, Field(description="Publish even when this fit is WEAKER evidence than the artifact it replaces — fewer prompts, or not converged where the published one converged. OFF by default: publishing is otherwise last-writer-wins, and a stale job that finishes last is still last. A 400-prompt non-converged fit once published over a 1097-prompt converged one this way")] = False,
        semantic_probe: Annotated[Optional[Dict[str, Any]], Field(description="Fixture for the SEMANTIC check: {prompt, expected_intermediate, control_prompt?, layer?, top_k?}. WITHOUT IT NOTHING IS PUBLISHED — the check cannot run and the suite fails closed. The intermediate must NOT appear in the prompt, or a lens encoding nothing would pass. Omit `layer` to SCAN every fitted layer (the default: which depth carries an intermediate is a property of the model, not something to assert); naming a layer pins the check to it. `control_prompt` is an unrelated prompt for which the intermediate would be absurd — if it surfaces there too the check FAILS, because a lens that answers the same thing to everything has shown nothing")] = None,
    ) -> Any:
        """Queue a J-lens fit. GPU-bound and long-running; poll get_task_status.

        Fitting is the PRIMARY path, not a fallback: pre-fitted lenses exist
        for a limited model set and most models this workbench runs are not in
        it.

        SUPPLY `semantic_probe` OR NOTHING IS PUBLISHED. The SEMANTIC class
        needs a loaded model and a fixture, and an artifact validated without
        it can never be published — by design, since publishing on an unrun
        check is the failure the suite exists to prevent. A fit without it
        succeeds, validates partially, and is discarded; the result says so in
        `unpublished_reason`.

        The result carries a per-check validation report. An artifact that is
        `serviceable` can be read out locally; `passed` additionally requires
        the two consumer-interop checks, which need a live external consumer
        and are deferred until handover.
        """
        body: dict[str, Any] = {
            "model_id": model_id,
            "prompts": prompts,
            "freeze_qk": freeze_qk,
            "corpus_name": corpus_name,
            "allow_coverage_loss": allow_coverage_loss,
            "allow_quality_regression": allow_quality_regression,
            "freeze_norms": freeze_norms,
            "target_layer": target_layer,
            **({"convergence_delta": convergence_delta} if convergence_delta else {}),
        }
        if layers:
            body["layers"] = layers
        if semantic_probe:
            body["semantic_probe"] = semantic_probe
        return await client.post("/jlens/fit", json_body=body)
