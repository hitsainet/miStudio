"""Circuit tools (category: circuits) — Feature 018, IDL-33/IDL-35.

Rung-aware access to circuits: every returned circuit/edge carries its
evidence rung AND the server-rendered rung_language string — agents MUST use
that language verbatim and never describe rung-0/1 artifacts with causal
words (the evidence-ladder discipline). The agent loop: run discovery (016
tools, when present) → review candidates → create/promote circuits here →
steer them (steer_combined with per-member sae_id) → export definitions or
per-layer v1 slices for today's single-SAE consumers.
"""

from typing import Annotated, Any, Optional

from pydantic import Field

from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def list_circuits(
        promoted: Annotated[Optional[bool], Field(description="Promotion is a BADGE, not a gate: it never restricts export or steering, and is reversible")] = None,
        min_rung: Annotated[Optional[int], Field(description="Filter to circuits at or above this evidence rung (0 mined, 1 attribution, 2 causally validated, 3 faithfulness-tested)")] = None,
        granularity: Annotated[Optional[str], Field(description="Filter: 'feature' | 'cluster'")] = None,
    ) -> Any:
        """List circuits with rung + rung_language on every row. min_rung
        filters by evidence rung (0 mined, 1 attribution-supported,
        2 causally validated, 3 faithfulness-tested)."""
        params: dict[str, Any] = {}
        if promoted is not None:
            params["promoted"] = promoted
        if min_rung is not None:
            params["min_rung"] = min_rung
        if granularity:
            params["granularity"] = granularity
        return await client.get("/circuits", **params)

    @mcp.tool()
    async def get_circuit(circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx) — NOT a miLLM circuit id")]) -> Any:
        """One circuit: members by layer, typed edges with full evidence
        (statistics, attribution, validation manifest refs), budget,
        faithfulness, rung + rung_language + what-moves-it-up."""
        return await client.get(f"/circuits/{circuit_id}")

    @mcp.tool()
    async def build_circuit_from_discovery(
        run_id: Annotated[str, Field(description="Discovery run id from run_circuit_discovery; must be status=completed")],
        name: Annotated[str, Field(description="Human-readable circuit name")],
        candidate_keys: Annotated[Optional[list], Field(description="[[up_layer, up_idx, down_layer, down_idx], ...] selecting which candidates become edges. EMPTY/omitted = ALL candidates, including any that failed validation — select explicitly unless you mean all")] = None,
        narrative: Annotated[Optional[str], Field(description="Free-text account of what this circuit is claimed to do")] = None,
    ) -> Any:
        """Build a circuit from a discovery run's candidates — PREFER THIS over
        create_circuit when the circuit came from discovery.

        This is the EVIDENCE-PRESERVING path. It carries onto each edge the
        coactivation statistics (pmi, lift, support, null percentile), the
        attribution score and gate, the rung-2 effect size, and the
        validation_manifest_ref — and derives each edge's rung from what
        actually passed rather than from what you assert.

        `create_circuit` does none of that. A circuit hand-assembled there
        exports with `effect_size`, `coactivation`, `attribution` and
        `validation_manifest_ref` all null: the RUNG survives the export but
        the evidence behind it does not, so a consumer gets "trust me, it is
        rung 2" with no way to check. It also sets `discovery_run_id`, without
        which run_circuit_faithfulness is refused 409 forever.

        Created UNPROMOTED. Promote separately; promotion is a badge, not a
        gate."""
        return await client.post(
            f"/circuit-discovery/{run_id}/build-circuit",
            json_body={"name": name, "narrative": narrative,
                       "candidate_keys": candidate_keys or []})

    @mcp.tool()
    async def create_circuit(
        name: Annotated[str, Field(description="Human-readable circuit name")],
        saes: Annotated[list, Field(description="[{layer, sae_id}] — the SAEs this circuit references, one per layer")],
        members: Annotated[list, Field(description="[{layer, member_kind, feature:{feature_idx, strength, label, max_activation}}]. STRENGTH IS WHAT STEERS; ~0.15 x the real max_activation per member, total ~3")],
        edges: Annotated[Optional[list], Field(description="[{up:{layer,feature_idx}, down:{...}, type, rung}] — EVIDENCE ONLY, carries no strength and does not steer")] = None,
        narrative: Annotated[Optional[str], Field(description="Free-text account of what this circuit is claimed to do")] = None,
        budget: Annotated[Optional[dict], Field(description="{intensity, intensity_range:[lo,hi]} — the dial envelope carried to miLLM")] = None,
        granularity: Annotated[str, Field(description="'feature' (individual SAE features) | 'cluster' (precomputed cluster supernodes)")] = "feature",
        discovery_run_id: Annotated[Optional[str], Field(description="Discovery run this came from. PASS IT IF YOU HAVE IT — omitting permanently forfeits run_circuit_faithfulness")] = None,
        discovery: Annotated[Optional[dict], Field(description="Provenance: {mode, granularity, corpus, split, thresholds, dates}")] = None,
        faithfulness: Annotated[Optional[dict], Field(description="Pre-computed circuit-level faithfulness, if you already have it")] = None,
        model_id: Annotated[Optional[str], Field(description="miStudio model row id")] = None,
        model_hf_id: Annotated[Optional[str], Field(description="HuggingFace repo id — the cross-instance-stable provenance carried by exports")] = None,
    ) -> Any:
        """Create a circuit from hand assembly.

        IF THIS CAME FROM A DISCOVERY RUN, USE `build_circuit_from_discovery`
        INSTEAD. This tool stores exactly the edges you hand it, so the
        coactivation statistics, attribution scores, effect sizes and
        validation_manifest_refs that discovery/validation measured are NOT
        carried: the exported definition keeps the rung and loses the evidence
        for it.

        PASS `discovery_run_id` IF YOU HAVE ONE. Omitting it PERMANENTLY
        forfeits `run_circuit_faithfulness`, which draws its prompts from that
        capture store and 409s without it. The failure surfaces later, at a
        different tool, and the only remedy is recreating the circuit. Any
        hand-assembled circuit therefore cannot be faithfulness-tested.

        Contract rules enforced server-side: per-layer member caps (20 PER
        LAYER), edges must ascend layers and reference members. Rejections
        return the exact violation. model_hf_id (HF repo id) is the
        cross-instance-stable model provenance carried by exports.

        STRENGTHS live on members (`member.feature.strength`) and are what
        actually steers — edges carry evidence only. Measured envelope: about
        0.15 x the feature's REAL max_activation per member, total ~3 across
        two layers; a circuit at ~50x that emitted pure token soup in
        production. See mistudio_howto('strength_calibration')."""
        return await client.post("/circuits", json_body={
            "name": name, "saes": saes, "members": members,
            "edges": edges or [], "narrative": narrative, "budget": budget,
            "granularity": granularity, "discovery_run_id": discovery_run_id,
            "discovery": discovery, "faithfulness": faithfulness,
            "model_id": model_id, "model_hf_id": model_hf_id,
        })

    @mcp.tool()
    async def promote_circuit(circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx) — NOT a miLLM circuit id")], promoted: Annotated[bool, Field(description="Promotion is a BADGE, not a gate — reversible, never restricts export or steering")] = True) -> Any:
        """Promote a circuit into a loadable multi-layer steering profile —
        or unpromote it (promoted=false). Promotion is a BADGE, not a gate —
        unvalidated circuits promote and carry their rung visibly everywhere."""
        return await client.post(f"/circuits/{circuit_id}/promote",
                                 json_body={"promoted": promoted})

    @mcp.tool()
    async def update_circuit(
        circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx) from list_circuits/create_circuit — NOT a miLLM circuit id")],
        name: Annotated[Optional[str], Field(description="Human-readable circuit name")] = None,
        narrative: Annotated[Optional[str], Field(description="Free-text account of what this circuit is claimed to do")] = None,
        members: Annotated[Optional[list], Field(description="[{layer, member_kind, feature:{feature_idx, strength, label, max_activation}}]. STRENGTH IS WHAT STEERS; ~0.15 x the real max_activation per member, total ~3")] = None,
        edges: Annotated[Optional[list], Field(description="[{up:{layer,feature_idx}, down:{...}, type, rung}] — EVIDENCE ONLY, carries no strength and does not steer")] = None,
        saes: Annotated[Optional[list], Field(description="[{layer, sae_id}] — the SAE each layer resolves to. REQUIRED for the export to be servable: an entry with no id exports as `mistudio_sae_id: null` and fails at miLLM as an unbound SAE. Read the ids from list_circuit_captures.")] = None,
        budget: Annotated[Optional[dict], Field(description="{intensity, intensity_range:[lo,hi]} — the dial envelope carried to miLLM")] = None,
        granularity: Annotated[Optional[str], Field(description="Filter: 'feature' | 'cluster'")] = None,
    ) -> Any:
        """Edit a circuit (rename, fix a narrative, drop a bad edge, adjust
        members, switch granularity) — the agent review loop is not
        create-only. Structural contract rules re-validate on every update.

        `saes` is repairable here. That matters for circuits created before
        `sae_id` was accepted as an alias: they persisted with a null SAE id,
        which exports clean and only fails at miLLM as an unbound SAE. The
        PATCH route always accepted `saes`; this tool simply did not expose it,
        so the only field that makes an export SERVABLE was unreachable from
        MCP."""
        body = {k: v for k, v in {
            "name": name, "narrative": narrative, "members": members,
            "edges": edges, "saes": saes, "budget": budget,
            "granularity": granularity}.items() if v is not None}
        return await client.patch(f"/circuits/{circuit_id}", json_body=body)

    @mcp.tool()
    async def delete_circuit(circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx) — NOT a miLLM circuit id")]) -> Any:
        """Delete a circuit permanently (its manifests survive — they are
        first-class records).

        IRREVERSIBLE, and this tool does NOT check whether the circuit is
        serving in miLLM — unlike its sibling `millm_delete_circuit`, which
        refuses a serving circuit without `acknowledge_serving`. Deleting one
        that miLLM has imported does not stop it serving there, but you lose
        the definition. Export first (`export_circuit_definition`) if you may
        want it back."""
        return await client.delete(f"/circuits/{circuit_id}")

    @mcp.tool()
    async def import_circuit_definition(definition: Annotated[dict, Field(description="A circuit-definition document. `kind` is 'mistudio.circuit-definition' with NO /v1 suffix")]) -> Any:
        """Import a mistudio.circuit-definition/v1 document (the BR-013
        round-trip). Unknown kinds are rejected explicitly."""
        return await client.post("/circuits/import", json_body=definition)

    @mcp.tool()
    async def export_circuit_definition(circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx) — NOT a miLLM circuit id")]) -> Any:
        """Export the portable circuit-definition JSON (lossless: rungs, edge
        types, attribution, manifest refs, provenance all travel). This is
        the raw contract document — for the human-readable rung_language
        string, use get_circuit/list_circuits.

        FEEDING THIS TO miLLM: the document's `saes[].mistudio_sae_id` must be
        the id MILLM knows (e.g.
        `mistudio--sae-<model>--layer_12--width_8k--res`), NOT miStudio's
        internal `sae_xxxxxxxx` row id. They are DIFFERENT NAMESPACES for the
        same SAE and no tool translates between them — read miLLM's ids from
        `millm_status`. A mismatched id imports fine and only fails at
        activation, as an unbound SAE.

        The `kind` field is `mistudio.circuit-definition` with NO `/v1`
        suffix, despite the schema file and docs saying v1."""
        return await client.get(f"/circuits/{circuit_id}/export")

    @mcp.tool()
    async def export_circuit_slices(circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx) — NOT a miLLM circuit id")]) -> Any:
        """Export per-layer cluster-definition/v1 slices (BR-014) for today's
        single-SAE consumers (miLLM). Each slice is a PARTIAL RENDERING —
        parent rung travels in the response and in each slice's provenance;
        never present a slice as the validated whole."""
        return await client.post(f"/circuits/{circuit_id}/export-slices")

    # ── Feature 016: capture → discovery → attribution ───────────────────

    @mcp.tool()
    async def start_circuit_capture(
        dataset_id: Annotated[str, Field(description="Dataset to capture activations over")],
        layers: Annotated[list, Field(description="[{layer, sae_id}] — which layers to capture, each with its SAE")],
        model_id: Annotated[Optional[str], Field(description="miStudio model row id")] = None,
        epsilon: Annotated[float, Field(gt=0, description="Activation threshold for recording a firing event; lower captures more, growing the store")] = 0.1,
        sample_cap: Annotated[int, Field(ge=1, description="Max documents to capture. GPU time scales with this")] = 2000,
        attention_capture: Annotated[Optional[dict], Field(description="Also record attention-mediated signals (Tier-2.5); costs more GPU time")] = None,
        confirm: Annotated[bool, Field(description="false = probe and return a COST ESTIMATE only; true = actually run the capture")] = False,
    ) -> Any:
        """Start a circuit-capture run (per-token multi-layer SAE activations
        with FIRST-CLASS positions + error norms). layers = [{layer, sae_id}].
        confirm=false returns a cost ESTIMATE only (probe); call again with
        confirm=true — or start_circuit_capture_confirm — to launch the full
        capture. Managed GPU task.

        POLL `list_circuit_captures` (NOT get_task_status). This endpoint
        returns a raw Celery task id and creates no task-queue row, so
        get_task_status cannot see it — status lives on the capture run.

        Serialised against ALL other circuit GPU work (attribution,
        validation, faithfulness) by one advisory lock, ACROSS UNRELATED
        circuits: a 409 may name a run you do not own. Back off and poll; do
        not retry immediately."""
        return await client.post("/circuit-capture", json_body={
            "dataset_id": dataset_id, "model_id": model_id, "layers": layers,
            "epsilon": epsilon, "sample_cap": sample_cap,
            "attention_capture": attention_capture, "confirm": confirm,
        })

    @mcp.tool()
    async def list_circuit_captures() -> Any:
        """List capture runs (status, corpus, layers, split, size, stale flag)."""
        return await client.get("/circuit-capture")

    @mcp.tool()
    async def run_circuit_discovery(
        capture_run_id: Annotated[str, Field(description="Capture run id from start_circuit_capture; must be status=completed")],
        granularity: Annotated[str, Field(description="'feature' (individual SAE features) | 'cluster' (precomputed cluster supernodes)")] = "feature",
        mode: Annotated[str, Field(description="'open' (mine all pairs) | 'seeded' (expand from seed_refs)")] = "open",
        seed_refs: Annotated[Optional[list], Field(description="Seed features/clusters to expand from. Only used when mode='seeded'")] = None,
        s_min: Annotated[int, Field(ge=1, description="Minimum support (co-occurrence count) for a pair to be considered. Higher = fewer, stronger candidates")] = 20,
        null_shuffles: Annotated[int, Field(ge=1, description="Circular-shift null samples per pair. DRIVES RUNTIME roughly linearly; 100 is the calibrated default")] = 100,
        fdr_q: Annotated[float, Field(gt=0, lt=1, description="Benjamini-Hochberg false-discovery rate. Lower = stricter, fewer surviving candidates")] = 0.05,
        force: Annotated[bool, Field(description="Re-run even if a completed discovery already exists for this capture")] = False,
    ) -> Any:
        """Mine a completed capture store for candidate cross-layer edges.
        granularity: feature | cluster (supernodes over curated profiles —
        recommended for seeded mode). mode: seeded (seed_refs = [{layer,
        feature_idx}|{layer, cluster_profile_id}]) | open. Statistically
        disciplined: PMI over raw counts, within-document circular-shift null,
        pooled-standardized BH-FDR, held-out replication. ALL co-activation is
        lag-0 (same token position) — attention-mediated structure is not
        found here (Tier-2.5). Returns a run id; get_discovery_results carries
        the first-class report."""
        return await client.post("/circuit-discovery", json_body={
            "capture_run_id": capture_run_id, "granularity": granularity,
            "mode": mode, "seed_refs": seed_refs, "s_min": s_min,
            "null_shuffles": null_shuffles, "fdr_q": fdr_q, "force": force,
        })

    @mcp.tool()
    async def get_discovery_results(run_id: Annotated[str, Field(description="Discovery run id from run_circuit_discovery")],
                                    include_candidates: Annotated[bool, Field(description="Include the full candidate list, not just the statistical report")] = True) -> Any:
        """A discovery run + its report (null-model summary, FDR discipline,
        held-out replication RATE, stage counts, caps, uncovered seeds, lag-0
        disclosure) + ranked candidates (PMI, support, null percentile,
        attribution when the pass has run, both orderings). The report is the
        trust surface — read it before trusting the list."""
        return await client.get(f"/circuit-discovery/{run_id}",
                                include_candidates=include_candidates)

    @mcp.tool()
    async def run_attribution_pass(run_id: Annotated[str, Field(description="Discovery run id from run_circuit_discovery")],
                                   prompt_limit: Annotated[Optional[int], Field(description="Prompts used for the gradient attribution pass")] = None) -> Any:
        """Tier-2 gradient-attribution pass over a discovery run's candidates:
        re-ranks the shortlist before 017's causal validation and gates rung-1
        (attribution_supported) by sign-agreement + magnitude. This is
        attribution, NOT causal proof — rung stays ≤1 until 017 intervenes.
        Both orderings are preserved so 017 can report survival-rate uplift.
        GPU task.

        POLL `get_discovery_results` and read `attribution_status` (NOT
        get_task_status — this returns a raw Celery id with no task-queue
        row). Terminal when it leaves pending/running.

        Holds the shared circuit-GPU lock (see start_circuit_capture); a 409
        may name unrelated work."""
        return await client.post(f"/circuit-discovery/{run_id}/attribution",
                                 json_body={"prompt_limit": prompt_limit})

    # ── Feature 017: validation + faithfulness + manifests ───────────────

    @mcp.tool()
    async def validate_circuit_edges(
        run_id: Annotated[str, Field(description="Discovery run id from run_circuit_discovery")],
        ordering: Annotated[str, Field(description="'coact' (co-activation rank) | 'attr' (attribution rank — REQUIRES a completed run_attribution_pass, else 409)")] = "coact",
        k: Annotated[int, Field(ge=1, description="Top-k edges to validate. GPU COST ~ k x prompts_per_edge x (1 + null_samples) forward passes")] = 20,
        prompts_per_edge: Annotated[int, Field(ge=1, description="Prompts per edge, drawn from the capture store. More = tighter effect size, linearly more GPU")] = 8,
        null_samples: Annotated[int, Field(ge=1, description="Support-matched non-edge pairs forming the null. Below 10 the verdict FAILS as underpowered rather than passing")] = 20,
        sign_frac: Annotated[float, Field(ge=0, le=1, description="Fraction of prompts whose effect must share the mean sign for rung 2. 0.8 = the BR-018 criterion")] = 0.8,
        baseline: Annotated[str, Field(description="Ablation baseline: 'zero' (clamp the feature to 0) | 'mean' (clamp to its mean activation)")] = "zero",
    ) -> Any:
        """Causally validate the top-K edges of a discovery run (rung 2):
        suppress the upstream feature, run the model, measure the downstream
        effect size vs a shuffled-non-edge null; rung-2 iff |ES| beats the null
        percentile AND is sign-consistent. This is the REAL causal tier (rung
        2) — the tier where causal language is earned. ordering: coact | attr
        (attr needs an attribution pass first). Returns a run id; poll
        get_task_status then get_discovery_results for the per-edge verdicts."""
        return await client.post(
            f"/circuit-discovery/{run_id}/validate", json_body={
                "ordering": ordering, "k": k, "prompts_per_edge": prompts_per_edge,
                "null_samples": null_samples, "sign_frac": sign_frac,
                "baseline": baseline})

    @mcp.tool()
    async def run_circuit_faithfulness(
        circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx) from list_circuits/create_circuit — NOT a miLLM circuit id")],
        mode: Annotated[str, Field(description="'both' (necessity + sufficiency) | 'necessity' (sufficiency marked untested)")] = "both",
        k_nonmembers: Annotated[int, Field(ge=1, description="Top-k NON-member features ablated for the sufficiency approximation; disclosed on the badge")] = 256,
        ablate_all_n: Annotated[int, Field(ge=1, description="Per-layer top-N ablated to approximate ablate-all, the necessity denominator")] = 1024,
        n_prompts: Annotated[int, Field(ge=1, description="Prompts per behaviour measurement (4 measurements are run)")] = 16,
        seed: Annotated[int, Field(description="RNG seed for prompt sampling; fix it to make a run reproducible")] = 0,
    ) -> Any:
        """Faithfulness-test a circuit (rung 3 — the HIGHEST tier): suppress
        its members and measure how much of the behavior they drive is
        NECESSARY (ablating them collapses it) and, with mode='both',
        SUFFICIENT (ablating only non-members leaves it standing) — vs a
        per-layer top-N 'ablate all' proxy (N recorded). Behavior metric v1 =
        the summed activation of the circuit's downstream-most members over
        prompts from the circuit's own capture store (SAME tokenization; never
        re-decoded). mode='necessity' skips the sufficiency probe (marked
        untested, never silently omitted). Rung 3 sits above rung 2 causal
        validation, so the score is a real faithfulness measurement — the
        manifest records the exact metric so the number is never trusted blind.
        Returns a task
        id.

        POLL `get_circuit` and read `faithfulness_status`, then
        `circuit.faithfulness` for the result (NOT get_task_status — raw
        Celery id, no task-queue row).

        REQUIRES the circuit to have a `discovery_run_id`: its prompts come
        from that capture store. A circuit created WITHOUT one — any
        hand-authored circuit — is refused 409 here and can never be
        faithfulness-tested. That is decided at create_circuit, not here.

        Holds the shared circuit-GPU lock; a 409 may name unrelated work."""
        return await client.post(
            f"/circuits/{circuit_id}/faithfulness", json_body={
                "mode": mode, "k_nonmembers": k_nonmembers,
                "ablate_all_n": ablate_all_n, "n_prompts": n_prompts,
                "seed": seed})

    @mcp.tool()
    async def calibrate_circuit_strength(
        circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx) from list_circuits — NOT a miLLM circuit id")],
        step_budget: Annotated[int, Field(ge=2, le=40, description="Max generations the bisection may spend finding the cliff; ~10 is plenty. Higher = tighter cliff, more GPU")] = 10,
        probe_count: Annotated[int, Field(ge=1, le=10, description="How many neutral-topic falsifiable probes to generate and judge at each dial")] = 3,
        margin: Annotated[float, Field(ge=0.0, le=1.0, description="Safety gap below the cliff for the sweet-spot (the shipped default intensity). 0.15 = default intensity sits 0.15 dial below where facts break")] = 0.15,
        seed: Annotated[int, Field(description="RNG seed for probe generation + sampling; fix it to reproduce a band")] = 0,
        judge_endpoint: Annotated[Optional[str], Field(description="OpenAI-compatible endpoint for the correctness JUDGE + probe generation (e.g. the miLLM instance's /v1). REQUIRED for a real run — the cliff cannot be found without a judge")] = None,
        judge_model: Annotated[Optional[str], Field(description="Model name the judge_endpoint serves (e.g. the served model id from millm_status)")] = None,
    ) -> Any:
        """Calibrate a circuit's usable steering STRENGTH and clamp its served
        dial to it (Feature 20). Finds two thresholds by two DIFFERENT tests:

          ONSET — the least dial that measurably changes output vs baseline
          (a drift test, no judge). Below it the circuit is inert.

          CORRECTNESS CLIFF — the most dial where the model's FACTS still hold,
          judged by an LLM against generated NEUTRAL-topic probes (topics the
          circuit should not touch, so degradation shows as the circuit's tint
          corrupting unrelated facts). Perplexity/theme metrics cannot find this
          — the cliff sits between adjacent dials one giving a correct answer,
          the next confidently false.

        On success it CLAMPS `budget.intensity_range` to [onset, cliff] (a served
        dial physically cannot reach the nonsense zone) and sets the default
        `intensity` to a sweet-spot `margin` below the cliff. The band is marked
        `provisional` because the probes are generated and it is measured on the
        discovery-plane model — the probes travel in the export so miLLM can
        re-verify cheaply. Badge, not gate: it never blocks promotion/steering.

        Holds the shared circuit-GPU lock (same guard as faithfulness); a 409 may
        name unrelated circuit work — poll and back off, do NOT retry. Returns a
        task id; POLL `get_circuit` and read `calibration_status`, then
        `circuit.calibration` for the band (NOT get_task_status — raw Celery id)."""
        return await client.post(
            f"/circuits/{circuit_id}/calibration", json_body={
                "step_budget": step_budget, "probe_count": probe_count,
                "margin": margin, "seed": seed,
                "judge_endpoint": judge_endpoint, "judge_model": judge_model})

    @mcp.tool()
    async def reproduce_calibration(manifest_id: Annotated[str, Field(description="A calibration manifest id (from list_validation_manifests, kind='calibration') to re-run")]) -> Any:
        """Re-run a calibration from its manifest — the SAME probes and seed —
        and record a reproduction manifest with the band-delta verdict (does the
        re-measured onset/sweet_spot/cliff land within tolerance of the
        original?). Proves the band is reproducible, not a one-off of model
        nondeterminism. Does NOT re-clamp the circuit — reproduction only checks
        the number. GPU pass; shares the single-GPU circuit guard, so a 409 may
        name unrelated work. Returns a task id; poll get_task_status, then read
        the reproduction manifest for the verdict."""
        return await client.post(
            f"/circuits/calibration-manifests/{manifest_id}/reproduce")

    @mcp.tool()
    async def record_steering_samples(
        artifact: Annotated[dict, Field(description="What to steer: {'kind':'circuit','circuit_id':'crc_...'} | {'kind':'cluster','cluster_profile_id':'clp_...'} | {'kind':'features','model_id':'m_...','features':[{layer,feature_idx,strength,sae_id}]}")],
        dials: Annotated[list[float], Field(description="Steering strengths to record, e.g. [0.4,0.6,1.0]. Baseline (dial 0) is recorded per prompt automatically. Max 8, each in [0, 2.0]")],
        prompts: Annotated[list[str], Field(description="Prompts to generate on (max 8). prompts × (1+dials) ≤ 64 total generations")],
        max_tokens: Annotated[Optional[int], Field(description="Max new tokens per greedy generation. Default 80, max 200")] = None,
        seed: Annotated[int, Field(description="Generation seed (greedy ⇒ deterministic; kept for reproducibility)")] = 0,
    ) -> Any:
        """Record (dial, prompt, unsteered_output, steered_output) transcripts on
        the GPU for a circuit, cluster, or ad-hoc feature set — the raw material
        for a SEPARATE, post-run meaning-analysis pass (hand the transcripts to a
        strong model like Opus 4.8 to read what steering this artifact
        semantically DID across the dial). NOT a correctness judge — this path is
        judge-free and needs no judge endpoint. Uses the same steering core and
        residual hook as calibration, so the transcripts match a calibrated band.
        Holds the shared single-GPU circuit lock (a 409 may name unrelated work —
        poll and back off, do NOT retry). Returns a record_run_id + task id; poll
        get_task_status, then get_steering_samples to read the transcripts."""
        return await client.post(
            "/circuits/steering-samples", json_body={
                "artifact": artifact, "dials": dials, "prompts": prompts,
                "max_tokens": max_tokens, "seed": seed})

    @mcp.tool()
    async def get_steering_samples(
        circuit_id: Annotated[Optional[str], Field(description="miStudio circuit id (crc_...) to fetch CIRCUIT-artifact steering-sample records for (they link to their circuit)")] = None,
        manifest_id: Annotated[Optional[str], Field(description="A specific steering_samples manifest id (vman_...) — REQUIRED for CLUSTER/FEATURE records, which are not linked to a circuit. The record_steering_samples result and get_task_status carry the manifest_ref")] = None,
    ) -> Any:
        """Fetch recorded steering-sample transcripts: per prompt, the unsteered
        baseline plus the steered output at each recorded dial — the raw material
        for an Opus meaning-analysis pass.

        Pass `circuit_id` for a circuit-artifact record (they link to the
        circuit), or `manifest_id` for a specific record — REQUIRED for
        cluster/feature records, which carry the artifact in the payload and are
        not circuit-linked (the record_steering_samples result returns the
        manifest_ref). Exactly one of circuit_id / manifest_id."""
        if manifest_id and circuit_id:
            return {"error": "pass EXACTLY one of circuit_id or manifest_id, "
                    "not both"}
        if manifest_id:
            return await client.get(f"/validation-manifests/{manifest_id}")
        if not circuit_id:
            return {"error": "pass circuit_id (circuit records) or manifest_id "
                    "(cluster/feature records)"}
        resp = await client.get("/validation-manifests", circuit_id=circuit_id)
        manifests = (resp or {}).get("manifests", []) if isinstance(resp, dict) else []
        return {"circuit_id": circuit_id,
                "manifests": [m for m in manifests
                              if m.get("kind") == "steering_samples"]}

    @mcp.tool()
    async def get_validation_manifest(manifest_id: Annotated[str, Field(description="Validation manifest id from list_validation_manifests")]) -> Any:
        """A validation manifest — the SELF-CONTAINED, reproducible record of a
        validation run (intervention config, baseline, prompts, seeds, null
        summary, per-edge effect sizes). The evidence behind a rung-2 claim."""
        return await client.get(f"/validation-manifests/{manifest_id}")

    @mcp.tool()
    async def list_validation_manifests(
        discovery_run_id: Annotated[Optional[str], Field(description="Discovery run this came from. PASS IT IF YOU HAVE IT — omitting permanently forfeits run_circuit_faithfulness")] = None,
        circuit_id: Annotated[Optional[str], Field(description="miStudio circuit id (circ_xxxxxxxx) — NOT a miLLM circuit id")] = None,
    ) -> Any:
        """List validation manifests for a discovery run or a circuit."""
        params: dict[str, Any] = {}
        if discovery_run_id:
            params["discovery_run_id"] = discovery_run_id
        if circuit_id:
            params["circuit_id"] = circuit_id
        return await client.get("/validation-manifests", **params)

    @mcp.tool()
    async def reproduce_validation(manifest_id: Annotated[str, Field(description="Validation manifest id from list_validation_manifests")]) -> Any:
        """Re-execute an edge_batch manifest from its payload and compare —
        the test that a rung-2 claim is reproducible, not a one-off. Returns a
        task id; the reproduction manifest carries per-edge deltas + a
        within-tolerance verdict."""
        return await client.post(
            f"/validation-manifests/{manifest_id}/reproduce")
