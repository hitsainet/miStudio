---
sidebar_position: 10
title: "Circuits"
description: "Discover cross-layer feature circuits — capture, statistically mine, and attribution-rank candidates"
---

# Circuits — Discovering Cross-Layer Structure

A cluster names features that mean the same thing at one layer. A **circuit** names how features at *different* layers work together — an upstream feature that drives a downstream one, across the model's depth. miStudio can now **discover** these, not just host hand-built ones.

Discovery is deliberately honest about what it has and hasn't proven. It produces **candidates** with disclosed statistics; it never dresses an association up as a proven mechanism. The evidence gets stronger in stages (the [evidence ladder](/concepts/evidence-ladder)), and every surface tells you which rung a circuit is on.

## The loop

1. **Capture** the per-token, multi-layer feature activations over an evaluation corpus.
2. **Discover** candidate edges by mining that store with statistically sound methods.
3. **Attribution-rank** the shortlist with a gradient pass before the expensive causal tier.
4. **Validate** the top edges causally, then **review, promote, calibrate, and steer** them.

## Capture

The **Circuits → Capture** tab records what discovery needs and today's extraction throws away: for every token, which features fired at each selected layer, **including the token's position** and the SAE's reconstruction-error norm.

Configure the layer set (each layer with the SAE trained on it), the evaluation dataset, a sparsity threshold, and a sample cap. Before committing GPU time you get a **cost estimate** — projected events, disk size, and wall-clock, extrapolated from a small probe batch. Confirm to launch the managed capture (progress, cancel, guardrails).

The store is built **Tier-2.5-ready**: token positions are first-class and an optional attention sidecar can be captured, so [attention-mediated discovery](/concepts/tier-2.5) is a future implementation, never a re-capture. At capture time the corpus is split **80/20 into discovery and held-out partitions** (seeded and recorded) — every downstream run uses the same split, so replication is measured, not assumed.

Capture stores are listed, reusable, and deletable. If a referenced SAE changes, the store is **flagged stale** (not deleted) — mining a stale store requires an explicit override.

## Discovery

The **Circuits → Discovery** tab mines a completed capture store. Two choices shape the run:

- **Granularity** — **feature-level** (individual features) or **cluster-level supernodes** (your curated clusters as single units; a supernode fires wherever any member fires). Cluster granularity is the recommended default for seeded runs.
- **Mode** — **seeded** (mine the upstream/downstream partners of chosen features or clusters) or **open-corpus** (mine broadly). Both are first-class.

### What makes a candidate trustworthy

Naive co-occurrence counting is worse than nothing — it surfaces whatever is frequent. Discovery instead:

- ranks by **PMI** (pointwise mutual information / lift), not raw counts, so two independently-frequent features do **not** rise to the top;
- requires a **minimum support** before a pair is ranked at all;
- tests each pair against a **null model** — a within-document circular shift that preserves each feature's firing rate and burstiness while destroying its alignment with the partner;
- applies **Benjamini–Hochberg FDR** for multiple comparisons;
- **re-tests survivors on the held-out partition** and reports the **replication rate** as a first-class number.

The **run report**, pinned above the candidate list, is the trust surface: it states the null method, the FDR discipline, the held-out replication rate, the counts at each stage, and — crucially — **any caps that were hit** (unit caps, candidate truncation). Nothing is silently dropped.

:::note Lag-0 limitation
All co-activation here is **same-token-position**. That finds within-position computation; it does **not** find attention-mediated, cross-position circuits (a feature three sentences back driving one here). Every discovery surface says so, and [Tier-2.5](/concepts/tier-2.5) is the designed path to lift it.
:::

## Attribution pass

A discovery candidate is a statistical association. The **Run attribution pass** action spends one forward + one backward per prompt (with the SAE reconstruction error held as a stop-gradient constant) to compute a **gradient-attribution score** per candidate — a second, independent evidence signal. It **re-ranks** the shortlist so the causal validation tier is spent on the most promising edges, and it records both orderings so validation can report whether the re-ranking actually raised the survival rate.

Attribution earns a candidate the **attribution-supported** rung — one step up the ladder, and explicitly *not* a causal claim. That rung comes only from intervention (below).

## Causal validation — the rung-2 tier

Discovery and attribution produce *associations*. **Validation** is where a candidate earns the word "causal". The **Validation tab** takes the top-K edges of a run (in either ordering) and, for each, **intervenes**: it suppresses the upstream feature — subtracting its decoder direction from the residual stream, *never* re-decoding, so the SAE's reconstruction error is left untouched — runs the model, and measures how much the downstream feature's activation drops on the tokens where the upstream one fired.

That drop, expressed as an **effect size** (mean drop over the downstream feature's own activation scale), is compared against a **null** built from shuffled non-edges — random support-matched upstream features that *aren't* connected to the downstream one. An edge reaches **rung 2 (causally validated)** only if its effect size beats the null percentile **and** is sign-consistent across prompts. An edge that's tested and doesn't clear the bar is recorded as **tested, did not validate** — history that never *demotes* a rung the edge earned another way.

Run over both orderings, validation reports a **survival rate** per ordering, and the difference — the **uplift** — answers the question attribution set up: did re-ranking by attribution actually raise the fraction of edges that survive causal testing?

### Faithfulness — the rung-3 tier

For a whole circuit, **faithfulness** asks whether its members are *necessary* and *sufficient* for a behavior: suppress all members and measure how much of the behavior collapses (necessity), suppress everything *except* the members and measure how much survives (sufficiency). The behavior metric and the "ablate-all" proxy are always recorded, so the measurement is legible and reproducible.

### Manifests — reproducible evidence

Every validation run writes a **manifest**: a self-contained record of the intervention config, baseline, prompts, seeds, null summary, and per-edge effect sizes — everything needed to **reproduce** it. The **Reproduce** button re-executes from the manifest and reports whether the effect sizes come back within tolerance. A rung-2 claim you can't reproduce isn't a rung-2 claim.

## Circuit strength calibration

A validated circuit still ships with an *arbitrary* steering dial. Calibration finds the **usable band** for that dial — the range where the circuit visibly acts *without* corrupting the model — and clamps the served dial to it. It runs two **different** detectors, because "the output changed" and "the facts still hold" are different questions:

- **Onset** — the least dial that measurably changes the output versus baseline. This is a plain **output-drift** test with **no judge**: below onset the circuit is inert.
- **Correctness cliff** — the most dial where the model's **facts still hold**, decided by an **LLM judge** against **generated neutral-topic falsifiable probes**. The probes are on topics the circuit should *not* touch, so degradation shows up as the circuit's tint corrupting unrelated facts. Perplexity or theme metrics can't find this — the cliff sits between two adjacent dials, one giving a correct answer and the next confidently false. An **adaptive bisection** narrows it within a small generation budget.

On success, calibration **clamps** the served dial to `[onset, cliff]` — a served dial physically cannot reach the nonsense zone — and sets the default intensity a safety `margin` below the cliff. The band is recorded on the circuit (a `calibration` block, with `calibration_status`) and travels in the export so miLLM can re-verify it cheaply against the served model.

Two disciplines matter:

- **Badge, not gate.** Calibration never blocks promotion, steering, or export. It annotates the circuit with a measured band; the choice to respect it stays with the operator.
- **A weak judge is reported, never guessed around.** If the judge model can't reliably tell correct from broken (it fails even the *unsteered* baseline), calibration reports **`judge_unreliable`** rather than inventing a cliff — and never emits a false **`no_band`**. A band you can't trust is labeled as such.

The **Reproduce** action re-runs a calibration from its manifest — the same probes and seed — and records a band-delta verdict: do onset, sweet-spot, and cliff land within tolerance of the original? Reproduction only *checks* the number; it does not re-clamp the circuit.

## Steered transcript recorder

Calibration judges; the recorder does **not**. The **steered transcript recorder** is an **instrument, not a judge**: it records `(dial, prompt, unsteered_output, steered_output)` transcripts on the GPU and hands them, unlabeled, to a separate meaning-analysis pass — typically a **strong model reading the transcripts after the run** to say what steering this artifact semantically *did* across the dial. It carries no correctness verdict of its own and needs no judge endpoint.

It works on any of three artifacts: a **circuit**, a **cluster** (profile), or an **ad-hoc feature set**. For each prompt it records the unsteered baseline plus the steered output at every requested dial, using the same steering core and residual hook as calibration — so what you read matches a calibrated band. Records are written as manifests; circuit-artifact records link back to their circuit, while cluster and feature records are fetched by manifest id.

## A note on the feature-level "ablation impact"

The impact number in the Feature Detail view is a **statistical estimate** (from a feature's activation frequency, magnitude, and consistency) — it runs *no* model inference and is labeled as such. It is a quick heuristic, never causal evidence. For a real causal measurement, validate the circuit edge.

## Doing it from an agent

The [MCP server](/advanced/mcp-server) exposes the whole loop in the `circuits` category (default-on):

- **Capture & mine:** `start_circuit_capture`, `list_circuit_captures`, `run_circuit_discovery`, `get_discovery_results`, `run_attribution_pass`.
- **Causal tier:** `validate_circuit_edges`, `get_validation_manifest`, `list_validation_manifests`, `reproduce_validation`.
- **Circuit lifecycle:** `create_circuit`, `build_circuit_from_discovery`, `get_circuit`, `update_circuit`, `list_circuits`, `delete_circuit`, `run_circuit_faithfulness`, `promote_circuit`, `import_circuit_definition`, `export_circuit_definition`, `export_circuit_slices`.
- **Calibration:** `calibrate_circuit_strength` and `reproduce_calibration` — calibrate the usable strength band and re-verify it from its manifest.
- **Steered transcript recorder:** `record_steering_samples` and `get_steering_samples` — record `(dial, prompt, unsteered, steered)` transcripts for a circuit, cluster, or feature set, then read them back for post-run analysis.

Tool descriptions carry the same lag-0 and rung-language discipline the UI does: only `validate_circuit_edges` results earn causal language. Start any agent circuit session with `mistudio_howto` — it holds the workflows and the failure modes.
