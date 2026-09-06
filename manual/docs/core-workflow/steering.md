---
sidebar_position: 6
title: "Model Steering"
description: "Proving causation through feature intervention"
---

# Model Steering — Proving Causation

Steering is the definitive proof of your research. By manipulating specific features during generation, you demonstrate that a feature causally influences model behavior — not just correlates with it.

![Steering Panel — Feature configuration with prompts and strength settings](/img/miStudio_Steering_Panel-Config.jpg)

## Steering Modes

miStudio provides three distinct steering modes:

| Mode | What It Does | Best For |
|------|-------------|----------|
| **Individual** | One feature at multiple strengths | Understanding a single feature's dose-response curve |
| **Comparison** | Multiple features at the same strength, side-by-side | Comparing related features (e.g., "French" vs "German") |
| **Combined** | All selected features applied simultaneously | Discovering synergistic effects between features |

## Strength Values

Steering strengths are **raw coefficients** added to the model's residual stream, compatible with Neuronpedia's scale:

| Range | Effect | Example |
|-------|--------|---------|
| **0** | No intervention (baseline) | Unsteered output |
| **0.07 – 5** | Subtle influence | Slight shift in topic or tone |
| **5 – 50** | Moderate effect | Clear behavioral change |
| **50 – 100** | Strong effect | Dominant feature influence |
| **100 – 200** | Very strong | Feature overwhelms other signals |
| **200 – 300** | Extreme | Often causes repetition or collapse |
| **Negative** | Suppression | Inhibits the feature's concept |

:::warning Strength Calibration
The effective range depends on the SAE and layer. Start with strengths around **5–20** and increase gradually. Values above ±100 frequently cause the model to "collapse" into repetitive or incoherent output.
:::

:::tip Multi-Strength Testing
Each feature supports up to 3 **additional strengths** tested simultaneously. Set a primary strength and additional values to see the dose-response curve in one generation pass. For example: primary=10, additional=[5, 20, 50].
:::

## Generation Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **Max Tokens** | 100 | 1–2,048 | Length of generated text |
| **Temperature** | 0.7 | 0–2.0 | Randomness. 0 = deterministic, 1.0 = creative |
| **Top-P** | 0.9 | 0–1.0 | Nucleus sampling threshold |
| **Top-K** | 50 | 0–500 | Vocabulary restriction per token |
| **Repetition Penalty** | 1.15 | 0.5–2.0 | Penalizes repeated tokens. Increase if output loops. |
| **Seed** | — | Optional | Set for reproducible results across runs |

## The Matrix Testing Workflow

Unlike tools with a single slider, miStudio uses a **grid approach**:

1. **Select Features:** Add up to 4 features (e.g., "Honesty" + "French Language")
2. **Add Prompts:** Multiple test prompts processed in batch
3. **Configure Strengths:** Set primary and additional strengths per feature
4. **Include Baseline:** Toggle "Include Unsteered" to compare against the natural output
5. **Execute:** miStudio runs all combinations, presenting results in a structured comparison view

:::info Combined Mode Synergies
Steering multiple features simultaneously is NOT the same as running them separately. "Scientific Tone" + "Excitement" combined may produce different text than either alone. Combined mode reveals **circuit behavior** where features interact.
:::

## Viewing Results

After execution, results are presented in a structured comparison view showing baseline and steered outputs side-by-side with perplexity metrics:

![Steering Session Results — Baseline vs steered outputs with perplexity comparison](/img/miStudio_Steering_Panel-SessionResults.jpg)

## Multi-layer circuits — steering across SAEs

Everything above steers features from **one** SAE at **one** layer. A discovered [circuit](/core-workflow/circuits), though, spans layers — its members live on the layers each was found on, and each layer has its own SAE. Feature 015 makes steering follow the circuit: **every member steers through the SAE trained on its own layer**, in a single generation.

### The own-layer rule

Each feature carries the SAE it belongs to. When you load a multi-layer circuit into the steering panel, the members arrive with their per-layer SAEs already set — an L13 member steers through the L13 SAE, an L14 member through the L14 SAE, in the same Blended run. You cannot steer an L14-trained feature at L10: that's rejected with a clear message naming the offending member, never quietly served through the wrong decoder (which would inject a direction from the wrong layer's basis — the bug this feature exists to fix).

### Per-layer budgets, one dial

The [budget model](/core-workflow/steering#the-budget) runs **independently per layer** — each layer gets its own budget bar (total budget B, cohesion gain G, similarity-weighted allocation) computed against that layer's SAE. One global intensity dial (λ) scales the whole circuit at once, so you tune the entire cross-layer behaviour with a single control while each layer keeps its principled starting strengths. The applied-features summary groups members by layer so you can see, and verify, that each one steered through its own SAE.

### Hazard warnings — compounding and cancellation

Steering an **upstream** feature that drives a **downstream** feature you're *also* steering makes their influences **compound** (or, with opposite signs, **cancel**). miStudio surfaces this before you generate — an amber banner listing the pairs — but **never silently corrects it**; you decide.

The warning is as strong as the evidence:

- If the circuit has a **validated edge** between the pair (a rung-2 edge that survived [causal validation](/core-workflow/circuits#causal-validation-the-rung-2-tier)), the warning is **quantified from the measured effect size** — "validated edge, ES=0.8 — combined influence ≈ higher than the naive sum."
- If there's no validated edge, a **weight-prior heuristic** (how aligned the upstream feature's output direction is with the downstream feature's input direction) still warns — but every such warning is explicitly **labeled `heuristic`**, never presented as causal. It's a hint to check, not a proven mechanism.

This is the payoff of the whole circuits arc: the same validated evidence that earns a circuit its rung-2 badge is what makes the steering hazard warning trustworthy instead of a guess.

:::info VRAM
Each distinct SAE the circuit references loads onto the GPU (~130 MB for an 8k-feature SAE). A typical two- or three-layer circuit adds well under a gigabyte; only the SAEs the circuit actually uses are loaded, and exiting steering mode frees them all.
:::
