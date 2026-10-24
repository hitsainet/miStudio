---
sidebar_position: 2
title: "Quickstart Tutorial"
description: "Your first SAE in about 30 minutes — TinyStories to steering, end to end"
---

# Quickstart: Your First SAE in ~30 Minutes

This tutorial runs the entire pipeline once, on purpose-picked small ingredients: **GPT-2 small** (124M params) and the **TinyStories** dataset. Everything fits in 8 GB of VRAM, and no step takes longer than a coffee. By the end you'll have trained an SAE, browsed its features, labeled one, and steered the model with it.

**Prerequisites:** a running miStudio instance ([installation](/getting-started/installation)) with a CUDA GPU.

## Step 1 — Download the Dataset (~2 min)

1. Open the **Datasets** panel and click **+ Download**
2. Enter repo ID: `roneneldan/TinyStories`
3. Start the download and watch the progress bar on the dataset card

TinyStories is small (~1 GB), synthetic children's stories — narratively rich but simple vocabulary, which makes SAE features unusually easy to interpret. Perfect first subject.

## Step 2 — Download the Model (~2 min)

1. Open the **Models** panel and click download
2. Enter repo ID: `gpt2`
3. Leave quantization at **FP16**

While it downloads, miStudio maps the model's structure — GPT-2 small has 12 layers with a 768-dimensional residual stream. We'll probe **layer 6**, the middle of the network, where features tend to be most abstract.

## Step 3 — Tokenize (~3 min)

1. Back in **Datasets**, open the TinyStories card and choose **Tokenize**
2. Select **gpt2** as the model and set **Max Length** to `512`
3. Start the job

Remember: [tokenizations are per-model](/core-workflow/dataset-management) — this one only feeds GPT-2 pipelines.

## Step 4 — Train the SAE (~10–15 min)

1. Open the **Training** panel and create a new training job
2. **Dataset:** the TinyStories tokenization you just made
3. **Model:** gpt2 — **Layer:** `6` — **Hook:** `residual`
4. **Framework:** Standard (SAELens) — the classic ReLU + L1 architecture
5. Hyperparameters for a quick-but-real run:

| Parameter | Value | Why |
|-----------|-------|-----|
| Expansion factor | 8× | 768 → 6,144 features; plenty for a demo |
| Total steps | 5,000 | Enough for coherent features on TinyStories |
| Batch size | 2048 | Fits comfortably in 8 GB |
| Learning rate | 3e-4 | Default |
| `l1_alpha` | 5e-4 | Default sparsity |

6. Start it, and watch the live metrics: **loss** should fall steadily, **L0** should settle somewhere in the 10–100 range, and **dead neurons** should stay under ~30%.

:::tip While you wait
Read the [Interpretability Primer](/concepts/interpretability-primer) — it explains what this training is actually doing, in about the time the job takes.
:::

## Step 5 — Import the SAE & Extract Features (~5 min)

1. When training completes, open the **SAEs** panel → **Import from training** → select your run
2. Open the **Extraction** panel and configure a **Feature Extraction** job:
   - **SAE:** the one you just imported
   - **Evaluation samples:** `10,000` — **Top-K examples:** `50`
   - Leave the token filters at their defaults
3. Run it

This scans the dataset through your SAE and records each feature's top activating examples — the evidence you'll interpret next. (Confused about the two kinds of "extraction"? [This page](/concepts/extraction-pipeline) untangles them.)

## Step 6 — Browse and Label a Feature (~5 min)

Open the **Feature Browser** in the Extraction panel and click through a few features. On TinyStories you'll quickly find crisply interpretable ones — features for character names, for "once upon a time" openings, for dialogue, for emotions.

Pick a feature whose examples show an obvious pattern and:

- Read its top activating examples — the highlighted tokens are where it fires
- Click into the feature detail and give it a label describing the pattern you see
- If you've configured a labeling LLM in [Settings](/advanced/settings-reference), try **Enhanced Labeling** on it — a two-pass LLM analysis that summarizes every example and synthesizes a label with supporting evidence

## Step 7 — Steer With It (~5 min)

The payoff: prove the feature *causes* the behavior it correlates with.

1. Open the **Steering** panel
2. Select your SAE and the feature you labeled
3. Enter a neutral prompt, e.g. `The little girl walked into the`
4. Set strengths `0` (baseline) and `20`, then generate

Compare the outputs. With a "sad/crying" feature at strength 20, the story turns melancholy; at negative strength, cheerfully upbeat. That side-by-side is a causal demonstration — you found a direction in GPT-2's residual stream that *means* something, and you proved it by intervening.

## Where to Go Next

- **Scale up:** the same pipeline on `gemma-2-2b` with a Gemma Scope comparison — see [SAE Management](/advanced/external-saes)
- **Understand the knobs:** the [SAE Training guide](/core-workflow/sae-training) covers all six frameworks and their hyperparameters
- **Automate interpretation:** [bulk auto-labeling](/core-workflow/auto-labeling) labels thousands of features overnight
- **Share findings:** [export to Neuronpedia](/advanced/exporting)
