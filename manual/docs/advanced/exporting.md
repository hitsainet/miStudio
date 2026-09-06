---
sidebar_position: 4
title: "Neuronpedia Export & Push"
description: "Package SAE findings for sharing, or push directly to a local Neuronpedia"
---

# Neuronpedia Export & Push

[Neuronpedia](https://neuronpedia.org) is the community platform for browsing SAE features. miStudio can share your findings two ways:

1. **Export to ZIP** — build a complete, Neuronpedia-compatible package for manual upload or offline archival
2. **Direct Push** — push features, weights, and dashboard data straight into a *local* Neuronpedia instance over its API

Both run as background jobs with live progress, so multi-thousand-feature exports don't tie up your browser.

## Export to ZIP

From an SAE card, choose **Export** and configure:

| Option | Description |
|--------|-------------|
| **Feature selection** | All features, extracted-only, or a custom index range |
| **Logit lens data** | Promoted/suppressed tokens per feature (top-K configurable) |
| **Activation histograms** | Distribution of each feature's activation values (bin count configurable) |
| **Top activating tokens** | Aggregated most-frequent tokens per feature |
| **Explanations** | Your feature labels, human and LLM-generated |
| **SAELens weights** | `cfg.json` + `sae_weights.safetensors` for Python interop |

### What's in the package

```
export.zip
├── metadata.json            # SAE config + export manifest
├── README.md                # human-readable summary
├── features/                # one JSON per feature
│   ├── 0.json               #   stats, logit lens, histogram,
│   ├── 1.json               #   top tokens, activation examples
│   └── ...
├── explanations/
│   └── explanations.json    # all feature labels
└── saelens/
    ├── cfg.json             # SAELens-compatible config
    └── sae_weights.safetensors
```

The `saelens/` directory loads directly in SAELens/TransformerLens research code, so a single export serves both Neuronpedia upload and Python analysis.

Progress reports the current stage (computing logit lens, generating histograms, packaging) and the export can be **cancelled** mid-run. Completed exports stay in the export history for re-download until deleted.

### Logit lens, briefly

For each feature, miStudio projects the feature's decoder direction through the model's unembedding matrix. Tokens with the highest projection are the ones the feature *promotes* when active; the most negative are *suppressed*. It's a fast, surprisingly informative summary of what a feature does to the model's output — see the [Interpretability Primer](/concepts/interpretability-primer) for context.

## Direct Push to Local Neuronpedia

If you run a local Neuronpedia instance (miStudio ships a K8s manifest for one), **Push to Neuronpedia** uploads everything in one step:

1. miStudio checks the instance is reachable (a status indicator shows this before you start)
2. A background job creates the model + SAE records in Neuronpedia with names derived from your SAE for discoverability
3. Dashboard data — logit lens, feature statistics, activation histograms — is computed on the fly and pushed per feature
4. Progress (`features pushed / total`) streams live; failures report the exact error

The push job survives browser refreshes — it runs server-side, and the panel re-attaches to its progress when you return.

:::tip Compute dashboard data separately
The **Compute Dashboard Data** action pre-computes logit lens + histograms for an SAE without pushing. Useful when you want the data cached before a large push, or for local inspection only.
:::

## Which Mode Should I Use?

| Situation | Use |
|-----------|-----|
| Sharing with the public community | Export ZIP → upload to neuronpedia.org |
| Lab-internal feature browsing | Direct push to your local instance |
| Archiving results / reproducibility | Export ZIP (self-contained) |
| Python analysis in SAELens | Export ZIP → use the `saelens/` folder |
