---
sidebar_position: 1
title: "The Researcher's Journey"
description: "The six-stage mechanistic interpretability pipeline"
---

# The Researcher's Journey

The mechanistic interpretability pipeline follows six stages:

```
Model → Dataset → Activations → SAE Training → Feature Discovery → Steering
  ↓         ↓          ↓              ↓                ↓              ↓
Select    Prepare    Extract       Disentangle      Interpret      Prove
the LLM   stimuli   internal      superposed       what each     causation
                     numbers       features         feature       with
                                                    means         intervention
```

1. **The Subject (Model):** Select an LLM — the "brain" you're dissecting
2. **The Stimuli (Dataset):** Text that "stimulates" the model to activate different concepts
3. **The Capture (Extraction):** Record internal activations as the model processes text
4. **The Disentanglement (SAE):** Train a Sparse Autoencoder to "untangle" polysemantic neurons
5. **The Interpretation (Feature Discovery):** Browse activation examples, run auto-labeling and enhanced per-feature labeling to understand what each feature encodes
6. **The Proof (Steering):** Manipulate discovered features to verify causal influence
