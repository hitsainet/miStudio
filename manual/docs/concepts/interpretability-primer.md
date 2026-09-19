---
sidebar_position: 2
title: "Interpretability Primer"
description: "Superposition, features, SAEs, and steering — the ideas behind the tool"
---

# Interpretability Primer

You can use miStudio without reading this page. But every design decision in the tool — why we train sparse autoencoders, why "features" and not "neurons", why steering works at all — comes from a small set of ideas in mechanistic interpretability research. Here they are, without the math.

## The Problem: Neurons Don't Mean Anything

The natural first instinct when opening up a neural network is to ask *what each neuron does*. This mostly fails. A single neuron in a language model might fire for French text, *and* DNA sequences, *and* legal boilerplate — a phenomenon called **polysemanticity**.

The leading explanation is **superposition**: a model wants to represent far more concepts than it has neurons, so it stores concepts as *directions* in activation space rather than as individual neurons, and packs many nearly-orthogonal directions into the same space. Each neuron then participates in many concepts, and each concept spreads across many neurons.

Consequence: to understand the model, you need a tool that un-mixes these directions.

## The Tool: Sparse Autoencoders

A **sparse autoencoder (SAE)** is that un-mixing tool. It's a small network trained on a frozen model's internal activations with two competing objectives:

1. **Reconstruction** — its output must faithfully reproduce the input activation
2. **Sparsity** — only a handful of its (many) internal units may be active at once

The SAE has far more internal units than the model has dimensions — typically 8–32× wider ("expansion factor"). Under sparsity pressure, the training converges on a dictionary where each unit captures one recurring direction in the activations. These units are the **features**, and empirically they tend to be **monosemantic**: one fires for expressions of love, another for Python decorators, another for hedging language.

The two objectives trade off directly — perfect reconstruction with no sparsity learns nothing interpretable; extreme sparsity destroys fidelity. That trade-off is what all the [training hyperparameters](/core-workflow/sae-training) tune, and different SAE architectures (L1 penalty, JumpReLU thresholds, TopK) are just different mechanisms for enforcing sparsity.

## Reading Features

A trained SAE gives you a feature dictionary, but each feature is still just "unit #7142 fires sometimes." Making it meaningful takes evidence:

- **Top activating examples** — the dataset snippets where the feature fires hardest. This is what [feature extraction](/core-workflow/feature-extraction) computes, and it's the primary evidence for what a feature detects.
- **Logit lens** — projecting the feature's direction onto the model's vocabulary shows which tokens it *promotes* or *suppresses* when active.
- **Labels** — an LLM (or you) reads the evidence and writes a description. That's [auto-labeling](/core-workflow/auto-labeling) and [enhanced labeling](/core-workflow/enhanced-labeling).

:::tip A healthy skepticism
A label is a hypothesis, not ground truth. The activation examples are the evidence — when a label surprises you, read the examples before believing it.
:::

## The Payoff: Steering as Proof

The strongest evidence that a feature means what you think comes from **intervention**: add the feature's direction into the model's activations during generation and watch the output change.

If the "politeness" feature is real, amplifying it should make generations measurably more polite — and suppressing it, less. This is exactly what the [Steering panel](/core-workflow/steering) does, with side-by-side steered vs. unsteered generations. Steering upgrades interpretability from *correlation* ("this fires on polite text") to *causation* ("this direction produces politeness").

## The Full Loop

```
train SAE on activations  →  extract top examples per feature
        →  label features  →  steer with features to verify
```

That loop is miStudio's core workflow, one panel per stage. The [Researcher's Journey](/core-workflow/researcher-journey) walks it end-to-end, and the [Quickstart Tutorial](/getting-started/quickstart-tutorial) runs it on a small model in about half an hour.

## Going Deeper

- *Toy Models of Superposition* (Anthropic, 2022) — the superposition hypothesis
- *Towards Monosemanticity* (Bricken et al., 2023) — the SAE approach miStudio's "Standard" framework implements
- *Scaling Monosemanticity* (Templeton et al., 2024) — SAEs on production-scale models
- *Gemma Scope* (Rajamanoharan et al., 2024) — the JumpReLU architecture and the public SAE suite you can [import directly](/advanced/external-saes)
