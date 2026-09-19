---
sidebar_position: 5
title: "Multi-Dataset Training"
description: "Train SAEs on multiple datasets or cached activations"
---

# Multi-Dataset Training

SAE features reflect their training data: an SAE trained only on children's stories will have wonderful "narrative" features and nothing for code. Training across multiple datasets produces more robust, general features.

## Selecting Multiple Datasets

The training configuration's dataset picker is multi-select. Choose any number of datasets, and the trainer interleaves their tokenized sequences during training.

**The compatibility rule:** every selected dataset must have a tokenization for the *training model* at the *same max length*. The picker only offers compatible tokenizations — if a dataset you expect is missing, go to [Dataset Management](/core-workflow/dataset-management) and tokenize it for that model first.

## Training from Cached Activations

Multi-dataset support extends to **cached activations**: instead of raw tokenized text, a training job can consume the output of one or more prior [activation extractions](/core-workflow/feature-extraction).

Why this matters:

- **Skip redundant forward passes.** Extracting activations is the expensive part (running the base model). If you already extracted layer-12 activations for an experiment, every subsequent SAE trained on that layer reuses them — the base model never loads.
- **Mix sources at the activation level.** Combine cached activations from several extractions (different datasets, same model+layer+hook) into a single training run.
- **Iterate on SAE hyperparameters cheaply.** Sweeping `l1_alpha` across 5 trainings against the same cached activations costs a fraction of 5 full pipelines.

Select **cached activations** as the data source in the training config and pick one or more completed extractions. The same compatibility rule applies: extractions must share the model, layer, and hook type.

## Practical Recipe

A common robust-features setup:

1. Download 2–3 diverse datasets (e.g., web text + code + dialogue)
2. Tokenize each for your model at the same max length
3. Run one activation extraction per dataset at your target layer
4. Train the SAE from the combined cached activations
5. Sweep sparsity settings against the same cache — no re-extraction needed
