---
sidebar_position: 1
title: "Introduction to miWorkbench"
description: "What miWorkbench is and why it exists"
---

# Introduction to miWorkbench

miWorkbench is an end-to-end mechanistic interpretability platform designed to replace the fragmented tooling typically associated with AI safety research — Jupyter notebooks, custom scripts, and manually tracked experiments — with a professional, database-backed workbench.

By providing a unified environment for data management, SAE training, feature discovery, and causal intervention testing, miWorkbench allows researchers to move from hypothesis to proven intervention in a fraction of the time required by traditional methods.

## The Scalability Spectrum: Edge to Cluster

miWorkbench is engineered with a "scale-agnostic" architecture:

- **At the Edge:** Run on an NVIDIA Jetson Orin or a laptop with a single RTX 3060. The software optimizes memory via quantization (4-bit, 8-bit) and micro-batching for 1B–3B parameter models.
- **In the Lab:** Deploy on multi-GPU workstations. miWorkbench detects CUDA devices and distributes extraction and training jobs across all available VRAM via its Celery/Redis task queue.
- **In the Cloud:** Deploy on GCP/AWS with Kubernetes for auto-scaling GPU access.

:::info Why Not Notebooks?
Jupyter notebooks suffer from "hidden state" — results depend on cell execution order. miWorkbench enforces structured workflows where every experiment is recorded in PostgreSQL with exact hyperparameters, making research reproducible by default.
:::

:::tip The Superposition Hypothesis
Modern LLMs represent more concepts than they have neurons through **superposition** — neurons become "polysemantic," firing for unrelated concepts. Sparse Autoencoders (SAEs) "unpack" these neurons into individual, monosemantic features. miWorkbench is the workbench for this science.
:::
