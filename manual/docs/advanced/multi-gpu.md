---
sidebar_position: 3
title: "Multi-GPU Support"
description: "GPU monitoring, extraction routing, and the multi-GPU roadmap"
---

# Multi-GPU Support

miStudio runs happily on a single GPU, but on multi-GPU hosts it can monitor every device and route heavy extraction work to a GPU of your choice. This page describes what works today and what is planned — honestly.

## What Works Today

### Per-GPU monitoring

The [Monitor page](/getting-started/dashboard) discovers all CUDA devices and streams per-GPU metrics (utilization, memory, temperature, power) every 2 seconds over WebSocket. A **view-mode toggle** switches between an aggregated view and per-GPU charts.

Under the hood, each GPU has its own WebSocket channel (`system/gpu/{id}` — see the [WebSocket reference](/reference/websocket-channels)), and the REST API exposes `gpu-list`, `gpu-metrics`, `gpu-info`, and `gpu-processes` endpoints for scripting.

![Monitor page — live system resources alongside per-GPU utilization, memory, temperature, power, and device info](/img/miStudio_Monitor_Panel-Resources.jpg)

### Extraction GPU routing

Activation extraction jobs accept a **`gpu_id`** — select which device runs the model forward passes. This lets you keep GPU 0 free for interactive steering while a long extraction saturates GPU 1. Each job validates the index against the discovered device list and cleans up its own device memory when finished.

### GPU watchdog

A background watchdog task (scheduled via Celery Beat) monitors for zombie GPU processes — for example, a steering worker that died without releasing memory — and cleans them up. This is what keeps long-running deployments from slowly leaking VRAM.

### Work partitioning via Celery

Even without explicit device selection everywhere, the Celery/Redis backbone gives you practical multi-GPU workflows:

- **Inference partition:** one GPU dedicated to the base model for interactive steering
- **Batch partition:** other GPUs consumed by queued extraction jobs via `gpu_id`

:::info The Celery/Redis Backbone
GPU jobs can take hours or days. Redis stores the task queue; Celery workers execute tasks. Queue 5 extraction runs, close your browser, and check results in the morning — research continues on the backend.
:::

## What Is Planned (Not Yet Implemented)

For transparency, these items appear in roadmap documents but are **not** in the current release:

| Capability | Status |
|-----------|--------|
| GPU selection for **training** jobs | Planned — training currently runs on the default CUDA device |
| Distributed data-parallel (DDP/NCCL) training across GPUs | Planned |
| Persisted GPU metrics history in the database | Planned — metrics are currently live-stream only |
| Multi-GPU steering | Planned — steering is pinned to GPU 0 |

If you need training on a specific device today, the workaround is setting `CUDA_VISIBLE_DEVICES` on the Celery worker process before starting it.
