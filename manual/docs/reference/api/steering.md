---
sidebar_position: 7
title: "Steering API"
description: "Async steering generation, mode control, and saved experiments"
---

# Steering API

Prefix: `/api/v1/steering` · UI: [Model Steering](/core-workflow/steering)

## Async generation (the primary interface)

Steering runs as background tasks with WebSocket progress — generations can take minutes for long outputs.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/async/compare` | One feature at multiple strengths, or several features side-by-side. Returns a `task_id` (202-style) |
| `POST` | `/async/sweep` | Strength sweep across a range for dose-response analysis |
| `POST` | `/async/combined` | All selected features applied simultaneously |
| `GET` | `/async/result/{task_id}` | Poll for the finished result |
| `DELETE` | `/async/task/{task_id}` | Cancel a running task |

**Parameter ranges:** `strength` ∈ [-300, +300] (raw residual-stream coefficients, Neuronpedia-compatible), `max_new_tokens` ∈ [1, 2048], `temperature` ∈ [0, 2], `top_p` ∈ [0, 1].

:::note Removed sync endpoints
`POST /compare` and `POST /sweep` (the old synchronous API) return **410 Gone**. If you hit them from the UI, hard-refresh the browser to load the current frontend.
:::

## Mode & operations

Steering mode pre-loads the model+SAE onto the GPU for low-latency interactive use.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/mode` | Is steering mode active? |
| `POST` | `/enter-mode` / `/exit-mode` | Enter/exit steering mode (loads/frees GPU memory) |
| `GET` | `/status` | Steering service health/resilience state |
| `POST` | `/reset` | Reset the resilience circuit-breaker |
| `POST` | `/cleanup` | Force-release steering GPU memory |

## Saved experiments

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/experiments` | List saved experiments |
| `POST` | `/experiments` | Save a steering result as an experiment |
| `GET` | `/experiments/{id}` | Get one experiment |
| `DELETE` | `/experiments/{id}` | Delete one |
| `POST` | `/experiments/delete` | Batch delete |

**Progress channel:** `steering/{task_id}` (events `steering:progress|completed|failed`).
