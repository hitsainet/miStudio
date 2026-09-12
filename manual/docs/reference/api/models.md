---
sidebar_position: 3
title: "Models API"
description: "Model download, architecture, and activation-extraction endpoints"
---

# Models API

Prefix: `/api/v1/models` · UI: [Model Management](/core-workflow/data-model-management)

## Model lifecycle

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/download` | Download a model from HuggingFace (202); body: `repo_id`, `quantization`, `trust_remote_code` |
| `POST` | `/{id}/redownload` | Re-download / re-quantize an existing model (202) |
| `DELETE` | `/{id}/cancel` | Cancel an in-progress download (revokes the running task) |
| `GET` | `` | List models |
| `GET` | `/local-cache/list` | Enumerate models present in the local HF cache |
| `GET` | `/{id}` | Get model details |
| `GET` | `/{id}/architecture` | Discovered layer/hook structure (dynamic layer discovery output) |
| `PATCH` | `/{id}` | Update model metadata |
| `DELETE` | `/{id}` | Delete model (204). Returns **409** if a training references it |
| `GET` | `/tasks/{task_id}` | Raw Celery task status |

## Activation extraction (Stage 1)

These run the *base model* over a tokenized dataset and cache raw activations — see [the extraction pipeline](/concepts/extraction-pipeline) for how this differs from SAE feature extraction.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/{id}/estimate-extraction` | Estimate VRAM/disk for an extraction config before running it |
| `POST` | `/{id}/extract-activations` | Start activation extraction (202); config includes layers, hook types, and optional `gpu_id` |
| `GET` | `/{id}/extractions` | List the model's activation extractions |
| `GET` | `/{id}/extractions/active` | The currently running extraction, if any (200 with `null` data when idle) |
| `POST` | `/{id}/extractions/{eid}/cancel` | Cancel a running extraction |
| `POST` | `/{id}/extractions/{eid}/retry` | Retry a failed extraction |
| `DELETE` | `/{id}/extractions` | Delete the model's extraction records/artifacts |

**Progress channels:** `models/{id}/progress` (events `model:progress|completed|error`) and `models/{id}/extraction`.
