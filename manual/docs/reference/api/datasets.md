---
sidebar_position: 2
title: "Datasets API"
description: "Dataset download, tokenization, and inspection endpoints"
---

# Datasets API

Prefix: `/api/v1/datasets` · UI: [Dataset Management](/core-workflow/dataset-management)

## Datasets

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `` | Create a dataset record (201) |
| `GET` | `` | List datasets (paginated) |
| `GET` | `/{id}` | Get dataset details |
| `PATCH` | `/{id}` | Update dataset metadata / filter settings |
| `DELETE` | `/{id}` | Delete dataset — cascades to its tokenizations (204) |
| `POST` | `/download` | Start a HuggingFace download (202); body includes `repo_id`, optional `split`/`config` |
| `DELETE` | `/{id}/cancel` | Cancel an in-progress download. During download: removes partial files; during processing: preserves raw files for retry |
| `GET` | `/{id}/task-status` | Status of the dataset's current background task |
| `GET` | `/{id}/samples` | Paginated raw samples (bytes fields sanitized to strings) |

## Tokenizations

A dataset has many tokenizations — unique per `(dataset, model, max_length)`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/{id}/tokenize` | Start a tokenization job (202); body: `model_id`, `max_length`, filter options |
| `POST` | `/tokenize-preview` | Preview tokenization of sample text without running a job |
| `GET` | `/{id}/tokenizations` | List the dataset's tokenizations |
| `GET` | `/{id}/tokenizations/{model_id}` | Get the tokenization for a specific model |
| `POST` | `/{id}/tokenizations/{tok_id}/cancel` | Cancel a running tokenization |
| `DELETE` | `/{id}/tokenizations/{tok_id}` | Delete one tokenization (204) |
| `DELETE` | `/{id}/tokenization` | Clear tokenization state (legacy single-tokenization path) |

**Progress channels:** `datasets/{id}/progress` (events `dataset:progress|completed|error`) and `datasets/{id}/tokenization/{tok_id}` (events `tokenization:progress|status`) — see the [WebSocket reference](/reference/websocket-channels).
