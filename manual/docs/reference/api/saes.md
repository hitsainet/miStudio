---
sidebar_position: 5
title: "SAEs API"
description: "SAE download, import, upload, and feature-extraction endpoints"
---

# SAEs API

Prefix: `/api/v1/saes` · UI: [SAE Management](/advanced/external-saes)

## Browse & acquire

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `` | List all SAEs (trained, downloaded, imported) |
| `GET` | `/{id}` | Get SAE details |
| `POST` | `/hf/preview` | Preview a HuggingFace repo's SAE files (grouped by directory) before downloading |
| `POST` | `/download` | Download SAE(s) from HuggingFace; supports multi-select |
| `POST` | `/upload` | Upload a trained SAE to HuggingFace (uses `layer_XX/width_{n}k/` layout) |
| `GET` | `/training/{training_id}/available` | SAEs a completed training produced that can be imported (already-imported ones flagged) |
| `POST` | `/import/training` | Import SAE(s) from a completed training |
| `POST` | `/import/file` | Import an SAE directory from local disk |

## Delete

| Method | Path | Description |
|--------|------|-------------|
| `DELETE` | `/{id}` | Delete an SAE. `?delete_files=true` (default) is a **hard delete** that cascades to extracted features; `?delete_files=false` is a reversible soft delete. `?force=true` unbinds any cluster profiles bound to this SAE and deletes anyway |
| `POST` | `/delete` | Batch delete — body is a list of SAE IDs |

:::info Bound cluster profiles (409)
If cluster profiles are bound to the SAE, `DELETE /{id}` returns **409** with a structured body `{ "code": "PROFILES_BOUND", "profile_count": <n>, "message": … }`. Delete those profiles first, or retry with `?force=true` — force **unbinds** the profiles (they survive as unbound and are steerable again after re-binding) rather than destroying user-authored work.
:::

## Feature extraction (Stage 2)

Runs the *SAE* over activations to find each feature's top examples — see [the extraction pipeline](/concepts/extraction-pipeline).

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/{id}/extract-features` | Start feature extraction for this SAE |
| `GET` | `/{id}/extraction-status` | Current extraction status |
| `POST` | `/{id}/cancel-extraction` | Cancel a running extraction |
| `POST` | `/batch-extract-features` | Queue feature extraction for multiple SAEs |
| `GET` | `/{id}/features` | Browse the SAE's extracted features |

**Progress channels:** `sae/{id}/download`, `sae/{id}/upload`, `sae/{id}/extraction`.
