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

#### `POST /{id}/extract-features`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_ids` | `string[]` | — | Corpora to draw samples from. All must share the same `max_length`. |
| `dataset_weights` | `number[]` | equal | Share of the evaluation samples per corpus, **positional over `dataset_ids`**. Normalised server-side. |
| `evaluation_samples` | `int` | 10,000 | Rows to scan (100 – 1,000,000). |
| `top_k_examples` | `int` | 100 | Examples kept per feature (10 – 1,000). |
| `min_activation_frequency` | `float` | 0.001 | Below this, a feature is dropped as dead. |
| `context_prefix_tokens` | `int` | 25 | Tokens before the peak (0 – 50). |
| `context_suffix_tokens` | `int` | 25 | Tokens after the peak (0 – 50). |
| `filter_special` · `filter_single_char` · `filter_punctuation` · `filter_numbers` · `filter_fragments` · `filter_stop_words` | `bool` | all `true` | Applied to the **prime token**; a filtered prime discards the whole example. |
| `gpu` | `string` | `auto` | `auto`, or a GPU UUID. Resolved to a UUID at submit and recorded. |
| `auto_nlp` | `bool` | `false` | Run NLP analysis when extraction completes. |

`dataset_id` remains accepted as a query parameter for the single-corpus form and is ignored when
`dataset_ids` is supplied. Every job records both, so a single-corpus request and a one-element
mixture are stored identically.

`batch-extract-features` takes the same fields plus `sae_ids`, and queues one job per SAE.
